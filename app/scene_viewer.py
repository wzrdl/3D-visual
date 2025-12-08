"""
SceneViewer - Scene Visualizer

PyVista-based renderer that supports:
1. Multi-object scene rendering
2. Debug visualization (AABB boxes, parent-child lines, grid)
3. Camera controls

Extends ThreeDViewer with scene rendering features.
"""

import copy
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Callable

import numpy as np
import pyvista as pv
from pyvistaqt import QtInteractor

from app.layout_engine import SceneNode, Transform, LayoutEngine
from app.scene_brain import SceneBrain, SceneObject
from app.viewer import ThreeDViewer


class SceneViewer(QtInteractor):
    """
    Scene visualizer.

    Features:
    1. Render multi-object 3D scenes
    2. Show debug info (AABB, connection lines, etc.)
    3. Support scene screenshots
    """
    
    # Debug visualization colors
    DEBUG_COLORS = {
        'aabb': '#00FF00',           # AABB box - green
        'parent_child_line': '#FF0000',  # Parent-child line - red
        'center_point': '#0000FF',   # Center point - blue
        'grid': '#888888',           # Grid - gray
    }
    
    def __init__(self, parent=None, data_manager=None):
        """
        Args:
            parent: Qt parent
            data_manager: optional ClientDataManager for on-demand model download
        """
        super().__init__(parent)
        self._setup_renderer()
        
        # Store loaded mesh actors
        self._scene_actors: List[Any] = []
        self._debug_actors: List[Any] = []
        
        # Debug mode toggle
        self.debug_mode = False
        
        # Model cache
        self._model_cache: Dict[str, pv.PolyData] = {}
        # Optional data manager for fetching models on demand
        self.data_manager = data_manager
        
        # Scene generation state
        self._scene_brain: Optional[SceneBrain] = None
        self._layout_engine: Optional[LayoutEngine] = None
        self._scene_nodes: List[SceneNode] = []
        self._iteration_timer = None
        self._iteration_step = 0
        self._status_callback: Optional[Callable[[str], None]] = None
        
    def _setup_renderer(self):
        """Configure renderer"""
        self.background_color = '#1a1a2e'  # Dark background
        self.enable_trackball_style()
        self.show_axes()
    
    # ===== 场景生成入口 =====
    def set_status_callback(self, cb: Callable[[str], None]):
        """注册状态更新回调（由上层页面传入）"""
        self._status_callback = cb
    
    def _emit_status(self, text: str):
        if self._status_callback:
            self._status_callback(text)
        else:
            print(text)
    
    def _ensure_engines(self):
        """Lazy-load SceneBrain / LayoutEngine"""
        if self._scene_brain is None:
            self._scene_brain = SceneBrain(data_manager=self.data_manager)
        if self._layout_engine is None:
            self._layout_engine = LayoutEngine()
    
    def generate_scene(self, text: str, debug_enabled: bool = False):
        """
        External entry: generate and render scene (with optional iteration playback)
        """
        text = (text or "").strip()
        if not text:
            self._emit_status("⚠️ Please enter a scene description")
            return
        
        self.debug_mode = debug_enabled
        self._emit_status("🔍 Analyzing scene description...")
        
        self._ensure_engines()
        
        is_forest = self._is_forest_preset(text)
        if is_forest:
            self._emit_status("🌲 Loading preset scene forest ...")
            scene_objects = self._build_forest_scene()
            summary = f"Forest preset: {len(scene_objects)} assets from testmodel"
            self._layout_engine.iterations = max(self._layout_engine.iterations, 120)
        else:
            self._emit_status("🧠 Performing semantic analysis...")
            scene_objects = self._scene_brain.parse_scene_description(text)
            if not scene_objects:
                self._emit_status("⚠️ No objects recognized; please try a more specific description")
                return
            summary = self._scene_brain.get_scene_summary(scene_objects)
            self._emit_status(f"✅ {summary}")
        
        self._emit_status(f"📐 Running force-directed layout algorithm ({len(scene_objects)} objects)...")
        models_dir = self._get_models_dir(use_testmodel=is_forest)
        model_dimensions = self._get_model_dimensions(scene_objects, models_dir)
        self._scene_nodes = self._layout_engine.layout_scene(scene_objects, model_dimensions)
        
        iteration_history = self._layout_engine.debug_data.get('iteration_history', []) if debug_enabled else []
        
        self._emit_status("🎨 Rendering 3D scene...")
        if iteration_history:
            self._emit_status("🎞️ Playing back force-directed iterations...")
            self._play_iteration_history(models_dir, iteration_history, summary, debug_enabled)
        else:
            self._render_final_scene(models_dir, summary, debug_enabled)
    
    def load_model_file(self, file_path: str) -> Optional[pv.PolyData]:
        """
        Load a model file and cache it.

        Args:
            file_path: Model file path

        Returns:
            PyVista PolyData mesh or None
        """
        # Check cache first
        if file_path in self._model_cache:
            return self._model_cache[file_path].copy()
        
        try:
            mesh = ThreeDViewer.load_model(file_path)
            mesh = self._ensure_polydata(mesh)
            self._model_cache[file_path] = mesh
            return mesh.copy()
        except Exception as e:
            print(f"[SceneViewer] Error loading model '{file_path}': {e}")
            return None

    def _ensure_polydata(self, mesh: pv.DataSet) -> pv.PolyData:
        """
        Ensure a PolyData is returned.
        - If MultiBlock: merge_blocks; on failure pick the first geometry block.
        - If already PolyData: return directly.
        """
        if isinstance(mesh, pv.MultiBlock):
            try:
                merged = pv.merge_blocks(mesh)
                return merged.extract_geometry()
            except Exception:
                # Fallback: pick the first usable geometry block
                for block in mesh:
                    if block is None:
                        continue
                    try:
                        return block.extract_geometry()
                    except Exception:
                        continue
                # If everything fails, create a placeholder
                return pv.PolyData()
        if isinstance(mesh, pv.PolyData):
            return mesh
        try:
            return mesh.extract_geometry()
        except Exception:
            return pv.PolyData()
    
    def render_scene(
        self,
        scene_nodes: List[SceneNode],
        models_dir: Path,
        debug_data: Optional[Dict] = None
    ):
        """
        Render the full scene.

        Args:
            scene_nodes: Scene nodes (from LayoutEngine)
            models_dir: Model directory
            debug_data: Debug data (optional)
        """
        # Clear current scene
        self.clear_scene()
        
        # Collect all nodes including children
        all_nodes = self._flatten_nodes(scene_nodes)
        
        # Render each node
        for node in all_nodes:
            self._render_node(node, models_dir)
        
        # Render debug info if enabled
        if self.debug_mode and debug_data:
            self._render_debug_info(debug_data)
        
        # Add ground grid
        self._add_ground_grid()
        
        # Position the camera
        self._setup_camera(all_nodes)
        
        # Add lighting
        self._setup_lighting()
    
    def _render_final_scene(self, models_dir: Path, summary: str, debug_enabled: bool):
        """Render final scene and update status"""
        self.render_scene(
            self._scene_nodes,
            models_dir,
            self._layout_engine.debug_data if debug_enabled else None
        )
        flat_nodes = self._layout_engine.get_all_nodes_flat(self._scene_nodes)
        self._emit_status(
            f"✅ Scene generation complete!\n"
            f"📊 {summary}\n"
            f"🔢 {len(flat_nodes)} model instances in total"
        )
    
    def render_iteration_snapshot(
        self,
        scene_nodes: List[SceneNode],
        models_dir: Path,
        positions: List[List[float]]
    ):
        """
        Render a single iteration snapshot (uses copies to avoid mutating final nodes).
        """
        # Adjust only top-level node positions; count matches recorded positions
        nodes_copy = copy.deepcopy(scene_nodes)
        for idx, pos in enumerate(positions):
            if idx < len(nodes_copy):
                nodes_copy[idx].transform.position = np.array(pos)
        
        # Use current debug_mode; do not reuse stale debug data during playback
        self.render_scene(
            nodes_copy,
            models_dir,
            None
        )
    
    def _play_iteration_history(self, models_dir: Path, iteration_history: List[Dict], summary: str, debug_enabled: bool):
        """Replay force-directed iterations using QTimer"""
        if not iteration_history:
            self._render_final_scene(models_dir, summary, debug_enabled)
            return
        
        # Stop any existing timer
        if self._iteration_timer:
            try:
                self._iteration_timer.stop()
            except Exception:
                pass
        
        from PyQt6.QtCore import QTimer  # Delayed import to avoid top-level dependency
        self._iteration_step = 0
        self._iteration_timer = QTimer(self)
        self._iteration_timer.setInterval(200)  # 200ms per frame
        
        def render_step():
            # Render final result after playback completes
            if self._iteration_step >= len(iteration_history):
                self._iteration_timer.stop()
                self._render_final_scene(models_dir, summary, debug_enabled)
                return
            
            entry = iteration_history[self._iteration_step]
            iteration_idx = entry.get('iteration', self._iteration_step)
            positions = entry.get('positions', [])
            
            self._emit_status(
                f"🎞️ Replaying iteration {iteration_idx}/{iteration_history[-1].get('iteration', len(iteration_history)-1)}"
            )
            
            try:
                self.render_iteration_snapshot(self._scene_nodes, models_dir, positions)
            except Exception as e:
                print(f"[SceneViewer] iteration playback error: {e}")
            
            self._iteration_step += 1
        
        self._iteration_timer.timeout.connect(render_step)
        render_step()  # Render the first frame immediately
        self._iteration_timer.start()
    
    def _flatten_nodes(self, nodes: List[SceneNode]) -> List[SceneNode]:
        """Flatten nested scene nodes into a list"""
        result = []
        
        def collect(node: SceneNode):
            result.append(node)
            for child in node.children:
                collect(child)
        
        for node in nodes:
            collect(node)
        
        return result
    
    def _render_node(self, node: SceneNode, models_dir: Path):
        """
        Render a single scene node.

        Applies transform (position, rotation, scale).

        Args:
            node: Scene node
            models_dir: Models directory
        """
        # Build model path; fetch via data_manager if missing locally
        model_path = models_dir / node.filename

        if not model_path.exists() and self.data_manager:
            try:
                dm_path = self.data_manager.get_model_path(node.model_id)
                if dm_path:
                    model_path = Path(dm_path)
            except Exception as e:
                print(f"[SceneViewer] Error fetching model from backend (id={node.model_id}): {e}")

        if not model_path.exists():
            print(f"[SceneViewer] Model file not found: {model_path}")
            # Create placeholder geometry
            mesh = self._create_placeholder(node.bbox_size)
        else:
            mesh = self.load_model_file(str(model_path))
            if mesh is None:
                mesh = self._create_placeholder(node.bbox_size)
        
        # === Geometry normalization ===
        try:
            mesh = self._normalize_mesh(mesh, node.placement_type)
        except Exception as e:
            print(f"[SceneViewer] normalize mesh failed: {e}")
        
        # Apply transform
        transformed_mesh = self._apply_transform(mesh, node.transform)

        # Add to scene with a unified base material (no extra textures)
        actor_name = getattr(node, "instance_id", None) or node.display_name or node.model_id
        actor = self.add_mesh(
            transformed_mesh,
            name=actor_name,  # Use unique instance ID to avoid actor overwrite
            show_edges=False,
            smooth_shading=True,
            pbr=True,  # Use PBR rendering
            metallic=0.1,
            roughness=0.5,
        )
        
        self._scene_actors.append(actor)
    
    def _create_placeholder(self, bbox_size: np.ndarray) -> pv.PolyData:
        """
        Create placeholder geometry when a model cannot be loaded.

        Args:
            bbox_size: Bounding box size

        Returns:
            PyVista Box mesh
        """
        # Build a cube with the same bbox size
        box = pv.Box(
            bounds=(
                -bbox_size[0]/2, bbox_size[0]/2,
                -bbox_size[1]/2, bbox_size[1]/2,
                -bbox_size[2]/2, bbox_size[2]/2
            )
        )
        return box
    
    def _apply_transform(self, mesh: pv.PolyData, transform: Transform) -> pv.PolyData:
        """
        Apply transform to mesh.

        Order: scale → rotate → translate.

        Args:
            mesh: Original mesh
            transform: Transform parameters

        Returns:
            Transformed mesh
        """
        result = mesh.copy()
        
        # 1. Scale
        if transform.scale != 1.0:
            result.points *= transform.scale
        
        # 2. Rotate (Euler XYZ)
        rx, ry, rz = transform.rotation
        if rx != 0:
            result.rotate_x(rx, inplace=True)
        if ry != 0:
            result.rotate_y(ry, inplace=True)
        if rz != 0:
            result.rotate_z(rz, inplace=True)
        
        # 3. Translate
        result.translate(transform.position, inplace=True)
        
        return result

    def _normalize_mesh(self, mesh: pv.PolyData, placement_type: str) -> pv.PolyData:
        """
        Geometry normalization:
        1) Move mesh center to origin
        2) For ground/character/prop, align bottom to Y=0 to avoid floating/penetration
        """
        norm_mesh = mesh.copy()
        center = norm_mesh.center
        norm_mesh.points -= center

        if placement_type in ['ground', 'character', 'prop']:
            y_min = norm_mesh.bounds[2]
            norm_mesh.points[:, 1] -= y_min

        return norm_mesh

    def _load_mesh_normalized_for_dimension(self, model_path: Path, placement_type: str) -> Optional[pv.PolyData]:
        """
        为尺寸测量加载并归一化 mesh，确保 obj / glb 使用同一流程
        """
        mesh = self.load_model_file(str(model_path))
        if mesh is None:
            return None
        return self._normalize_mesh(mesh, placement_type)
    
    def _render_debug_info(self, debug_data: Dict):
        """
        Render debug visualization information.

        Includes:
        - AABB bounding boxes (green wireframe)
        - Parent-child lines (red)
        - Center point markers (blue)

        Args:
            debug_data: Debug data dictionary
        """
        # Render AABB boxes
        aabb_boxes = debug_data.get('aabb_boxes', [])
        for aabb in aabb_boxes:
            self._render_aabb(aabb)
        
        # Render parent-child lines
        lines = debug_data.get('parent_child_lines', [])
        for line_data in lines:
            self._render_parent_child_line(line_data)
    
    def _render_aabb(self, aabb_data: Dict):
        """
        Render an AABB bounding box.

        Args:
            aabb_data: AABB data {center, size, corners}
        """
        center = np.array(aabb_data['center'])
        size = np.array(aabb_data['size'])
        
        # Create wireframe box
        box = pv.Box(
            bounds=(
                center[0] - size[0]/2, center[0] + size[0]/2,
                center[1] - size[1]/2, center[1] + size[1]/2,
                center[2] - size[2]/2, center[2] + size[2]/2
            )
        )
        
        # Add wireframe
        actor = self.add_mesh(
            box,
            style='wireframe',
            color=self.DEBUG_COLORS['aabb'],
            line_width=2,
            opacity=0.8
        )
        self._debug_actors.append(actor)
        
        # Add center point
        point = pv.PolyData([center])
        point_actor = self.add_mesh(
            point,
            color=self.DEBUG_COLORS['center_point'],
            point_size=10,
            render_points_as_spheres=True
        )
        self._debug_actors.append(point_actor)
        
        # Add label
        self.add_point_labels(
            [center + np.array([0, size[1]/2 + 0.5, 0])],
            [aabb_data.get('display_name', '')],
            font_size=12,
            text_color='white',
            shape_color='#333333',
            shape_opacity=0.7
        )
    
    def _render_parent_child_line(self, line_data: Dict):
        """
        Render a parent-child connection line.

        Args:
            line_data: Connection data {parent_pos, child_pos}
        """
        parent_pos = np.array(line_data['parent_pos'])
        child_pos = np.array(line_data['child_pos'])
        
        # Create line segment
        line = pv.Line(parent_pos, child_pos)
        
        # Add to scene
        actor = self.add_mesh(
            line,
            color=self.DEBUG_COLORS['parent_child_line'],
            line_width=3
        )
        self._debug_actors.append(actor)
        
        # Add arrow at midpoint to indicate direction
        mid_point = (parent_pos + child_pos) / 2
        direction = child_pos - parent_pos
        if np.linalg.norm(direction) > 0.1:
            direction = direction / np.linalg.norm(direction)
            arrow = pv.Arrow(
                start=mid_point - direction * 0.3,
                direction=direction,
                scale=0.5
            )
            arrow_actor = self.add_mesh(
                arrow,
                color=self.DEBUG_COLORS['parent_child_line'],
                opacity=0.8
            )
            self._debug_actors.append(arrow_actor)
    
    def _add_ground_grid(self):
        """Add ground grid"""
        # Create ground plane
        grid_size = 40
        grid = pv.Plane(
            center=(0, -0.01, 0),  # Slightly below Y=0 to avoid Z-fighting
            direction=(0, 1, 0),
            i_size=grid_size,
            j_size=grid_size,
            i_resolution=20,
            j_resolution=20
        )
        
        # Add grid lines
        actor = self.add_mesh(
            grid,
            style='wireframe',
            color=self.DEBUG_COLORS['grid'],
            line_width=1,
            opacity=0.3
        )
        self._debug_actors.append(actor)
    
    def _setup_camera(self, nodes: List[SceneNode]):
        """
        Position the camera to view the entire scene.

        Args:
            nodes: All scene nodes
        """
        if not nodes:
            self.reset_camera()
            return
        
        # 计算场景包围盒
        all_positions = [node.transform.position for node in nodes]
        positions_array = np.array(all_positions)
        
        center = np.mean(positions_array, axis=0)
        
        # 计算场景范围
        min_pos = np.min(positions_array, axis=0)
        max_pos = np.max(positions_array, axis=0)
        extent = np.max(max_pos - min_pos)
        
        # 设置相机距离
        distance = max(extent * 2, 15)
        
        # 45度俯视角
        camera_height = distance * 0.7
        camera_dist = distance * 0.7
        
        camera_pos = (
            center[0] + camera_dist,
            center[1] + camera_height,
            center[2] + camera_dist
        )
        
        self.camera_position = [
            camera_pos,
            (center[0], center[1], center[2]),
            (0, 1, 0)  # Y-up
        ]
    
    def _setup_lighting(self):
        """Configure scene lighting"""
        # Key light from upper right
        main_light = pv.Light(
            position=(20, 30, 20),
            focal_point=(0, 0, 0),
            color='white',
            intensity=1.0
        )
        self.add_light(main_light)
        
        # Fill light from left
        fill_light = pv.Light(
            position=(-15, 10, -10),
            focal_point=(0, 0, 0),
            color='#CCE0FF',
            intensity=0.4
        )
        self.add_light(fill_light)
        
        # Ambient light
        ambient_light = pv.Light(
            light_type='headlight',
            intensity=0.3
        )
        self.add_light(ambient_light)
    
    def clear_scene(self):
        """Clear the scene"""
        # Remove all actors
        self.clear()
        self._scene_actors.clear()
        self._debug_actors.clear()
    
    def toggle_debug_mode(self) -> bool:
        """
        Toggle debug mode.

        Returns:
            New debug mode state
        """
        self.debug_mode = not self.debug_mode
        return self.debug_mode
    
    def show_grid(self):
        """Show coordinate grid"""
        self.show_bounds(
            grid='back',
            location='outer',
            ticks='outside'
        )
    
    def take_scene_screenshot(self, filename: str = None) -> str:
        """
        Capture a scene screenshot.

        Args:
            filename: Output filename (optional)

        Returns:
            Screenshot file path
        """
        if filename is None:
            from datetime import datetime
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f'scene_screenshot_{timestamp}.png'
        
        # 确保输出目录存在
        output_dir = Path(__file__).parent.parent / 'assets' / 'screenshots'
        output_dir.mkdir(parents=True, exist_ok=True)
        
        output_path = output_dir / filename
        
        # 截图
        self.screenshot(str(output_path))
        
        print(f"[SceneViewer] Screenshot saved: {output_path}")
        return str(output_path)
    
    def get_camera_info(self) -> Dict:
        """获取当前相机信息"""
        pos = self.camera_position
        return {
            'position': list(pos[0]) if pos else [0, 10, 20],
            'focal_point': list(pos[1]) if pos else [0, 0, 0],
            'up_vector': list(pos[2]) if pos else [0, 1, 0]
        }
    
    # ===== 辅助：模型目录、尺寸、预设 =====
    def _get_models_dir(self, use_testmodel: bool = False) -> Path:
        base = Path(__file__).parent.parent / 'assets'
        if use_testmodel:
            return base / 'testmodel'
        return base / 'models'
    
    def _get_model_dimensions(self, scene_objects: List["SceneObject"], models_dir: Path) -> dict:
        """
        Measure real model dimensions, preferring actual mesh.bounds.
        To keep obj / glb consistent, measurement uses the same load + normalize flow:
        1) load_model_file (PyVista)
        2) _normalize_mesh (center to origin, drop to ground)
        3) Compute size from normalized bounds
        """
        import numpy as np
        dimensions: Dict[str, np.ndarray] = {}
        print("📏 Measuring actual model dimensions...")

        for obj in scene_objects:
            if obj.model_id in dimensions:
                continue

            model_path = models_dir / obj.filename

            # If missing locally, try fetching via data_manager
            if (not model_path.exists()) and self.data_manager:
                try:
                    dm_path = self.data_manager.get_model_path(obj.model_id)
                    if dm_path:
                        model_path = Path(dm_path)
                except Exception as e:
                    print(f"[SceneViewer] fetch model for dimension failed (id={obj.model_id}): {e}")

            if model_path.exists():
                mesh = self._load_mesh_normalized_for_dimension(model_path, obj.placement_type)
                if mesh is not None:
                    bounds = mesh.bounds
                    size = np.array([
                        bounds[1] - bounds[0],
                        bounds[3] - bounds[2],
                        bounds[5] - bounds[4],
                    ])
                    dimensions[obj.model_id] = size
                    continue

            # Fallback defaults
            name = obj.display_name.lower()
            if 'table' in name:
                dimensions[obj.model_id] = np.array([2.0, 1.0, 1.0])
            elif 'chair' in name:
                dimensions[obj.model_id] = np.array([0.6, 1.0, 0.6])
            elif 'tree' in name:
                dimensions[obj.model_id] = np.array([2.0, 4.0, 2.0])
            elif 'rock' in name or 'stone' in name:
                dimensions[obj.model_id] = np.array([1.0, 0.8, 1.0])
            elif any(c in name for c in ['soldier', 'knight', 'ninja', 'person']):
                dimensions[obj.model_id] = np.array([0.8, 1.8, 0.8])
            else:
                dimensions[obj.model_id] = np.array([1.0, 1.0, 1.0])
        return dimensions
    
    def _is_forest_preset(self, text: str) -> bool:
        return text.strip().lower() == "forest"
    
    def _build_forest_scene(self) -> List[SceneObject]:
        """Build scene objects using all glb files under assets/testmodel"""
        test_dir = self._get_models_dir(use_testmodel=True)
        if not test_dir.exists():
            raise FileNotFoundError(f"testmodel directory does not exist: {test_dir}")
        
        scene_objects: List[SceneObject] = []
        for glb_path in sorted(test_dir.glob("*.glb")):
            model_id = glb_path.stem
            filename = glb_path.name
            lower_name = model_id.lower()
            if "tree" in lower_name:
                placement_type = "ground"
            elif "rock" in lower_name:
                placement_type = "ground"
            else:
                placement_type = "ground"
            
            scene_obj = SceneObject(
                model_id=model_id,
                display_name=model_id,
                filename=filename,
                count=1,
                tags=["forest"],
                placement_type=placement_type,
                is_parent=False,
                parent_id=None
            )
            scene_objects.append(scene_obj)
        
        return scene_objects

