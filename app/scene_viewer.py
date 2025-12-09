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
    
    # Models known to cause crashes or issues
    BLACKLISTED_MODELS = [
        'Wizard.obj',
        'Wizard',
        'teapot_simple.obj',
        'teapot_simple',
        #'Christmas Tree.obj',
        #'Christmas Tree',
    ]
    
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
        try:
            text = (text or "").strip()
            if not text:
                self._emit_status("⚠️ Please enter a scene description")
                return
            
            self.debug_mode = debug_enabled
            self._emit_status("🔍 Analyzing scene description...")
            
            self._ensure_engines()
            
            self._emit_status("🧠 Performing semantic analysis...")
            scene_objects = self._scene_brain.parse_scene_description(text)
            if not scene_objects:
                self._emit_status("⚠️ No objects recognized; please try a more specific description")
                return
            summary = self._scene_brain.get_scene_summary(scene_objects)
            self._emit_status(f"✅ {summary}")
            
            self._emit_status(f"📐 Running force-directed layout algorithm ({len(scene_objects)} objects)...")
            models_dir = self._get_models_dir()
            model_dimensions = self._get_model_dimensions(scene_objects, models_dir)
            semantic_vectors = self._get_model_embeddings(scene_objects)
            self._scene_nodes = self._layout_engine.layout_scene(scene_objects, model_dimensions, semantic_vectors)
            
            iteration_history = self._layout_engine.debug_data.get('iteration_history', []) if debug_enabled else []
            
            self._emit_status("🎨 Rendering 3D scene...")
            if iteration_history:
                self._emit_status("🎞️ Playing back force-directed iterations...")
                self._play_iteration_history(models_dir, iteration_history, summary, debug_enabled)
            else:
                self._render_final_scene(models_dir, summary, debug_enabled)
        except Exception as e:
            import traceback
            traceback.print_exc()
            self._emit_status(f"❌ Scene generation failed: {e}")
    
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
            try:
                self._render_node(node, models_dir)
            except Exception as e:
                print(f"[SceneViewer] render_node failed for {node.model_id}: {e}")
                continue
        
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
        # Skip blacklisted models
        if node.filename in self.BLACKLISTED_MODELS or node.model_id in self.BLACKLISTED_MODELS:
            print(f"[SceneViewer] Skipping render for blacklisted model: {node.filename}")
            return

        # Build model path; fetch via data_manager if missing locally
        model_path = models_dir / node.filename

        if not model_path.exists() and self.data_manager:
            try:
                dm_path = self.data_manager.get_model_path(node.model_id)
                if dm_path:
                    model_path = Path(dm_path)
                else:
                    failed = getattr(self.data_manager, "_failed_downloads", None)
                    if failed is not None and node.model_id in failed:
                        print(f"[SceneViewer] Skipping render for failed model download: {node.model_id}")
                        return
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

        self._add_mesh_safe(mesh, node)

    def _add_mesh_safe(self, mesh: pv.PolyData, node: SceneNode):
        """Normalize, transform, and add mesh with exception safety."""
        # === Geometry normalization ===
        try:
            # Pass display_name for semantic scaling
            mesh = self._normalize_mesh(mesh, node.placement_type, node.display_name)
        except Exception as e:
            print(f"[SceneViewer] normalize mesh failed: {e}")
        
        # Apply transform
        try:
            transformed_mesh = self._apply_transform(mesh, node.transform)
        except Exception as e:
            print(f"[SceneViewer] apply transform failed: {e}")
            transformed_mesh = mesh

        # Add to scene with a unified base material (no extra textures)
        actor_name = getattr(node, "instance_id", None) or node.display_name or node.model_id
        try:
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
        except Exception as e:
            print(f"[SceneViewer] add_mesh failed for {actor_name}: {e}")
    
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

    def _normalize_mesh(self, mesh: pv.PolyData, placement_type: str, display_name: str = "") -> pv.PolyData:
        """
        Geometry normalization and auto-scaling.
        
        1) Move mesh center to origin
        2) For ground/character/prop, align bottom to Y=0 to avoid floating/penetration
        3) Normalize scale to a unit size (max_dim=1.0) and then apply semantic scaling
        """
        norm_mesh = mesh.copy()
        
        # 1. Center geometry
        center = norm_mesh.center
        norm_mesh.points -= center
        
        # 2. Normalize scale (Unit Box Normalization)
        # This fixes "abnormally large" models by pulling everything to 1.0 first
        bounds = norm_mesh.bounds
        size = np.array([
            bounds[1] - bounds[0],
            bounds[3] - bounds[2],
            bounds[5] - bounds[4]
        ])
        max_dim = np.max(size)
        
        if max_dim > 0:
            scale_factor = 1.0 / max_dim
            norm_mesh.points *= scale_factor
            
            # 3. Apply semantic scaling (Heuristic based on object type)
            # Since everything is now 1.0, we scale it to a "Reasonable" size in meters
            semantic_scale = self._get_semantic_scale(display_name)
            norm_mesh.points *= semantic_scale

        # 4. Ground alignment (after scaling)
        if placement_type in ['ground', 'character', 'prop']:
            y_min = norm_mesh.bounds[2]
            norm_mesh.points[:, 1] -= y_min

        return norm_mesh

    def _get_semantic_scale(self, display_name: str) -> float:
        """
        Get expected size (max dimension in meters) based on object name.
        """
        name = display_name.lower()
        
        if any(k in name for k in ['tree']):
            return 4.0
        if any(k in name for k in ['house', 'building', 'castle']):
            return 8.0
        if any(k in name for k in ['car', 'truck', 'vehicle']):
            return 3.0
        if any(k in name for k in ['person', 'soldier', 'knight', 'wizard', 'man', 'woman']):
            return 1.8
        if any(k in name for k in ['table', 'desk']):
            return 1.5
        if any(k in name for k in ['chair', 'bench']):
            return 1.0
        if any(k in name for k in ['rock', 'stone']):
            return 1.2
        if any(k in name for k in ['cup', 'mug', 'pot', 'bottle', 'lamp']):
            return 0.4
            
        return 1.2  # Default size


    def _load_mesh_normalized_for_dimension(self, model_path: Path, placement_type: str, display_name: str = "") -> Optional[pv.PolyData]:
        """
        为尺寸测量加载并归一化 mesh，确保 obj / glb 使用同一流程
        """
        mesh = self.load_model_file(str(model_path))
        if mesh is None:
            return None
        # Use the same normalization as rendering (unit+semantic scaling + ground align)
        return self._normalize_mesh(mesh, placement_type, display_name)
    
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
    
    # ===== 辅助：模型目录、尺寸 =====
    def _get_models_dir(self) -> Path:
        """Get the models directory path."""
        return Path(__file__).parent.parent / 'assets' / 'models'
    
    def _get_model_dimensions(self, scene_objects: List["SceneObject"], models_dir: Path) -> dict:
        """
        Measure real model dimensions, preferring actual mesh.bounds.
        To keep obj / glb consistent, measurement uses the same load + normalize flow:
        1) load_model_file (PyVista)
        2) _normalize_mesh (center to origin, drop to ground)
        3) Compute size from normalized bounds

        Large or unavailable models are skipped and will use fallback sizes to avoid crashes.
        """
        import numpy as np
        dimensions: Dict[str, np.ndarray] = {}
        print("📏 Measuring actual model dimensions...")

        for obj in scene_objects:
            if obj.model_id in dimensions:
                continue
            
            # Skip blacklisted models
            if obj.filename in self.BLACKLISTED_MODELS or obj.model_id in self.BLACKLISTED_MODELS:
                print(f"[SceneViewer] Skipping blacklisted model: {obj.filename}")
                continue

            model_path = models_dir / obj.filename

            # If missing locally, try fetching via data_manager
            if (not model_path.exists()) and self.data_manager:
                try:
                    dm_path = self.data_manager.get_model_path(obj.model_id)
                    if dm_path:
                        model_path = Path(dm_path)
                    else:
                        failed = getattr(self.data_manager, "_failed_downloads", None)
                        if failed is not None and obj.model_id in failed:
                            continue
                except Exception as e:
                    print(f"[SceneViewer] fetch model for dimension failed (id={obj.model_id}): {e}")

            if model_path.exists():
                try:
                    # Pass display_name to ensure correct semantic scaling
                    mesh = self._load_mesh_normalized_for_dimension(model_path, obj.placement_type, obj.display_name)
                    
                    if mesh is not None:
                        bounds = mesh.bounds
                        size = np.array([
                            bounds[1] - bounds[0],
                            bounds[3] - bounds[2],
                            bounds[5] - bounds[4],
                        ])
                        dimensions[obj.model_id] = size
                        continue
                except Exception as e:
                    print(f"[SceneViewer] dimension fallback for {obj.model_id}: {e}")

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

    def _get_model_embeddings(self, scene_objects: List["SceneObject"]) -> Dict[str, np.ndarray]:
        """
        Fetch normalized semantic embeddings for the given scene objects (best-effort).
        """
        if not self.data_manager:
            return {}
        try:
            model_ids = list({obj.model_id for obj in scene_objects})
            return self.data_manager.get_model_embeddings(model_ids)
        except Exception as e:
            print(f"[SceneViewer] semantic embedding fetch failed: {e}")
            return {}

