"""
LayoutEngine - Spatial Layout Algorithm Engine

Responsible for spatial computation, physics-inspired simulation, and
matrix transforms to build intelligent scene layouts.

Core algorithms:
1. Force-directed global layout - keeps multiple objects naturally spread
   out and prevents overlap
2. Anchor-based hierarchy - places children precisely relative to parents

Input: list of model IDs + SceneObject metadata
Output: transform matrices (position, rotation, scale)
"""

import math
import random
from typing import List, Dict, Tuple, Optional, Any
from dataclasses import dataclass, field
import numpy as np

from app.scene_brain import SceneObject


@dataclass
class Transform:
    """3D transform data container"""
    position: np.ndarray = field(default_factory=lambda: np.array([0.0, 0.0, 0.0]))
    rotation: np.ndarray = field(default_factory=lambda: np.array([0.0, 0.0, 0.0]))  # Euler angles (degrees)
    scale: float = 1.0
    
    def to_dict(self) -> Dict:
        """Convert to a serializable dict"""
        return {
            'pos': self.position.tolist(),
            'rot': self.rotation.tolist(),
            'scale': self.scale
        }


@dataclass
class SceneNode:
    """Scene graph node that includes transform information"""
    model_id: str
    instance_id: str
    filename: str
    display_name: str
    transform: Transform
    bbox_size: np.ndarray  # Bounding box size [length, width, height]
    placement_type: str
    children: List['SceneNode'] = field(default_factory=list)
    
    def to_dict(self) -> Dict:
        """Convert to scene-graph JSON format"""
        result = {
            'model_id': self.model_id,
            'instance_id': self.instance_id,
            'filename': self.filename,
            'display_name': self.display_name,
            'transform': self.transform.to_dict(),
            'bbox_size': self.bbox_size.tolist(),
            'placement_type': self.placement_type,
        }
        if self.children:
            result['children'] = [child.to_dict() for child in self.children]
        return result


class LayoutEngine:
    """
    Layout engine that implements automatic 3D scene layout algorithms.

    Physical intuition:
    - Treat each object as a charged particle
    - Repulsion prevents mesh intersection
    - Attraction keeps objects inside the view frustum
    """
    
    # Force-directed algorithm parameters
    DEFAULT_REPULSION_K = 80.0       # Repulsion coefficient (lower to avoid blow-ups)
    DEFAULT_ATTRACTION_K = 1.2       # Attraction coefficient (higher to pull together)
    DEFAULT_DAMPING = 0.9            # Damping factor (higher reduces oscillation)
    DEFAULT_ITERATIONS = 50          # Iteration count (higher = more evolution steps)
    DEFAULT_TIME_STEP = 0.05         # Time step (smaller = more stable)
    DEFAULT_MIN_DISTANCE = 0.5       # Minimum distance to avoid divide-by-zero
    
    # Layout space parameters
    DEFAULT_LAYOUT_RADIUS = 8.0      # Initial layout radius
    DEFAULT_SCENE_BOUNDS = 30.0      # Scene boundary
    
    # Anchor parameters
    DEFAULT_ANCHOR_PADDING = 0.3     # Padding between anchor and parent object
    
    # Default bounding box size (used when real size is unavailable)
    DEFAULT_BBOX_SIZE = np.array([1.0, 1.0, 1.0])

    # Semantic clustering parameters (cosine similarity on embeddings)
    DEFAULT_SEMANTIC_ATTRACTION_STRENGTH = 8.0
    DEFAULT_SEMANTIC_ATTRACTION_THRESHOLD = 0.45

    # ------------------ Facing helpers ------------------
    def _calculate_facing_rotation(self, position: np.ndarray, display_name: str, method: str = "auto") -> float:
        """
        Compute facing angle around Y-axis.
        - Characters/directional objects: face a fixed "viewer" direction
        - Natural objects: random heading
        """
        name = display_name.lower()
        if method == "auto":
            is_nature = any(k in name for k in ["tree", "rock", "stone", "grass", "plant"])
            is_character = any(k in name for k in ["soldier", "knight", "ninja", "zombie", "wizard", "person", "player", "npc", "car", "chair"])
            if is_nature:
                method = "random"
            elif is_character:
                method = "viewer"
            else:
                method = "viewer"

        if method == "random":
            return random.uniform(0, 360)

        if method == "viewer":
            # Define a global "viewer" direction (camera roughly at +X/+Z looking down).
            # Models face +Z by default, so rotate toward (1, 0, 1).
            direction = np.array([1.0, 0.0, 1.0])
            angle_rad = math.atan2(direction[0], direction[2])  # atan2(x, z)
            angle_deg = math.degrees(angle_rad)  # about -135 / 225 degrees
            return angle_deg + random.uniform(-10, 10)  # slight jitter

        # fallback
        return 0.0
    
    def __init__(
        self,
        repulsion_k: float = DEFAULT_REPULSION_K,
        attraction_k: float = DEFAULT_ATTRACTION_K,
        iterations: int = DEFAULT_ITERATIONS,
        layout_radius: float = DEFAULT_LAYOUT_RADIUS,
        semantic_attraction_strength: float = DEFAULT_SEMANTIC_ATTRACTION_STRENGTH,
        semantic_attraction_threshold: float = DEFAULT_SEMANTIC_ATTRACTION_THRESHOLD,
    ):
        """
        Initialize the layout engine.

        Args:
            repulsion_k: Repulsion coefficient k_rep
            attraction_k: Attraction coefficient k_att
            iterations: Number of force-directed iterations
            layout_radius: Initial layout radius R
        """
        self.repulsion_k = repulsion_k
        self.attraction_k = attraction_k
        self.iterations = iterations
        self.layout_radius = layout_radius
        self._instance_counter = 0

        # Semantic attraction controls
        self.semantic_attraction_strength = semantic_attraction_strength
        self.semantic_attraction_threshold = semantic_attraction_threshold
        
        # Storage for debug visualization artifacts
        self.debug_data: Dict[str, Any] = {
            'aabb_boxes': [],      # Axis-aligned bounding boxes
            'parent_child_lines': [],  # Parent-child connection lines
            'force_vectors': [],    # Force vectors (optional)
            'iteration_history': [],  # Iteration history (for animation)
        }
    
    def layout_scene(
        self, 
        scene_objects: List[SceneObject],
        model_dimensions: Optional[Dict[str, np.ndarray]] = None,
        semantic_vectors: Optional[Dict[str, np.ndarray]] = None,
    ) -> List[SceneNode]:
        """
        Build scene layout.

        Main entry that coordinates force-directed layout and anchor constraints.

        Args:
            scene_objects: Scene objects (from SceneBrain)
            model_dimensions: Mapping from model_id to bbox size {model_id: [L, W, H]}
            semantic_vectors: Optional mapping {model_id: embedding}; used for clustering

        Returns:
            List of SceneNode (scene graph roots)
        """
        if not scene_objects:
            return []
        
        # Reset debug data
        self._clear_debug_data()
        
        # 1. Expand objects based on count to create individual instances
        expanded_objects = self._expand_objects(scene_objects)
        
        # 2. Separate parents and children
        parent_objects = [obj for obj in expanded_objects if obj.is_parent or obj.parent_id is None]
        child_objects = [obj for obj in expanded_objects if obj.parent_id is not None]
        
        # 3. Run force-directed layout for parents/independent objects
        parent_nodes = self._force_directed_layout(parent_objects, model_dimensions, semantic_vectors)
        
        # 4. Apply anchor constraints for children and collect orphans
        orphan_nodes = self._apply_anchor_constraints(parent_nodes, child_objects, model_dimensions)
        if orphan_nodes:
            parent_nodes.extend(orphan_nodes)
        
        # 5. Apply placement_type rules
        self._apply_placement_rules(parent_nodes)
        
        # 6. Store debug data
        self._store_debug_data(parent_nodes)
        
        return parent_nodes
    
    def _expand_objects(self, scene_objects: List[SceneObject]) -> List[SceneObject]:
        """
        Expand objects list based on count.

        Example: count=3 for a tree becomes three independent tree instances.
        """
        expanded = []
        for obj in scene_objects:
            for i in range(obj.count):
                # Clone and append an index suffix for each instance
                expanded_obj = SceneObject(
                    model_id=obj.model_id,
                    display_name=f"{obj.display_name}_{i+1}" if obj.count > 1 else obj.display_name,
                    filename=obj.filename,
                    count=1,
                    tags=obj.tags.copy(),
                    placement_type=obj.placement_type,
                    is_parent=obj.is_parent,
                    parent_id=obj.parent_id
                )
                expanded.append(expanded_obj)
        return expanded
    
    def _force_directed_layout(
        self,
        objects: List[SceneObject],
        model_dimensions: Optional[Dict[str, np.ndarray]] = None,
        semantic_vectors: Optional[Dict[str, np.ndarray]] = None,
    ) -> List[SceneNode]:
        """
        Force-directed global layout.

        Physical model:
        - Repulsion: F_rep = k_rep × (1/d²) × n_ba
          Triggered when distance d < (R_a + R_b) to avoid overlap
        - Attraction: F_att = -k_att × DistanceToCenter
          Keeps objects near the scene center

        Iteration loop:
        1. Initialize: random coordinates within radius R
        2. Iterate: compute net force → update velocity → update position
        3. Stop when system energy converges

        Args:
            objects: Objects to layout
            model_dimensions: Model dimension dictionary

        Returns:
            List of SceneNode
        """
        if not objects:
            return []
        
        n = len(objects)
        model_dimensions = model_dimensions or {}

        # Normalize semantic vectors once for pairwise cosine similarity
        normalized_vectors: Optional[Dict[str, np.ndarray]] = None
        if semantic_vectors:
            normalized_vectors = {}
            for mid, vec in semantic_vectors.items():
                arr = np.array(vec, dtype=float)
                norm = np.linalg.norm(arr)
                if norm > 0:
                    normalized_vectors[mid] = arr / norm
        
        # initialize nodes
        nodes: List[SceneNode] = []
        positions = np.zeros((n, 3))
        velocities = np.zeros((n, 3))
        radii = np.zeros(n)  # Effective radii for collision detection

        # === 定义聚落中心 (Cluster Centers) ===
        # 这一步是为了让同类物体一开始就聚在一起，而不是散落在地图各处
        # key: model_id, value: (angle_base, radius_base)
        cluster_bases = {}

        for i, obj in enumerate(objects):
            # Get bounding box size
            bbox = model_dimensions.get(obj.model_id, self.DEFAULT_BBOX_SIZE.copy())
            if isinstance(bbox, list):
                bbox = np.array(bbox)

            # === 智能分区初始化 ===
            name = obj.display_name.lower()

            # 1. 识别物体角色
            # Hero: 核心物体 (房子, 桌子, 篝火, 汽车) -> 放在正中心 (0,0)
            is_hero = any(k in name for k in ['house', 'table', 'desk', 'campfire', 'car', 'fountain', 'bed'])

            # Nature: 环境背景 (树, 石头, 墙) -> 放在外圈
            is_nature = any(k in name for k in ['tree', 'rock', 'stone', 'bush', 'wall', 'fence'])

            # Companion: 伴随物体 (人, 椅子) -> 放在中圈
            is_companion = any(k in name for k in ['chair', 'person', 'dog', 'bench'])

            # 2. 计算位置
            if is_hero:
                # 核心物体：强制在极小的中心范围内 (0-2米)
                # 这样场景就有了“视觉焦点”
                angle = random.uniform(0, 2 * math.pi)
                radius = random.uniform(0.0, 1.5)

            elif is_nature:
                # 环境物体：在外圈聚落分布
                # 如果这个种类的物体（比如 Pine Tree）还没有聚落中心，就随机定一个
                if obj.model_id not in cluster_bases:
                    cluster_bases[obj.model_id] = (random.uniform(0, 2*math.pi), random.uniform(5.0, 10.0))

                base_angle, base_radius = cluster_bases[obj.model_id]

                # 在聚落中心附近偏移 (高斯分布)
                angle = base_angle + random.uniform(-0.5, 0.5) # 弧度偏移
                radius = base_radius + random.uniform(-2.0, 2.0)

            else: # is_companion / default
                # 其他物体：均匀散布在中圈 (2-5米)
                angle = random.uniform(0, 2 * math.pi)
                radius = random.uniform(2.0, 5.0)

            # 3. 设置坐标
            positions[i] = np.array([
                math.cos(angle) * radius,
                0,
                math.sin(angle) * radius
            ])

            # Compute effective radius (half diagonal on XZ plane)
            radii[i] = math.sqrt(bbox[0]**2 + bbox[2]**2) / 2 + 0.5  # Add safety padding
            
            # Smart facing
            rotation_y = self._calculate_facing_rotation(positions[i], obj.display_name, method="auto")

            # 创建场景节点
            transform = Transform(
                position=positions[i].copy(),
                rotation=np.array([0.0, rotation_y, 0.0]),
                scale=1.0
            )
            
            node = SceneNode(
                model_id=obj.model_id,
                instance_id=self._next_instance_id(obj.model_id),
                filename=obj.filename,
                display_name=obj.display_name,
                transform=transform,
                bbox_size=bbox.copy(),
                placement_type=obj.placement_type,
                children=[]
            )
            nodes.append(node)
        
        MAX_FORCE = 20.0  # Cap maximum force to prevent runaway speeds
        MAX_VEL = 5.0     # Cap maximum velocity
        
        # Force-directed iterations
        for iteration in range(self.iterations):
            forces = np.zeros((n, 3))
            
            # Compute repulsion between objects
            for i in range(n):
                for j in range(i + 1, n):
                    delta = positions[i] - positions[j]
                    delta[1] = 0  # Ignore Y component
                    distance = np.linalg.norm(delta)
                    
                    # Avoid divide-by-zero
                    if distance < 0.01:
                        distance = 0.01
                        delta = np.array([random.uniform(-1, 1), 0, random.uniform(-1, 1)])
                        delta = delta / np.linalg.norm(delta) * distance
                    
                    # Collision distance threshold
                    collision_dist = (radii[i] + radii[j]) * 1.05  # Add buffer
                    
                    # Repulsion formula: softened version with force cap
                    if distance < collision_dist * 1.1:  # Within influence range
                        force_direction = delta / distance
                        strength = self.repulsion_k / ((distance + 0.1) ** 2)
                        strength = min(strength, MAX_FORCE)
                        force = strength * force_direction
                        
                        forces[i] += force
                        forces[j] -= force

                    # 额外的全局引力，防止过度疏离
                    gravity_strength = 0.05
                    gravity_dir = delta / distance
                    forces[i] -= gravity_dir * gravity_strength
                    forces[j] += gravity_dir * gravity_strength

                    # [Semantic attraction] Pull objects with high embedding similarity
                    if normalized_vectors:
                        v_i = normalized_vectors.get(objects[i].model_id)
                        v_j = normalized_vectors.get(objects[j].model_id)
                        if v_i is not None and v_j is not None and distance > (radii[i] + radii[j]):
                            sim = float(np.dot(v_i, v_j))
                            if sim >= self.semantic_attraction_threshold:
                                attract_strength = self.semantic_attraction_strength * sim
                                attract_force = (delta / distance) * attract_strength
                                forces[i] -= attract_force
                                forces[j] += attract_force
            
            # Compute attraction toward center
            for i in range(n):
                dist_to_center = np.linalg.norm(positions[i])
                if dist_to_center > 0.1:
                    # Attraction formula: F_att = -k_att × DistanceToCenter
                    center_force = -self.attraction_k * positions[i]
                    center_force[1] = 0  # Keep Y at zero
                    forces[i] += center_force
            
            # Update velocity and position
            velocities = velocities * self.DEFAULT_DAMPING + forces * self.DEFAULT_TIME_STEP
            
            # Clamp velocity to prevent jumping outside in one step
            vel_norm = np.linalg.norm(velocities, axis=1, keepdims=True)
            velocities = np.where(vel_norm > MAX_VEL, velocities / vel_norm * MAX_VEL, velocities)
            positions += velocities * self.DEFAULT_TIME_STEP
            
            # Keep inside scene bounds
            for i in range(n):
                positions[i] = np.clip(
                    positions[i],
                    -self.DEFAULT_SCENE_BOUNDS,
                    self.DEFAULT_SCENE_BOUNDS
                )
                positions[i][1] = 0  # Force Y = 0
            
            # Record iteration history (for debug animation)
            self.debug_data['iteration_history'].append({
                'iteration': iteration,
                'positions': positions.copy().tolist(),
                'total_energy': np.sum(velocities ** 2)
            })
        
        # Update node positions
        for i, node in enumerate(nodes):
            node.transform.position = positions[i].copy()
        
        return nodes
    
    def _apply_anchor_constraints(
        self,
        parent_nodes: List[SceneNode],
        child_objects: List[SceneObject],
        model_dimensions: Optional[Dict[str, np.ndarray]] = None
    ) -> List[SceneNode]:
        """
        Anchor-based hierarchical constraint algorithm (assumes model origin
        near geometric center; if origin is at the base, anchor heights may offset).

        Math principle: local coordinate frame transforms.

        Steps:
        1. AABB analysis: compute parent's axis-aligned bounding box
        2. Anchor definition:
           - Slot_front = Center + (0, 0, Depth/2 + Padding)
           - Slot_right = Center + (Width/2 + Padding, 0, 0)
           - Slot_back = Center + (0, 0, -(Depth/2 + Padding))
           - Slot_left = Center + (-(Width/2 + Padding), 0, 0)
        3. Matrix transform:
           P_child = P_parent + (R_parent · V_offset)
           where R_parent is parent's rotation matrix and V_offset is anchor offset

        Returns:
            List of orphan nodes (children whose parent is missing are promoted)
        """
        if not child_objects:
            return []
        
        model_dimensions = model_dimensions or {}
        orphans: List[SceneNode] = []
        
        # Map parent model_id to nodes
        parent_map = {node.model_id: node for node in parent_nodes}
        
        # Define anchor slots for each parent
        anchor_slots: Dict[str, List[np.ndarray]] = {}
        slot_index: Dict[str, int] = {}
        
        for node in parent_nodes:
            bbox = node.bbox_size
            padding = self.DEFAULT_ANCHOR_PADDING
            
            # Define four main anchor positions relative to parent center
            slots = [
                np.array([0, 0, bbox[2]/2 + padding]),   # front
                np.array([bbox[0]/2 + padding, 0, 0]),   # right
                np.array([0, 0, -(bbox[2]/2 + padding)]), # back
                np.array([-(bbox[0]/2 + padding), 0, 0]), # left
            ]
            anchor_slots[node.model_id] = slots
            slot_index[node.model_id] = 0
        
        # Process each child object
        for child_obj in child_objects:
            parent_id = child_obj.parent_id
            if parent_id not in parent_map:
                # Parent not found: promote to root node for global layout
                bbox = model_dimensions.get(child_obj.model_id, self.DEFAULT_BBOX_SIZE.copy())
                if isinstance(bbox, list):
                    bbox = np.array(bbox)
                orphan_node = SceneNode(
                    model_id=child_obj.model_id,
                    instance_id=self._next_instance_id(child_obj.model_id),
                    filename=child_obj.filename,
                    display_name=child_obj.display_name,
                    transform=Transform(
                        position=np.array([0.0, 0.0, 0.0]),
                        rotation=np.array([0.0, 0.0, 0.0]),
                        scale=1.0
                    ),
                    bbox_size=bbox.copy(),
                    placement_type=child_obj.placement_type,
                    children=[]
                )
                orphans.append(orphan_node)
                continue
            
            parent_node = parent_map[parent_id]
            
            # Get child bounding box
            child_bbox = model_dimensions.get(child_obj.model_id, self.DEFAULT_BBOX_SIZE.copy())
            if isinstance(child_bbox, list):
                child_bbox = np.array(child_bbox)
            
            # Get next available anchor slot
            slots = anchor_slots[parent_id]
            idx = slot_index[parent_id] % len(slots)
            slot_index[parent_id] += 1
            
            # Compute anchor offset
            v_offset = slots[idx]
            
            # Parent rotation matrix
            parent_rotation = parent_node.transform.rotation
            R_parent = self._euler_to_rotation_matrix(parent_rotation)
            
            # Child world position: P_child = P_parent + (R_parent · V_offset)
            rotated_offset = R_parent @ v_offset
            child_position = parent_node.transform.position + rotated_offset
            
            # Child rotation faces parent center
            dir_to_parent = parent_node.transform.position - child_position
            dir_to_parent[1] = 0
            if np.linalg.norm(dir_to_parent) > 0.01:
                angle_y = math.degrees(math.atan2(dir_to_parent[0], dir_to_parent[2]))
            else:
                angle_y = parent_rotation[1]
            
            # Create child node
            child_transform = Transform(
                position=child_position,
                rotation=np.array([0.0, angle_y, 0.0]),
                scale=1.0
            )
            
            child_node = SceneNode(
                model_id=child_obj.model_id,
                instance_id=self._next_instance_id(child_obj.model_id),
                filename=child_obj.filename,
                display_name=child_obj.display_name,
                transform=child_transform,
                bbox_size=child_bbox,
                placement_type=child_obj.placement_type,
                children=[]
            )
            
            # Attach to parent's children
            parent_node.children.append(child_node)
            
            # Record parent-child line (for debugging)
            self.debug_data['parent_child_lines'].append({
                'parent_pos': parent_node.transform.position.tolist(),
                'child_pos': child_position.tolist(),
                'parent_name': parent_node.display_name,
                'child_name': child_node.display_name
            })
        
        return orphans
    
    def _euler_to_rotation_matrix(self, euler_angles: np.ndarray) -> np.ndarray:
        """
        Convert Euler angles to rotation matrix (Y-X-Z order, degrees).

        Mainly used for Y-axis rotation (vertical axis).

        Args:
            euler_angles: [rx, ry, rz] in degrees

        Returns:
            3x3 rotation matrix
        """
        rx, ry, rz = np.radians(euler_angles)
        
        # Y-axis rotation matrix (most common)
        cy, sy = math.cos(ry), math.sin(ry)
        Ry = np.array([
            [cy, 0, sy],
            [0, 1, 0],
            [-sy, 0, cy]
        ])
        
        # X-axis rotation matrix
        cx, sx = math.cos(rx), math.sin(rx)
        Rx = np.array([
            [1, 0, 0],
            [0, cx, -sx],
            [0, sx, cx]
        ])
        
        # Z-axis rotation matrix
        cz, sz = math.cos(rz), math.sin(rz)
        Rz = np.array([
            [cz, -sz, 0],
            [sz, cz, 0],
            [0, 0, 1]
        ])
        
        # Compose: R = Rz @ Rx @ Ry
        return Rz @ Rx @ Ry
    
    def _apply_placement_rules(self, nodes: List[SceneNode]):
        """
        Apply placement_type rules.

        Rules:
        - ground: keep Y=0
        - character: lock rotation_x/z = 0 (stay upright)
        - floating: set pos_y = random(5, 10)
        - prop: place on parent surface (simplified: keep current Y or default 0)

        Args:
            nodes: Scene nodes
        """
        def apply_rules(node: SceneNode):
            ptype = node.placement_type
            
            if ptype == 'character':
                # Keep character upright
                node.transform.rotation[0] = 0  # rx = 0
                node.transform.rotation[2] = 0  # rz = 0
                node.transform.position[1] = 0
                
            elif ptype == 'floating':
                # Floating objects at random height within the band [5, 15]
                node.transform.position[1] = random.uniform(5.0, 15.0)
                
            elif ptype == 'prop':
                # Prop placement: keep current Y (simplified)
                pass
                
            elif ptype == 'ground':
                # Grounded objects at Y=0
                node.transform.position[1] = 0
            
            # Recurse into children
            for child in node.children:
                apply_rules(child)
        
        for node in nodes:
            apply_rules(node)
    
    def _store_debug_data(self, nodes: List[SceneNode]):
        """
        Store debug visualization data.

        Includes:
        - AABB bounding boxes
        - Parent-child connection lines (already recorded in _apply_anchor_constraints)
        """
        def collect_aabb(node: SceneNode):
            pos = node.transform.position
            size = node.bbox_size
            
            # Center should be at geometry center; current pos is ground-level anchor,
            # so lift by half height to center the box.
            center = pos + np.array([0.0, size[1] / 2.0, 0.0])
            half_size = size / 2
            corners = []
            for dx in [-1, 1]:
                for dy in [-1, 1]:
                    for dz in [-1, 1]:
                        corner = center + np.array([
                            dx * half_size[0],
                            dy * half_size[1],
                            dz * half_size[2]
                        ])
                        corners.append(corner.tolist())
            
            self.debug_data['aabb_boxes'].append({
                'model_id': node.model_id,
                'display_name': node.display_name,
                'center': center.tolist(),
                'size': size.tolist(),
                'corners': corners
            })
            
            # Recurse into children
            for child in node.children:
                collect_aabb(child)
        
        for node in nodes:
            collect_aabb(node)
    
    def _clear_debug_data(self):
        """Reset debug data container"""
        self.debug_data = {
            'aabb_boxes': [],
            'parent_child_lines': [],
            'force_vectors': [],
            'iteration_history': [],
        }
    
    def get_scene_graph_json(self, nodes: List[SceneNode]) -> List[Dict]:
        """
        Convert scene nodes to JSON scene graph.

        Output format follows the Scene Graph spec from the design doc.

        Args:
            nodes: Scene nodes

        Returns:
            Scene graph JSON list
        """
        return [node.to_dict() for node in nodes]
    
    def _next_instance_id(self, base: str) -> str:
        """Generate unique instance ID for render naming to avoid actor overwrite"""
        self._instance_counter += 1
        return f"{base}_{self._instance_counter}"
    
    def get_all_nodes_flat(self, nodes: List[SceneNode]) -> List[SceneNode]:
        """
        Get a flattened list of all nodes (including children).

        Useful for rendering traversal.

        Args:
            nodes: Root node list

        Returns:
            Flat list of all nodes
        """
        result = []
        
        def collect(node: SceneNode):
            result.append(node)
            for child in node.children:
                collect(child)
        
        for node in nodes:
            collect(node)
        
        return result

