"""
LayoutEngine - Spatial Layout Algorithm Engine
This module is used to layout the scene using the force-directed algorithm.
It has core algorithm to keep the objects from overlapping and to keep the objects in the view frustum.
It also has a semantic clustering algorithm to keep the objects with high embedding similarity together.
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
    rotation: np.ndarray = field(default_factory=lambda: np.array([0.0, 0.0, 0.0]))
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

    Physical intuition:Treat each object as a charged particle, and repulsion prevents mesh intersection, and attraction keeps objects inside the view frustum
    """
    
    # Force-directed algorithm parameters
    DEFAULT_REPULSION_K = 80.0       
    DEFAULT_ATTRACTION_K = 1.2       
    DEFAULT_DAMPING = 0.9            
    DEFAULT_ITERATIONS = 50         
    DEFAULT_TIME_STEP = 0.05        
    DEFAULT_MIN_DISTANCE = 0.2      
    
    # Layout space parameters
    DEFAULT_LAYOUT_RADIUS = 8.0      
    DEFAULT_SCENE_BOUNDS = 30.0     
    
    # Anchor parameters
    DEFAULT_ANCHOR_PADDING = 0.3    
    
    # Default bounding box size (used when real size is unavailable)
    DEFAULT_BBOX_SIZE = np.array([1.0, 1.0, 1.0])

    # Semantic clustering parameters (cosine similarity on embeddings)
    # The semantic attraction strength is used to pull objects with high embedding similarity
    DEFAULT_SEMANTIC_ATTRACTION_STRENGTH = 8.0
    DEFAULT_SEMANTIC_ATTRACTION_THRESHOLD = 0.45

    # Facing helpers
    def _calculate_facing_rotation(self, position: np.ndarray, display_name: str, method: str = "auto") -> float:
        """
        Compute facing angle around Y-axis.
        This fucntion forces the object to face the viewer or a random direction
        For natural objects, it will be facing a random direction
        For other objects, it will be facing the viewer direction
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
            angle_rad = math.atan2(direction[0], direction[2])
            angle_deg = math.degrees(angle_rad)  
            return angle_deg + random.uniform(-10, 10)

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
        self.repulsion_k = repulsion_k
        self.attraction_k = attraction_k
        self.iterations = iterations
        self.layout_radius = layout_radius
        self._instance_counter = 0

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
        This function is the main entry that coordinates force-directed layout and anchor constraints.
        """
        if not scene_objects:
            return []
        
        self._clear_debug_data()
        
        expanded_objects = self._expand_objects(scene_objects)

        # Anchor-based hierarchy temporarily disabled: lay out everything with force-directed.
        parent_nodes = self._force_directed_layout(expanded_objects, model_dimensions, semantic_vectors)
        
        self._apply_placement_rules(parent_nodes)
        
        self._store_debug_data(parent_nodes)
        
        return parent_nodes
    
    def _expand_objects(self, scene_objects: List[SceneObject]) -> List[SceneObject]:
        """
        Expand objects list based on count.
        For example, if the count of a tree is 3, it will be expanded to three independent tree instances.
        """
        expanded = []
        for obj in scene_objects:
            for i in range(obj.count):
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
        This function is used to layout the scene using the force-directed algorithm.
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
        
        nodes: List[SceneNode] = []
        positions = np.zeros((n, 3))
        velocities = np.zeros((n, 3))
        radii = np.zeros(n)  # Effective radii for collision detection

        # Define cluster centers
        # This step is to keep similar objects together at the beginning
        # instead of scattered all over the map
        cluster_bases = {}

        for i, obj in enumerate(objects):
            bbox = model_dimensions.get(obj.model_id, self.DEFAULT_BBOX_SIZE.copy())
            if isinstance(bbox, list):
                bbox = np.array(bbox)

            name = obj.display_name.lower()
     
            is_hero = any(k in name for k in ['house', 'table', 'desk', 'campfire', 'car', 'fountain', 'bed'])

            is_nature = any(k in name for k in ['tree', 'rock', 'stone', 'bush', 'wall', 'fence'])

            is_companion = any(k in name for k in ['chair', 'person', 'dog', 'bench'])

            
            if is_hero:
                angle = random.uniform(0, 2 * math.pi)
                radius = random.uniform(0.0, 1.5)

            elif is_nature:
                if obj.model_id not in cluster_bases:
                    cluster_bases[obj.model_id] = (random.uniform(0, 2*math.pi), random.uniform(5.0, 10.0))

                base_angle, base_radius = cluster_bases[obj.model_id]

                angle = base_angle + random.uniform(-0.5, 0.5)
                radius = base_radius + random.uniform(-2.0, 2.0)

            else: # is_companion / default
                angle = random.uniform(0, 2 * math.pi)
                radius = random.uniform(2.0, 5.0)

            positions[i] = np.array([
                math.cos(angle) * radius,
                0,
                math.sin(angle) * radius
            ])

            radii[i] = math.sqrt(bbox[0]**2 + bbox[2]**2) / 2 + 0.5
            
            rotation_y = self._calculate_facing_rotation(positions[i], obj.display_name, method="auto")

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
        
        MAX_FORCE = 20.0
        MAX_VEL = 5.0
        
        for iteration in range(self.iterations):
            forces = np.zeros((n, 3))

            for i in range(n):
                for j in range(i + 1, n):
                    delta = positions[i] - positions[j]
                    delta[1] = 0
                    distance = np.linalg.norm(delta)
                    
                    if distance < 0.01:
                        distance = 0.01
                        delta = np.array([random.uniform(-1, 1), 0, random.uniform(-1, 1)])
                        delta = delta / np.linalg.norm(delta) * distance
                    
                    collision_dist = (radii[i] + radii[j]) * 1.05
                    
                    if distance < collision_dist * 1.1:
                        force_direction = delta / distance
                        strength = self.repulsion_k / ((distance + 0.1) ** 2)
                        strength = min(strength, MAX_FORCE)
                        force = strength * force_direction
                        
                        forces[i] += force
                        forces[j] -= force

                    gravity_strength = 0.05
                    gravity_dir = delta / distance
                    forces[i] -= gravity_dir * gravity_strength
                    forces[j] += gravity_dir * gravity_strength

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
            
            for i in range(n):
                dist_to_center = np.linalg.norm(positions[i])
                if dist_to_center > 0.1:
                    center_force = -self.attraction_k * positions[i]
                    center_force[1] = 0
                    forces[i] += center_force
            
            velocities = velocities * self.DEFAULT_DAMPING + forces * self.DEFAULT_TIME_STEP
            
            vel_norm = np.linalg.norm(velocities, axis=1, keepdims=True)
            velocities = np.where(vel_norm > MAX_VEL, velocities / vel_norm * MAX_VEL, velocities)
            positions += velocities * self.DEFAULT_TIME_STEP
            
            for i in range(n):
                positions[i] = np.clip(
                    positions[i],
                    -self.DEFAULT_SCENE_BOUNDS,
                    self.DEFAULT_SCENE_BOUNDS
                )
                positions[i][1] = 0
            
            self.debug_data['iteration_history'].append({
                'iteration': iteration,
                'positions': positions.copy().tolist(),
                'total_energy': np.sum(velocities ** 2)
            })
        
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
        This function is used to apply the anchor constraints to the scene.
        """
        # Anchor-based hierarchy temporarily disabled.
        return []
    
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
        Apply placement_type rules. Here we set the position and rotation of the objects based on the placement_type.
        Based on the placement_type, we set the position and rotation of the objects.
        For example, if the placement_type is 'character', we keep the character upright.
        If the placement_type is 'floating', we set the position of the object to a random height within the band [5, 15].
        If the placement_type is 'prop', we keep the current Y position.
        If the placement_type is 'ground', we set the position of the object to Y=0.
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
        """
        return [node.to_dict() for node in nodes]
    
    def _next_instance_id(self, base: str) -> str:
        """Generate unique instance ID for render naming to avoid actor overwrite"""
        self._instance_counter += 1
        return f"{base}_{self._instance_counter}"
    
    def get_all_nodes_flat(self, nodes: List[SceneNode]) -> List[SceneNode]:
        """
        Get a flattened list of all nodes including children.
        """
        result = []
        
        def collect(node: SceneNode):
            result.append(node)
            for child in node.children:
                collect(child)
        
        for node in nodes:
            collect(node)
        
        return result

