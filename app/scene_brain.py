"""
SceneBrain - Semantic Analysis Module

Handles NLP parsing and vector retrieval to map unstructured natural language
to the local asset library using sentence-transformers + cosine similarity
for semantic matching.

Input: text string
Output: target model IDs and quantity info
"""

import re
from typing import List, Dict, Tuple, Optional, Any
from dataclasses import dataclass
import numpy as np


@dataclass
class SceneObject:
    """Scene object descriptor"""
    model_id: str           # Model ID
    display_name: str       # Display name
    filename: str           # File name
    count: int              # Quantity
    tags: List[str]         # Tags
    placement_type: str     # Placement type: ground, prop, character, floating
    is_parent: bool         # Whether it is a parent object (e.g., table)
    parent_id: Optional[str] = None  # Parent object ID (if it is a child)


class SceneBrain:
    """
    Semantic engine that understands user input and maps it to 3D assets.

    Features:
    1. Parse natural language descriptions
    2. Extract object keywords and counts
    3. Match the local asset library via semantic similarity
    4. Identify parent-child constraints
    """
    
    # 预定义的物体关系映射（父物体 -> 子物体列表）
    PARENT_CHILD_RELATIONS = {
        'table': ['chair', 'cup', 'plate', 'book', 'lamp'],
        'desk': ['chair', 'computer', 'lamp', 'book', 'pen'],
        'bed': ['pillow', 'blanket', 'lamp'],
        'sofa': ['cushion', 'pillow', 'lamp', 'table'],
        'tree': ['bird', 'squirrel'],
        'car': ['wheel', 'person'],
        'house': ['door', 'window', 'person'],
    }
    
    # 放置类型关键词映射
    PLACEMENT_TYPE_KEYWORDS = {
        'ground': ['table', 'chair', 'desk', 'tree', 'car', 'building', 'house', 
                   'rock', 'stone', 'bench', 'lamp', 'fence', 'wall'],
        'prop': ['cup', 'book', 'pen', 'plate', 'vase', 'clock', 'phone'],
        'character': ['person', 'human', 'man', 'woman', 'child', 'animal',
                      'soldier', 'knight', 'ninja', 'zombie', 'elf', 'wizard'],
        'floating': ['cloud', 'bird', 'plane', 'helicopter', 'balloon', 'star'],
    }
    
    # 数量词映射
    QUANTITY_WORDS = {
        'a': 1, 'an': 1, 'one': 1, 'single': 1,
        'two': 2, 'couple': 2, 'pair': 2,
        'three': 3, 'few': 3,
        'four': 4, 'several': 4,
        'five': 5, 'many': 5,
        'six': 6, 'seven': 7, 'eight': 8,
        'nine': 9, 'ten': 10,
        'some': 3, 'multiple': 4,
    }
    
    def __init__(self, data_manager=None, similarity_threshold: float = 0.4):
        """
        Initialize the semantic engine.

        Args:
            data_manager: Data manager instance for accessing the model library
            similarity_threshold: Semantic similarity threshold
        """
        self.data_manager = data_manager
        self.similarity_threshold = similarity_threshold
        # Lowered from 0.35 to 0.25 to include more relevant items (like trees/spheres)
        self.fallback_threshold = 0.25
        self.top_k = 20  # number of candidates to consider per keyword
        self._fallback_model_id = "cube"
        self._model_cache: List[Dict] = []
        self._embeddings_cache: Dict[str, np.ndarray] = {}
    
    def _load_model_cache(self, force_refresh: bool = False):
        """
        Load/refresh model cache from the data manager.

        force_refresh=True ensures newly added models (e.g., AI generation) are visible.
        """
        if self.data_manager and (force_refresh or not self._model_cache):
            try:
                self._model_cache = self.data_manager.get_all_models()
            except Exception as e:
                print(f"[SceneBrain] Failed to load model cache: {e}")
    
    def parse_scene_description(self, text: str) -> List[SceneObject]:
        """
        Parse scene description text and return a list of scene objects.

        Algorithm:
        1. Text preprocessing (lowercase, tokenization)
        2. Extract object keywords and quantities
        3. Semantic match against the model library
        4. Identify parent-child relations

        Args:
            text: User-entered scene description, e.g., "a room with a table and two chairs"

        Returns:
            List of SceneObject
        """
        if not text or not text.strip():
            return []
        
        # Load model cache
        self._load_model_cache(force_refresh=True)
        
        # 1. Text preprocessing
        text_lower = text.lower().strip()
        
        # 2. Extract objects and counts
        extracted_items = self._extract_objects_and_quantities(text_lower)
        
        # 3. Match against model library
        scene_objects = []
        parent_map = {}  # Track parent objects for relationship building
        
        for keyword, count in extracted_items:
            matched_list = self._match_models(keyword)
            
            for idx, (matched_model, matched_score) in enumerate(matched_list):
                tags = matched_model.get('tags', [])
                if isinstance(tags, str):
                    tags = [tags]
                placement_type = matched_model.get('placement_type') or self._get_placement_type(keyword)
                is_parent = self._is_parent_object(keyword)
                
                scene_obj = SceneObject(
                    model_id=matched_model['id'],
                    display_name=matched_model.get('display_name', matched_model.get('name', keyword)),
                    filename=matched_model['filename'],
                    count=count if idx == 0 else 1,  # preserve user count only for top hit
                    tags=tags,
                    placement_type=placement_type,
                    is_parent=is_parent,
                    parent_id=None
                )
                
                # 记录父物体（仅首个匹配用于父映射）
                if is_parent and idx == 0:
                    parent_map[keyword] = scene_obj.model_id
                    
                scene_objects.append(scene_obj)
        
        # 4. 建立父子关系
        scene_objects = self._establish_parent_child_relations(scene_objects, parent_map)
        
        return scene_objects
    
    def _extract_objects_and_quantities(self, text: str) -> List[Tuple[str, int]]:
        """
        Extract object keywords and quantities from text.

        Uses regex and quantity word mapping.

        Args:
            text: Preprocessed text

        Returns:
            List of (keyword, count) tuples
        """
        results = []
        
        # Pattern 1: "quantity word + noun" (e.g., "three trees", "a table")
        # Pattern 2: "number + noun" (e.g., "3 chairs")
        
        # Handle number + noun first
        number_pattern = r'(\d+)\s+(\w+s?)'
        for match in re.finditer(number_pattern, text):
            count = int(match.group(1))
            noun = match.group(2).rstrip('s')  # Remove plural s
            if count > 0 and count <= 20:  # Reasonable range guard
                results.append((noun, count))
        
        # Handle quantity-word + noun
        for quantity_word, count in self.QUANTITY_WORDS.items():
            pattern = rf'\b{quantity_word}\s+(\w+s?)\b'
            for match in re.finditer(pattern, text):
                noun = match.group(1).rstrip('s')
                # Avoid duplicates
                if not any(noun == r[0] for r in results):
                    results.append((noun, count))
        
        # If no counted objects were found, try extracting standalone nouns
        if not results:
            # Common 3D object nouns
            common_objects = [
                'table', 'chair', 'tree', 'house', 'car', 'person',
                'lamp', 'desk', 'bed', 'sofa', 'rock', 'stone',
                'building', 'fence', 'wall', 'door', 'window',
                'soldier', 'knight', 'ninja', 'zombie', 'wizard',
                'elf', 'goblin', 'cow', 'dog', 'cat', 'bird',
                'cube', 'sphere', 'cone', 'cylinder', 'pyramid'
            ]
            
            words = re.findall(r'\b\w+\b', text)
            for word in words:
                word_base = word.rstrip('s')
                if word_base in common_objects:
                    if not any(word_base == r[0] for r in results):
                        results.append((word_base, 1))

        # If still nothing, fall back to treating any token as a candidate keyword
        if not results:
            fallback_words = re.findall(r'\b\w+\b', text)
            seen = set()
            for w in fallback_words:
                w_base = w.rstrip('s')
                if w_base and w_base not in seen:
                    results.append((w_base, 1))
                    seen.add(w_base)
        
        return results
    
    def _match_models(self, keyword: str) -> List[Tuple[Dict, float]]:
        """
        Match a model in the library by combining semantic similarity with
        keyword fallbacks. Returns a ranked list of candidates.
        """
        if not self._model_cache:
            return []

        candidates: List[Tuple[Dict, float]] = []

        semantic_list = self._semantic_match(keyword)
        best_score = semantic_list[0][1] if semantic_list else 0.0

        # If semantic score is below threshold, try deterministic keyword matching
        keyword_model = self._keyword_match(keyword)
        if keyword_model and best_score < self.similarity_threshold:
            candidates.append((keyword_model, 1.0))

        # Accept semantic matches above fallback threshold
        for model, score in semantic_list:
            if score >= self.fallback_threshold:
                candidates.append((model, score))

        # If nothing yet, but keyword match exists, use it
        if not candidates and keyword_model:
            candidates.append((keyword_model, 1.0))

        # Final safety fallback
        if not candidates:
            fallback = self._fallback_model()
            if fallback:
                candidates.append((fallback, 0.0))

        # Limit to top_k
        return candidates[: self.top_k]

    def _keyword_match(self, keyword: str) -> Optional[Dict]:
        """Simple keyword-based matching used as a deterministic fallback."""
        keyword_lower = keyword.lower()

        # Exact match on display_name / filename
        for model in self._model_cache:
            display_name = model.get('display_name', '').lower()
            filename = model.get('filename', '').lower()
            if keyword_lower == display_name or keyword_lower in filename:
                return model

        # Tag match
        for model in self._model_cache:
            tags = model.get('tags', [])
            if isinstance(tags, str):
                tags = [tags]
            for tag in tags:
                if keyword_lower == tag.lower() or keyword_lower in tag.lower():
                    return model

        # Partial match
        for model in self._model_cache:
            display_name = model.get('display_name', '').lower()
            filename = model.get('filename', '').lower()
            if keyword_lower in display_name or keyword_lower in filename:
                return model

        return None

    def _semantic_match(self, keyword: str) -> List[Tuple[Dict, float]]:
        """Semantic similarity search against DataManager embedding index."""
        if not self.data_manager:
            return []
        results = self.data_manager.semantic_search(keyword, top_k=self.top_k)
        if not results:
            return []
        return [(item["model"], item["score"]) for item in results]

    def _fallback_model(self) -> Optional[Dict]:
        """Return a safe placeholder model to avoid crashes."""
        if not self._model_cache:
            return None

        # Prefer cube / placeholder assets
        for model in self._model_cache:
            name = model.get("display_name", "").lower()
            if self._fallback_model_id in name or "placeholder" in name:
                return model
        return self._model_cache[0] if self._model_cache else None
    
    def _get_placement_type(self, keyword: str) -> str:
        """
        Determine placement type based on keyword.

        Placement types:
        - ground: ground objects (trees, tables, buildings, etc.)
        - prop: props (cups, books, small items)
        - character: characters (people, animals, etc.)
        - floating: floating objects (clouds, birds, etc.)

        Args:
            keyword: Object keyword

        Returns:
            Placement type string
        """
        keyword_lower = keyword.lower()
        
        for ptype, keywords in self.PLACEMENT_TYPE_KEYWORDS.items():
            for kw in keywords:
                if keyword_lower in kw or kw in keyword_lower:
                    return ptype
        
        # 默认为地面物体
        return 'ground'
    
    def _is_parent_object(self, keyword: str) -> bool:
        """Determine if keyword represents a parent object (can host children)"""
        return keyword.lower() in self.PARENT_CHILD_RELATIONS
    
    def _establish_parent_child_relations(
        self, 
        scene_objects: List[SceneObject],
        parent_map: Dict[str, str]
    ) -> List[SceneObject]:
        """
        Establish parent-child constraints.

        Based on predefined relation mappings and existing scene objects.

        Args:
            scene_objects: Scene object list
            parent_map: Mapping from parent keyword to ID

        Returns:
            Scene objects with parent_id updated
        """
        # Traverse all objects to see if they should attach to a parent
        for obj in scene_objects:
            if obj.is_parent:
                continue
            
            # Check whether this object should belong to a parent
            obj_name_lower = obj.display_name.lower()
            
            for parent_keyword, children in self.PARENT_CHILD_RELATIONS.items():
                if parent_keyword in parent_map:
                    for child_keyword in children:
                        if child_keyword in obj_name_lower:
                            obj.parent_id = parent_map[parent_keyword]
                            break
                            
        return scene_objects
    
    def get_scene_summary(self, scene_objects: List[SceneObject]) -> str:
        """
        Generate a scene summary description.

        Args:
            scene_objects: Scene object list

        Returns:
            Scene summary string
        """
        if not scene_objects:
            return "Empty scene"
        
        items = []
        for obj in scene_objects:
            if obj.count > 1:
                items.append(f"{obj.count}x {obj.display_name}")
            else:
                items.append(obj.display_name)
        
        return f"Scene contains: {', '.join(items)}"

