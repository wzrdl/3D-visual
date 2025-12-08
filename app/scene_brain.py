"""
SceneBrain - Semantic Analysis Module

Handles NLP parsing and vector retrieval to map unstructured natural language
to the local asset library.
Tech: SBERT (all-MiniLM-L6-v2) + cosine similarity

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
        self._model_cache: List[Dict] = []
        self._embeddings_cache: Dict[str, np.ndarray] = {}
        
        # NLP模型 (延迟加载)
        self._encoder = None
        self._nlp_available = False
        
        # 尝试初始化NLP组件
        self._init_nlp()
        
    def _init_nlp(self):
        """
        Initialize NLP components.

        TODO: Implement SBERT semantic encoder.
        Current version uses keyword matching as a placeholder.
        """
        try:
            # TODO: Enable after adding sentence-transformers dependency
            # from sentence_transformers import SentenceTransformer
            # self._encoder = SentenceTransformer('all-MiniLM-L6-v2')
            # self._nlp_available = True
            self._nlp_available = False
            print("[SceneBrain] NLP module placeholder - using keyword matching")
        except ImportError:
            self._nlp_available = False
            print("[SceneBrain] sentence-transformers not available, using keyword matching")
    
    def _load_model_cache(self):
        """Load model cache from the data manager"""
        if self.data_manager and not self._model_cache:
            self._model_cache = self.data_manager.get_all_models()
    
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
        self._load_model_cache()
        
        # 1. Text preprocessing
        text_lower = text.lower().strip()
        
        # 2. Extract objects and counts
        extracted_items = self._extract_objects_and_quantities(text_lower)
        
        # 3. Match against model library
        scene_objects = []
        parent_map = {}  # Track parent objects for relationship building
        
        for keyword, count in extracted_items:
            matched_model = self._match_model(keyword)
            
            if matched_model:
                placement_type = self._get_placement_type(keyword)
                is_parent = self._is_parent_object(keyword)
                
                scene_obj = SceneObject(
                    model_id=matched_model['id'],
                    display_name=matched_model.get('display_name', matched_model.get('name', keyword)),
                    filename=matched_model['filename'],
                    count=count,
                    tags=matched_model.get('tags', []),
                    placement_type=placement_type,
                    is_parent=is_parent,
                    parent_id=None
                )
                
                # 记录父物体
                if is_parent:
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
        
        return results
    
    def _match_model(self, keyword: str) -> Optional[Dict]:
        """
        Match a model in the library by keyword.

        Current implementation: simple keyword matching.
        TODO: implement SBERT-based semantic vector matching.

        Matching algorithm:
        1. Exact match on filename or display name
        2. Tag match
        3. Partial match
        4. (TODO) Semantic cosine similarity

        Args:
            keyword: Object keyword

        Returns:
            Matched model dict or None
        """
        if not self._model_cache:
            return None
        
        keyword_lower = keyword.lower()
        
        # 1. 精确匹配显示名或文件名
        for model in self._model_cache:
            display_name = model.get('display_name', '').lower()
            filename = model.get('filename', '').lower()
            
            if keyword_lower == display_name or keyword_lower in filename:
                return model
        
        # 2. 标签匹配
        for model in self._model_cache:
            tags = model.get('tags', [])
            if isinstance(tags, str):
                tags = [tags]
            
            for tag in tags:
                if keyword_lower == tag.lower() or keyword_lower in tag.lower():
                    return model
        
        # 3. 部分匹配（显示名或文件名包含关键词）
        for model in self._model_cache:
            display_name = model.get('display_name', '').lower()
            filename = model.get('filename', '').lower()
            
            if keyword_lower in display_name or keyword_lower in filename:
                return model
        
        # 4. TODO: 语义向量匹配
        # if self._nlp_available and self._encoder:
        #     return self._semantic_match(keyword)
        
        return None
    
    def _semantic_match(self, keyword: str) -> Optional[Dict]:
        """
        Semantic vector-based model matching.

        TODO: Implement:
        1. Get semantic vector for keyword: V_query = Encoder(keyword)
        2. Compute cosine similarity with all models
        3. Return the highest-scoring model above threshold

        Cosine similarity:
        S = (V_query · V_model) / (||V_query|| × ||V_model||)
        """
        # 占位实现
        # TODO: 实现SBERT语义匹配
        # query_embedding = self._encoder.encode(keyword)
        # best_match = None
        # best_score = 0
        # 
        # for model in self._model_cache:
        #     model_text = f"{model.get('display_name', '')} {' '.join(model.get('tags', []))}"
        #     model_embedding = self._get_or_compute_embedding(model['id'], model_text)
        #     
        #     # 余弦相似度
        #     score = np.dot(query_embedding, model_embedding) / (
        #         np.linalg.norm(query_embedding) * np.linalg.norm(model_embedding)
        #     )
        #     
        #     if score > best_score and score >= self.similarity_threshold:
        #         best_score = score
        #         best_match = model
        # 
        # return best_match
        return None
    
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

import torch
from typing import List  # in order to make sure that the keywords input is a list

from sentence_transformers import util
from app.client_data_manager import ClientDataManager


class SceneBrainLegacy:
    """
    Legacy keyword-based SceneBrain (kept for reference).

    Note: The active SceneBrain is defined above with data_manager support.
    """

    def __init__(self):
        self.keywords = []
        self.top_matches = []
        self.stopwords = set()

        # as these are used often
        self.cdm = ClientDataManager()  # cdm for ease
        self.embedder = self.cdm.miniM_model

    def stop_words_set(self):

        file = "assets/stopwords.txt"
        myFile = open(file, "r")
        for line in myFile:
            self.stopwords.add(line.strip())
        myFile.close()

        return self.stopwords

    # takes user input and finds keywords from it
    def extract_keywords(self, user_input: str):
        """
        From user input, separates keywords that we can later use
        """

        user_input = user_input.lower()
        user_input = user_input.strip()
        split_input = user_input.split(" ") # splits based on a space

        # just in case there are irregular spaces
        for i in range(len(split_input)-1):
            split_input[i] = split_input[i].strip()

        clean_list = []

        """ 
        OPTIONAL
        sentence transformer SHOULD be able to tell articles apart
        If we wanted a list that is usable in other contexts, though, this could be userufl
        """
        remove_stopwords = True  # adjust this for what we want

        if remove_stopwords == True:
            self.stopwords = self.stop_words_set()

            for i in range(len(split_input) - 1):
                if split_input[i] not in self.stop_words_set():
                    clean_list.append(split_input[i])
        else:
            for i in range(len(split_input) - 1):
                clean_list.append(split_input[i])

        return clean_list

    def keywords_to_vector(self, user_input: str):
        self.keywords = self.extract_keywords(user_input)

        queries = " ".join(self.keywords)
        query_embedding = self.embedder.encode_query(queries)

        # seeing the scores or calculations for the test
        similarity_score = util.cos_sim(query_embedding, self.cdm.vector_database)
        # makes it a 2D grid with the first [0] being the query and the second holding the
        #       different temp_data lines

        all_names = self.cdm.name_order
        k = len(all_names)

        if k > 0:
            scores, indices = torch.topk(similarity_score[0], k=k)
        else:
            return ["placeholder"]

        """Edit this as we see fit"""
        match_accuracy = 0.4
        # 0.5 was too high for the pytest

        # checking which scores are more than 0.5
        for score, index in zip(scores, indices):
            if score > match_accuracy:
                match_name = self.cdm.name_order[index]
                self.top_matches.append(match_name)
                print(f"MATCH: {match_name} has a score of ({score:4f})")

        # if there is no good match -- prevents error
        if len(self.top_matches) == 0:
            print("No matches found, placeholder set as match.")
            self.top_matches.append("placeholder")

        return self.top_matches
