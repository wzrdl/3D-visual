"""
DataManager - handles the local dataset stuff
reads/writes metadata.json to keep track of all the 3D models
"""
import json
import os
from pathlib import Path
from typing import List, Dict, Optional


class DataManager:
    """manages 3D models and their metadata"""
    
    def __init__(self, assets_dir: Optional[str] = None):
        """setup paths and load metadata"""
        if assets_dir is None:
            project_root = Path(__file__).parent.parent
            self.assets_dir = project_root / "assets"
        else:
            self.assets_dir = Path(assets_dir)
        
        self.models_dir = self.assets_dir / "models"
        self.metadata_file = self.assets_dir / "metadata.json"
        self.metadata: List[Dict] = []
        
        self.models_dir.mkdir(parents=True, exist_ok=True)
        self.load_metadata()
    
    def load_metadata(self) -> bool:
        """load metadata from json file"""
        if not self.metadata_file.exists():
            self.metadata = []
            self.save_metadata()
            return True
        
        try:
            with open(self.metadata_file, 'r', encoding='utf-8') as f:
                self.metadata = json.load(f)
            
            if not isinstance(self.metadata, list):
                self.metadata = []
                return False
            
            return True
        except (json.JSONDecodeError, IOError) as e:
            print(f"Error loading metadata: {e}")
            self.metadata = []
            return False
    
    def save_metadata(self) -> bool:
        """save metadata to json"""
        try:
            with open(self.metadata_file, 'w', encoding='utf-8') as f:
                json.dump(self.metadata, f, indent=2, ensure_ascii=False)
            return True
        except IOError as e:
            print(f"Error saving metadata: {e}")
            return False
    
    def get_model_path(self, model_id: str) -> Optional[Path]:
        """get file path for a model by id"""
        for entry in self.metadata:
            if entry.get('id') == model_id:
                filename = entry.get('filename')
                if filename:
                    return self.models_dir / filename
        return None
    
    def get_all_models(self) -> List[Dict]:
        """get all models"""
        return self.metadata.copy()
    
    def search_models(self, query: str) -> List[Dict]:
        """search models by name or tags"""
        query_lower = query.lower()
        results = []
        
        for entry in self.metadata:
            name = entry.get('name', '').lower()
            tags = [tag.lower() for tag in entry.get('tags', [])]
            
            if query_lower in name or any(query_lower in tag for tag in tags):
                results.append(entry)
        
        return results
    
    def add_model(self, model_id: str, filename: str, name: str, 
                  tags: List[str]) -> bool:
        """add a new model to metadata"""
        # check if id already exists
        if any(entry.get('id') == model_id for entry in self.metadata):
            print(f"Model with ID '{model_id}' already exists")
            return False
        
        new_entry = {
            'id': model_id,
            'filename': filename,
            'name': name,
            'tags': tags
        }
        
        self.metadata.append(new_entry)
        return self.save_metadata()
    
    def get_next_id(self) -> str:
        """generate next available model id"""
        if not self.metadata:
            return "model_001"
        
        existing_ids = []
        for entry in self.metadata:
            id_str = entry.get('id', '')
            if id_str.startswith('model_'):
                try:
                    num = int(id_str.replace('model_', ''))
                    existing_ids.append(num)
                except ValueError:
                    pass
        
        if existing_ids:
            next_num = max(existing_ids) + 1
        else:
            next_num = 1
        
        return f"model_{next_num:03d}"

