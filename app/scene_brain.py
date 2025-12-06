import os
import json
from sentence_transformers import SentenceTransformer, util
from app.client_data_manager import ClientDataManager

# app = QApplication(sys.argv)

# making temporary meta.json file to test with
temp_data = [
    {"filename": "tree.obj", "tags": ["leaves", "natural", "wood", "green"]},
    {"filename": "flower.obj", "tags": ["petals", "natural", "colorful"]},
    {"filename": "chair.obj", "tags": ["wood", "furniture", "legs"]},
    {"filename": "keyboard.obj", "tags": ["mechanical", "silicon", "keys"]}
]

temp_data_path = "assets/pytest assets/test_meta.json"

os.makedirs(os.path.dirname(temp_data_path), exist_ok=True)  # make sure it exists? if not will make
with open(temp_data_path, "w") as f:
    json.dump(temp_data, f)

# running function
vector = ClientDataManager(load_without_test=False)
vector.concatenate_name_tags(temp_data_path)

# testing
test_word = "forest"
test_embedding = vector.miniM_model.encode(test_word)

# seeing the scores or calculations for the test
calculations = util.cos_sim(test_embedding, vector.vector_database)

