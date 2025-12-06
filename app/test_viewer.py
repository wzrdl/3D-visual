import torch
import os
import sys

import pytest
from PyQt6.QtWidgets import QApplication

"""
Pytest #1 
This tests if the generate thumbnail function works as intended
It passes in a test object in the pytest assets and will make it a thumbnail
By the end, it will delete the thumbnail to not add clutter
"""
from app.viewer import ThreeDViewer

# WORKS
@pytest.mark.skip(reason="successful, no need to see this each time for now")
def test_generate_thumbnail():
    app = QApplication(sys.argv)

    file_name = "assets/pytest assets/cone_test.obj"
    object = ThreeDViewer()
    object.generate_thumbnail(file_name)

    # checks if it makes the png file
    check = os.path.exists("assets/thumbnails/cone_test.png")

    # cleans up the new file
    if check == True:
        os.remove("assets/thumbnails/cone_test.png")

    assert check == True

"""
Pytest #2
Passing the add_model function something that does exist
"""
from app.data_manager import DataManager

# WORKS
@pytest.mark.skip(reason="successful, no need to see this each time for now")
def test_add_model_exists():
    object = DataManager()

    id_number = "model_001"
    file_name = "cube.obj"
    name = "Cube"
    tags = [
      "geometric",
      "primitive",
      "box",
      "square"
    ]

    assert object.add_model(id_number, file_name, name, tags) == False

"""
Pytest #3
Vector Database logic works correctly
"""

from sentence_transformers import SentenceTransformer, util
import json
from app.client_data_manager import ClientDataManager

def test_vector_database():

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
    name_order = vector.concatenate_name_tags(temp_data_path)
    embedder = vector.miniM_model  # to reduce the amount of __.__.__ we have

    # testing test_word -- SBERT.net recommends using query for this
    query = "forest"
    query_embedding = embedder.encode_query(query)

    # seeing the scores or calculations for the test
    similarity_score = util.cos_sim(query_embedding, vector.vector_database)
    # makes it a 2D grid with the first [0] being the query and the second holding the
    # different temp_data lines

    # find the closest matches
    top_k = min(2, len(similarity_score))

    scores, indices = torch.topk(similarity_score[0], k=top_k)

    #print("Query: ", query)
    for score, index in zip(scores, indices):
        filename = name_order[index]
        #print(f"Filename: {filename}, (Score: {score:4f}) at index: {index}")

    assert similarity_score[0][0] > similarity_score[0][1] # tree > flower
    assert filename == "tree.obj"

from app.pages import GalleryPage
"""
Pytest #4
Search bar returns the right amount of models per the tag searched 
"""
@pytest.mark.skip(reason = "Not finished")
def test_tag_gallery_search():
    app = QApplication(sys.argv)

    object = GalleryPage()
    object.filter_models("round")
