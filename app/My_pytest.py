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

#@pytest.mark.skip(reason="successful")
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
tests the function in the layout_engine file that focuses on model rotation
"""

import numpy as np
from app.layout_engine import LayoutEngine

#@pytest.mark.skip(reason ="successful")
def test_model_rotation():

    temp_position = np.array([0, 0, 0])

    lg = LayoutEngine() # object for the class

    natural_model = lg._calculate_facing_rotation(temp_position, "tree")
    unnatural_model = lg._calculate_facing_rotation(temp_position, "ninja")

    assert unnatural_model < natural_model
    assert natural_model > 0
"""
Pytest #3
Vector Database logic works correctly
This pytest had to use the mock module in order to skip the api and cloud calls
A lot of values were set to make it work, but this test is to test the logic 
"""

from sentence_transformers import SentenceTransformer, util
import json
from app.client_data_manager import ClientDataManager
from unittest.mock import MagicMock, patch  # as with SentenceTransformer it takes a while to load

# loading times are still very long, so trying to further reduce it
#@patch('app.client_data_manager.SentenceTransformer') # first attempt -- ignore st in .py
#@patch('sentence_transformers.SentenceTransformer') # second attempt -- ignore all st
@patch.object(ClientDataManager, '__init__', return_value=None)
def test_vector_database(mock_init):
    # need to "catch" the value from patch ^^
    query = "a forest"
    mock_model = MagicMock() # to replace sentence transformer
    # we need to manually set a few member variables to continue

    # the encoder -- to avoid long loading times we are using mock
    #model = SentenceTransformer(cdm._semantic_model_name)
    mock_model.encode.return_value = np.array([[1.0, 0.0, 1.0]])

    cdm = ClientDataManager()  # actual object
    cdm._encoder = mock_model

    # temp database
    cdm._semantic_embeddings = np.array( [
        [0.9, 0, 0.9], # close to encoder value?
        [0.0, 1.0, 0.0] # far from it
    ])

    # IDs
    cdm._semantic_ids = ["model_1", "model_2"] # arbitrary to lookup?

    # directory for the IDs to lookup information for
    cdm._model_lookup = {
        "model_1": {"name": "tree"},
        "model_2": {"name": "ninja"}
    }

    results = cdm.semantic_search(query)

    assert len(results) > 0
    assert results[0]["model"]["name"] == "tree"

from app.scene_brain import SceneBrain
"""
Pytest #4 
Keyword extraction from user input (updated to SceneBrain)
"""
#@pytest.mark.skip(reason="successful")
def test_user_input_keywords_extraction():
    sb = SceneBrain()
    extracted = sb._extract_objects_and_quantities("I want a tree and a flower")
    keywords = {k for k, _ in extracted}
    assert "tree" in keywords
    assert "flower" in keywords


"""
Pytest #5
Placement type inference
"""
#@pytest.mark.skip(reason="successful")
def test_placement_type_inference():
    sb = SceneBrain()
    assert sb._get_placement_type("bird") == "floating"
    assert sb._get_placement_type("chair") == "ground"


"""
Pytest #6
Quantity parsing with numbers and words
"""
#@pytest.mark.skip(reason="successful")
def test_extract_objects_and_quantities_counts():
    sb = SceneBrain()
    extracted = sb._extract_objects_and_quantities("3 trees and two chairs")
    counts = {k: v for k, v in extracted}
    assert counts.get("tree") == 3
    assert counts.get("chair") == 2


"""
Pytest #7
Cosine similarity sanity check (tree vs forest closer than chair)
"""
#@pytest.mark.skip(reason="successful")
def test_cosine_similarity_basic():
    model = SentenceTransformer("paraphrase-MiniLM-L3-v2")
    embeddings = model.encode(["tree", "forest", "chair"], normalize_embeddings=True)
    scores = util.cos_sim(embeddings[0], embeddings[1:]).squeeze()  # shape (2,)
    assert float(scores[0]) > float(scores[1])


"""
Pytest #8
Placement rule application on scene nodes
"""
from app.layout_engine import SceneNode, Transform


def test_apply_placement_rules():
    lg = LayoutEngine()

    # Character: should zero out x/z rotation and ground on y=0
    char_node = SceneNode(
        model_id="m1",
        instance_id="i1",
        filename="f1",
        display_name="Hero",
        transform=Transform(position=np.array([0.0, 2.0, 0.0]), rotation=np.array([15.0, 30.0, 20.0]), scale=1.0),
        bbox_size=np.array([1.0, 1.0, 1.0]),
        placement_type="character",
        children=[]
    )

    # Floating: should set y to [5, 15]
    float_node = SceneNode(
        model_id="m2",
        instance_id="i2",
        filename="f2",
        display_name="Balloon",
        transform=Transform(position=np.array([0.0, 0.0, 0.0]), rotation=np.array([0.0, 0.0, 0.0]), scale=1.0),
        bbox_size=np.array([1.0, 1.0, 1.0]),
        placement_type="floating",
        children=[]
    )

    # Ground: should force y=0
    ground_node = SceneNode(
        model_id="m3",
        instance_id="i3",
        filename="f3",
        display_name="Rock",
        transform=Transform(position=np.array([0.0, 3.0, 0.0]), rotation=np.array([0.0, 0.0, 0.0]), scale=1.0),
        bbox_size=np.array([1.0, 1.0, 1.0]),
        placement_type="ground",
        children=[]
    )

    lg._apply_placement_rules([char_node, float_node, ground_node])

    # Character upright and grounded
    assert char_node.transform.rotation[0] == 0
    assert char_node.transform.rotation[2] == 0
    assert char_node.transform.position[1] == 0

    # Floating height in band
    assert 5.0 <= float_node.transform.position[1] <= 15.0

    # Ground node forced to ground
    assert ground_node.transform.position[1] == 0


"""
Pytest #9
Expand objects respects count and naming
"""
from app.scene_brain import SceneObject


def test_expand_objects():
    lg = LayoutEngine()
    scene_objects = [
        SceneObject(
            model_id="tree01",
            display_name="Tree",
            filename="tree.glb",
            count=2,
            tags=["nature"],
            placement_type="ground"
        )
    ]

    expanded = lg._expand_objects(scene_objects)

    # Expect two entries with suffixed names
    assert len(expanded) == 2
    names = {obj.display_name for obj in expanded}
    assert "Tree_1" in names and "Tree_2" in names