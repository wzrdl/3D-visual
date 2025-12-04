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

@pytest.mark.skip(reason="successful, no need to see this each time for now")
# WORKS
def test_generate_thumbnail():
    app = QApplication(sys.argv)

    file_name = "assets\\pytest assets\\cone_test.obj"
    object = ThreeDViewer()
    object.generate_thumbnail(file_name)

    # checks if it makes the png file
    check = os.path.exists("assets\\thumbnails\\cone_test.png")

    # cleans up the new file
    if check == True:
        os.remove("assets\\thumbnails\\cone_test.png")

    assert check == True

"""
Pytest #2
Passing the add_model function something that does exist
"""
from app.data_manager import DataManager

@pytest.mark.skip(reason="successful, no need to see this each time for now")
# WORKS
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

from app.pages import GalleryPage
"""
Pytest #3
Search bar returns the right amount of models per the tag searched 
"""
@pytest.skip(reason = "Not finished")
def test_tag_gallery_search():
    app = QApplication(sys.argv)

    object = GalleryPage()
    object.filter_models("round")



""" 
Pytest #4

"""

"""
Pytest #5

"""
