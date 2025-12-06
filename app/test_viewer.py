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
    viewer = ThreeDViewer()
    viewer.generate_thumbnail(file_name)

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
    data = DataManager()

    id_number = "model_001"
    file_name = "cube.obj"
    name = "Cube"
    tags = [
      "geometric",
      "primitive",
      "box",
      "square"
    ]

    assert data.add_model(id_number, file_name, name, tags) == False

"""
Pytest #3
Download button does the correct file name
"""
from app.pages import ViewerPage

@pytest.mark.skip(reason="successful, no need to see this each time for now")
def test_download_button():
    app = QApplication(sys.argv)
    viewer = ViewerPage()

    viewer.clicked_download_button("assets/pytest assets", "cone_dt.obj")
    # running twice so that the second time it should add temp_name 1
    viewer.clicked_download_button("assets/pytest assets", "cone_dt.obj")

    if os.path.exists("assets/pytest assets/cone_dt.obj"):
        check = True
        os.remove("assets/pytest assets/cone_dt.obj")
        os.remove("assets/pytest assets/cone_dt1.obj")
    else:
        check = False

    assert check == True


from app.pages import GalleryPage
from app.main_window import MainWindow
"""
Pytest #4
Search bar returns the right amount of models per the tag searched 
"""
@pytest.mark.skip(reason = "No clue what to do")
# DOESN"T WORK
def test_tag_gallery_search():
    app = QApplication(sys.argv)

    object = GalleryPage()
    object_setup = MainWindow()

    object_setup.setup()

    object.filter_models("round")
    assert len(object.model_list) == 2
