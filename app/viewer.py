"""
3D viewer using PyVista
widget for displaying 3D models in the main window
"""
import sys

from PyQt6.QtWidgets import QWidget, QApplication
from PyQt6.QtCore import Qt
import pyvista as pv
from pyvista import examples
from pyvistaqt import QtInteractor


class ThreeDViewer(QtInteractor):
    """3D model viewer widget"""

    def __init__(self, parent=None):

        super().__init__(parent)
        self.setup_renderer() # calls the function to set up renderer settings

        self.plotter = pv.Plotter() # sets up a variable to hold the 3D model plot

        # this calls the load_model function, with the file path as an input
        #self.load_model('C:\\Users\lizzy\OneDrive\Documents\GitHub\\3D-visual\\assets\models\cone.obj')
        #self.load_model('/Users/newt/Desktop/3D-visual/assets/models/cone.obj')
        # need the ABSOLUTE location for this to work ^^ (this would be mine)
        # as of 11/9/25, do not need to load model here. Subject to change as we figure stuff out

        self.clear() # clears everything

    def setup_renderer(self):
        """setup renderer settings"""
        self.background_color = '#2b2b2b'
        self.enable_trackball_style()
        #self.add_axes_widget() # ERROR for some reason, this would not let it run

    def load_model(self, file_path: str) -> bool:
        """load and show a 3D model"""
        """
        This function takes in a file path, reads it with Pyvista, and them displays it in a new window
        param: file_path -- the absolute path of the file that contains the 3D model
        """

        print("reading file path", file_path)
        mesh = pv.read(file_path)
        #self.plotter.add_mesh(mesh)
        #self.plotter.show()
        return mesh

    def clear(self):
        """clear all models"""
        # Clear all meshes from the plotter
        self.plotter.clear()

"""
# to test if it works -- remove quotations
app = QApplication(sys.argv) # main app setup?

window = ThreeDViewer() # variable to hold main window
window.show() # IMPORTANT -- so we can actually see it

app.exec_() # executing the app
"""