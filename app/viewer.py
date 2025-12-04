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
        # QtInteractor IS a plotter, so we use self directly for plotter operations
        self.clear() # clears everything

    def setup_renderer(self):
        """setup renderer settings"""
        self.background_color = '#ffffff'
        self.enable_trackball_style()
        self.show_axes()

    @staticmethod
    def load_model(file_path: str):
        """load a 3D model from file path and return the mesh
        
        Args:
            file_path: The absolute or relative path of the file that contains the 3D model
            
        Returns:
            pyvista mesh object
        """
        print("reading file path", file_path)
        mesh = pv.read(file_path)
        return mesh

    @staticmethod
    def setup_light():
        """Create a light that shines straight down from above the model."""
        light = pv.Light(
            position=(0, 0, 10),   # directly above in +Z
            focal_point=(0, 0, 0), # pointing to scene center
            show_actor=True,
            positional=True,
            cone_angle=45,
            exponent=10,
        )
        return light

    def generate_thumbnail(self, file_path: str):
        """generate a screenshot/2D thumbnail from a 3D model"""

        # off_screen means that the user will not see this happening
        # window_size is pixel x pixel
        thumbnail = pv.Plotter(off_screen=True, window_size=[600,600])
        mesh = pv.read(file_path)
        thumbnail.add_mesh(mesh)
        thumbnail.set_background("white")

        # works for windows -- idk about mac
        file_name = file_path.split('\\')[-1].replace('.obj', '.png')
        print(file_name)
        # positioning the camera
        thumbnail.view_isometric()
        # to include the full model
        thumbnail.reset_camera()
        # optional zoom
        thumbnail.camera.zoom(0.9)
        thumbnail.screenshot(f"assets\\thumbnails\\{file_name}")
        thumbnail.close() # IMPORTANT

    def clear(self):
        """clear all models from the viewer"""
        # Clear all meshes from the plotter
        # Since QtInteractor IS a plotter, we use self directly
        super().clear()

"""
# to test if it works -- remove quotations
app = QApplication(sys.argv) # main app setup?

window = ThreeDViewer() # variable to hold main window
window.show() # IMPORTANT -- so we can actually see it

app.exec_() # executing the app
"""