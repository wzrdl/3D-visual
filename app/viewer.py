"""
3D viewer using PyVista
widget for displaying 3D models in the main window
"""
from PyQt6.QtWidgets import QWidget
from PyQt6.QtCore import Qt
import pyvista as pv
from pyvistaqt import QtInteractor


class ThreeDViewer(QtInteractor):
    """3D model viewer widget"""
    
    def __init__(self, parent=None):
        """init viewer"""
        super().__init__(parent)
        self.setup_renderer()
    
    def setup_renderer(self):
        """setup renderer settings"""
        self.background_color = '#2b2b2b'
        self.enable_trackball_style()
        self.add_axes_widget()
    
    def load_model(self, file_path: str) -> bool:
        """load and show a 3D model"""
        # TODO: implement in week 2 - load from assets folder
        pass
    
    def clear(self):
        """clear all models"""
        self.renderer.clear()

