"""
3D viewer using PyVista
widget for displaying 3D models in the main window.

We deliberately keep the rendering stack simple:
- All 3D files (OBJ, GLB, etc.) are loaded via `pyvista.read` and rendered
  without any custom GLB parsing logic.
- This avoids extra dependencies and keeps the behavior predictable.
"""
import contextlib
import os
import sys
from pathlib import Path

import pyvista as pv
from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QApplication, QWidget
from pyvistaqt import QtInteractor


@contextlib.contextmanager
def suppress_vtk_warnings():
    """
    Temporarily silence VTK/OpenGL warnings that are printed to stderr,
    e.g. wglMakeCurrent errors on Windows during off‑screen rendering.
    """
    original_stderr = sys.stderr
    try:
        with open(os.devnull, "w") as devnull:
            sys.stderr = devnull
            yield
    finally:
        sys.stderr = original_stderr


class ThreeDViewer(QtInteractor):
    """3D model viewer widget"""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setup_renderer()  # calls the function to set up renderer settings
        # QtInteractor IS a plotter, so we use self directly for plotter operations
        self.clear()  # clears everything

    def setup_renderer(self):
        """setup renderer settings"""
        self.background_color = "#ffffff"
        self.enable_trackball_style()
        self.show_axes()

    @staticmethod
    def load_model(file_path: str) -> pv.PolyData:
        """load a 3D model from file path and return the mesh"""
        print("reading file path", file_path)
        # Let PyVista/VTK decide which reader to use based on extension.
        return pv.read(file_path)

    @staticmethod
    def setup_light():
        """Create a light that shines straight down from above the model."""
        light = pv.Light(
            position=(0, 0, 10),  # directly above in +Z
            focal_point=(0, 0, 0),  # pointing to scene center
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
        with suppress_vtk_warnings():
            thumbnail = pv.Plotter(off_screen=True, window_size=[600, 600])
            mesh = ThreeDViewer.load_model(file_path)
            thumbnail.add_mesh(mesh)
            thumbnail.set_background("white")

        # Cross‑platform file/thumbnail paths
        src_path = Path(file_path)
        file_name = f"{src_path.stem}.png"
        print(file_name)

        # Thumbnails live under project_root/assets/thumbnails
        project_root = Path(__file__).parent.parent
        thumbnails_dir = project_root / "assets" / "thumbnails"
        thumbnails_dir.mkdir(parents=True, exist_ok=True)
        thumbnail_path = thumbnails_dir / file_name

        # positioning the camera
        thumbnail.view_isometric()
        # to include the full model
        thumbnail.reset_camera()
        # optional zoom
        thumbnail.camera.zoom(0.9)
        thumbnail.screenshot(str(thumbnail_path))
        thumbnail.close()  # IMPORTANT

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