"""
3D viewer using PyVista
widget for displaying 3D models in the main window.

We deliberately keep the rendering stack simple:
- All 3D files (OBJ, GLB) are loaded via `pyvista.read` and rendered
  without any custom GLB parsing logic.
- This avoids extra dependencies and keeps the behavior predictable.
"""
import contextlib
import os
import sys
from pathlib import Path

import pyvista as pv
import numpy as np
from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QApplication, QWidget
from pyvistaqt import QtInteractor


@contextlib.contextmanager
def suppress_vtk_warnings():
    """
    Temporarily silence VTK/OpenGL warnings that are printed to stderr,
    When using windows, it will show a lot of warnings that are not important,
    so we suppress them.
    """
    # Try to use VTK's own silence mechanism
    try:
        import vtkmodules.all as vtk
        output = vtk.vtkOutputWindow.GetInstance()
        if output:
            output.GlobalWarningDisplayOff()
    except ImportError:
        pass
    
    original_stderr = sys.stderr
    try:
        with open(os.devnull, "w") as devnull:
            sys.stderr = devnull
            yield
    finally:
        sys.stderr = original_stderr



class ThreeDViewer(QtInteractor):
    """3D model viewer widget, inherits from QtInteractor"""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setup_renderer()
        self.clear()  # clears everything

    def setup_renderer(self):
        """setup renderer settings"""
        self.background_color = "#ffffff"
        self.enable_trackball_style()
        self.show_axes()

    @staticmethod
    def load_model(file_path: str) -> pv.PolyData:
        """load a 3D model from file path and return the mesh"""
        #print("reading file path", file_path)
        return pv.read(file_path)

    @staticmethod
    def ensure_polydata(mesh: pv.DataSet) -> pv.PolyData:
        """
        Ensure a PolyData is returned.
        """
        if isinstance(mesh, pv.PolyData):
            return mesh
            
        if mesh is None:
            return pv.PolyData()

        try:
            return mesh.extract_geometry()
        except Exception:
            return pv.PolyData()

    @staticmethod
    def normalize_mesh(mesh: pv.PolyData) -> pv.PolyData:
        """
        Normalize mesh to unit size and center it.
        This prevents huge models from clipping or being hard to navigate. This function is used for glb data.
        """
        norm_mesh = ThreeDViewer.ensure_polydata(mesh).copy()
        
        # 1. Center
        center = norm_mesh.center
        norm_mesh.points -= center
        
        # 2. Normalize scale
        bounds = norm_mesh.bounds
        size = np.array([
            bounds[1] - bounds[0],
            bounds[3] - bounds[2],
            bounds[5] - bounds[4]
        ])
        max_dim = np.max(size)
        
        if max_dim > 0:
            scale_factor = 1.0 / max_dim
            norm_mesh.points *= scale_factor
            
            # Scale up slightly to be visible in default view (e.g. size 2.0 approx)
            norm_mesh.points *= 2.0
            
        return norm_mesh


    # TODO: If possible, change the light setup， right now it faces the object directly
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
            mesh = ThreeDViewer.ensure_polydata(mesh)
            thumbnail.add_mesh(mesh)
            thumbnail.set_background("white")

        # Cross‑platform file/thumbnail paths, works on both windows and macos
        src_path = Path(file_path)
        file_name = f"{src_path.stem}.png"
        print(file_name)


        project_root = Path(__file__).parent.parent
        thumbnails_dir = project_root / "assets" / "thumbnails"
        thumbnails_dir.mkdir(parents=True, exist_ok=True)
        thumbnail_path = thumbnails_dir / file_name

        # Position camera : upper-left diagonal
        bounds = mesh.bounds
        center = mesh.center
        extent = max(bounds[1] - bounds[0], bounds[3] - bounds[2], bounds[5] - bounds[4], 1.0)
        distance = max(extent * 2.0, 2.5)
        camera_height = distance * 0.7
        camera_dist = distance * 0.7
        camera_pos = (
            center[0] - camera_dist,
            center[1] + camera_height,
            center[2] + camera_dist,
        )
        thumbnail.camera_position = [camera_pos, center, (0, 1, 0)]
        thumbnail.reset_camera()
        thumbnail.screenshot(str(thumbnail_path))
        thumbnail.close()  # IMPORTANT

    def clear(self):
        """clear all models from the viewer"""
        # Clear all meshes from the plotter
        super().clear()


"""
# to test if it works -- remove quotations
app = QApplication(sys.argv) # main app setup?

window = ThreeDViewer() # variable to hold main window
window.show() # IMPORTANT -- so we can actually see it

app.exec_() # executing the app
"""