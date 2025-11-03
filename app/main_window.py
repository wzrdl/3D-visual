"""
main window for the app
has gallery browsing and AI generation stuff
"""
from PyQt6.QtWidgets import QMainWindow, QWidget, QVBoxLayout, QHBoxLayout
from PyQt6.QtCore import Qt
from app.viewer import ThreeDViewer
from app.data_manager import DataManager


class MainWindow(QMainWindow):
    """main window"""
    
    def __init__(self):
        """init main window"""
        super().__init__()
        self.setWindowTitle("3D Model Generator & Library")
        self.setGeometry(100, 100, 1200, 800)
        
        self.data_manager = DataManager()
        self.setup_ui()
    
    def setup_ui(self):
        """setup UI layout"""
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        main_layout = QHBoxLayout(central_widget)
        
        # left panel - gallery/search (week 3)
        left_panel = QWidget()
        left_panel.setMaximumWidth(300)
        left_layout = QVBoxLayout(left_panel)
        # TODO: gallery UI in week 3
        
        main_layout.addWidget(left_panel)
        
        # right panel - 3D viewer (week 4)
        # TODO: integrate viewer in week 4
        viewer_placeholder = QWidget()
        viewer_placeholder.setStyleSheet("background-color: #2b2b2b;")
        
        main_layout.addWidget(viewer_placeholder, stretch=1)

