"""
main window for the app
manages the tabbed interface and coordinates between pages
"""
import sys
import os
from pathlib import Path
from PyQt6.QtWidgets import QMainWindow, QWidget, QVBoxLayout, QTabWidget
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QCloseEvent
from app.data_manager import DataManager
from app.pages import GalleryPage, AIGenerationPage, ViewerPage

# Suppress VTK warnings during cleanup
# os.environ['VTK_LOGGING_LEVEL'] = 'ERROR'


class MainWindow(QMainWindow):
    """The main window that holds all the tabs"""
    
    def __init__(self):
        """Create the main window"""
        super().__init__()
        self.setWindowTitle("3D Model Generator & Library")
        self.setGeometry(100, 100, 1200, 800)
        
        self.data_manager = DataManager()
        self.setup_ui()
    
    def setup_ui(self):
        """Create the tabs and add all the pages"""
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        main_layout = QVBoxLayout(central_widget)
        main_layout.setContentsMargins(0, 0, 0, 0)
        
        # Create tab widget for multiple pages
        self.tabs = QTabWidget()
        self.tabs.setTabPosition(QTabWidget.TabPosition.North)
        
        # Page 1: Gallery (with callback to viewer page)
        self.gallery_page = GalleryPage(
            data_manager=self.data_manager,
            viewer_page_callback=self.on_model_selected_callback
        )
        self.tabs.addTab(self.gallery_page, "Gallery")
        
        # Page 2: AI Generation
        self.ai_generation_page = AIGenerationPage()
        self.tabs.addTab(self.ai_generation_page, "AI Generation")
        
        # Page 3: 3D Viewer
        self.viewer_page = ViewerPage()
        self.tabs.addTab(self.viewer_page, "3D Viewer")

        """
        # link buttons to this
        self.viewer_page.download_button.clicked.connect(self.clicked_download_button)
        self.viewer_page.light_button.clicked.connect(self.toggle_light_button)
        self.viewer_page.gallery_button.clicked.connect(self.clicked_gallery_button)
        """

        main_layout.addWidget(self.tabs)
        central_widget.setLayout(main_layout)
    
    def on_model_selected_callback(self, model_path: Path, model_name: str):
        """Called when someone picks a model from the gallery - switch to viewer and load it"""
        # Switch to viewer tab (index 2 - third tab)
        self.tabs.setCurrentIndex(2)
        # Load model in viewer
        self.viewer_page.load_model(str(model_path))
        print(f"Loaded model: {model_name}")

    def closeEvent(self, event: QCloseEvent):
        """Clean up everything when the window closes"""
        # Clean up viewer page
        if hasattr(self, 'viewer_page') and self.viewer_page:
            self.viewer_page.cleanup()
        
        # Close database connection
        if hasattr(self, 'data_manager') and self.data_manager:
            self.data_manager.close()
        
        event.accept()