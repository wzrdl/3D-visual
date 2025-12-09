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
from app.client_data_manager import ClientDataManager
from app.pages import GalleryPage, AIGenerationPage, ViewerPage, SceneGeneratorPage

# Don't show VTK warnings during cleanup 
os.environ['VTK_SILENCE_GET_VOID_POINTER_WARNINGS'] = '1'
# 0 means no logging, 9 means all logging
os.environ['VTK_LOGGING_LEVEL'] = 'OFF'

class MainWindow(QMainWindow):
    """The main window that holds all the tabs"""
    
    def __init__(self):
        """Class constructor, we set the window title and geometry, and create the data manager"""
        super().__init__()
        self.setWindowTitle("3D Model Generator & Library")
        self.setGeometry(100, 100, 1200, 800)
        
        # Use the FastAPI-based client DataManager (local cache only)
        self.data_manager = ClientDataManager()
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
        
        # Page 1: Gallery Page
        self.gallery_page = GalleryPage(
            data_manager=self.data_manager,
            viewer_page_callback=self.on_model_selected_callback,
        )
        self.tabs.addTab(self.gallery_page, "Gallery")

        # Page 2: AI Generation Page
        self.ai_generation_page = AIGenerationPage(
            # We need to pass the data manager 
            # so the AI generation page can upload the model to the backend
            data_manager=self.data_manager, 
            # We need to pass the gallery page so the AI generation page can refresh the gallery
            gallery_page=self.gallery_page,
        )
        self.tabs.addTab(self.ai_generation_page, "AI Generation")
        
        # Page 3: 3D Viewer Page
        self.viewer_page = ViewerPage()
        self.tabs.addTab(self.viewer_page, "3D Viewer")

        # Page 4: Scene Generator Page (Smart Scene Composer)
        self.scene_generator_page = SceneGeneratorPage(
            data_manager=self.data_manager,
        )
        self.tabs.addTab(self.scene_generator_page, "🎬 Scene Generator")

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
        
        # Clean up scene generator page
        if hasattr(self, 'scene_generator_page') and self.scene_generator_page:
            self.scene_generator_page.cleanup()

        # Clear local cache and close the client
        if hasattr(self, 'data_manager') and self.data_manager:
            try:
                # Delete cached model files
                if hasattr(self.data_manager, "clear_cache"):
                    self.data_manager.clear_cache()
            except Exception:
                pass
            self.data_manager.close()
        
        event.accept()