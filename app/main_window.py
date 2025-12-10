"""
main window for the app
manages the tabbed interface and coordinates between pages
"""
import os
from pathlib import Path
from PyQt6.QtWidgets import QMainWindow, QWidget, QVBoxLayout, QTabWidget, QLabel, QProgressBar
from PyQt6.QtCore import Qt, QThread, pyqtSignal
from PyQt6.QtGui import QCloseEvent
from app.client_data_manager import ClientDataManager
from app.pages import GalleryPage, AIGenerationPage, ViewerPage, SceneGeneratorPage

# Don't show VTK warnings during cleanup 
os.environ['VTK_SILENCE_GET_VOID_POINTER_WARNINGS'] = '1'
# 0 means no logging, 9 means all logging
os.environ['VTK_LOGGING_LEVEL'] = 'OFF'


class InitWorker(QThread):
    """Background worker to initialize data manager without freezing UI."""

    progress_update = pyqtSignal(str)
    finished = pyqtSignal(object)  # Emits the initialized ClientDataManager

    def run(self):
        manager = ClientDataManager(defer_initialization=True)

        def callback(msg: str):
            self.progress_update.emit(msg)

        try:
            manager.initialize(progress_callback=callback)
            self.finished.emit(manager)
        except Exception as e:
            self.progress_update.emit(f"Error: {e}")
            self.finished.emit(manager)


class LoadingWidget(QWidget):
    """Simple startup screen with status text."""

    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setAlignment(Qt.AlignmentFlag.AlignCenter)

        title = QLabel("3D Model Generator\n& Library")
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        title.setStyleSheet(
            "font-size: 32px; font-weight: bold; color: #333; margin-bottom: 20px;"
        )
        layout.addWidget(title)

        self.progress_bar = QProgressBar()
        self.progress_bar.setFixedWidth(400)
        self.progress_bar.setRange(0, 0)  # Indeterminate mode (loading animation)
        self.progress_bar.setStyleSheet(
            """
            QProgressBar {
                border: 2px solid grey;
                border-radius: 5px;
                text-align: center;
                height: 25px;
            }
            QProgressBar::chunk {
                background-color: #4CAF50;
                width: 20px;
            }
        """
        )
        layout.addWidget(self.progress_bar)

        self.status_label = QLabel("Initializing...")
        self.status_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.status_label.setStyleSheet("color: #666; font-size: 14px; margin-top: 10px;")
        layout.addWidget(self.status_label)

    def update_status(self, text: str):
        self.status_label.setText(text)


class MainWindow(QMainWindow):
    """The main window that holds all the tabs."""

    def __init__(self):
        """Class constructor."""
        super().__init__()
        self.setWindowTitle("3D Model and Scene Generator & Gallery")
        self.setGeometry(100, 100, 1200, 800)

        self.data_manager = None
        self.init_thread: QThread | None = None

        self.loading_widget = LoadingWidget()
        self.setCentralWidget(self.loading_widget)

        self.start_initialization()

    def start_initialization(self):
        self.init_thread = InitWorker()
        self.init_thread.progress_update.connect(self.loading_widget.update_status)
        self.init_thread.finished.connect(self.on_initialization_complete)
        self.init_thread.start()

    def on_initialization_complete(self, data_manager: ClientDataManager):
        self.data_manager = data_manager
        if self.init_thread:
            self.init_thread.deleteLater()
            self.init_thread = None
        self.setup_ui()

    def setup_ui(self):
        """Create the tabs and add all the pages."""
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
            data_manager=self.data_manager,
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
        self.tabs.setCurrentIndex(2)
        self.viewer_page.load_model(str(model_path))
        print(f"Loaded model: {model_name}")

    def closeEvent(self, event: QCloseEvent):
        """Clean up everything when the window closes"""
        if self.init_thread and self.init_thread.isRunning():
            self.init_thread.quit()
            self.init_thread.wait(2000)

        if hasattr(self, "viewer_page") and self.viewer_page:
            self.viewer_page.cleanup()

        if hasattr(self, "scene_generator_page") and self.scene_generator_page:
            self.scene_generator_page.cleanup()

        if hasattr(self, "data_manager") and self.data_manager:
            try:
                if hasattr(self.data_manager, "clear_cache"):
                    self.data_manager.clear_cache()
            except Exception:
                pass
            self.data_manager.close()

        event.accept()