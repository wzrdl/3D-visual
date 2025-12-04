"""
Page classes for the application
Each page is a separate class with its own functionality
"""
import os

from PyQt6.QtGui import QImage, QIcon
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QLineEdit, QListWidget,
    QListWidgetItem, QPushButton, QTextEdit, QLabel, QHBoxLayout, QGridLayout, QListView
)
from PyQt6.QtCore import Qt, QSize
from app.viewer import ThreeDViewer
from app.data_manager import DataManager

# to download files
import shutil
import pathlib


class BasePage(QWidget):
    """Base class that all pages inherit from"""
    
    def __init__(self, parent=None):
        """Create a new page"""
        super().__init__(parent)
        self.setup_ui()
    
    def setup_ui(self):
        """Build the UI - each page needs to implement this"""
        raise NotImplementedError("Subclasses must implement setup_ui()")


class GalleryPage(BasePage):
    """The gallery page where you can browse and search through your 3D models"""
    
    def __init__(self, data_manager: DataManager, viewer_page_callback=None, parent=None):
        """Create the gallery page. Needs a data manager to load models and a callback to open them in the viewer"""
        self.data_manager = data_manager
        self.viewer_page_callback = viewer_page_callback
        self.all_models = []
        super().__init__(parent)
        self.load_models()
    
    def setup_ui(self):
        """Put together the search bar and model list"""
        layout = QGridLayout(self) # was QVBox Layout
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(10)

        # Search bar
        self.search_bar = QLineEdit()
        self.search_bar.setMaxLength(50)
        self.search_bar.setPlaceholderText("Search models by name or tags...")
        self.search_bar.setAlignment(Qt.AlignmentFlag.AlignLeft)
        self.search_bar.textChanged.connect(self.filter_models)
        layout.addWidget(self.search_bar)

        # Model list widget
        self.model_list = QListWidget()
        self.model_list.setAlternatingRowColors(True)
        self.model_list.itemClicked.connect(self.on_model_selected)
        self.model_list.setFlow(QListView.Flow.LeftToRight)
        self.model_list.setWrapping(True)
        self.model_list.setResizeMode(QListWidget.ResizeMode.Adjust)

        layout.addWidget(self.model_list)
    
    def load_models(self):
        """Grab all models from the database and show them in the list"""

        # Currently we just get all models from the database and show them in the list
        # TODO: We may find a better way to load models in the future
        self.all_models = self.data_manager.get_all_models()
        self.populate_model_list(self.all_models)
    
    def populate_model_list(self, models):
        """Fill the list with model names"""
        self.model_list.clear()

        # setup for the gallery view
        self.model_list.setWordWrap(True)
        self.model_list.setIconSize(QSize(150, 150))
        self.model_list.setGridSize(QSize(175, 175))
        self.model_list.setViewMode(QListWidget.ViewMode.IconMode)

        for model in models:
            # model name
            item = QListWidgetItem()
            item.setText(model['display_name'])
            item.setData(Qt.ItemDataRole.UserRole, model)

            # model thumbnail
            model_name = model['filename']
            thumbnail_path = ".\\assets\\thumbnails\\" + model_name[:-4] + ".png"
            if os.path.exists(thumbnail_path) == False:
                thumbnail_object = ThreeDViewer()
                model_path = ".\\assets\\models\\" + model_name
                thumbnail_object.generate_thumbnail(model_path)
            thumbnail = QIcon(thumbnail_path)
            item.setIcon(thumbnail)

            self.model_list.addItem(item)

    
    def filter_models(self, query: str):
        """Update the list as the user types in the search box"""
        query = query.strip()
        if not query:
            self.populate_model_list(self.all_models)
        else:
            filtered_models = self.data_manager.search_models(query)
            self.populate_model_list(filtered_models)
    
    def on_model_selected(self, item: QListWidgetItem):
        """When someone clicks a model, open it in the 3D viewer"""
        model = item.data(Qt.ItemDataRole.UserRole)
        if not model:
            return
        
        # Get model path using DataManager
        model_path = self.data_manager.get_model_path(model['id'])
        if not model_path or not model_path.exists():
            print(f"Model file not found: {model_path}")
            return
        
        # Call callback to switch to viewer page and load model
        if self.viewer_page_callback:
            self.viewer_page_callback(model_path, model['display_name'])


class AIGenerationPage(BasePage):
    """Page where you can type a description and have AI generate a 3D model"""
    
    def __init__(self, parent=None):
        """Create the AI generation page"""
        super().__init__(parent)
    
    def setup_ui(self):
        """Build the prompt input and generate button"""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(20, 20, 20, 20)
        layout.setSpacing(15)

        # Title/Header
        title_label = QLabel("AI 3D Model Generation")
        title_label.setStyleSheet("font-size: 18px; font-weight: bold; margin-bottom: 10px;")
        layout.addWidget(title_label)

        # Description
        desc_label = QLabel("Enter a text prompt to generate a 3D model using AI")
        desc_label.setStyleSheet("color: #666; margin-bottom: 20px;")
        layout.addWidget(desc_label)

        # Text input area (like ChatGPT)
        self.prompt_input = QTextEdit()
        self.prompt_input.setPlaceholderText(
            "Describe the 3D model you want to generate...\n\n"
            "Example: 'A red sports car' or 'A medieval castle with towers'"
        )
        self.prompt_input.setMinimumHeight(200)
        layout.addWidget(self.prompt_input)

        # Generate button
        self.generate_button = QPushButton("Generate")
        self.generate_button.setMinimumHeight(40)
        self.generate_button.setStyleSheet("""
            QPushButton {
                background-color: #4CAF50;
                color: white;
                border: none;
                border-radius: 5px;
                font-size: 14px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #45a049;
            }
            QPushButton:pressed {
                background-color: #3d8b40;
            }
        """)
        # TODO: Connect to API client in Week 5
        self.generate_button.clicked.connect(self.on_generate_clicked)
        layout.addWidget(self.generate_button)

        # Add stretch to push content to top
        layout.addStretch()
    
    def get_prompt(self) -> str:
        """Returns whatever text the user typed in"""
        return self.prompt_input.toPlainText().strip()

    def on_generate_clicked(self):
        """When the generate button is pressed"""
        print("generate clicked")
        return

    def clear_prompt(self):
        """Wipe out the text in the input box"""
        self.prompt_input.clear()


class ViewerPage(BasePage):
    """The 3D viewer page where models get displayed"""
    
    def __init__(self, parent=None):
        """Create the viewer page"""
        self.viewer = None
        super().__init__(parent)
    
    def setup_ui(self):
        """Set up buttons in the viewer tab"""
        button_layout = QHBoxLayout(self)
        button_layout.setContentsMargins(0, 0, 0, 0)
        button_layout.setSpacing(0)

        # button_style_template copied from generate_button to be consistent
        button_style_template = """
            QPushButton {
                background-color: #4CAF50;
                color: white;
                border: none;
                border-radius: 5px;
                font-size: 14px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #45a049;
            }
            QPushButton:pressed {
                background-color: #3d8b40;
            }
        """
        # Download button -- copied from generate_button
        self.download_button = QPushButton("Download")
        self.download_button.setMinimumHeight(40)
        self.download_button.setStyleSheet(button_style_template)

        # Toggle Light button
        self.light_button = QPushButton("Light: On")
        self.light_button.setMinimumHeight(40)
        self.light_button.setStyleSheet(button_style_template)

        # Toggle Gallery button
        self.gallery_button = QPushButton("Add to Gallery")
        self.gallery_button.setMinimumHeight(40)
        self.gallery_button.setStyleSheet(button_style_template)

        # combining the buttons in a layout
        button_layout.addWidget(self.download_button)
        button_layout.addWidget(self.light_button)
        button_layout.addWidget(self.gallery_button)

        # making a widget hold the layout so we can nest layouts
        button_layout_widget = QWidget(self)
        button_layout_widget.setLayout(button_layout)

        """Set up the 3D viewer widget"""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        # 3D viewer - QtInteractor IS a QWidget, so we can add it directly
        self.viewer = ThreeDViewer(self)
        # Set minimum size to ensure viewer is visible
        self.viewer.setMinimumSize(400, 400)
        self.viewer.show_grid()

        # Add viewer directly to layout (QtInteractor is already a widget)
        layout.addWidget(button_layout_widget) # adding the top row of buttons
        layout.addWidget(self.viewer, stretch=1)

        self.download_button.clicked.connect(self.clicked_download_button)
        self.light_button.clicked.connect(self.toggle_light_button)
        self.gallery_button.clicked.connect(self.clicked_gallery_button)
    
    def load_model(self, model_path: str):
        """Load a model file and show it in the viewer"""
        """ takes in the path to load and the name to set for a file download name"""
        if not self.viewer:
            return
        
        try:
            # Clear previous model
            self.viewer.clear()
            # Load new model
            self.model_path = str(model_path) # so that download function can access it
            mesh = ThreeDViewer.load_model(model_path)
            self.viewer.add_mesh(mesh)
            self.light = ThreeDViewer.setup_light(self) # lighting
            self.viewer.add_light(self.light)
            self.viewer.reset_camera()
            print(f"Loaded model: {model_path}")
        except Exception as e:
            print(f"Error loading model: {e}")

        # resetting button text upon new model loading
        self.download_button.setText("Download")
        self.gallery_button.setText("Add to Gallery")
        self.light_button.setText("Light: On")

    # makes the file and display name from the model path
    def file_name_from_model_path(self):
        """ takes the model path and gets the [name].obj from it """

        if self.model_path != None:
            file_name = self.model_path.split('\\')[-1]
            return file_name
        else:
            return None

    # function for the next button click
    def file_name_exists(self, file_name):
        """ changes the filename so that it does not produce errors
            It does so by checking the last value for if it is a number or not """

        if file_name[-5].isdigit() == True:
            number = int(file_name[-5]) + 1
            string_number = str(number)
            file_name = file_name[:-5] + string_number + file_name[-4:]
        else:
            file_name = file_name[:-4] + "1" + file_name[-4:]

        return file_name

        # for buttons
    def clicked_download_button(self):
        """When the download button is pressed"""
        print("download clicked")
        self.download_button.setText("File Downloaded")
        file_name = self.file_name_from_model_path()

        download_path = pathlib.Path.home() / 'Downloads' # works for windows computer

        if os.path.exists(str(download_path / file_name)):
            file_name = self.file_name_exists(file_name)

        shutil.copyfile(self.model_path, file_name)
        shutil.move(file_name, download_path)

        return

    def toggle_light_button(self):
        """Toggles the light on or off"""

        if self.light.on:
            self.light.switch_off()
            self.light_button.setText("Light: Off")
        else:
            self.light.switch_on()
            self.light_button.setText("Light: On")
        return

    def clicked_gallery_button(self):
        """When the gallery button is pressed"""
        print("gallery clicked")

        # the following two lines tests the thumbnail generation
        #threeD = ThreeDViewer()
        #threeD.generate_thumbnail(self.model_path)

        gallery = DataManager()

        # need the following values to add to gallery

        #model id
        model_id = gallery.get_next_id()

        # filename
        filename = self.file_name_from_model_path()
        # display name
        display_name = filename.upper()[0] + filename[1:]

        """
        TO do
        Add tags and model data after AI implementation
        """
        # tags
        tags = [] # IMPORTANT ADD THIS --------------------------------

        #model data
        model_data = None # ADD THIS AFTER AI -------------------------

        """
        # gallery.add_model(model_id, filename, display_name, tags)
        added = gallery.save_model_to_gallery(model_id, filename, display_name, tags, model_data)
        if added == True:
            self.gallery_button.setText("Added to Gallery") # giving user feedback
        else:
            self.gallery_button.setText("Failed to add to Gallery")
        """
        return

    def clear(self):
        """Remove everything from the viewer"""
        if self.viewer:
            self.viewer.clear()
    
    def cleanup(self):
        """Free up OpenGL stuff when closing the window"""
        if self.viewer:
            try:
                self.viewer.clear()
                if hasattr(self.viewer, 'render_window') and self.viewer.render_window:
                    try:
                        self.viewer.render_window.Finalize()
                    except:
                        pass
                self.viewer.close()
            except Exception:
                pass

