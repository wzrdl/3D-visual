"""
Page classes for the application
Each page is a separate class with its own functionality
"""
import os
import asyncio
import re
from pathlib import Path

from PyQt6.QtGui import QImage, QIcon
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QLineEdit, QListWidget,
    QListWidgetItem, QPushButton, QTextEdit, QLabel, QHBoxLayout, QGridLayout, QListView
)
from PyQt6.QtCore import Qt, QSize
from app.viewer import ThreeDViewer
from app.meshy_client import MeshyClient
from app.client_data_manager import ClientDataManager


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
    
    def __init__(self, data_manager, viewer_page_callback=None, parent=None):
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

            # Ensure we have a local copy of the model before generating thumbnail.
            # This will download from backend/GCS if needed.
            if not os.path.exists(thumbnail_path):
                model_id = model.get("id")
                model_path_fs: str | None = None
                try:
                    if model_id is not None and hasattr(self.data_manager, "get_model_path"):
                        path_obj = self.data_manager.get_model_path(model_id)
                        if path_obj is not None:
                            model_path_fs = str(path_obj)
                except Exception as e:
                    print(f"Error resolving model path for thumbnail (id={model_id}): {e}")
                    model_path_fs = None

                if model_path_fs and os.path.exists(model_path_fs):
                    try:
                        thumbnail_object = ThreeDViewer()
                        thumbnail_object.generate_thumbnail(model_path_fs)
                    except Exception as e:
                        print(f"Error generating thumbnail for '{model_name}': {e}")

            if os.path.exists(thumbnail_path):
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
    
    def __init__(self, data_manager: ClientDataManager, gallery_page: GalleryPage | None = None, parent=None):
        """Create the AI generation page

        Args:
            data_manager: client-side data manager used to talk to the backend
            gallery_page: optional reference to gallery page so we can refresh it after upload
        """
        self.data_manager = data_manager
        self.gallery_page = gallery_page
        self.meshy_client: MeshyClient | None = None
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

        # Status label for basic feedback
        self.status_label = QLabel("")
        self.status_label.setStyleSheet("color: #444; margin-top: 8px;")
        layout.addWidget(self.status_label)

        # Add stretch to push content to top
        layout.addStretch()
    
    def get_prompt(self) -> str:
        """Returns whatever text the user typed in"""
        return self.prompt_input.toPlainText().strip()

    def _ensure_meshy_client(self) -> MeshyClient | None:
        """Create Meshy client if needed, handling missing API key gracefully."""
        if self.meshy_client is not None:
            return self.meshy_client
        try:
            self.meshy_client = MeshyClient()
            return self.meshy_client
        except Exception as e:
            # Most likely missing MESHY_API_KEY or network issues
            msg = f"Meshy client init failed: {e}"
            print(msg)
            self.status_label.setText(
                "Error: Meshy API is not configured. Please set the MESHY_API_KEY environment variable."
            )
            return None

    def _next_model_id(self) -> str:
        """Generate next model id, mirroring backend DataManager.get_next_id() behavior."""
        models = self.data_manager.get_all_models()
        ids = []
        for m in models:
            mid = str(m.get("id") or "")
            if mid.startswith("model_"):
                try:
                    ids.append(int(mid.replace("model_", "")))
                except ValueError:
                    continue
        if not ids:
            return "model_001"
        return f"model_{max(ids) + 1:03d}"

    def _build_filename(self, model_id: str, prompt: str) -> str:
        """Create a reasonably safe filename from model id + prompt."""
        safe = re.sub(r"[^a-zA-Z0-9_-]+", "_", prompt.strip().lower()) or "model"
        if len(safe) > 40:
            safe = safe[:40]
        return f"{model_id}_{safe}.obj"

    async def _generate_and_upload(self, prompt: str) -> None:
        """Async workflow: Meshy -> download OBJ -> upload to backend (which mirrors to GCS)."""
        client = self._ensure_meshy_client()
        if client is None:
            return
        
        self.status_label.setText("Calling Meshy to generate 3D model (this may take some time)...")
        result = await client.generate_model(prompt)
        if not result.get("success"):
            error = result.get("error") or "unknown error"
            print(f"Meshy generation failed: {error}")
            self.status_label.setText(f"Meshy generation failed: {error}")
            return

        obj_url = result.get("obj_url")
        if not obj_url:
            self.status_label.setText("Meshy result does not contain an OBJ download URL.")
            return

        # Decide local save path (also acts as client cache)
        assets_dir = Path(__file__).parent.parent / "assets"
        models_dir = assets_dir / "models"
        models_dir.mkdir(parents=True, exist_ok=True)

        model_id = self._next_model_id()
        filename = self._build_filename(model_id, prompt)
        local_path = models_dir / filename

        self.status_label.setText("Downloading OBJ model from Meshy...")
        ok = await client.download_model(obj_url, str(local_path))
        if not ok:
            self.status_label.setText("Failed to download OBJ from Meshy.")
            return

        # Read bytes to upload to backend; backend will save locally and upload to GCS (if configured)
        try:
            file_bytes = local_path.read_bytes()
        except OSError as e:
            print(f"Error reading downloaded OBJ file: {e}")
            self.status_label.setText("Failed to read downloaded OBJ file.")
            return

        display_name = prompt.strip() or "AI Generated Model"
        tags = ["ai", "meshy"]

        self.status_label.setText("Uploading model to backend (will write to DB and sync to GCS)...")
        try:
            # Uses FastAPI backend /models endpoint; backend will call DataManager.save_model_to_gallery
            resp = self.data_manager.api.upload_model(
                name=display_name,
                tags=tags,
                file_bytes=file_bytes,
                filename=filename,
                model_id=model_id,
            )
            print(f"Upload response: {resp}")
        except Exception as e:
            print(f"Error uploading model to backend: {e}")
            self.status_label.setText("Upload to backend / GCS failed.")
            return

        # Refresh in-memory list and optionally refresh gallery UI
        self.data_manager.get_all_models()
        if self.gallery_page is not None:
            try:
                self.gallery_page.load_models()
            except Exception as e:
                print(f"Error refreshing gallery page: {e}")

        self.status_label.setText("Generation complete: model has been added to Gallery and uploaded to GCP.")

    def on_generate_clicked(self):
        """When the generate button is pressed"""
        prompt = self.get_prompt()
        if not prompt:
            self.status_label.setText("Please enter a text description to generate a 3D model.")
            return

        # Basic UI feedback
        self.generate_button.setEnabled(False)
        self.generate_button.setText("Generating...")
        self.status_label.setText("Calling Meshy API to generate model...")

        try:
            # Run the async workflow in a fresh event loop (PyQt main thread has no asyncio loop)
            asyncio.run(self._generate_and_upload(prompt))
            self.generate_button.setText("Generate Again")
        except Exception as e:
            print(f"Error during AI generation flow: {e}")
            self.status_label.setText(f"Error occurred during generation flow: {e}")
            self.generate_button.setText("Generate")
        finally:
            self.generate_button.setEnabled(True)

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
            self.model_path = str(model_path)  # so that download function can access it
            mesh = ThreeDViewer.load_model(model_path)
            self.viewer.add_mesh(mesh)

            # Top-down light from above (positive Z direction)
            self.light = ThreeDViewer.setup_light()
            self.viewer.add_light(self.light)

            # Default to viewing the model from the "front"
            # Assume the model's forward direction is the Z axis (+Z forward) and Y is the vertical up axis
            cx, cy, cz = mesh.center
            distance = max(getattr(mesh, "length", 1.0), 1.0)
            camera_pos = (cx, cy, cz + 1.5 * distance)  # In +Z direction, looking straight at the model
            # Use +Y as the up direction to keep the model upright in the view
            self.viewer.camera_position = [camera_pos, (cx, cy, cz), (0, 1, 0)]

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
        # print("gallery clicked")

        # the following two lines tests the thumbnail generation
        #threeD = ThreeDViewer()
        #threeD.generate_thumbnail(self.model_path)

        # gallery = DataManager()

        # need the following values to add to gallery

        #model id
        #model_id = gallery.get_next_id()

        # filename
        #filename = self.file_name_from_model_path()
        # display name
        #display_name = filename.upper()[0] + filename[1:]

        """
        TO do
        Add tags and model data after AI implementation
        """
        # tags
        # tags = [] # IMPORTANT ADD THIS --------------------------------

        #model data
        # model_data = None # ADD THIS AFTER AI -------------------------

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

