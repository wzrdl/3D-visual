"""
main window for the app
has gallery browsing and AI generation stuff
"""
import sys # to test the application at the bottom
from PyQt6.QtWidgets import QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QApplication, QLineEdit
from PyQt6.QtCore import Qt
from viewer import ThreeDViewer
from app.data_manager import DataManager
import pyvista as pv
import pyvistaqt as pvqt


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

        # for the search bar: 
        widget_search = QLineEdit()
        widget_search.setMaxLength(50) # setting the size
        widget_search.setPlaceholderText("Enter your text") # placeholder text that appears in the search bar
        widget_search.setAlignment(Qt.AlignmentFlag.AlignLeft) # aligned left for now

        left_layout.addWidget(widget_search)
        """
        # this connects this widget to the below function
        # so that when a user hits enter, we can then do something (hopefully save the data)
        widget_search.returnPressed.connect(self.return_pressed) 
        """

        main_layout.addWidget(left_panel)
        
        # right panel - 3D viewer (week 4)
        # TODO: integrate viewer in week 4

        self.plotter = pvqt.QtInteractor(self)
        #right_panel = ThreeDViewer()

        main_layout.addWidget(self.plotter.interactor)
        #mesh = pv.read('C:\\Users\lizzy\OneDrive\Documents\GitHub\\3D-visual\\assets\models\cone.obj')
        self.plotter.add_mesh(ThreeDViewer.load_model(self, 'C:\\Users\lizzy\OneDrive\Documents\GitHub\\3D-visual\\assets\models\cone.obj'))
        self.plotter.show_grid()

        viewer_placeholder = QWidget()
        viewer_placeholder.setStyleSheet("background-color: #2b2b2b;")

        main_layout.addWidget(viewer_placeholder, stretch=1)

        central_widget.setLayout(main_layout)

    def return_pressed(self):
        print("Return pressed")
        pass # as it has no use right now



# to test to see if it works
app = QApplication(sys.argv) # main app setup

window = MainWindow() # variable to hold main window
window.show() # IMPORTANT -- so we can actually see it

app.exec_() # executing the app


