import sys

from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QMainWindow, QLabel, QLineEdit, QVBoxLayout, QWidget, QApplication, QGridLayout, QHBoxLayout


class MainWindow(QMainWindow): # to setup a window
    def __init__(self, *args, **kwargs): # extend the class
        super(MainWindow, self).__init__(*args, **kwargs) # must call super init function

        self.setWindowTitle("Text to 3D Image") # title of window

# first set of widgets -- title and description (t and d)
        # widget for the title
        widget_title = QLabel("Image")
        font_title = widget_title.font()
        font_title.setPointSize(20)
        widget_title.setFont(font_title)
        widget_title.setAlignment(Qt.AlignmentFlag.AlignLeft)

        # widget for description
        widget_desc = QLabel("Type the prompt of the image you want to create")
        font_desc = widget_desc.font()
        font_desc.setPointSize(12)
        widget_desc.setFont(font_desc)
        widget_desc.setAlignment(Qt.AlignmentFlag.AlignLeft)

        # setting layout 1-- title and description
        layout_t_d = QVBoxLayout()

        layout_t_d.addWidget(widget_title)
        layout_t_d.addWidget(widget_desc)

# second set of widgets -- placeholder for search bar and image

        # widget for search bar
        widget_search = QLineEdit()
        widget_search.setMaxLength(10) # setting the size
        widget_search.setPlaceholderText("Enter your text")
        widget_search.setAlignment(Qt.AlignmentFlag.AlignLeft)

        widget_search.returnPressed.connect(self.return_pressed)

        #for the image, for now will be text
        widget_image = QLabel("image placeholder")
        widget_image.setAlignment(Qt.AlignmentFlag.AlignCenter)
        #widget_image.setMaximumSize(500, 500) # this is for when we have the image showing

        # setting layout for the search bar and image showing
        layout_search = QVBoxLayout()

        layout_search.addWidget(widget_search)
        layout_search.addWidget(widget_image)

    # combined layout
        layout = QVBoxLayout()
        layout.addLayout(layout_t_d)
        layout.addLayout(layout_search)

        #display the layout
        displayWidget = QWidget()
        displayWidget.setLayout(layout)
        self.setCentralWidget(displayWidget)

    # function for search bar -- if return was pressed
    def return_pressed(self):
        print("Return pressed")

app = QApplication(sys.argv) # main app setup?

window = MainWindow() # variable to hold main window
window.show() # IMPORTANT -- so we can actually see it

app.exec_() # executing the app