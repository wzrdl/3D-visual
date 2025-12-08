"""
main entry point
"""
import torch # temporary solution to get the viewer to open
import sys
from PyQt6.QtWidgets import QApplication
from app.main_window import MainWindow


def main():
    """start the app"""
    app = QApplication(sys.argv)
    app.setApplicationName("3D Model Generator & Library") # Now we set the name of the application
    
    window = MainWindow()
    window.show()
    
    sys.exit(app.exec())


if __name__ == "__main__":
    main()

