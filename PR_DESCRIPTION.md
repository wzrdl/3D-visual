# Refactor: Page-based UI Architecture

## Summary
Refactored the application to use a modular page-based architecture with tabbed navigation, improving code organization and maintainability.

## Changes

### New Features
- **Page-based UI structure**: Introduced `pages.py` with separate page classes (`GalleryPage`, `AIGenerationPage`, `ViewerPage`) inheriting from a `BasePage` base class
- **Tabbed interface**: Main window now uses `QTabWidget` to organize different sections (Gallery, AI Generation, 3D Viewer)
- **Model selection flow**: Gallery page can now switch to viewer tab and load selected models automatically

### Code Improvements
- **Simplified viewer**: Refactored `ThreeDViewer` to be a cleaner widget using `QtInteractor` directly
- **Better separation of concerns**: Each page manages its own UI and logic independently
- **Improved cleanup**: Added proper cleanup methods for viewer resources on window close

### Documentation & Dependencies
- **Updated README**: Added troubleshooting section for PyQt6 DLL load errors on Windows
- **Refined requirements**: Added version constraints and comments to `requirements.txt` for better dependency management

## Files Changed
- `app/pages.py` (new) - Page classes for modular UI
- `app/main_window.py` - Refactored to use tabbed interface with pages
- `app/viewer.py` - Simplified viewer implementation
- `README.md` - Added Windows troubleshooting guide
- `requirements.txt` - Added version constraints and documentation

