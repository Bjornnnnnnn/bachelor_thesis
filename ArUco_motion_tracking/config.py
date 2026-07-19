# config.py

from pathlib import Path
import cv2

VIDEO = Path(r"C:\Users\bjorn\Downloads\ArUco_motion_tracking\slow_with_hand.mp4") # CHANGE THIS to your own video file path

BASE_DIR = Path(__file__).parent

OUTPUT_FOLDER = BASE_DIR / "output"

OUTPUT_FOLDER.mkdir(
    parents=True,
    exist_ok=True
)

FPS = None          # None = use video's FPS

MARKER_SIZE_MM = 20.0 # CHANGE THIS depending on your size

ARUCO_DICT = cv2.aruco.DICT_4X4_50 

# Coordinates of the top left of each marker (mm)
# Fill these in once after placing the markers.

BOARD = {

    0: (0.0, 0.0),          # bottom-left

    1: (90.0, 0.0),        # bottom-right

    2: (125.0, 0.0),         # top-left

    3: (160.0, 0.0),       # top-right

    4: (203.0, 20.0),          # bottom-left

    5: (0.0, -37.0),        # bottom-right

    6: (90.0, -37.0),         # top-left

    7: (125.0, -37.0),        # top-right

    8: (160.0, -37.0),         # top-left

    9: (203.0, -37.0)         # top-left

}