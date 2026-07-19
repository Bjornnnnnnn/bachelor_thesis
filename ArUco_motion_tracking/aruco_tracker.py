"""
aruco_tracker.py

Detects multiple ArUco markers and computes the transformation from
image coordinates to board coordinates.

"""

import cv2  # import OpenCV for marker detection and drawing
import numpy as np  # import NumPy for array math

from config import BOARD, MARKER_SIZE_MM, ARUCO_DICT  # import board layout and marker settings


class ArucoTracker:

    def __init__(self):

        dictionary = cv2.aruco.getPredefinedDictionary(ARUCO_DICT)  # get the ArUco marker dictionary

        parameters = cv2.aruco.DetectorParameters()  # create default detector settings

        self.detector = cv2.aruco.ArucoDetector(
            dictionary,
            parameters
        )  # create the marker detector object

    

    def detect(self, frame):
        """
        Detect all ArUco markers.

        Returns
        -------
        markers : dict

        markers[id] =

        {
            "corners": ndarray (4x2),

            "center": ndarray (2,)
        }
        """

        corners, ids, rejected = self.detector.detectMarkers(frame)  # find markers in the image

        markers = {}  # start with an empty dictionary for marker data

        if ids is None:
            return markers  # no markers were found

        ids = ids.flatten()  # convert ids to a simple list

        for marker_id, marker_corners in zip(ids, corners):

            pts = marker_corners.reshape(4, 2)  # reshape the marker corners to (4,2)

            center = np.mean(pts, axis=0)  # compute the marker center point

            markers[int(marker_id)] = {

                "corners": pts,

                "center": center

            }  # save marker corners and center by id

        return markers  # return all detected marker info

    

    def draw(self, frame, markers):
        """
        Draw detected markers.
        """

        for marker_id, data in markers.items():

            pts = data["corners"].astype(int)  # corners as integer pixel positions

            cv2.polylines(
                frame,
                [pts],
                True,
                (0,255,0),
                2
            )  # draw a green border around the marker

            c = tuple(data["center"].astype(int))  # center position as integers

            cv2.circle(
                frame,
                c,
                4,
                (0,0,255),
                -1
            )  # draw a small red circle at the marker center

            cv2.putText(
                frame,
                str(marker_id),
                (c[0]+5,c[1]-5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255,0,0),
                2
            )  # draw the marker id next to the marker

    

    def compute_homography(self, markers):
        """
        Computes the homography from image coordinates
        to board coordinates.

        Returns
        -------
        H

        or None
        """

        image_points = []  # points found in the image frame

        board_points = []  # corresponding points on the physical board

        for marker_id, data in markers.items():

            if marker_id not in BOARD:
                continue  # skip markers not defined in the board layout

            # Marker top-left corner in board coordinates

            bx, by = BOARD[marker_id]  # board coordinates for this marker id

            board_corners = np.array([

                [bx, by],

                [bx + MARKER_SIZE_MM, by],

                [bx + MARKER_SIZE_MM,
                 by + MARKER_SIZE_MM],

                [bx,
                 by + MARKER_SIZE_MM]

            ], dtype=np.float32)  # define the four corners of the marker in board units

            image_points.extend(
                data["corners"]
            )  # add the marker corners from the image

            board_points.extend(
                board_corners
            )  # add the matching board corners

        if len(image_points) < 3:

            return None  # need at least 3 points to make a transformation

        image_points = np.array(
            image_points,
            dtype=np.float32
        )  # convert image points to a float array

        board_points = np.array(
            board_points,
            dtype=np.float32
        )  # convert board points to a float array

        H, mask = cv2.findHomography(

            image_points,

            board_points,

            cv2.RANSAC,

            3.0

        )  # compute the homography matrix mapping image to board points

        if H is None:
            return None  # no valid homography could be found


        # Reject unstable homographies
        det = np.linalg.det(H)  # compute determinant of the transformation

        if abs(det) < 1e-6:
            return None  # reject if the homography is nearly singular


        return H  # return the transformation matrix

    

    @staticmethod
    def image_to_board(point, H):
        """
        Convert image coordinates
        to board coordinates.
        """

        if H is None:
            return None  # cannot convert without a homography

        p = np.array([
            [[point[0], point[1]]]
        ], dtype=np.float32)  # prepare the point in the format OpenCV expects

        transformed = cv2.perspectiveTransform(
            p,
            H
        )  # apply the homography to the point

        return transformed[0,0]  # return the converted board point
