"""
color_tracker.py

Tracks a coloured circular marker.

The tracker learns the colour from one mouse click
and then follows the marker through the video.

"""

import cv2  # import OpenCV for image and video processing
import numpy as np  # import NumPy for number arrays and math
from typing import Optional, Tuple, Dict  # import type hints for clarity

positions: Dict[int, Tuple[float, float]] = {}  # dictionary to store positions if needed

class ColorTracker:

    def __init__(self):

        self.lower: Optional[np.ndarray] = None  # lower HSV color limit for detection
        self.upper: Optional[np.ndarray] = None  # upper HSV color limit for detection

        self.previous_position = None  # last known marker position

        self.learned = False  # flag to know if color has been learned yet

    

    def learn_color(self, frame):

        clicked: dict[str, Optional[Tuple[int, int]]] = {
            "point": None
        }  # store the clicked point coordinates

        display = frame.copy()  # make a copy of the frame to draw on

        def mouse(event, x, y, flags, param):
            if event == cv2.EVENT_LBUTTONDOWN:
                clicked["point"] = (x, y)  # save the point when the user clicks

        cv2.namedWindow("Select coloured marker")  # create a window for selecting the color

        cv2.setMouseCallback(
            "Select coloured marker",
            mouse
        )  # connect the mouse click to our function

        while True:

            temp = display.copy()  # copy the frame again for each loop

            if clicked["point"] is not None:
                cv2.circle(
                    temp,
                    clicked["point"],
                    5,
                    (0,255,0),
                    2
                )  # draw a small green circle on the clicked point

            cv2.imshow(
                "Select coloured marker",
                temp
            )  # show the frame to the user

            key = cv2.waitKey(20) & 0xFF  # wait a short time so the window stays responsive

            # Exit immediately after selecting a point
            if clicked["point"] is not None:
                break  # stop waiting once the user clicked

        cv2.destroyAllWindows()  # close the selection window
        cv2.waitKey(1)  # small pause to finish closing the window

        if clicked["point"] is None:
            raise RuntimeError("No colour point selected")  # error if click never happened

        x, y = clicked["point"]  # get the clicked x,y position

        hsv = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2HSV
        )  # change the image to HSV color space

        radius = 8  # how far around the point to sample colors

        roi = hsv[
            max(0, y-radius):y+radius,
            max(0, x-radius):x+radius
        ]  # take a small square around the click point

        pixels = roi.reshape(-1, 3)  # turn that square into a list of pixels

        mean = np.mean(
            pixels,
            axis=0
        )  # average the HSV values in that square

        H, S, V = mean  # separate the average into hue, saturation, value

        self.lower = np.array([
            max(H-12, 0),
            max(S-60, 40),
            max(V-60, 40)
        ], dtype=np.uint8)  # make the lower color boundary

        self.upper = np.array([
            min(H+12, 179),
            255,
            255
        ], dtype=np.uint8)  # make the upper color boundary

        self.learned = True  # mark that the color has been learned

        print("HSV learned")  # inform the user
        print(self.lower)  # show the lower boundary values
        print(self.upper)  # show the upper boundary values

    

    def detect(self, frame):

        if not self.learned:

            raise RuntimeError(
                "Colour not learned."
            )  # do not run detection before learning the marker color

        hsv = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2HSV
        )  # convert the frame to HSV color space for color filtering

        if self.lower is None or self.upper is None:
            return None  # if color boundaries are missing, return nothing

        mask = cv2.inRange(
            hsv,
            self.lower,
            self.upper
        )  # create a mask where the learned color is white

        kernel = np.ones((5,5), np.uint8)  # create a small kernel for cleaning the mask

        mask = cv2.morphologyEx(
            mask,
            cv2.MORPH_OPEN,
            kernel
        )  # remove tiny white noise from the mask

        mask = cv2.morphologyEx(
            mask,
            cv2.MORPH_CLOSE,
            kernel
        )  # fill small black holes inside the white areas

        contours, _ = cv2.findContours(

            mask,

            cv2.RETR_EXTERNAL,

            cv2.CHAIN_APPROX_SIMPLE

        )  # find outer contours in the mask

        if len(contours) == 0:

            return None  # no shapes found, so nothing to detect

        best_score = -1  # initial score is very low

        best_contour = None  # no best contour yet

        for contour in contours:

            area = cv2.contourArea(contour)  # how big the contour is

            if area < 20:

                continue  # skip tiny shapes

            perimeter = cv2.arcLength(
                contour,
                True
            )  # length of the contour boundary

            if perimeter == 0:
                continue  # skip invalid contours

            circularity = (
                4*np.pi*area
            ) / (perimeter*perimeter)  # how round the contour is

            M = cv2.moments(contour)  # compute moments to find the center

            if M["m00"] == 0:
                continue  # skip if area is zero in moments

            cx = M["m10"]/M["m00"]
            cy = M["m01"]/M["m00"]

            score = area  # start score based on area

            score += 300*circularity  # add more score if the shape is round

            if self.previous_position is not None:

                distance = np.linalg.norm(

                    np.array([cx,cy])

                    -

                    self.previous_position

                )  # distance from the last known position

                score -= distance  # prefer shapes closer to previous position

            if score > best_score:

                best_score = score  # update best score

                best_contour = contour  # remember this contour as the best

        if best_contour is None:

            return None  # no good contour was found

        M = cv2.moments(best_contour)  # compute moments for the chosen contour

        cx = M["m10"]/M["m00"]
        cy = M["m01"]/M["m00"]

        self.previous_position = np.array(
            [cx,cy]
        )  # save the current marker position for the next frame

        return {

            "position": np.array(
                [cx,cy],
                dtype=float
            ),  # return the center position of the marker

            "score": best_score,  # return the score of the best candidate

            "contour": best_contour,  # return the selected contour

            "mask": mask  # return the cleaned mask used for detection

        }