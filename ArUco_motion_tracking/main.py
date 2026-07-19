"""
main.py

Main motion tracking pipeline.

Measures coloured marker displacement relative to
the ArUco board coordinate system.

The first stable detected marker position is defined as (0,0).
"""

import cv2
import numpy as np
import pandas as pd

from config import VIDEO, OUTPUT_FOLDER, FPS

from aruco_tracker import ArucoTracker
from color_tracker import ColorTracker

############## Setup ############


cap = cv2.VideoCapture(str(VIDEO))

if not cap.isOpened():
    raise RuntimeError("Could not open video")


video_fps = cap.get(cv2.CAP_PROP_FPS)

if FPS is not None:
    video_fps = FPS


width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))



########## Output video ###############

output_video = OUTPUT_FOLDER / "tracked_video.mp4"


writer = cv2.VideoWriter(
    str(output_video),
    cv2.VideoWriter_fourcc(*"mp4v"),
    video_fps,
    (width, height)
)



############## Initialize trackers ######################

aruco = ArucoTracker()
color = ColorTracker()

############ Learn coloured marker##############

ret, first_frame = cap.read()

if not ret:
    raise RuntimeError("Could not read first frame")


color.learn_color(first_frame)


cap.set(
    cv2.CAP_PROP_POS_FRAMES,
    0
)



########### Data storage #########################

results = []

frame_number = 0



########## Zero calibration################

zero_position = None

zero_samples = []

ZERO_SAMPLE_COUNT = 30



############# Tracking stability settings ############

last_valid_position = None


# Maximum physical movement per frame
# 15 mm/frame at 60 FPS = 900 mm/s
MAX_MOVEMENT_PER_FRAME = 15 #otherwise reject jump


# Recovery settings
lost_counter = 0

MAX_LOST_FRAMES = 15 # after 15 frames of losing target reset marker to recover



# Position smoothing
position_history = []

FILTER_SIZE = 5 # moving average filter size



########## Main loop ###############

while True:


    ret, frame = cap.read()

    if not ret:
        break



    ########### ArUco detection #############

    markers = aruco.detect(frame)

    H = aruco.compute_homography(markers)


    aruco.draw(
        frame,
        markers
    )



    ######## Colour detection ###########

    dot = color.detect(frame)


    displacement = None

    board_position = None



    if dot is not None:


        image_position = dot["position"]


        if H is not None:


            board_position = aruco.image_to_board(
                image_position,
                H
            )



        if board_position is not None:


            board_position = np.array(
                board_position,
                dtype=float
            )



            # Flip Y axis
            board_position[1] = -board_position[1]



            ########## Reject impossible jumps ##############

            valid_position = True


            if last_valid_position is not None:


                movement = np.linalg.norm(
                    board_position -
                    last_valid_position
                )


                if movement > MAX_MOVEMENT_PER_FRAME:


                    print(
                        "Rejected jump:",
                        round(movement,2),
                        "mm"
                    )


                    valid_position = False



            if valid_position:


                lost_counter = 0


                last_valid_position = board_position.copy()



                ######### Zero calibration #########

                if zero_position is None:


                    zero_samples.append(
                        board_position.copy()
                    )


                    if len(zero_samples) >= ZERO_SAMPLE_COUNT:


                        zero_position = np.mean(
                            zero_samples,
                            axis=0
                        )


                        print(
                            "Zero position set:",
                            zero_position
                        )



                ############### Calculate displacement #######################

                if zero_position is not None:


                    displacement = (
                        board_position -
                        zero_position
                    )



                    ############ Smooth displacement #################

                    position_history.append(
                        displacement.copy()
                    )


                    if len(position_history) > FILTER_SIZE:

                        position_history.pop(0)



                    displacement = np.mean(
                        position_history,
                        axis=0
                    )



            else:

                lost_counter += 1



                # Allow recovery after bad tracking
                if lost_counter > MAX_LOST_FRAMES:


                    print(
                        "Attempting tracker recovery..."
                    )


                    last_valid_position = None

                    position_history.clear()

                    lost_counter = 0




        # Draw detected marker

        cv2.circle(
            frame,
            (
                int(image_position[0]),
                int(image_position[1])
            ),
            8,
            (0,0,255),
            -1
        )



    else:

        lost_counter += 1



    ############# Display values ################

    if displacement is not None:


        cv2.putText(
            frame,
            f"X: {displacement[0]:.2f} mm",
            (20,40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0,255,0),
            2
        )


        cv2.putText(
            frame,
            f"Y: {displacement[1]:.2f} mm",
            (20,80),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0,255,0),
            2
        )


    elif zero_position is None:


        cv2.putText(
            frame,
            "Calibrating zero...",
            (20,40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0,255,255),
            2
        )



    ############ Save data ###############

    results.append({

        "frame": frame_number,

        "time_s":
            frame_number / video_fps,


        "image_x_px":
            None if dot is None
            else dot["position"][0],


        "image_y_px":
            None if dot is None
            else dot["position"][1],


        "board_x_mm":
            None if displacement is None
            else displacement[0],


        "board_y_mm":
            None if displacement is None
            else displacement[1],


        "tracking_score":
            0 if dot is None
            else dot["score"]

    })



    writer.write(frame)



    cv2.imshow(
        "Tracking",
        frame
    )



    if cv2.waitKey(1) == 27:
        break



    frame_number += 1




####### Finish###########

cap.release()

writer.release()

cv2.destroyAllWindows()



df = pd.DataFrame(results)


csv_path = OUTPUT_FOLDER / "tracking_results.csv"


df.to_csv(
    csv_path,
    index=False,
    sep=";",
    decimal=","
)



print("==============================")
print("Tracking finished")
print()
print(f"Frames processed: {frame_number}")
print()
print("Saved:")
print(csv_path)
print(output_video)
print("==============================")
