# Explanation how to use
If you want to use this code, upload all files in the nucleoboard folder unto your nucleoboard. (assuming you have the right setup with a hx711 connected to a load cell). If you then complete a measurement (while filming!) make sure to move the measurement.csv from the nucleoboard to your laptop. you may need to reconnect your nucleoboard after a measurement before the file pops up, and make sure to remove it from the nucleoboard before restarting measurements.

The video should also be uploaded to the laptop. Next the ArUco_motion_tracking code can detect the markers from the video and generate a video where the location of the markers and the colored dot is marked. The code will output a csv file which describes the movement

next take the measurement csv and the movement csv and copy their paths into the graph maker file. This file will read both csv's, make graphs and extract some features
