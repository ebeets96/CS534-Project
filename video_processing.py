import cv2
import os
import subprocess

vidcap = cv2.VideoCapture('BlackManInAWhiteWorld.mp4')
success,image = vidcap.read()
count = 0
os.makedirs("video_frames")

command = "ffmpeg -i BlackManInAWhiteWorld.mp4 -vn -acodec copy audio.aac"
subprocess.call(command, shell=True)

while success:
	cv2.imwrite("video_frames/frame" + str(count) + ".jpg", image)     # save frame as JPEG file
	success,image = vidcap.read()

fps = vidcap.get(cv2.CAP_PROP_FPS)
command = "ffmpeg -r " + str(fps) + " -i video_frames/frame%d.jpg -i audio.aac -y combined_ouput.mp4"
subprocess.call(command, shell=True)
