
#code to get the key frame from the video and save it as a png file.
import cv2
import os

#videopath- video file path
#frames_path: path of frames
#count: to assign the video order to the frame.

def frameExtractor(video_path,frames_path,count):
    # using factor to differentiate between my videos and given gestures

    if not os.path.exists(frames_path):
        os.mkdir(frames_path)
    cap = cv2.VideoCapture(video_path)
    video_length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) - 1
    frame_no= int(video_length/2)

    # print("Extracting frame..\n")

    ret,frame=cap.read()

    cv2.imwrite(frames_path + "/%#05d.png" % (count+1), frame)



