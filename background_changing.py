import cv2
import cvzone
from cvzone.SelfiSegmentationModule import SelfiSegmentation
import os

video_capture = cv2.VideoCapture(0)
video_capture.set(3, 640)
video_capture.set(4, 480)
segmentor = SelfiSegmentation()

background_images = os.listdir("backgrounds")
background_list = []
for background_path in background_images:
    background = cv2.imread(f"backgrounds/{background_path}")
    background_list.append(background)

current_background = 0

while True:
    success, frame = video_capture.read()
    output_frame = segmentor.removeBG(frame, background_list[current_background], threshold=0.8)
    stack_frames = cvzone.stackImages([frame, output_frame], 2, 1)
    cv2.imshow("Background Changer", stack_frames)

    key = cv2.waitKey(1)

    # Press 'd' for the next background
    if key == ord('d'):
        if current_background > 0:
            current_background -= 1

    # Press 'a' for the previous background
    elif key == ord('a'):
        if current_background < len(background_list) - 1:
            current_background += 1

    # Press 'q' to stop the execution
    elif key == ord('q'):
        cv2.destroyAllWindows()
        break
