import cv2
import sys
from lanedetection import perspectiveWarp,processImage,plotHistogram,slide_window_search,general_search,draw_lane_lines
# sys.path.append(r"C:\Users\Aman Sheikh\Desktop\3dObjectDetection\3D-BoundingBox")
sys.path.append(r"3D-BoundingBox")
from Run import main,plot_regressed_3d_bbox
from enum import Enum


class cv_colors(Enum):
    RED = (0,0,255)
    GREEN = (0,255,0)
    BLUE = (255,0,0)
    PURPLE = (247,44,200)
    ORANGE = (44,162,247)
    MINT = (239,255,66)
    YELLOW = (2,255,250)
 
def plot_3d_box(ptslist,img):
    for box_3d in ptslist:
        cv2.line(img, (box_3d[0][0], box_3d[0][1]), (box_3d[2][0],box_3d[2][1]), cv_colors.GREEN.value, 1)
        cv2.line(img, (box_3d[4][0], box_3d[4][1]), (box_3d[6][0],box_3d[6][1]), cv_colors.GREEN.value, 1)
        cv2.line(img, (box_3d[0][0], box_3d[0][1]), (box_3d[4][0],box_3d[4][1]), cv_colors.GREEN.value, 1)
        cv2.line(img, (box_3d[2][0], box_3d[2][1]), (box_3d[6][0],box_3d[6][1]), cv_colors.GREEN.value, 1)

        cv2.line(img, (box_3d[1][0], box_3d[1][1]), (box_3d[3][0],box_3d[3][1]), cv_colors.GREEN.value, 1)
        cv2.line(img, (box_3d[1][0], box_3d[1][1]), (box_3d[5][0],box_3d[5][1]), cv_colors.GREEN.value, 1)
        cv2.line(img, (box_3d[7][0], box_3d[7][1]), (box_3d[3][0],box_3d[3][1]), cv_colors.GREEN.value, 1)
        cv2.line(img, (box_3d[7][0], box_3d[7][1]), (box_3d[5][0],box_3d[5][1]), cv_colors.GREEN.value, 1)
        
        for i in range(0,7,2):
            cv2.line(img, (box_3d[i][0], box_3d[i][1]), (box_3d[i+1][0],box_3d[i+1][1]), cv_colors.GREEN.value, 1)


        front_mark = [(box_3d[i][0], box_3d[i][1]) for i in range(4)]
        cv2.circle(img, (box_3d[0][0], box_3d[0][1]), 10, cv_colors.BLUE.value, -1)
        cv2.circle(img, (box_3d[1][0], box_3d[1][1]), 10, cv_colors.RED.value, -1)
        cv2.circle(img, (box_3d[4][0], box_3d[4][1]), 10, cv_colors.GREEN.value, -1)
        cv2.circle(img, (box_3d[5][0], box_3d[5][1]), 10, cv_colors.MINT.value, -1)

        cv2.line(img, front_mark[0], front_mark[3], cv_colors.GREEN.value, 1)
        cv2.line(img, front_mark[1], front_mark[2], cv_colors.BLUE.value, 1)
        cv2.imshow("frame",img)

# cap = cv2.VideoCapture(r"C:\Users\Aman Sheikh\Desktop\3dObjectDetection\FinalFolder\output_video.avi")

cap = cv2.VideoCapture(r"C:\Users\Aman Sheikh\Desktop\3dObjectDetection\FinalFolder\WhatsApp Video 2022-12-16 at 2.18.33 PM.mp4")


while cap.read():
    rate,frame = cap.read()
    frame = cv2.resize(frame, (1280, 720))
    ##lane detectiom

    birdView, minverse = perspectiveWarp(frame)
    # birdView, birdViewL, birdViewR, minverse = perspectiveWarp(frame)
    img, hls, grayscale, thresh, blur, canny = processImage(birdView)
    hist= plotHistogram(thresh)
    ploty, left_fit, right_fit, left_fitx, right_fitx = slide_window_search(thresh, hist)
    draw_info = general_search(thresh, left_fit, right_fit)
    result = draw_lane_lines(frame, thresh, minverse, draw_info)
    # 3d object detection
    base_ptslist,ptslist,locations = main(result)
    result = cv2.resize(result, (1242, 375))
    plot_3d_box(ptslist,result)


