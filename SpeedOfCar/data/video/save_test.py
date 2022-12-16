import cv2

cap = cv2.VideoCapture(r"G:\drive_e\Programming\AI\Competitions\hackathons\tractor\yolov4-deepsort\data\video\road_way_2.mp4")
save_path = r"G:\drive_e\Programming\AI\Competitions\hackathons\tractor\yolov4-deepsort\data\video\frame_5.png"
i = 0 
while True:
    ret, vid = cap.read()

    if ret:
        if i ==5:
            cv2.imwrite(save_path,vid)
        i+=1
        if cv2.waitKey(0) == ord("q"):
            break
    else:
        print("no frames")
        break
    
cap.release()
cv2.destroyAllWindows()