import cv2
import numpy as np
import torch
import time
from PIL import Image
from utils.plots import Annotator, colors, save_one_box

# Open camera
pipeline = "nvarguscamerasrc sensor-id=0 !" "video/x-raw(memory:NVMM), width=(int)1920, height=(int)1080, framerate=(fraction)30/1 ! " "nvvidconv flip-method=2 ! " "video/x-raw, width=(int)640, height=(int)480, format=(string)BGRx ! " "videoconvert ! " "video/x-raw, format=(string)BGR ! appsink"
cap = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)

# Load model
device = torch.device('cuda')
# model = torch.load(r'/home/electronic/Desktop/AntiDrone/yolov5_csi/yolov5n.pt', map_location = torch.device('cuda:0'))
# model.eval()
model = torch.hub.load('ultralytics/yolov5', 'yolov5s')

if cap.isOpened():
    while True:
        _, im = cap.read()
        annotator = Annotator(im, line_width=3, example=str('Stream'))
        # cv2.imshow('Stream', im)
        # im = np.asanyarray(im, dtype = float)
        # im = torch.from_numpy(im).to( )
        # im = im.float()
        # im /= 255
        result = model([im])
        result.print()
        print(result.pandas().xyxy[0])
        for i in range(len(result.pandas().xyxy[0])):
            conf = result.pandas().xyxy[0].iloc[i]['confidence']
            if conf <= 0.3:
                continue
            else:
                cls_name = result.pandas().xyxy[0].iloc[i]['name']
                cls_id = result.pandas().xyxy[0].iloc[i]['class']
                xyxy = (result.pandas().xyxy[0].iloc[i]['xmin'],
                        result.pandas().xyxy[0].iloc[i]['ymin'],
                        result.pandas().xyxy[0].iloc[i]['xmax'],
                        result.pandas().xyxy[0].iloc[i]['ymax']
                        )
                annotator.box_label(xyxy, f'{cls_name} {conf:.2f}', color=colors(i, True))
        im0s = annotator.result()
        cv2.imshow('Stream', im0s)
        keycode = cv2.waitKey(1)
        if keycode == ord('q'):
            break
        time.sleep(0.0)

    cap.release()
    cv2.destroyAllWindows()
else:
    print("Error: Unable to open camera")
