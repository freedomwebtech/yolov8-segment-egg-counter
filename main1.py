import cv2
from yolo_segmentation import YOLOSEG
import cvzone
from tracker import*
import numpy as np
ys = YOLOSEG("best.pt")

my_file = open("coco1.txt", "r")
data = my_file.read()
class_list = data.split("\n")

cap=cv2.VideoCapture('egg.mp4')
count=0
def RGB(event, x, y, flags, param):
    if event == cv2.EVENT_MOUSEMOVE :  
        point = [x, y]
        print(point)
  
        

cv2.namedWindow('RGB')
cv2.setMouseCallback('RGB', RGB)
tracker=Tracker()
area=[(434,39),(453,469),(473,474),(456,36)]
counter1=[]
while True:
    ret,frame=cap.read()
    if not ret:
        break

    frame=cv2.resize(frame,(1020,500))
    overlay = frame.copy()
    alpha = 0.5

    bboxes, classes, segmentations, scores = ys.detect(frame)
    bbox_idx=tracker.update(bboxes)
    for bbox,seg in zip(bbox_idx,segmentations):
        x3,y3,x4,y4,id=bbox
        cx=int(x3+x4)//2
        cy=int(y3+y4)//2
        result=cv2.pointPolygonTest(np.array(area,np.int32),((cx,cy)),False)
        if result>=0:
 #       cv2.rectangle(frame, (x3, y3), (x4, y4), (255, 0, 0), 2)    
#           cv2.polylines(frame, [seg], True, (0, 0, 255), 4)
           cv2.circle(frame,(cx,cy),4,(0,255,0),-1)
           cv2.fillPoly(overlay, [seg], (0,0,255))
           cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 2, frame)
           cvzone.putTextRect(frame, f'{id}', (x3,y3),1,1)
           if counter1.count(id)==0:
              counter1.append(id) 
    cv2.polylines(frame,[np.array(area,np.int32)],True,(255,0,0),2)
    ca1=len(counter1)
    cvzone.putTextRect(frame, f'CA:-{ca1}', (50,60),2,2)

    cv2.imshow("RGB",frame)
    if cv2.waitKey(1)&0xFF==27:
        break
cap.release()
cv2.destroyAllWindows()