import cv2 
import numpy as np 
from elements.yolo import OBJ_DETECTION 
import time

Object_classes = ['fire']
print(len(Object_classes))
Object_colors = list(np.random.rand(40,3)*255) 
Object_detector = OBJ_DETECTION('best.pt', Object_classes) 

cap = cv2.imread('000043.png')

                        # detection process 
objs = Object_detector.detect(cap) 
print(objs)
# plotting 
for obj in objs:

        label = obj['label'] 

        score = obj['score'] 
        
        
        if label == 'fire':
                print(score)
                print('fire')
        
                
                labels = 'fire'
        [(xmin,ymin),(xmax,ymax)] = obj['bbox'] 
        color = Object_colors[Object_classes.index(label)] 
        frame = cv2.rectangle(cap, (xmin+50,ymin+50), (xmax-50,ymax-50), color, 2) 
        frame = cv2.putText(frame, f'{labels} ({str(score)})', (xmin+50,ymin+50),cv2.FONT_HERSHEY_SIMPLEX , 0.75, color, 1, cv2.LINE_AA) 

        cv2.imshow("window", frame) 
        keyCode = cv2.waitKey(0) 
        if keyCode == ord('q'): 
            break 
        
cv2.destroyAllWindows() 

 
