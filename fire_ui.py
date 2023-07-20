import numpy as np
from os import listdir
import sys
import os
import pickle
from PyQt5 import QtWidgets
from PyQt5.QtWidgets import QDialog, QApplication, QFileDialog
from PyQt5.uic import loadUi
from PyQt5 import QtCore, QtGui, QtWidgets
import cv2 
import numpy as np 
from elements.yolo import OBJ_DETECTION 

import time
import pyttsx3 as voice
#serialcomm = serial.Serial('COM6', 9600)

#serialcomm.timeout = 1


class MainWindow(QDialog):
    def __init__(self):
        super(MainWindow,self).__init__()
        loadUi("1.ui",self)
        self.mainbutton1.clicked.connect(self.gotoimage)
        self.mainbutton2.clicked.connect(self.gotodata)
        
   
    def gotoimage(self):
        image = imagedetection()
        widget.addWidget(image)
        widget.setCurrentIndex(widget.currentIndex()+1)
   
    def gotodata(self):
        data = dataprediction()
        widget.addWidget(data)
        widget.setCurrentIndex(widget.currentIndex()+1)   
         

class imagedetection(QDialog):



    def __init__(self):
        super(imagedetection,self).__init__()
        loadUi("three.ui",self)
        self.button1.clicked.connect(self.browsefiles)
        self.button2.clicked.connect(self.detect)
        
        self.back.clicked.connect(self.mainback)


    def browsefiles(self):
        
        global fname
        fname, _ = QFileDialog.getOpenFileName(self, 'Open file', '', ' ( *.png *. *.jpeg *.jpg)')                
        
        image_name = os.path.basename(fname)
        
        self.label.setPixmap(QtGui.QPixmap(fname))

    def detect(self):
        
        Object_classes = ['fire']
        print(len(Object_classes))
        Object_colors = list(np.random.rand(40,3)*255) 
        Object_detector = OBJ_DETECTION('best.pt', Object_classes) 

        cap = cv2.imread(fname)

                                # detection process 
        objs = Object_detector.detect(cap) 
        print(objs)
        # plotting 
        
        if len(objs) == 0:
            
                self.prediction.setText('        no fire')
        for obj in objs:

                label = obj['label'] 

                score = obj['score'] 
                
                
                if label == 'fire':
                    
                    
                        print(score)
                        print('fire')
                        
                        
                        labels = 'fire'
                        self.prediction.setText('         fire')
                        i = "on#"
                        for g in range(5):
                            engine = voice.init()
                            engine.say('fire')
                            engine.runAndWait()
                            
                            #serialcomm.write(i.encode())
                            
                            time.sleep(0.5)
                            
                            #print(serialcomm.readline().decode('ascii'))
                
                [(xmin,ymin),(xmax,ymax)] = obj['bbox'] 
                color = Object_colors[Object_classes.index(label)] 
                frame = cv2.rectangle(cap, (xmin+50,ymin+50), (xmax-50,ymax-50), color, 2) 
                frame = cv2.putText(frame, f'{labels} ({str(score)})', (xmin+50,ymin+50),cv2.FONT_HERSHEY_SIMPLEX , 0.75, color, 1, cv2.LINE_AA) 

                cv2.imshow("window", frame) 
                keyCode = cv2.waitKey(0) 
                if keyCode == ord('q'): 
                    break 
                
        cv2.destroyAllWindows() 
        #serialcomm.close()
        
        

    def mainback(QDialog):
        mainwindow = MainWindow()
        widget.addWidget(mainwindow)
        widget.setCurrentIndex(widget.currentIndex()+1)
   
         
         
class dataprediction(QDialog):

    
    def __init__(self):
        super(dataprediction,self).__init__()
        loadUi("ttwo.ui",self)

        
        self.predict2.clicked.connect(self.predict)
        self.back.clicked.connect(self.gotoback)
      
               
    def predict(self):
        
        vec = open('mlmodellr.pickle', 'rb')
        model = pickle.load(vec)

        C =['fire','not fire']
        
        
        
        
        
        x=[]
        
        temp = float(self.Tline1.text())
        
        RH = float(self.line2RH.text())
        ws = float(self.line3WS.text())
        R = float(self.line4Rain.text())
        FFMC = float(self.line5FFMC.text())
        DMC = float(self.line6DMC.text())
        DC = float(self.line7DC.text())
        ISI = float(self.line8ISI.text())
        BUI = float(self.line9BUI.text())
        FWI = float(self.line10FWI.text())        
                  
        x.append(temp)
        x.append(RH)
        x.append(ws)
        x.append(R)
        x.append(FFMC)
        x.append(DMC)
        x.append(DC)
        x.append(ISI)
        x.append(BUI)
        x.append(FWI)
        
        x = [x]
        pred = model.predict(x)
        a =C[pred[0]]
        
        
        
        self.labelcrop.setText(a) 
        
    def gotoback(QDialog):
         mainwindow = MainWindow()
         widget.addWidget(mainwindow)
         widget.setCurrentIndex(widget.currentIndex()+1)
        
app=QApplication(sys.argv)
mainwindow = MainWindow()
widget=QtWidgets.QStackedWidget()

widget.addWidget(mainwindow)
widget.setFixedWidth(1300)
widget.setFixedHeight(750)
widget.show()
sys.exit(app.exec_())




































