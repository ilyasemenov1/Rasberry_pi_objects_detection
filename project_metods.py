import sys
import time
import cv2
import tf_labels

tf_labels.initLabels("opencv-extra/mscoco_label_map.pbtxt")
cvNet = cv2.dnn.readNetFromTensorflow("resourses/frozen_inference_graph.pb", "opencv-extra/ssd_mobilenet_v1_coco.pbtxt")


class Main:
    
    def __init__(self, img):
        self.img = img

        
    def object_recognition(self):
        rows = self.img.shape[0]
        cols = self.img.shape[1]
        cvNet.setInput(cv2.dnn.blobFromImage(self.img, 1.0/127.5, (300, 300), (127.5, 127.5, 127.5), swapRB=True, crop=False))
        cvOut = cvNet.forward()

        for detection in cvOut[0,0,:,:]:
            score = float(detection[2])
            if score > 0.4:
                left = int(detection[3] * cols)
                top = int(detection[4] * rows)
                right = int(detection[5] * cols)
                bottom = int(detection[6] * rows)
                label = tf_labels.getLabel(int(detection[1]))
                print(label, score, left, top, right, bottom)
                text_color = (23, 230, 210)
                cv2.rectangle(self.img, (left, top), (right, bottom), text_color, thickness=2)
                cv2.putText(self.img, label, (left, top), cv2.FONT_HERSHEY_SIMPLEX, 1, text_color, 2)
                
        return self.img
    
    def road_recognition(self):
        pass
