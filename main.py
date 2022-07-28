from picamera.array import PiRGBArray
from picamera import PiCamera
from threading import Thread
import time
import cv2
import json
import string_int_label_map_pb2
import label_map_util
import tf_labels
import project_metods

class Camera:
    
    def __init__(self, resolution=(320, 240), framerate=32):
        self.cam = PiCamera()
        
        #with open("camera_settings.json", "r") as read_file:
        #    data = json.load(read_file)
        #    print(data)
        
        self.cam.resolution = resolution
        self.cam.framerate = framerate
        self.cam.sharpness = 0
        self.cam.contrast = 0
        self.cam.brightness = 50
        self.cam.saturation = 0
        self.cam.ISO = 0
        self.cam.video_stabilization = True
        self.cam.exposure_compensation = 0
        self.cam.exposure_mode = 'auto'
        self.cam.meter_mode = 'average'
        self.cam.awb_mode = 'auto'
        self.cam.image_effect = 'none'
        self.cam.color_effects = None
        self.cam.rotation = 0
        self.cam.hflip = False
        self.cam.vflip = False
        self.cam.crop = (0.0, 0.0, 1.0, 1.0)
        
        self.raw_capture = PiRGBArray(self.cam, size=resolution)
        self.stream = self.cam.capture_continuous(self.raw_capture, format="bgr", use_video_port=True)
        
        self.key = None
        self.img = None
        
        time.sleep(0.1)
        
    def video(self):
        
        for frame in self.stream:
            self.capture(frame)
            
            if self.key == ord("q"):
                break
        
    def capture(self, cap):
        self.img = cap.array
    
        self.img = project_metods.Main(self.img).object_recognition()
    
        cv2.imshow("Frame", self.img)
    
        self.key = cv2.waitKey(1) & 0xFF
        self.raw_capture.truncate(0)
    
        
if __name__ == "__main__":

    camera = Camera()
    camera.video()
