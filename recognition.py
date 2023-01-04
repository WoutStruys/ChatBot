#!/usr/bin/env python3

import json
import cv2
import numpy as np

class Recognition:
    def __init__(self):
        self.data = {}
        self.rec = cv2.face.LBPHFaceRecognizer_create()
        self.rec.read("trainingData.yml")
        self.face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
        self.read_json()

    def read_json(self):
        with open('labels.json') as json_file:
            # write to global variable data
            self.data = json.load(json_file)

    def get_label(self, id):
        for label in self.data:
            if label["id"] == id:
                return label["name"]
        return "Unknown"

    def detect_and_crop_faces(self, frame):
        if frame is None:
            return None
        
        gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
        
        faces = self.face_cascade.detectMultiScale(gray,1.3,5);
        
        # Draw rectangle around the faces
        for (x, y, w, h) in faces: 
            crop_img = gray[y:y+h, x:x+w]
            return crop_img
        return None
        
    def face_rec(self, frame):
        if frame is None:
            return "Unknown", False
        access = False
        name = "Unknown"
        face_id, conf = self.rec.predict(frame)
        if(conf < 100):
            name = self.get_label(face_id)
            if name != "Unknown":
                access = True
        return name, access
