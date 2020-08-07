import cv2
import tensorflow as tf
import numpy as np
import pafy
import time

def prepare(img_array):
        img_size = 200
        new_array = cv2.resize(img_array, (img_size, img_size))
        return new_array.reshape(-1, img_size, img_size, 3)


face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('haarcascade_righteye_2splits.xml')
model = tf.keras.models.load_model("mask_detector_model")
categories = ["with_mask", "without_mask"]
flag = 0
cap = cv2.VideoCapture(0)
get_time = float(time.time())
while True:
        ret, frame = cap.read()
        new_time = float(time.time())
        cv2.imshow('image', frame)
        if new_time >= get_time + 0.5:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                faces = face_cascade.detectMultiScale(gray, 1.3, 5)
                eyes = eye_cascade.detectMultiScale(gray, 1.3, 5)
                for (ex, ey, ew, eh) in eyes:
                        for (x, y, w, h) in faces:
                                if ex > x and ey < y and ex+ew < x+w and ey+eh <y+h:
                                        cv2.rectangle(frame, (x,y), (x+w,y+h), (0, 0, 255), 2)
                                        flag = 1
                        if flag == 0:
                                check_face = frame[ey-60:ey+eh+150, ex-60:ex+ew+140]
                                prediction = model.predict([prepare(check_face)])
                                if categories[int(prediction[0][0])] == "with_mask":
                                    cv2.rectangle(frame, (ex-60,ey-60), (ex+ew+140,ey+eh+160), (0, 255, 0), 2)
                                else:
                                    cv2.rectangle(frame, (ex-60,ey-60), (ex+ew+140,ey+eh+160), (0, 0, 255), 2)
                        else:
                                flag = 0
                                
                cv2.imshow('image', frame)
                get_time = float(time.time())

                                        
        k = cv2.waitKey(30) & 0xff
        if k == 27:
                break
cap.release()
cv2.destroyAllWindows()

