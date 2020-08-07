import tensorflow as tf
import boto3
import csv
import os
import io
import numpy as np
import cv2
from PIL import Image, ImageDraw
import time

model = tf.keras.models.load_model("mask_detector_model")


def prepare(img_array):
        img_size = 200
        new_array = cv2.resize(img_array, (img_size, img_size))
        return new_array.reshape(-1, img_size, img_size, 3)

        


with open('new_user_credentials.csv', 'r') as creds:
    next(creds)
    reader = csv.reader(creds)
    for keys in reader:
        access_key = keys[2]
        secret_access_key = keys[3]

client = boto3.client('rekognition',
                      aws_access_key_id = access_key,
                      aws_secret_access_key = secret_access_key,
                      region_name = 'us-east-1')


categories = ["with_mask", "without_mask"]

cap = cv2.VideoCapture(0)
get_time = int(time.time())
while(True):
    ret, frame = cap.read()
    
    new_time = int(time.time())
    if new_time >= get_time + 1:
        get_time = int(time.time())
        cv2.imwrite("temp2.jpg", frame)
        image = Image.open("temp2.jpg")
        imgWidth, imgHeight = image.size
        draw = ImageDraw.Draw(image)

        with open("temp2.jpg", 'rb') as source:
                img_bytes = source.read()

        response = client.detect_faces(Image = {'Bytes':img_bytes}, Attributes = ['DEFAULT'])
        os.remove("temp2.jpg")  

        for faceDetail in response['FaceDetails']:
            
            box = faceDetail['BoundingBox']
            left = imgWidth * box['Left']
            top = imgHeight * box['Top']
            width = imgWidth * box['Width']
            height = imgHeight * box['Height']
            points = (
                    (left,top),
                    (left + width, top),
                    (left + width, top + height),
                    (left , top + height),
                    (left, top)
                    )
            cv2.imshow('video', frame)

            img = image.crop((left, top, left+width, top+height))
            img1 = np.array(img)
            img1 = img1[:,:,::-1].copy()

            prediction = model.predict([prepare(img1)])
                    

            if categories[int(prediction[0][0])] == "with_mask":
                    draw.line(points, fill='#00d400', width=2)
            else:
                    draw.line(points, fill='#FF0000', width=2)

            new_frame = np.array(image)
            new_frame = new_frame[:,:,::-1].copy()
            cv2.imshow('video', new_frame)
    

        

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
