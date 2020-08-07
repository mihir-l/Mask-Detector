import tensorflow as tf
import numpy as np
import cv2
import boto3
from PIL import Image, ImageDraw
import csv
import os
import io
import time

model = tf.keras.models.load_model("mask_delector_model")


def prepare(file):
        img_size = 200
        img_array = cv2.imread(file)
        new_array = cv2.resize(img_array, (img_size, img_size))
        return new_array.reshape(-1, img_size, img_size, 3)


def check(photo):
        image = Image.open(photo)
        imgWidth, imgHeight = image.size
        draw = ImageDraw.Draw(image)

        with open(photo, 'rb') as source:
                img_bytes = source.read()

        response = client.detect_faces(Image = {'Bytes':img_bytes}, Attributes = ['DEFAULT'])

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

                img = image.crop((left, top, left+width, top+height))
                img1 = np.array(img)
                img1 = img1[:,:,::-1].copy()
                draw.line(points, fill='#00d400', width=2)

##                img.save("temp1.jpg")
##                prediction = model.predict([prepare("temp1.jpg")])
##
##                if categories[int(prediction[0][0])] == "with_mask":
##                        draw.line(points, fill='#00d400', width=2)
##                else:
##                        draw.line(points, fill='#FF0000', width=2)
##                os.remove("temp1.jpg")


        image.show()
        cv2.imshow('img',img1)





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

photo = input("Image")
check(photo)




