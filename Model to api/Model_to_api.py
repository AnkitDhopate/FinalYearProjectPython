from fastapi import FastAPI, File, UploadFile
from pydantic import BaseModel  
import json

from PIL import Image
import io

import cv2
import numpy as np
from keras.models import model_from_json
from fastapi.middleware.cors import CORSMiddleware

emotion_dict = {0: "Angry", 1: "Disgusted", 2: "Fearful", 3: "Happy", 4: "Neutral", 5: "Sad", 6: "Surprised"}

# load json and create model
json_file = open('emotion_model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
emotion_model = model_from_json(loaded_model_json)

# load weights into new model
emotion_model.load_weights("emotion_model.h5")

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class model_input(BaseModel):
    img : str  

@app.post('/predict')
async def emotion_pred(file: UploadFile = File(...)):
    with open("image.jpg", "wb") as f:
        f.write(await file.read())
    
    # read the image file
    frame = cv2.imread("image.jpg") 

    height, width, _ = frame.shape
    aspect_ratio = width / height
    new_height = 240
    new_width = int(new_height * aspect_ratio)
    frame = cv2.resize(frame, (new_width, new_height))

    face_detector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    result = "none"	

    # detect faces available on camera
    num_faces = face_detector.detectMultiScale(gray_frame, scaleFactor=1.3, minNeighbors=5)
    # take each face available on the camera and Preprocess it
    for (x, y, w, h) in num_faces:
        cv2.rectangle(frame, (x, y-50), (x+w, y+h+10), (0, 255, 0), 4)
        roi_gray_frame = gray_frame[y:y + h, x:x + w]
        cropped_img = np.expand_dims(np.expand_dims(cv2.resize(roi_gray_frame, (48, 48)), -1), 0)

        # predict the emotions
        emotion_prediction = emotion_model.predict(cropped_img)
        maxindex = int(np.argmax(emotion_prediction))
        result = emotion_dict[maxindex]
        cv2.putText(frame, emotion_dict[maxindex], (x+5, y-20), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
    
    if (result == "none"):
        return 'No face detected'
    else:
        return result