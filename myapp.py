# importing libraries
import streamlit as st
from streamlit_webrtc import webrtc_streamer
import av
import cv2 
import numpy as np
import random
import time
from twilio.rest import Client
import config
import base64

#creating Twilio account for the video access
account_sid = config.TWILIO_ACCOUNT_SID
auth_token = config.TWILIO_AUTH_TOKEN
client = Client(account_sid, auth_token)
token = client.tokens.create()
path = os.path.dirname(__file__)

# laoding the model and all the variables
emotions=["angry", "happy", "sad", "neutral"]
fishface = cv2.face.FisherFaceRecognizer_create()
font = cv2.FONT_HERSHEY_SIMPLEX
facedict={}
facecascade=cv2.CascadeClassifier("WD_INNOVATIVE/haarcascade_frontalface_default.xml")

if "run" not in st.session_state:
	st.session_state["run"] = "true"
#Face recognition functions
def crop(clahe_image, face):
    for (x, y, w, h) in face:
        faceslice=clahe_image[y:y+h, x:x+w]
        faceslice=cv2.resize(faceslice, (350, 350))
        facedict["face%s" %(len(facedict)+1)]=faceslice
    return faceslice

def grab_face():
    ret, frame=video_capture.read()
    #cv2.imshow("Video", frame)
    cv2.imwrite('WD_INNOVATIVE/test.jpg', frame)
    cv2.imwrite("WD_INNOVATIVE/main%s.jpg" %count, frame)
    gray=cv2.imread('WD_INNOVATIVE/test.jpg',0)
    #gray=cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    clahe=cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    clahe_image=clahe.apply(gray)
    return clahe_image

def detect_face():
    clahe_image=grab_face()
    face=facecascade.detectMultiScale(clahe_image, scaleFactor=1.1, minNeighbors=15, minSize=(10, 10), flags=cv2.CASCADE_SCALE_IMAGE)
    if len(face)>=1:
        faceslice=crop(clahe_image, face)
        #return faceslice
    else:
        print("No/Multiple faces detected!!, passing over the frame")

def save_face(emotion):
    print("\n\nLook "+emotion+" untill the timer expires and keep the same emotion for some time.")
    #winsound.Beep(frequency, duration)
    print('\a')
    
    
    for i in range(0, 5):
        print(5-i)
        time.sleep(1)
    
    while len(facedict.keys())<16:
        detect_face()

    for i in facedict.keys():
        path, dirs, files = next(os.walk("dataset/%s" %emotion))
        file_count = len(files)+1
        cv2.imwrite("dataset/%s/%s.jpg" %(emotion, (file_count)), facedict[i])
    facedict.clear()
def identify_emotions():
  prediction=[]
  confidence=[]

  for i in facedict.keys():
      pred, conf=fishface.predict(facedict[i])
      cv2.imwrite("WD_INNOVATIVE/%s.jpg" %i, facedict[i])
      prediction.append(pred)
      confidence.append(conf)
  output=emotions[max(set(prediction), key=prediction.count)]    
  print("You seem to be %s" %output) 
  facedict.clear()
  return output;
def getEmotion():
    count=0
    while True:
        count=count+1
        detect_face()
        if count==10:
            fishface.read("WD_INNOVATIVE/model.xml")
            return identify_emotions()
            break
webrtc_streamer(key="key", desired_playing_state=True,
				video_processor_factory=getEmotion,
				media_stream_constraints={"video": True, "audio": False},rtc_configuration={
      "iceServers": token.ice_servers
  })
