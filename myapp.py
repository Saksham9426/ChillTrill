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

# laoding the model and all the variables
emotions=["angry", "happy", "sad", "neutral"]
fishface = cv2.face.FisherFaceRecognizer_create()
fishface.read("model.xml")
font = cv2.FONT_HERSHEY_SIMPLEX
facedict={}
facecascade=cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
emo=''

if "run" not in st.session_state:
	st.session_state["run"] = "true"
#Face recognition functions
def crop(clahe_image, face):
    for (x, y, w, h) in face:
        faceslice=clahe_image[y:y+h, x:x+w]
        faceslice=cv2.resize(faceslice, (350, 350))
        facedict["face%s" %(len(facedict)+1)]=faceslice
    return faceslice

def detect_face(frame):
	#cv2.imshow("Video", frame)
	cv2.imwrite('test.jpg', frame)
	cv2.imwrite("main%s.jpg" %count, frame)
	gray=cv2.imread('test.jpg',0)
	#gray=cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
	clahe=cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
	clahe_image=clahe.apply(gray)
	face=facecascade.detectMultiScale(clahe_image, scaleFactor=1.1, minNeighbors=15, minSize=(10, 10), flags=cv2.CASCADE_SCALE_IMAGE)
	if len(face)>=1:
		faceslice=crop(clahe_image, face)
	#return faceslice
	else:
		print("No/Multiple faces detected!!, passing over the frame")

def identify_emotions():
  prediction=[]
  confidence=[]

  for i in facedict.keys():
      pred, conf=fishface.predict(facedict[i])
      cv2.imwrite("%s.jpg" %i, facedict[i])
      prediction.append(pred)
      confidence.append(conf)
  output=emotions[max(set(prediction), key=prediction.count)]    
  print("You seem to be %s" %output) 
  facedict.clear()
  return output;
class VideoProcessor:
	def recv(self,frame):
		frm = frame.to_ndarray(format="bgr24")
		frm = cv2.flip(frm, 1)
		cv2.imwrite('test.jpg', frm)
		cv2.imwrite("main%s.jpg" %count, frm)
		gray=cv2.imread('test.jpg',0)
		return av.VideoFrame.from_ndarray(gray, format="bgr24")
		count=0
		'''while True:
			count=count+1
			detect_face(frm)
			if count==10:
				emo = identify_emotions()
				cv2.putText(frm, pred, (50,50),cv2.FONT_ITALIC, 1, (255,0,0),2)
				return av.VideoFrame.from_ndarray(frm, format="bgr24")'''
webrtc_streamer(key="key", desired_playing_state=True,
				video_processor_factory=VideoProcessor,
				media_stream_constraints={"video": True, "audio": False},rtc_configuration={
      "iceServers": token.ice_servers
  })
