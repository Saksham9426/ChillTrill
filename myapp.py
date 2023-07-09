# importing libraries
import streamlit as st
from streamlit_webrtc import webrtc_streamer,RTCConfiguration,VideoTransformerBase
import av,cv2,os
import numpy as np
import random,time,base64
from twilio.rest import Client
import config
from tensorflow import keras
from keras.models import model_from_json
from keras.preprocessing.image import img_to_array

path = os.path.dirname(__file__)



#creating Twilio account for the video access
account_sid = config.TWILIO_ACCOUNT_SID
auth_token = config.TWILIO_AUTH_TOKEN
client = Client(account_sid, auth_token)
token = client.tokens.create()

# load model
fishface = cv2.face.FisherFaceRecognizer_create()
font = cv2.FONT_HERSHEY_SIMPLEX
try:
	fishface.read(os.path.join(path,"model.xml"))
except:
	st.write("No trained model found... --update will create one.")
try:
	facecascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
except:
	st.write("classifier not loaded")
st.write(fish)
class VideoTransformer(VideoTransformerBase):
	def transform(self, frame):
		img = frame.to_ndarray(format="bgr24")
		img = cv2.flip(img,1)
		img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
		faces=facecascade.detectMultiScale(img_gray, scaleFactor=1.1, minNeighbors=15, minSize=(10, 10), flags=cv2.CASCADE_SCALE_IMAGE)
		for (x, y, w, h) in faces:
			cv2.rectangle(img=img, pt1=(x, y), pt2=(x + w, y + h), color=(255, 0, 0), thickness=2)
			roi_gray = img_gray[y:y + h, x:x + w]
			roi_gray = cv2.resize(roi_gray, (350,350))
			if np.sum([roi_gray]) != 0:
				pred, conf=fishface.predict(roi_gray)
				finalout = emotion[pred]
				output = str(finalout)
			label_position = (x, y)
			cv2.putText(img, output, label_position, cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
		
		return img

webrtc_streamer(key="key", desired_playing_state=True,
				video_transformer_factory=VideoTransformer,
				media_stream_constraints={"video": True, "audio": False},rtc_configuration={
      "iceServers": token.ice_servers
  })
