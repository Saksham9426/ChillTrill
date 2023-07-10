# importing libraries
import streamlit as st
import cv2
from keras.models import load_model
import numpy as np
import random,time,base64
from twilio.rest import Client
import config
from tensorflow import keras
from tensorflow.keras.preprocessing.image import img_to_array
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase, RTCConfiguration, VideoProcessorBase

st.title('-----------')
st.header('Description')
st.markdown('fill in the details here')

#creating lists of songs for different emotions
angry = ['m1.mp3','m2.mp3','m3.mp3']
sad = ['m4.mp3','m5.mp3','m6.mp3']
happy = ['m7.mp3','m8.mp3','m9.mp3']
neutral = ['m10.mp3','m11.mp3','m12.mp3']

#creating Twilio account for the video access
account_sid = config.TWILIO_ACCOUNT_SID
auth_token = config.TWILIO_AUTH_TOKEN
client = Client(account_sid, auth_token)
token = client.tokens.create()

# Define the emotions.
emotion_labels = ['Angry','Disgust','Fear','Happy','Neutral', 'Sad', 'Surprise']

# Load model.
classifier =load_model('model_78.h5')

# load weights into new model
classifier.load_weights("model_weights_78.h5")

# Load face using OpenCV
try:
	face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
except Exception:
	st.write("Error loading cascade classifiers")

class VideoTransformer(VideoTransformerBase):
	def transform(self, frame):
		img = frame.to_ndarray(format="bgr24")
		
		#image gray
		img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
		faces = face_cascade.detectMultiScale(
		    image=img_gray, scaleFactor=1.3, minNeighbors=5)
		for (x, y, w, h) in faces:
			cv2.rectangle(img=img, pt1=(x, y), pt2=(
			x + w, y + h), color=(0, 255, 255), thickness=2)
			roi_gray = img_gray[y:y + h, x:x + w]
			roi_gray = cv2.resize(roi_gray, (48, 48), interpolation=cv2.INTER_AREA)
			if np.sum([roi_gray]) != 0:
				roi = roi_gray.astype('float') / 255.0
				roi = img_to_array(roi)
				roi = np.expand_dims(roi, axis=0)
				prediction = classifier.predict(roi)[0]
				maxindex = int(np.argmax(prediction))
				finalout = emotion_labels[maxindex]
				output = str(finalout)
			np.save("emotion.npy", np.array([output]))
			label_position = (x, y-10)
			cv2.putText(img, output, label_position, cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
	
		return img


webrtc_streamer(key="key", desired_playing_state=True,
				video_processor_factory=VideoTransformer,
				media_stream_constraints={"video": True, "audio": False},rtc_configuration={
      "iceServers": token.ice_servers
  })
st.write("# Auto-playing Audio!")

def auto(file_path: str):
	placeholder.empty()
	with open(file_path, "rb") as f:
		data = f.read()
	b64 = base64.b64encode(data).decode()
	md = f"""
		<audio controls autoplay="true">
		<source src="data:audio/mp3;base64,{b64}" type="audio/mp3">
		</audio>
		"""
	placeholder.markdown(
            md,
            unsafe_allow_html=True,
        )
placeholder = st.empty()
t = st.empty()
while True:
	final = []
	for i in range(0,20):
		emo = np.load("emotion.npy")[0]
		final.append(emo)
	t.empty()
	my = max(set(final), key = final.count)
	t.write('You seem'+my)
	if my =='Happy':
		auto(random.choice(happy))
	if my == 'Angry':
		auto(random.choice(angry))
	if my == 'Sad':
		auto(random.choice(sad))
	if my == 'Neutral':
		auto(random.choice(neutral))
	if my =='happy'):
		auto('m13.mp3')
	time.sleep(120)

