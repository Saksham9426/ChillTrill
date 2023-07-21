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

st.title('ChillTrill')
st.subheader('An advanced webapp that utilizes deep learning and AI-ML technology to detect micro-changes in your facial expressions, allowing it to assess if the currently played music is inducing a calming effect for you. It can detect different emotions for example happy, sad, angry, calm, neutral; based on which, it dynamically consistently changes the music selection according to the data it collects of songs which seem to calm your and help you focus on your tasks.')
st.caption('While working for long hours or even short durations on your laptops/computers/phones, the webcam on your device can be used to take in your facial input in real-time. We dynamically curate and play songs to help users focus taking into account what seems to be working and what does not; thus creating the perfect environment for your needs. More specifically, it assists people with attention deficit disorders -ADD, ADHD- to focus and concentrate on important tasks consistently when it may seem tough to do so. Your privacy is of the utmost importance, therefore none of the video inputs taken in real-time are stored or sent anywhere, it is only used to determine your current sentiment analysis to better curate music playlists automatically for our users and help them focus and aid in productivity with scientifically researched and tested data.')
#st.markdown('')

#creating lists of songs for different emotions
angry = ['m1.mp3','m2.mp3','m3.mp3','All These Things That Ive Done.mp3','All You Ever Wanted.mp3','Arizona.mp3','Beautiful War.mp3','Chopin_ Etude in E Major Op.10 No.3.mp3','Chopin_ Nocturne in E flat Major Op.9 No.2.mp3','Cold Desert.mp3','Cruel World.mp3']
sad = ['m4.mp3','m5.mp3','m6.mp3','Grains of Sand.mp3','Grapevine Fires.mp3','Green Light (feat. André 3000).mp3','Hard Sun.mp3','Hey Brother.mp3','Honky Cat.mp3','I Smile.mp3','If No One Will Listen.mp3','Inside Out.mp3','Chopin_ Prelude Raindrop in D flat Major Op.28 No.15.mp3']
happy = ['m7.mp3','m8.mp3','m9.mp3','Lost.mp3','m y . l i f e (with 21 Savage & Morray).mp3','My Girl.mp3','Never Give Up.mp3','Ragoo.mp3','Renegades Of Funk.mp3','Revelry.mp3','Seven Nation Army.mp3','Simple Man.mp3','Smells Like Teen Spirit.mp3','St Jude.mp3','Sugar.mp3']
neutral = ['m10.mp3','m11.mp3','m12.mp3','Sunday Morning.mp3','Super Rich Kids.mp3','Teenage Dream.mp3','The Grand Optimist.mp3','The Lion Sleeps Tonight.mp3','Victory.mp3','Wake Me up When September Ends.mp3','Winning.mp3','Wish You Were Here.mp3','Within.mp3','Wonder Woman Theme.mp3']
fear = ['m1.mp3','m5.mp3','m8.mp3','Dont Stop Believin.mp3','Flaws and All.mp3','Free Bird.mp3']
surprise = ['m9.mp3','m3.mp3','m12.mp3','Intentional.mp3','Just Dance.mp3','Killing Strangers.mp3']

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
	t.write('You seem '+my)
	if my =='Happy':
		auto(random.choice(happy))
	if my == 'Angry':
		auto(random.choice(angry))
	if my == 'Sad':
		auto(random.choice(sad))
	if my == 'Neutral':
		auto(random.choice(neutral))
	if my =='happy':
		auto('Chopin_ Etude in E Major Op.10 No.3.mp3')
	if my == 'Fear':
		auto(random.choice(fear))
	if my == 'Surprise':
		auto(random.choice(surprise))
	time.sleep(90)

