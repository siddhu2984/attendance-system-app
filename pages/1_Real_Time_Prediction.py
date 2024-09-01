import streamlit as st 
from Home import face_rec
from streamlit_webrtc import webrtc_streamer
import av
import time

# st.set_page_config(page_title='Real-Time Attendance System',layout='centered')
st.subheader('Real-Time Attendance System')

# Retrive the data from Redis Database
with st.spinner('Retriving Data from Redis DB ...'):    
    redis_face_db = face_rec.retrive_data(name='academy:register')
    st.dataframe(redis_face_db)
    
st.success("Data sucessfully retrived from Redis")

waitTime = 10 
setTime = time.time()
realtimepred = face_rec.RealTimePred() 

# Real Time Prediction
def video_frame_callback(frame):
    global setTime
    
    img = frame.to_ndarray(format="bgr24") 
    pred_img = realtimepred.face_prediction(img,redis_face_db,'facial_features',['Name','Role'],thresh=0.5)
   #  print(pred_img)
    timenow = time.time()
    difftime = timenow - setTime
    if difftime >= waitTime:
        realtimepred.saveLogs_redis()
        setTime = time.time()       
        print('Save Data to redis database')
    

    return av.VideoFrame.from_ndarray(pred_img, format="bgr24")


webrtc_streamer(key="realtimePrediction", video_frame_callback=video_frame_callback,
rtc_configuration={
        "iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]
    }
)
