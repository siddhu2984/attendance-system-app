import streamlit as st
from Home import face_rec
import cv2
import numpy as np
from streamlit_webrtc import webrtc_streamer
import av


# st.set_page_config(page_title='Registration Form',layout='centered')
st.subheader('Registration Form')

registration_form = face_rec.RegistrationForm()

# Collect name role

person_name = st.text_input(label='Name',placeholder='First & Last Name')
role = st.selectbox(label='Select your Role',options=('Student',
                                                      'Teacher'))


# Collect facial embedding 
def video_callback_func(frame):
    img = frame.to_ndarray(format='bgr24') 
    reg_img, embedding = registration_form.get_embedding(img)
    if embedding is not None:
        with open('face_embedding.txt',mode='ab') as f:
            np.savetxt(f,embedding)
    
    return av.VideoFrame.from_ndarray(reg_img,format='bgr24')

webrtc_streamer(key='registration',video_frame_callback=video_callback_func,
rtc_configuration={
        "iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]
    }
)


#save in redis
if st.button('Submit'):
    return_val = registration_form.save_data_in_redis_db(person_name,role)
    if return_val == True:
        st.success(f"{person_name} registered sucessfully")
    elif return_val == 'name_false':
        st.error('Please enter the name: Name cannot be empty or spaces')
        
    elif return_val == 'file_false':
        st.error('face_embedding.txt is not found. Please refresh the page and execute again.')
        
