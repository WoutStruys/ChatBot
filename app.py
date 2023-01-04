#!/usr/bin/env python3

from chatbot import Chatbot
from recognition import Recognition

import cv2
import random
import numpy as np
import streamlit as st
from streamlit_chat import message

def face_rec(rec, frame):
    crop = rec.detect_and_crop_faces(frame)
    name, access = rec.face_rec(crop)
    return name, access
    
def detect_face():
    # Get a frame from the webcam
    img_file_buffer = st.camera_input("Start Chatting", key="camera")  
    
    if img_file_buffer is not None:
        # To read image file buffer with OpenCV:
        bytes_data = img_file_buffer.getvalue()
        cv2_img = cv2.imdecode(np.frombuffer(bytes_data, np.uint8), cv2.IMREAD_COLOR)
        return cv2_img
    return None

def get_text():
    input_text = st.text_input("", key="input", label_visibility="hidden")
    return input_text 

  
@st.experimental_singleton(show_spinner=False) 
def load_chatbot():
    return Chatbot() 
    
@st.experimental_singleton(show_spinner=False) 
def load_recognition():
    return Recognition() 
         
def main():
    st.set_page_config(
        page_title="Streamlit Chat - Demo",
        page_icon="🤖",
        initial_sidebar_state="expanded",
        menu_items={
        'Get help': 'https://github.com/PXLAIRobotics/AI2223Jaar3_Groep1'
    }
    )
    
    if "access" not in st.session_state:
        st.session_state.access = False
        st.session_state.name = "Unknown"
    
    if 'bot' not in st.session_state:
        st.session_state.bot = []

    if 'user' not in st.session_state:
        st.session_state.user = []
        
    if "seed" not in st.session_state:
        st.session_state.seed = random.randint(0, 1000)
    
    TITLE = st.empty()
    
    with st.spinner("Loading Chatbot..."):
        chatbot = load_chatbot()
    with st.spinner("Loading Face Recognition..."):
        rec = load_recognition()
    
    if st.session_state.access == False:
        TITLE.title("Face Recognition")
        with st.spinner("Recognizing faces in video frames from a webcam..."):
            frame = detect_face()
            if frame is not None:
                name, access = face_rec(rec, frame)
                if access:
                    st.session_state.access = access
                    st.session_state.name = name
                    if len(st.session_state.bot) == 0:
                        st.session_state.bot.append("Hello " + st.session_state.name + "!") 
                        st.experimental_rerun()
                else:
                    st.error("Access denied " + name + "!")
                    st.session_state.camera
    else:
        TITLE.title("Chat with me")
        
        if st.session_state.bot:
            for i in range(0, len(st.session_state.bot), 1):
                if len(st.session_state.bot) > i:
                    message(st.session_state.bot[i], key=str(i), seed=st.session_state.seed)
                if len(st.session_state.user) > i:
                    message(st.session_state.user[i], is_user=True, key=str(i) + '_user', seed=st.session_state.seed)
                
        placeholder = st.empty()
        
        input_text = placeholder.text_input("", value="", key="input", label_visibility="hidden")
        
        if len(input_text) > 0:
            st.session_state.user.append(input_text)
            # message(user_input, is_user=True, key=str(i) + '_user')
            with st.spinner("Generating response..."):
                try:
                    response = chatbot.get_emoji(input_text)
                except Exception:
                    response = "I'm sorry, I didn't understand that."
                
            print("response: " + response)
            st.session_state.bot.append(response)
            placeholder.empty()
            del st.session_state.input
            input_text = ""

            st.experimental_rerun()
    
 
if __name__ == "__main__":
    main()
