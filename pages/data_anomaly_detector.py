import os
import uuid
import ollama
import socket
import logging
import streamlit as st


os.environ['NO_PROXY'] = 'localhost,127.0.0.1,::1'

logger = logging.getLogger(__name__)

# Set Page Config
st.set_page_config(
    page_title = "DAD",
    page_icon = "🕵",
    layout = "wide",
    initial_sidebar_state = "expanded"
)

cols4 = st.columns([1, 20])

with cols4[0]:
    if st.button(label = "🏠︎"):
        st.switch_page("Home.py")
with cols4[1]:
    st.title("🕵 Data Anomaly Detector (DAD)", anchor = False)
st.subheader("Identify row and column level data mismatches between source and target tables", divider = "red", anchor = False)

