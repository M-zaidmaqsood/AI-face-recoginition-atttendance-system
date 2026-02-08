import streamlit as st
import requests
import pandas as pd

API_URL = "http://127.0.0.1:8000"

st.title("🎓 AI Attendance System")

if st.button("Start Recognition"):
    r = requests.post(f"{API_URL}/start")
    st.success(r.json()["message"])

if st.button("Stop Recognition"):
    r = requests.post(f"{API_URL}/stop")
    st.warning(r.json()["message"])

st.subheader("📊 Attendance Data")

try:
    df = pd.read_csv("attendance/attendance.csv")
    st.dataframe(df)
except:
    st.info("No attendance data yet")
