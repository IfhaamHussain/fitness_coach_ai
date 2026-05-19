import streamlit as st
import pandas as pd
import requests
import cv2
import mediapipe as mp
import numpy as np
import av
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase
from datetime import datetime

# -----------------------------
# CONFIG
# -----------------------------
st.set_page_config(
    page_title="AI Fitness Coach Pro",
    page_icon="💪",
    layout="wide"
)

# -----------------------------
# CUSTOM STYLING
# -----------------------------
st.markdown("""
<style>
.main {
    background-color: #0f172a;
    color: white;
}
h1, h2, h3 {
    color: #22c55e;
}
.stButton>button {
    background-color: #22c55e;
    color: white;
    border-radius: 10px;
    padding: 10px 20px;
    font-weight: bold;
}
section[data-testid="stSidebar"] {
    background-color: #111827;
}
</style>
""", unsafe_allow_html=True)

# -----------------------------
# HEADER
# -----------------------------
st.title("💪 AI Virtual Personal Fitness Coach Pro")
st.subheader("Personalized Workout + Diet + Live Workout Tracker")

# -----------------------------
# SIDEBAR USER PROFILE
# -----------------------------
st.sidebar.header("👤 User Profile")

age = st.sidebar.slider("Age", 10, 80, 21)
gender = st.sidebar.selectbox("Gender", ["Male", "Female"])
height = st.sidebar.slider("Height (cm)", 100, 220, 170)
weight = st.sidebar.slider("Weight (kg)", 30, 200, 70)

goal = st.sidebar.selectbox(
    "Fitness Goal",
    ["Weight Loss", "Muscle Gain", "Maintenance"]
)

fitness_level = st.sidebar.selectbox(
    "Fitness Level",
    ["Beginner", "Intermediate", "Advanced"]
)

workout_location = st.sidebar.selectbox(
    "Workout Location",
    ["Gym", "Home", "No Equipment"]
)

diet_pref = st.sidebar.selectbox(
    "Diet Preference",
    ["Vegetarian", "Non-Vegetarian"]
)

days_per_week = st.sidebar.slider("Workout Days Per Week", 3, 7, 5)
workout_duration = st.sidebar.selectbox(
    "Daily Workout Duration",
    ["30 min", "45 min", "60 min", "90 min"]
)

# -----------------------------
# BMI CALCULATION
# -----------------------------
height_m = height / 100
bmi = weight / (height_m ** 2)
protein_goal = round(weight * 1.6)

if bmi < 18.5:
    bmi_status = "Underweight"
elif bmi < 25:
    bmi_status = "Normal"
elif bmi < 30:
    bmi_status = "Overweight"
else:
    bmi_status = "Obese"

# -----------------------------
# DASHBOARD METRICS
# -----------------------------
col1, col2, col3 = st.columns(3)

with col1:
    st.metric("BMI", f"{bmi:.2f}", bmi_status)

with col2:
    st.metric("Protein Goal", f"{protein_goal}g/day")

with col3:
    st.metric("Workout Days", f"{days_per_week}/week")

# -----------------------------
# OPENROUTER API FUNCTION
# -----------------------------
def generate_ai_plan(prompt):
    url = "https://openrouter.ai/api/v1/chat/completions"

    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json"
    }

    payload = {
        "model": "openai/gpt-3.5-turbo",
        "messages": [
            {
                "role": "user",
                "content": prompt
            }
        ]
    }

    response = requests.post(url, headers=headers, json=payload)

    if response.status_code == 200:
        return response.json()["choices"][0]["message"]["content"]
    else:
        return "Error generating plan."

# -----------------------------
# GENERATE WORKOUT + DIET PLAN
# -----------------------------
if st.button("🚀 Generate My AI Fitness Plan"):
    prompt = f"""
    Create a detailed weekly workout plan and daily diet chart for:
    Age: {age}
    Gender: {gender}
    Height: {height} cm
    Weight: {weight} kg
    Goal: {goal}
    Fitness Level: {fitness_level}
    Workout Location: {workout_location}
    Diet Preference: {diet_pref}
    Workout Days Per Week: {days_per_week}
    Daily Workout Duration: {workout_duration}

    Include:
    - Day-wise gym/home exercises
    - Sets and reps
    - Diet plan
    - Hydration
    - Supplement suggestions
    - Recovery tips
    """

    with st.spinner("Generating your personalized AI transformation plan..."):
        ai_plan = generate_ai_plan(prompt)

    st.header("📅 Your Personalized AI Fitness Transformation Plan")
    st.write(ai_plan)

# -----------------------------
# LIVE BICEP CURL TRACKER
# -----------------------------
st.header("🎥 Live Workout Tracker (Bicep Curl Counter)")

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

def calculate_angle(a, b, c):
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)

    radians = np.arctan2(
        c[1]-b[1], c[0]-b[0]
    ) - np.arctan2(
        a[1]-b[1], a[0]-b[0]
    )

    angle = np.abs(radians * 180.0 / np.pi)

    if angle > 180:
        angle = 360 - angle

    return angle


class FitnessCoach(VideoProcessorBase):
    def __init__(self):
        self.pose = mp_pose.Pose(
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        self.counter = 0
        self.stage = None

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")

        image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = self.pose.process(image)

        if results.pose_landmarks:
            landmarks = results.pose_landmarks.landmark

            shoulder = [
                landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x,
                landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y
            ]

            elbow = [
                landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].x,
                landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].y
            ]

            wrist = [
                landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].x,
                landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].y
            ]

            angle = calculate_angle(shoulder, elbow, wrist)

            if angle > 150:
                self.stage = "down"

            if angle < 50 and self.stage == "down":
                self.stage = "up"
                self.counter += 1

            cv2.rectangle(img, (0, 0), (320, 100), (34, 197, 94), -1)

            cv2.putText(
                img,
                f"REPS: {self.counter}",
                (10, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (255,255,255),
                2
            )

            cv2.putText(
                img,
                f"STAGE: {self.stage}",
                (10, 80),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (255,255,255),
                2
            )

            mp_drawing.draw_landmarks(
                img,
                results.pose_landmarks,
                mp_pose.POSE_CONNECTIONS
            )

        return av.VideoFrame.from_ndarray(img, format="bgr24")


webrtc_streamer(
    key="fitness-coach",
    video_processor_factory=FitnessCoach
)

# -----------------------------
# FOOTER
# -----------------------------
st.success("🔥 Your AI-powered fitness ecosystem is ready!")

