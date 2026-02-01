import cv2
import numpy as np
import mediapipe as mp
import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase
import time

# =========================
# Helper Functions
# =========================
def eye_aspect_ratio(eye):
    A = np.linalg.norm(eye[1] - eye[5])
    B = np.linalg.norm(eye[2] - eye[4])
    C = np.linalg.norm(eye[0] - eye[3])
    return (A + B) / (2.0 * C)

def mouth_aspect_ratio(mouth):
    A = np.linalg.norm(mouth[2] - mouth[10])
    B = np.linalg.norm(mouth[4] - mouth[8])
    C = np.linalg.norm(mouth[0] - mouth[6])
    return (A + B) / (2.0 * C)

# =========================
# Thresholds
# =========================
EAR_THRESHOLD = 0.25       # Eye closed threshold
MAR_THRESHOLD = 0.6        # Yawning threshold
DROWSY_TIME = 2.0          # Seconds before alert

# =========================
# MediaPipe Setup
# =========================
mp_face_mesh = mp.solutions.face_mesh

LEFT_EYE = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]
MOUTH = [61, 291, 81, 178, 13, 14, 311, 402, 308, 324, 318, 91]

# =========================
# Video Processor Class
# =========================
class DrowsinessDetector(VideoProcessorBase):
    def __init__(self):
        self.face_mesh = mp_face_mesh.FaceMesh(
            static_image_mode=False,
            refine_landmarks=True,
            max_num_faces=1
        )
        self.start_time = None
        self.alert = False

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        h, w, _ = img.shape

        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(rgb)

        if results.multi_face_landmarks:
            face = results.multi_face_landmarks[0]
            landmarks = np.array(
                [[int(lm.x * w), int(lm.y * h)] for lm in face.landmark]
            )

            left_eye = landmarks[LEFT_EYE]
            right_eye = landmarks[RIGHT_EYE]
            mouth = landmarks[MOUTH]

            ear = (eye_aspect_ratio(left_eye) + eye_aspect_ratio(right_eye)) / 2
            mar = mouth_aspect_ratio(mouth)

            if ear < EAR_THRESHOLD or mar > MAR_THRESHOLD:
                if self.start_time is None:
                    self.start_time = time.time()
                elif time.time() - self.start_time >= DROWSY_TIME:
                    self.alert = True
            else:
                self.start_time = None
                self.alert = False

            status_text = "DROWSY! STOP DRIVING"
            alert_color = (0, 0, 255)

            if not self.alert:
                status_text = "ALERT"
                alert_color = (0, 255, 0)

            cv2.putText(
                img,
                status_text,
                (30, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.2,
                alert_color,
                3
            )

        return img

# =========================
# Streamlit UI
# =========================
st.set_page_config(
    page_title="Driver Drowsiness Detection",
    layout="centered"
)

st.title("🚗 Driver Drowsiness Detection System")
st.write(
    "This application detects driver drowsiness using **eye closure** "
    "and **yawning detection** in real time to help prevent accidents."
)

webrtc_streamer(
    key="driver-drowsiness",
    video_processor_factory=DrowsinessDetector,
    media_stream_constraints={
        "video": True,
        "audio": False
    }
)
