import math
import cv2
import numpy as np
import streamlit as st
import mediapipe as mp
from streamlit_webrtc import RTCConfiguration, VideoProcessorBase, webrtc_streamer

# Akses modul FaceMesh secara dinamis untuk menghindari issue import di server Linux
mp_face_mesh = getattr(mp.solutions, "face_mesh")

# --- PAGE SETUP ---
st.set_page_config(page_title="AI FocusGuard", page_icon="👁️", layout="centered")

st.title("👁️ AI FocusGuard")
st.caption("Precise Face & Eye Tracking menggunakan Streamlit + MediaPipe")


# Helper Distance & EAR
def calculate_distance(p1, p2):
  return math.sqrt((p1.x - p2.x) ** 2 + (p1.y - p2.y) ** 2)


def get_eye_aspect_ratio(landmarks, eye_indices):
  h = calculate_distance(landmarks[eye_indices[0]], landmarks[eye_indices[1]])
  v1 = calculate_distance(landmarks[eye_indices[2]], landmarks[eye_indices[3]])
  v2 = calculate_distance(landmarks[eye_indices[4]], landmarks[eye_indices[5]])
  return (v1 + v2) / (2.0 * h)


# --- VIDEO PROCESSOR CLASS ---
class FocusGuardProcessor(VideoProcessorBase):

  def __init__(self):
    self.closed_eye_frames = 0
    self.face_mesh = mp_face_mesh.FaceMesh(
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )

  def recv(self, frame):
    img = frame.to_ndarray(format="bgr24")

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = self.face_mesh.process(img_rgb)

    status_text = "TERALIHKAN (Wajah Tidak Terlihat)"
    color = (0, 165, 255)  # Orange

    if results.multi_face_landmarks:
      landmarks = results.multi_face_landmarks[0].landmark

      # Index Mata Resmi MediaPipe
      left_eye = [33, 133, 160, 144, 158, 153]
      right_eye = [362, 263, 385, 380, 387, 373]

      left_ear = get_eye_aspect_ratio(landmarks, left_eye)
      right_ear = get_eye_aspect_ratio(landmarks, right_eye)
      avg_ear = (left_ear + right_ear) / 2.0

      # Pose Wajah (Yaw)
      nose_tip = landmarks[1]
      left_cheek = landmarks[234]
      right_cheek = landmarks[454]
      dist_left = calculate_distance(nose_tip, left_cheek)
      dist_right = calculate_distance(nose_tip, right_cheek)
      yaw_ratio = dist_left / (dist_right + 1e-6)

      # Threshold Evaluasi
      is_eyes_closed = avg_ear < 0.25
      is_looking_away = yaw_ratio < 0.55 or yaw_ratio > 1.80

      if is_eyes_closed:
        self.closed_eye_frames += 1
      else:
        self.closed_eye_frames = 0

      # Penentuan Status
      if self.closed_eye_frames >= 2:
        status_text = f"MENGANTUK! (EAR: {avg_ear:.2f})"
        color = (0, 0, 255)  # Merah
      elif is_looking_away:
        status_text = f"TERALIHKAN (Yaw: {yaw_ratio:.2f})"
        color = (0, 255, 255)  # Kuning
      else:
        status_text = f"FOKUS (EAR: {avg_ear:.2f})"
        color = (0, 255, 0)  # Hijau

    cv2.putText(
        img,
        f"Status: {status_text}",
        (20, 50),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        color,
        2,
        cv2.LINE_AA,
    )

    return frame.from_ndarray(img, format="bgr24")


# --- WEBRTC STREAMER ---
RTC_CONFIGURATION = RTCConfiguration({
    "iceServers": [
        {"urls": ["stun:stun.l.google.com:19302"]},
        {"urls": ["stun:stun1.l.google.com:19302"]},
        {"urls": ["stun:stun2.l.google.com:19302"]},
    ]
})

webrtc_streamer(
    key="focus-guard",
    video_processor_factory=FocusGuardProcessor,
    rtc_configuration=RTC_CONFIGURATION,
    media_stream_constraints={"video": True, "audio": False},
)