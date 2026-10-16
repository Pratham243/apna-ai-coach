# 💪 Apna AI Coach — Real-Time AI Gym Coach

> **Full perception-to-reasoning-to-voice pipeline**: live webcam → pose detection → rep counting → LLM coaching → voice feedback, all in-browser.

[![Python](https://img.shields.io/badge/Python-3.11-3776AB?style=flat&logo=python&logoColor=white)](https://python.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-WebRTC-FF4B4B?style=flat&logo=streamlit&logoColor=white)](https://streamlit.io)
[![MediaPipe](https://img.shields.io/badge/MediaPipe-Pose%20Landmarker-00897B?style=flat)](https://mediapipe.dev)
[![Groq](https://img.shields.io/badge/Groq-Llama%203.3%2070B-F55036?style=flat)](https://groq.com)
[![OpenCV](https://img.shields.io/badge/OpenCV-Computer%20Vision-5C3EE8?style=flat&logo=opencv)](https://opencv.org)

---

## What This Does

A real-time AI personal trainer that runs entirely in your browser. It watches you exercise via webcam, counts your reps, detects form errors (squat depth, elbow drift, back arch, balance), tells you about them in natural language, and speaks the feedback aloud — all live, frame by frame.

**No model API for the vision part** — all pose detection runs locally via MediaPipe.

---

## Demo

*(Add a GIF here of the live webcam view with skeleton overlay + rep counter)*

---

## How It Works

```
Webcam (WebRTC, in-browser)
          │
          ▼
   streamlit-webrtc
   (frame-by-frame stream)
          │
          ▼
   MediaPipe Pose Landmarker
   (33 body keypoints, local)
          │
          ▼
   Joint-Angle Geometry
   (vector math, cosine law)
          │
     ┌────┴────┐
     │         │
     ▼         ▼
 Rep Counter  Form Checker
 (state       (squat depth,
  machine)     elbow drift,
               back arch,
               shoulder swing,
               balance)
     │         │
     └────┬────┘
          │
          ▼
   Form event → Groq API
   (Llama 3.3 70B)
   Natural language coaching
          │
          ▼
   gTTS → MP3 → Autoplay
   (voice feedback, in-browser)
          │
          ▼
   OpenCV overlay
   (skeleton + angles + status
    drawn onto video feed)
          │
          ▼
   SQLite (workout history)
```

---

## Exercises Supported

| Exercise | Form Checks | Rep Counting Method |
|---|---|---|
| **Squats** | Knee depth, back arch, balance | Hip angle threshold |
| **Push-ups** | Elbow flare, body alignment | Elbow angle threshold |
| **Bicep Curls** | Shoulder swing, elbow drift | Elbow angle threshold |
| **Shoulder Press** | Back arch, lockout check | Wrist height threshold |
| **Lunges** | Knee-over-toe, balance | Knee angle threshold |

All 5 detectors built from scratch using an OOP abstract base class pattern — each exercise is a class inheriting a common `ExerciseDetector` interface.

---

## Stack

| Component | Technology |
|---|---|
| Pose Detection | MediaPipe Pose Landmarker (33 keypoints) |
| Video Streaming | streamlit-webrtc (WebRTC, in-browser) |
| Joint Geometry | NumPy (vector math, cosine law) |
| Visual Overlay | OpenCV (skeleton + metric rendering) |
| LLM Coaching | Groq API (Llama 3.3 70B) |
| Text-to-Speech | gTTS (Google Text-to-Speech) |
| Frontend | Streamlit |
| Database | SQLite (workout history per user) |

---

## Key Engineering Highlights

- **Zero-latency local vision**: MediaPipe runs in-process — no cloud round-trip for pose detection, so coaching feedback appears in real time
- **Joint-angle geometry from scratch**: rep counting and form validation use vector math (cosine law on 3-keypoint angles), not ML classifiers — more interpretable and tunable
- **OOP exercise architecture**: adding a new exercise means subclassing `ExerciseDetector` and implementing 3 methods — the rest of the pipeline picks it up automatically
- **Full perception-to-voice loop**: the system goes from raw pixels to spoken coaching feedback entirely on its own, chaining computer vision → NLP → TTS

---

## How to Run

```bash
git clone https://github.com/Pratham243/apna-ai-coach
cd apna-ai-coach
pip install -r requirements.txt
# Add your Groq API key to .env
streamlit run app.py
```

Allow camera access in the browser prompt. Select an exercise and start moving.

---

*M.Sc. AI project · by [Prathamesh Shimpi](https://linkedin.com/in/prathamesh-shimpi0501)*
