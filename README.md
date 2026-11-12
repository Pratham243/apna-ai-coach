# 💪 Apna AI Coach - Real-Time AI Gym Coach

> **Full perception-to-reasoning-to-voice pipeline**: live webcam → pose detection → rep counting → LLM coaching → voice feedback, all in-browser.

[![Python](https://img.shields.io/badge/Python-3.11-3776AB?style=flat&logo=python&logoColor=white)](https://python.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-WebRTC-FF4B4B?style=flat&logo=streamlit&logoColor=white)](https://streamlit.io)
[![MediaPipe](https://img.shields.io/badge/MediaPipe-Pose%20Landmarker-00897B?style=flat)](https://mediapipe.dev)
[![Groq](https://img.shields.io/badge/Groq-Llama%203.3%2070B-F55036?style=flat)](https://groq.com)
[![OpenCV](https://img.shields.io/badge/OpenCV-Computer%20Vision-5C3EE8?style=flat&logo=opencv)](https://opencv.org)
[![CI](https://github.com/Pratham243/apna-ai-coach/actions/workflows/ci.yml/badge.svg)](https://github.com/Pratham243/apna-ai-coach/actions)

---

## Live Demo

- **Hosted App:** [https://pratham243-apna-ai-coach.streamlit.app/](https://pratham243-apna-ai-coach.streamlit.app/)

---

## Demo Preview

*(GIF showing webcam feed with skeleton overlay + rep counter — captured from Streamlit Cloud after deploy)*

---

## What This Does

A real-time AI personal trainer that runs entirely in your browser. It watches you exercise via webcam, counts your reps, detects form errors (squat depth, elbow flare, back arch, knee caving, balance), speaks feedback aloud in natural language, answers mid-workout voice questions, and performs on-demand multimodal visual inspection — all live, frame by frame.

**No model API for the vision part** — all pose detection runs locally via MediaPipe.

---

## How It Works

```
Webcam (WebRTC, in-browser via streamlit-webrtc)
          │
          ▼
   streamlit-webrtc
   (frame-by-frame stream with STUN: stun.l.google.com:19302)
          │
          ▼
   MediaPipe Pose Landmarker
   (33 body keypoints, local CPU inference)
          │
          ▼
   Joint-Angle Geometry
   (pure NumPy vector math, cosine law)
          │
     ┌────┴──────────────────────────┐
     │                               │
     ▼                               ▼
 Rep Counter (State Machine)   Form Checker (Geometry Checks)
 (START -> TRANSITION ->       (squat depth, knees caving,
  INFLECTION -> RETURN)         elbow flare, sagging hips,
     │                          elbow drift, torso swing)
     │                               │
     └──────────────┬────────────────┘
                    │
                    ▼
     Every 5 Reps Trigger → Groq API
     (Llama 3.3 70B via groq SDK)
     [Fallback: Rule-based coaching]
                    │
                    ▼
     gTTS → MP3 → In-Browser Audio Autoplay
                    │
                    ▼
     OpenCV HUD Overlay
     (skeleton + live angles + stage + rep counter badge)
```

---

## Exercises Supported

| Exercise | Form Checks | Rep Counting Method |
|---|---|---|
| **Squats** | Hip crease depth (<90°), knees caving inward (valgus), excessive forward torso lean | Knee flexion & hip angle state machine |
| **Push-ups** | Chest depth (elbow ≤90°), sagging hips / piking, elbow flare (>75° from torso) | Elbow angle & plank alignment |
| **Bicep Curls** | Elbow drift from ribcage, torso swinging (momentum cheat), full extension & contraction | Elbow flexion & torso vertical angle |
| **Lunges** | Front knee-over-toes tracking, back knee depth towards floor, front knee 90° bend | Dual knee angle tracking |
| **Shoulder Press** | Overhead lockout extension, asymmetric arm press, excessive lumbar arch | Elbow angle & vertical wrist displacement |

All 5 detectors are built using an OOP abstract base class pattern — each exercise is an independent class inheriting from `ExerciseDetector` (`src/exercises/base.py`).

---

## Stack

| Component | Technology |
|---|---|
| **Pose Detection** | MediaPipe Pose Landmarker (33 keypoints, local CPU) |
| **Video Streaming** | streamlit-webrtc (WebRTC with STUN `stun:stun.l.google.com:19302`) |
| **Joint Geometry** | NumPy (vector math, cosine law) |
| **Visual Overlay** | OpenCV (skeleton overlay + HUD metrics + warning banner) |
| **Primary LLM** | Groq Llama 3.3 70B (`llama-3.3-70b-versatile`) |
| **Secondary VLM** | HuggingFace LLaVA (`llava-hf/llava-1.5-7b-hf`, 1 call/min rate limit) |
| **Voice STT** | faster-whisper (`base.en` model) |
| **Voice TTS** | gTTS (Google Text-to-Speech) |
| **Observability** | Langfuse SDK (traces all Groq + VLM inferences) |
| **Frontend** | Streamlit |
| **CI / Quality** | GitHub Actions, pytest, ruff |

---

## Voice Interaction (Whisper + gTTS)

Apna AI Coach supports real-time, hands-free voice interactions:
1. **Push-To-Talk Mic Input**: Athlete taps the recording button during their workout and asks questions like *"What am I doing wrong?"* or *"How many reps have I finished?"*.
2. **Speech-to-Text**: `faster-whisper` (`base.en`) transcribes the speech locally.
3. **Context-Aware Coaching**: Groq Llama 3.3 70B synthesizes an immediate 1-2 sentence spoken reply incorporating current rep counts and recent form flaws.
4. **Text-to-Speech Streaming**: `gTTS` converts the response to MP3 audio and streams it directly to the browser for hands-free audio playback.

---

## Multimodal Form Analysis (LLaVA VLM)

In addition to frame-by-frame geometric keypoint tracking, athletes can trigger deeper visual reasoning:
- Press **"Analyze My Form"** on the Streamlit dashboard.
- The pipeline captures a 3-frame temporal burst of the current movement.
- Frames are formatted and passed to the local LLaVA vision-language model (`llava-hf/llava-1.5-7b-hf`).
- Returns a structured verdict: `"pose looks correct: <detail>"` or `"form issue noticed: <detail>"`.
- **Rate Limited**: Gated to strictly 1 call per minute per user to conserve local GPU/CPU compute.

---

## Observability (Langfuse)

Every call to the LLM (coaching cues, voice Q&A) and VLM is traced using the **Langfuse Python SDK**:
- Traces model name, system prompts, user context, token generation, latency, and status.
- Configured via environment variables: `LANGFUSE_PUBLIC_KEY`, `LANGFUSE_SECRET_KEY`, `LANGFUSE_HOST`.
- **Silent Degradation**: If keys are absent from the environment, Langfuse telemetry automatically and silently disables itself with zero runtime overhead or warnings.

---

## Results (measured on 5 test videos)

The perception and form-tracking engine was rigorously evaluated across 5 real public Creative Commons exercise videos (20,619 total frames processed). Detailed ground truth annotations and raw benchmarks are recorded in [tests/evaluation/results.json](tests/evaluation/results.json).

| Exercise | Video | Frames | Pose Detection Rate | True Reps | Pred Reps | Rep Accuracy | p50 Latency | Form Recall |
|---|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **Squats** | `squats.mp4` | 4,046 | 89.6% | 58 | 61 | **94.8%** | 28.2 ms | 66.7% |
| **Push-ups** | `pushups.mp4` | 6,510 | 95.2% | 102 | 105 | **97.1%** | 72.0 ms | 33.3% |
| **Bicep Curls** | `bicep_curls.mp4` | 878 | 96.8% | 10 | 12 | **80.0%** | 73.8 ms | 100.0% |
| **Lunges** | `lunges.mp4` | 5,615 | 83.6% | 57 | 62 | **91.2%** | 72.3 ms | 66.7% |
| **Shoulder Press** | `shoulder_press.mp4` | 3,570 | 91.6% | 38 | 41 | **92.1%** | 70.7 ms | 100.0% |
| **Overall Mean** | *5 videos* | **20,619** | **91.4%** | **265** | **281** | **91.1%** | **63.4 ms** | **73.3%** |

*All metrics are measured directly from the evaluation harness ([tests/evaluation/run_evaluation.py](tests/evaluation/run_evaluation.py)) comparing against manually verified hand counts ([tests/evaluation/ground_truth.json](tests/evaluation/ground_truth.json)).*

---

## How to Run

### 1. Clone & Set Up

```bash
git clone https://github.com/Pratham243/apna-ai-coach.git
cd apna-ai-coach
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Configure Environment

Copy `.env.example` to `.env` and fill in your keys:

```bash
cp .env.example .env
```

```ini
GROQ_API_KEY=gsk_...
# Optional Langfuse tracing:
LANGFUSE_PUBLIC_KEY=
LANGFUSE_SECRET_KEY=
LANGFUSE_HOST=https://cloud.langfuse.com
```

### 3. Run Streamlit Application

```bash
streamlit run app.py
```

Allow webcam permissions in your browser. Select an exercise from the sidebar and start exercising.

### 4. Run Unit Tests & Evaluation Harness

```bash
# Run unit tests
pytest -v

# Run benchmark evaluation across all test videos
python tests/evaluation/run_evaluation.py
```

---

*Author: [Prathamesh Shimpi](https://linkedin.com/in/prathamesh-shimpi0501)*
