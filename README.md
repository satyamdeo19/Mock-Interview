.

📘 AI Mock Interview System — README
🚀 A Complete Multimodal AI Interview Evaluation System

Text + Audio + Video → Behavioral Scoring → SHAP Explainability → Personalized Feedback

🧩 Project Overview

This project is an AI-powered Mock Interview System that analyzes video, audio, and text responses to evaluate a candidate across 10 behavioral dimensions such as Confidence, Fluency, Engagement, Professionalism, Cognitive Complexity, etc.

It extracts over 120 multimodal features, performs feature engineering, fuses modalities using a weighted scoring model, and provides detailed, explainable interview feedback using SHAP.

The system consists of:

Backend: FastAPI + Python (feature extraction + scoring + model inference)

Frontend: React.js (webcam recording, dashboard, UI)

Machine Learning: RandomForest models trained on multimodal interview datasets

📂 Project Structure
____venv
├── backend
│
│   ├── __pycache__
│   ├── .env
│   ├── benchmark_data.json
│   ├── config.py
│   ├── feature_config.json
│   ├── feature_weights.json
│   ├── main.py
│   ├── process_session.py
│   ├── question_generator.py
│   ├── resume_parser.py
│   ├── feature_engineering
│   │   ├── init.py
│   │   ├── multimodal_fusion.py
│   │   └── video_aggregator.py
│   ├── feature_extractors
│   │   ├── audio_extractor.py
│   │   ├── text_extractor.py
│   │   └── video_extractor.py
│   ├── feedback_engine
│   │   ├── feedback_generator.py
│   │   ├── predictor.py
│   │   ├── shap_analyzer.py
│   │   └── weighted_scorer.py
│   ├── feedback_results
│   │   ├── models
│   │   │   ├── human_aligned_model.joblib
│   │   │   └── scaler.joblib
│   │   └── processed_features
│   ├── recordings
│   │   └── sessions
│   ├── temp
│   │   └── visualization
│   │       ├── chart_generator.py
│   │       └── report_builder.py
│   └── __init__.py
└── frontend
    ├── node_modules
    ├── public
    ├── package-lock.json
    ├── package.json
    ├── src
    │   ├── App.jsx
    │   ├── components
    │   ├── hooks
    │   ├── services
    │   ├── styles
    │   └── utils
    └── README

⚙️ System Requirements
Backend Requirements

Python 3.10+

FFmpeg (mandatory for audio/video processing)

pip 23+

Virtual environment (venv)

Requirements from requirement.txt

Frontend Requirements

Node.js 18+

npm or yarn

🛠️ Backend Setup
1️⃣ Navigate to Backend Folder
cd backend

2️⃣ Create Virtual Environment
python -m venv .venv

Activate venv

Windows

.venv\Scripts\activate


Linux/Mac

source .venv/bin/activate

3️⃣ Install Dependencies

Make sure you have a requirements.txt file in backend.
Then run:

pip install -r requirements.txt

4️⃣ Install FFmpeg

This is required for audio extraction and merging video chunks.

Windows:
Download from: https://www.gyan.dev/ffmpeg/builds/

Add FFmpeg bin/ path to your System PATH.

Linux

sudo apt install ffmpeg


Mac

brew install ffmpeg

5️⃣ Create .env File

Inside /backend/.env:

GEMINI_API_KEY=YOUR_KEY_HERE


(Or leave blank to use template fallback question generator.)

6️⃣ Run Backend Server
cd backend
uvicorn main:app --reload


You should see:

http://127.0.0.1:8000
Docs: http://127.0.0.1:8000/docs

🖥️ Frontend Setup
1️⃣ Navigate to Frontend Folder
cd frontend

2️⃣ Install Dependencies
npm install

3️⃣ Run Frontend
npm start


Frontend runs on:

http://localhost:3000

🔄 End-to-End Flow
1. Upload Resume

Backend parses skills, projects, experience

Question generator (Gemini API or fallback)

2. Start Interview Session

Frontend:

Captures webcam video in chunks

Records full-session audio

Sends Q&A transcript to backend

3. Backend Pipeline

Text feature extraction

Audio feature extraction

Video feature extraction

Video feature aggregation

Multimodal feature fusion

Weighted scoring

SHAP explainability

Feedback report generation (JSON + visualizations)

4. Frontend Dashboard

Radar charts

Strengths & weaknesses

Improvement tips

Score breakdown

📦 Important Backend Directories
✔ recordings/sessions/

Stores:

Video chunks

Full session audio

Transcript CSV/JSON

✔ processed_features/

Stores:

audio_features.csv

video_features_raw.csv

video_features_aggregated.csv

text_features.csv

final_multimodal_features.csv

✔ feedback_results/

Stores:

weighted_scores.json

feedback_report.json

visualizations (radar charts, bar charts)

model joblib files

🧮 Key ML Components
✔ Feature Extractors

feature_extractors/

audio_extractor.py

video_extractor.py

text_extractor.py

✔ Feature Engineering

feature_engineering/

video_aggregator.py

multimodal_fusion.py

✔ ML Models

feedback_engine/

predictor.py

weighted_scorer.py

shap_analyzer.py

feedback_generator.py

📊 Models Included

Located in:

backend/feedback_results/models


Includes:

human_aligned_model.joblib

scaler.joblib

These models output:

Final score

10 dimension scores

SHAP explanations

🛑 Common Issues
❌ Video chunks not merging

→ Install FFmpeg
→ Add to PATH
→ Enable correct permissions

❌ No radar_chart.png

→ Ensure static route is mapped
→ Save visualizations in:
feedback_results/<session_id>/visualizations

❌ CORS errors

→ Add frontend URL to FastAPI CORS middleware

❌ Gemini API failure

→ Falls back to template question generator automatically

🧪 Testing the Backend
Test resume upload
curl -X POST -F "file=@resume.pdf" http://127.0.0.1:8000/upload_resume

Test session creation
curl -X POST http://127.0.0.1:8000/api/session/create

Test feedback retrieval
curl http://127.0.0.1:8000/api/session/<session_id>/feedback

📜 Scripts Summary
Module	Purpose
main.py	FastAPI routing, session orchestration
process_session.py	Full pipeline execution
audio_extractor.py	Extract 50+ audio features
video_extractor.py	468 landmark processing, head pose, facial metrics
text_extractor.py	NLP features, sentiment, filler ratio
video_aggregator.py	Frame → session aggregation
multimodal_fusion.py	10-dimension fusion + final score
weighted_scorer.py	Weighted scoring model
shap_analyzer.py	SHAP explanations
feedback_generator.py	Natural language feedback
chart_generator.py	Radar charts, bar charts
report_builder.py	(Optional) PDF/HTML report builder
📚 Technologies Used
Backend

FastAPI

Python 3.10

ffmpeg-python

MediaPipe

Librosa

OpenCV

Scikit-learn

Pandas + NumPy

SHAP

Joblib

Frontend

React.js

Custom hooks (useCamera, useAudioRecorder, etc.)

Axios

CSS Modules

🏁 Running Both Servers
Start backend:
cd backend
.venv\Scripts\activate
uvicorn main:app --reload

Start frontend:
cd frontend
npm start


Frontend automatically connects to:

VITE_BACKEND_URL=http://127.0.0.1:8000
