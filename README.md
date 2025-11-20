📘 AI Mock Interview System — README
🚀 Multimodal AI Interview Evaluation (Text + Audio + Video)

This project is a complete AI-powered mock interview platform that analyzes video, audio, and text responses to evaluate a candidate across 10 behavioral dimensions such as:

Confidence

Fluency

Engagement

Communication

Professionalism

Cognitive Complexity

Emotional Stability

Body-Language Cues

Voice Features

Overall Delivery

The system extracts 120+ multimodal features, performs feature engineering, fuses features using a weighted scoring model, and generates explainable feedback using SHAP.

🧩 Project Overview
System Includes:

Backend — FastAPI + Python

Frontend — React.js

Machine Learning — RandomForest + Feature Engineering + SHAP

📂 Project Structure
____venv
├── backend
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
│   │   ├── __init__.py
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
Backend

Python 3.10+

FFmpeg (required)

pip 23+

Virtual environment

requirements.txt dependencies

Frontend

Node.js 18+

npm or yarn

🛠️ Backend Setup
1️⃣ Navigate to Backend
cd backend

2️⃣ Create & Activate Virtual Environment

Windows:

python -m venv .venv
.venv\Scripts\activate


Linux/Mac:

python -m venv .venv
source .venv/bin/activate

3️⃣ Install Dependencies
pip install -r requirements.txt

4️⃣ Install FFmpeg

Windows:
Download: https://www.gyan.dev/ffmpeg/builds/

Add bin/ to PATH.

Linux:

sudo apt install ffmpeg


Mac:

brew install ffmpeg

5️⃣ Create .env
GEMINI_API_KEY=YOUR_KEY_HERE

6️⃣ Run Backend Server
uvicorn main:app --reload


Backend URL:

http://127.0.0.1:8000
http://127.0.0.1:8000/docs

🖥️ Frontend Setup
1️⃣ Navigate
cd frontend

2️⃣ Install packages
npm install

3️⃣ Start frontend
npm start


Frontend runs at:

http://localhost:3000

🔄 End-to-End Flow
1. Resume Upload

Extracts skills, projects, experience

Generates personalized interview questions

2. Interview Session

Webcam video chunks

Full session audio

Text transcript

3. Backend Processing

Audio feature extraction

Video landmark feature extraction

Text NLP analysis

Multimodal aggregation

Weighted scoring

SHAP explanations

Feedback report generation

4. Frontend Dashboard

Radar charts

Scores by dimension

Strengths & weaknesses

Improvement suggestions

📦 Important Backend Directories
recordings/sessions/

Stores:

Video chunks

Full audio

Transcripts

processed_features/

audio_features.csv

video_features_raw.csv

video_features_aggregated.csv

text_features.csv

final_multimodal_features.csv

feedback_results/

Scores

Feedback report

Visualizations

SHAP data

ML models

🧮 Machine Learning Components
Feature Extractors

audio_extractor.py

video_extractor.py

text_extractor.py

Feature Engineering

video_aggregator.py

multimodal_fusion.py

ML Models

predictor.py

weighted_scorer.py

shap_analyzer.py

feedback_generator.py

Included Models

Located in:

backend/feedback_results/models


Models:

human_aligned_model.joblib

scaler.joblib

Outputs:

Final Score

10 Dimension Scores

SHAP Explainability

🛑 Common Issues & Fixes
❌ Video chunk merge failure

✔ Install FFmpeg
✔ Add to PATH

❌ Missing radar chart

✔ Ensure visualization folder exists
✔ Configure static file route

❌ CORS issues

✔ Add frontend URL to FastAPI CORS

❌ Gemini API not working

✔ Falls back to template-based generator

🧪 Testing Backend with curl
Upload Resume
curl -X POST -F "file=@resume.pdf" http://127.0.0.1:8000/upload_resume

Create Session
curl -X POST http://127.0.0.1:8000/api/session/create

Get Feedback
curl http://127.0.0.1:8000/api/session/<session_id>/feedback

📜 Scripts Summary
Module	Purpose
main.py	FastAPI routes & orchestration
process_session.py	Complete pipeline driver
audio_extractor.py	Extracts 50+ audio features
video_extractor.py	Mediapipe landmarks + head pose
text_extractor.py	NLP features + filler detection
video_aggregator.py	Frame → Interview-level metrics
multimodal_fusion.py	Feature merging
weighted_scorer.py	Scoring logic
shap_analyzer.py	Explainability
feedback_generator.py	Natural language feedback
chart_generator.py	Radar & bar charts
report_builder.py	(Optional) Build PDF/HTML reports
📚 Technologies Used
Backend

FastAPI

ffmpeg-python

MediaPipe

Librosa

OpenCV

Scikit-learn

Pandas / NumPy

SHAP

Frontend

React.js

Custom hooks (camera, audio recorder)

Axios

CSS Modules

🏁 Running Both Servers
Backend
cd backend
.venv\Scripts\activate   # or source .venv/bin/activate
uvicorn main:app --reload

Frontend
cd frontend
npm start


Environment variable:

VITE_BACKEND_URL=http://127.0.0.1:8000
