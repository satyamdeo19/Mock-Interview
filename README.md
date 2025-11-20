AI Mock Interview System

A complete multimodal AI interview evaluation system that analyzes text, audio, and video to score candidates across 10 behavioral dimensions.
The system extracts 120+ multimodal features, performs feature fusion, applies ML scoring, and generates SHAP-based explainable feedback.

Project Overview

The system includes:

Backend: FastAPI + Python

Frontend: React.js

ML Models: RandomForest + SHAP + Feature Engineering

Modalities: Text, Audio, Video

Output: Scores, SHAP insights, natural-language feedback, charts

Project Structure
____venv
├── backend
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
    ├── public
    ├── package.json
    ├── src
    │   ├── App.jsx
    │   ├── components
    │   ├── hooks
    │   ├── services
    │   ├── styles
    │   └── utils

System Requirements
Backend

Python 3.10+

FFmpeg installed

pip 23+

Virtual environment

requirements.txt

Frontend

Node.js 18+

npm or yarn

Backend Setup
1. Navigate to backend
cd backend

2. Create virtual environment

Windows:

python -m venv .venv
.venv\Scripts\activate


Linux/Mac:

python -m venv .venv
source .venv/bin/activate

3. Install dependencies
pip install -r requirements.txt

4. Install FFmpeg

Windows: Download from https://www.gyan.dev/ffmpeg/builds/
 and add bin/ to PATH
Linux:

sudo apt install ffmpeg


Mac:

brew install ffmpeg

5. Create .env
GEMINI_API_KEY=YOUR_KEY_HERE

6. Run backend
uvicorn main:app --reload


Backend URLs:

http://127.0.0.1:8000
http://127.0.0.1:8000/docs

Frontend Setup
1. Navigate
cd frontend

2. Install packages
npm install

3. Run frontend
npm start


Frontend URL:

http://localhost:3000

End-to-End Workflow

Resume Upload
Extract skills, roles, projects → Generate interview questions.

Interview Session
Webcam video chunks, full audio, transcript sent to backend.

Backend Processing

Text feature extraction

Audio feature extraction

Video landmark extraction

Session-level aggregation

Multimodal fusion

Weighted scoring

SHAP explanation

Feedback generation

Frontend Dashboard
Radar chart, scores, strengths, weaknesses, suggestions.

Important Backend Directories
recordings/sessions/

Stores:

Video chunks

Full-session audio

Transcripts

processed_features/

audio_features.csv

video_features_raw.csv

video_features_aggregated.csv

text_features.csv

final_multimodal_features.csv

feedback_results/

weighted_scores.json

feedback_report.json

visualizations

model joblib files

ML Components
Feature Extractors

audio_extractor.py

video_extractor.py

text_extractor.py

Feature Engineering

video_aggregator.py

multimodal_fusion.py

Models

weighted_scorer.py

predictor.py

shap_analyzer.py

Feedback

feedback_generator.py

chart_generator.py

Models located in:

backend/feedback_results/models

Common Issues
Video not merging

Install FFmpeg and ensure PATH is set.

Visualizations missing

Check:

feedback_results/<session_id>/visualizations

CORS errors

Add frontend URL to FastAPI CORS settings.

Gemini API not working

System falls back to template generator automatically.

Testing Backend (curl)
Upload resume
curl -X POST -F "file=@resume.pdf" http://127.0.0.1:8000/upload_resume

Create session
curl -X POST http://127.0.0.1:8000/api/session/create

Get feedback
curl http://127.0.0.1:8000/api/session/<session_id>/feedback

Technologies Used
Backend

FastAPI

MediaPipe

ffmpeg-python

Librosa

OpenCV

Scikit-learn

Pandas, NumPy

SHAP

Frontend

React.js

Axios

Custom hooks

CSS Modules

Running Both Servers

Backend:

cd backend
.venv\Scripts\activate
uvicorn main:app --reload


Frontend:

cd frontend
npm start


Environment:

VITE_BACKEND_URL=http://127.0.0.1:8000
