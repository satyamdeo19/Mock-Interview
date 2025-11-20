# 🚀 AI Mock Interview Platform

A multimodal AI-powered system that evaluates candidates through video, audio, text, and resume analysis to generate real-time feedback, behavioral insights, and detailed scoring reports—simulating a real HR interview panel.

![License](https://img.shields.io/badge/license-MIT-blue.svg)
![Python](https://img.shields.io/badge/python-3.10+-blue.svg)
![Node](https://img.shields.io/badge/node-18+-green.svg)

---

## 🔥 Key Highlights

### 🎥 Multimodal Input
- Real-time webcam video capture
- Live audio recording + transcription
- NLP-based answer quality evaluation
- AI-powered resume parsing

### 🧠 AI Interviewer
- Dynamically adjusts difficulty
- Domain-specific technical questions
- Behavior & tone-aware follow-ups

### 📊 Feedback Engine
- Confidence score analysis
- Communication clarity metrics
- Technical correctness evaluation
- Bias & filler-word detection
- Attitude, fluency, and delivery assessment

### 📈 Analytics & Reports
- SHAP explainability
- Skill-wise performance charts
- Strength–weakness summary
- PDF/JSON report export

---

## 🗂️ Project Structure

```
.
├── __venv/
├── backend/
│   ├── .env
│   ├── benchmark_data.json
│   ├── config.py
│   ├── feature_config.json
│   ├── feature_weights.json
│   ├── main.py
│   ├── process_session.py
│   ├── question_generator.py
│   ├── resume_parser.py
│   ├── feature_engineering/
│   │   ├── __init__.py
│   │   ├── multimodal_fusion.py
│   │   └── video_aggregator.py
│   ├── feature_extractors/
│   │   ├── audio_extractor.py
│   │   ├── text_extractor.py
│   │   └── video_extractor.py
│   ├── feedback_engine/
│   │   ├── feedback_generator.py
│   │   ├── predictor.py
│   │   ├── shap_analyzer.py
│   │   └── weighted_scorer.py
│   ├── models/
│   │   ├── human_aligned_model.joblib
│   │   └── scaler.joblib
│   ├── processed_features/
│   ├── recordings/
│   │   └── sessions/
│   │       └── {session_id}/
│   │           ├── audio/
│   │           │   └── full_session.webm
│   │           ├── video/
│   │           │   ├── chunk0.webm
│   │           │   └── chunkN.webm
│   │           └── extracted_frames/
│   ├── temp/
│   └── utils/
│
└── frontend/
    ├── public/
    ├── package.json
    └── src/
        ├── App.jsx
        ├── components/
        ├── hooks/
        ├── services/
        ├── styles/
        └── utils/
```

---

## ⚙️ System Requirements

### Backend
- Python 3.10+
- FFmpeg
- PyTorch / Transformers
- Joblib
- OpenCV

### Frontend
- Node.js 18+
- Browser with WebRTC support

### Hardware
- GPU recommended (for faster processing)
- Webcam + Microphone

---

## 🔑 Environment Variables

### Backend `.env`

Create `backend/.env` with the following:

```env
OPENAI_API_KEY=your_openai_api_key_here
MODEL_PATH=models/human_aligned_model.joblib
SCALER_PATH=models/scaler.joblib
TEMP_DIR=temp
RECORDINGS_DIR=recordings/sessions
```

### Frontend `.env` (Optional)

Create `frontend/.env` if needed:

```env
VITE_API_URL=http://localhost:8000
```

---

## 🛠️ Installation & Setup

### Backend Setup

1. **Create Virtual Environment**

```bash
python -m venv __venv

# Activate virtual environment
# Linux/macOS:
source __venv/bin/activate

# Windows:
__venv\Scripts\activate
```

2. **Install Dependencies**

```bash
cd backend
pip install -r requirements.txt
```

3. **Start Backend Server**

```bash
uvicorn main:app --reload
```

Backend will run at: **http://localhost:8000**

---

### Frontend Setup

1. **Install Dependencies**

```bash
cd frontend
npm install
```

2. **Start Development Server**

```bash
npm run dev
```

Frontend will run at: **http://localhost:5173**

---
### Installation Issues: Use this venv folder
Link:
## 🚦 API Endpoints

### 🔹 Session Handling

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/start_session` | Create new interview session |
| POST | `/upload_audio` | Upload audio chunk |
| POST | `/upload_video` | Upload video chunk |
| POST | `/process_session` | Run full processing pipeline |
| GET | `/get_feedback/{session_id}` | Retrieve final results |

### 🔹 Resume Parsing

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/parse_resume` | Parse and extract resume data |

---

## 🧠 Processing Pipeline

### 1️⃣ **Capture**
- Video chunks → WebRTC
- Audio stream → WebM
- Transcript → Whisper
- Resume PDF → Parser

### 2️⃣ **Extract Features**
- **Video**: EAR (eye aspect ratio), MAR (mouth aspect ratio), brow raise, smile intensity
- **Audio**: MFCC, pitch, speed, energy
- **Text**: Coherence, sentiment, verbosity
- **Resume**: Skills, education, experience

### 3️⃣ **Multimodal Fusion**
`multimodal_fusion.py` merges all feature vectors into a unified representation.

### 4️⃣ **Prediction**
ML model outputs:
- Confidence score
- Communication quality
- Delivery metrics
- Overall performance score

### 5️⃣ **Feedback Generation**
`feedback_generator.py` produces:
- Strengths
- Weaknesses
- Actionable improvements
- SHAP explanatory charts
- Timeline graphs

---

## 📊 Generated Outputs

- ✅ JSON logs
- ✅ Full feature dump
- ✅ SHAP value visualizations
- ✅ Performance charts
- ✅ Summary text report
- ✅ Final interview score
- ✅ Step-by-step improvement suggestions

---

## 🖼️ Architecture Diagram

```
┌─────────────────────────────────────┐
│       Frontend (React)              │
│                                     │
│  ├── Webcam → Video Chunks          │
│  ├── Mic → Audio Chunks             │
│  └── Text Answers                   │
└──────────────┬──────────────────────┘
               │
┌──────────────▼──────────────────────┐
│       Backend (FastAPI)             │
│                                     │
│  ├── Video Extractor                │
│  ├── Audio Extractor                │
│  ├── Text Extractor                 │
│  ├── Resume Parser                  │
│  │                                  │
│  └── Multimodal Fusion              │
│      │                              │
│      └── ML Predictor               │
│          │                          │
│          └── Feedback Engine        │
│              │                      │
│              └── Report Generator   │
└─────────────────────────────────────┘
```

---

## 🧩 Tech Stack

### Frontend
- React
- WebRTC
- TailwindCSS
- Axios

### Backend
- FastAPI
- Whisper (Audio transcription)
- OpenCV (Video processing)
- NumPy, Pandas (Data processing)
- SHAP (Explainability)
- XGBoost / Custom ML models

---

## 🤝 Contributing

We welcome contributions! Here's how you can help:

1. **Fork** the repository
2. Create a **feature branch** (`git checkout -b feature/amazing-feature`)
3. **Commit** your changes (`git commit -m 'Add some amazing feature'`)
4. **Push** to the branch (`git push origin feature/amazing-feature`)
5. Submit a **Pull Request**

---

## 📜 License

This project is licensed under the **MIT License** — free to use and modify.

---

## 📧 Contact & Support

For questions, issues, or feature requests, please open an issue on GitHub.

---

## 🌟 Star Us!

If you find this project helpful, please consider giving it a ⭐ on GitHub!

---

**Built with ❤️ by the AI Interview Platform Team**
