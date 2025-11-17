from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import pdfplumber
import os
import uvicorn
import logging
from resume_parser import extract_resume_info
from question_generator import generate_personalized_questions
from typing import List, Dict, Any
from config import MAX_FILE_SIZE, ALLOWED_EXTENSIONS, SESSION_EXPIRY_HOURS
import pandas as pd
from fastapi import FastAPI, UploadFile, File, HTTPException, BackgroundTasks
from typing import List, Optional
import json
import csv
import shutil
from datetime import datetime, timedelta
import uuid
from fastapi import Body

from fastapi import BackgroundTasks
from fastapi.responses import FileResponse
import subprocess
import sys
import threading

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Store uploaded resume data in JSON file for persistence
import json
import time

RESUME_DATA_FILE = "temp/resume_data.json"

def load_resume_data():
    """Load resume data from file"""
    try:
        if os.path.exists(RESUME_DATA_FILE):
            with open(RESUME_DATA_FILE, 'r') as f:
                return json.load(f)
    except Exception as e:
        logger.warning(f"Could not load resume data: {e}")
    return {}

def save_resume_data(data):
    """Save resume data to file"""
    try:
        os.makedirs("temp", exist_ok=True)
        with open(RESUME_DATA_FILE, 'w') as f:
            json.dump(data, f, indent=2)
    except Exception as e:
        logger.error(f"Could not save resume data: {e}")

# Load existing data on startup
uploaded_resumes = load_resume_data()

class QuestionRequest(BaseModel):
    mode: str = "technical"

@app.post("/upload_resume")
async def upload_resume(file: UploadFile = File(...)):
    logger.info(f"Received file upload request: {file.filename}")
    logger.info(f"File content type: {file.content_type}")
    
    try:
        if not file or not file.filename:
            logger.error("No file provided")
            raise HTTPException(status_code=400, detail="No file provided")
        
        file_ext = os.path.splitext(file.filename)[1].lower()
        if file_ext not in ALLOWED_EXTENSIONS:
            logger.error(f"Invalid file type: {file.filename}")
            raise HTTPException(status_code=400, detail=f"Only {', '.join(ALLOWED_EXTENSIONS)} files are supported.")
        
        content = await file.read()
        if len(content) > MAX_FILE_SIZE:
            logger.error(f"File too large: {len(content)} bytes")
            raise HTTPException(status_code=400, detail=f"File size exceeds maximum limit of {MAX_FILE_SIZE // (1024*1024)}MB.")
        
        os.makedirs("temp", exist_ok=True)
        logger.info("Temp directory created/verified")
        
        file_path = f"temp/temp_{file.filename}"
        logger.info(f"Saving file to: {file_path}")
        
        try:
            with open(file_path, "wb") as f:
                f.write(content)
            logger.info(f"File saved successfully. Size: {len(content)} bytes")
        except Exception as save_error:
            logger.error(f"Error saving file: {str(save_error)}")
            raise HTTPException(status_code=500, detail=f"Error saving file: {str(save_error)}")
        
        text = ""
        try:
            logger.info("Starting PDF text extraction")
            with pdfplumber.open(file_path) as pdf:
                logger.info(f"PDF opened. Number of pages: {len(pdf.pages)}")
                for i, page in enumerate(pdf.pages):
                    page_text = page.extract_text()
                    if page_text:
                        text += page_text + "\n"
                        logger.info(f"Extracted text from page {i+1}: {len(page_text)} characters")
            logger.info(f"Total extracted text length: {len(text)} characters")
        except Exception as pdf_error:
            logger.error(f"Error reading PDF: {str(pdf_error)}")
            if os.path.exists(file_path):
                os.remove(file_path)
            raise HTTPException(status_code=500, detail=f"Error reading PDF: {str(pdf_error)}")
        
        try:
            if os.path.exists(file_path):
                os.remove(file_path)
                logger.info("Temp file cleaned up")
        except Exception as cleanup_error:
            logger.warning(f"Could not clean up temp file: {str(cleanup_error)}")
        
        if not text.strip():
            logger.error("No text extracted from PDF")
            raise HTTPException(status_code=400, detail="No text could be extracted from the PDF. The file might be corrupted or contain only images.")
        
        logger.info("Starting resume information extraction")
        try:
            data = extract_resume_info(text)
            logger.info(f"Resume info extracted successfully. Found: {len(data.get('skills', []))} skills, {len(data.get('experience', []))} experiences")
        except Exception as extract_error:
            logger.error(f"Error extracting resume info: {str(extract_error)}")
            raise HTTPException(status_code=500, detail=f"Error processing resume: {str(extract_error)}")
        
        session_id = f"resume_{int(time.time())}"
        uploaded_resumes[session_id] = {
            **data,
            "uploaded_at": time.time(),
            "expires_at": time.time() + (SESSION_EXPIRY_HOURS * 60 * 60)
        }
        save_resume_data(uploaded_resumes)
        logger.info(f"Resume data stored with session ID: {session_id}")
        
        result = {
            "status": "success",
            "message": "Resume processed successfully",
            "data": data,
            "session_id": session_id
        }
        
        logger.info("Request processed successfully")
        return JSONResponse(content=result, status_code=200)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        if 'file_path' in locals() and os.path.exists(file_path):
            try:
                os.remove(file_path)
                logger.info("Temp file cleaned up after error")
            except:
                pass
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@app.post("/generate_questions")
async def generate_questions(request: QuestionRequest):
    """Generate personalized questions based on the latest uploaded resume"""
    logger.info(f"Generating {request.mode} questions")
    
    try:
        if not uploaded_resumes:
            raise HTTPException(status_code=400, detail="No resume uploaded. Please upload a resume first.")
        
        current_time = time.time()
        valid_sessions = {k: v for k, v in uploaded_resumes.items() 
                         if v.get('expires_at', 0) > current_time}
        
        if not valid_sessions:
            raise HTTPException(status_code=400, detail="All uploaded resumes have expired. Please upload a new resume.")
        
        latest_session = max(valid_sessions.keys())
        resume_data = valid_sessions[latest_session]
        
        uploaded_resumes.clear()
        uploaded_resumes.update(valid_sessions)
        save_resume_data(uploaded_resumes)
        
        logger.info(f"Using resume data from session: {latest_session}")
        
        try:
            questions = generate_personalized_questions(resume_data, request.mode)
            logger.info(f"Generated {len(questions)} questions")
            
            if not questions:
                raise Exception("No questions generated")
                
        except Exception as gen_error:
            logger.error(f"Error generating questions: {str(gen_error)}")
            from question_generator import QuestionGenerator
            generator = QuestionGenerator()
            questions = generator._get_fallback_questions(request.mode)
            logger.info("Using fallback questions")
        
        result = {
            "status": "success",
            "message": f"Generated {len(questions)} {request.mode} questions",
            "questions": questions,
            "mode": request.mode,
            "session_id": latest_session
        }
        
        return JSONResponse(content=result, status_code=200)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected error in question generation: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error generating questions: {str(e)}")

@app.get("/get_resume_info")
async def get_resume_info():
    """Get information about uploaded resumes"""
    if not uploaded_resumes:
        return {"message": "No resumes uploaded", "count": 0}
    
    latest_session = max(uploaded_resumes.keys())
    resume_data = uploaded_resumes[latest_session]
    
    return {
        "message": "Resume data available",
        "count": len(uploaded_resumes),
        "latest_session": latest_session,
        "summary": {
            "name": resume_data.get("name", "Not Found"),
            "skills_count": len(resume_data.get("skills", [])),
            "experience_count": len(resume_data.get("experience", [])),
            "education_count": len(resume_data.get("education", [])),
            "projects_count": len(resume_data.get("projects", []))
        }
    }

@app.delete("/clear_resume_data")
async def clear_resume_data():
    """Clear all stored resume data"""
    global uploaded_resumes
    count = len(uploaded_resumes)
    uploaded_resumes.clear()
    save_resume_data(uploaded_resumes)
    
    return {
        "message": f"Cleared {count} resume sessions",
        "status": "success"
    }

# ============================================
# CRITICAL FIX: Use backend/recordings consistently
# ============================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_RECORDINGS_DIR = os.path.normpath(os.path.join(BASE_DIR, 'recordings', 'sessions'))
os.makedirs(BASE_RECORDINGS_DIR, exist_ok=True)

logger.info(f"✅ Recordings directory: {BASE_RECORDINGS_DIR}")

# Models
class SessionCreate(BaseModel):
    mode: str
    resume_uploaded: bool

class QuestionAnswer(BaseModel):
    question_number: int
    question: str
    answer: str
    timestamp: str
    duration_seconds: Optional[float] = None
    ideal_answer: Optional[str] = None

class SessionData(BaseModel):
    session_id: str
    qa_pairs: List[QuestionAnswer]
    mode: Optional[str] = None
    total_questions: Optional[int] = None

# ============================================
# ROOT ENDPOINT
# ============================================

@app.get("/")
async def root():
    return {
        "status": "running",
        "message": "AI Mock Interview Backend is running",
        "recordings_path": os.path.abspath(BASE_RECORDINGS_DIR),
        "sessions_count": len(os.listdir(BASE_RECORDINGS_DIR)) if os.path.exists(BASE_RECORDINGS_DIR) else 0
    }

# ============================================
# CREATE SESSION
# ============================================

@app.post("/api/session/create")
async def create_session(session: SessionCreate):
    """Create a new interview session"""
    session_id = f"session_{uuid.uuid4().hex[:12]}_{int(datetime.now().timestamp())}"
    
    logger.info(f"🆕 Creating session: {session_id}")
    logger.info(f"📋 Mode: {session.mode}, Resume: {session.resume_uploaded}")
    
    session_path = os.path.join(BASE_RECORDINGS_DIR, session_id)
    audio_path = os.path.join(session_path, "audio")
    video_path = os.path.join(session_path, "video")
    video_chunks_path = os.path.join(video_path, "chunks")
    transcripts_path = os.path.join(session_path, "transcripts")
    
    os.makedirs(audio_path, exist_ok=True)
    os.makedirs(video_chunks_path, exist_ok=True)
    os.makedirs(transcripts_path, exist_ok=True)
    
    logger.info(f"📁 Created directories:")
    logger.info(f"   - Audio: {os.path.abspath(audio_path)}")
    logger.info(f"   - Video: {os.path.abspath(video_chunks_path)}")
    logger.info(f"   - Transcripts: {os.path.abspath(transcripts_path)}")
    
    metadata = {
        "session_id": session_id,
        "mode": session.mode,
        "resume_uploaded": session.resume_uploaded,
        "created_at": datetime.now().isoformat(),
        "status": "active",
        "total_questions": 0,
        "completed": False
    }
    
    metadata_path = os.path.join(session_path, "metadata.json")
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)
    
    logger.info(f"✅ Session created successfully: {session_id}")
    
    return {
        "session_id": session_id,
        "message": "Session created successfully",
        "paths": {
            "base": session_path,
            "audio": audio_path,
            "video": video_path,
            "transcripts": transcripts_path
        }
    }

@app.post("/api/session/{session_id}/upload_audio")
async def upload_audio(session_id: str, file: UploadFile = File(...)):
    """Upload and save full-session audio"""
    session_path = os.path.join(BASE_RECORDINGS_DIR, session_id)
    if not os.path.exists(session_path):
        raise HTTPException(status_code=404, detail=f"Session not found: {session_id}")

    audio_dir = os.path.join(session_path, "audio")
    os.makedirs(audio_dir, exist_ok=True)

    filename = file.filename or "full_session.webm"
    audio_path = os.path.join(audio_dir, filename)
    logger.info(f"🎧 Saving audio file: {audio_path}")

    try:
        content = await file.read()
        if not content:
            raise HTTPException(status_code=400, detail="Empty audio file received.")

        with open(audio_path, "wb") as f:
            f.write(content)

        logger.info(f"✅ Full-session audio saved successfully ({len(content)} bytes)")
        return {"message": "Audio uploaded successfully", "path": audio_path, "size": len(content)}

    except Exception as e:
        logger.error(f"❌ Audio upload failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Audio upload error: {str(e)}")

# ============================================
# UPLOAD VIDEO CHUNK
# ============================================

@app.post("/api/session/{session_id}/upload_video_chunk")
async def upload_video_chunk(session_id: str, chunk_number: int, file: UploadFile = File(...)):
    """Upload video chunk"""
    logger.info(f"📤 VIDEO CHUNK UPLOAD:")
    logger.info(f"   Session ID: {session_id}")
    logger.info(f"   Chunk #: {chunk_number}")
    logger.info(f"   Filename: {file.filename}")
    
    session_path = os.path.join(BASE_RECORDINGS_DIR, session_id)
    
    if not os.path.exists(session_path):
        logger.error(f"❌ Session not found: {session_path}")
        raise HTTPException(status_code=404, detail=f"Session not found: {session_id}")
    
    chunk_path = os.path.join(session_path, "video", "chunks", f"chunk_{chunk_number}.webm")
    logger.info(f"💾 Saving video chunk to: {chunk_path}")
    
    try:
        content = await file.read()
        file_size = len(content)
        
        logger.info(f"📦 Chunk size: {file_size} bytes ({file_size / 1024 / 1024:.2f} MB)")
        
        if file_size == 0:
            logger.warning("⚠️ Empty video chunk received")
        
        with open(chunk_path, "wb") as f:
            f.write(content)
        
        if os.path.exists(chunk_path):
            logger.info(f"✅ Video chunk saved successfully")
        
        return {
            "message": "Video chunk uploaded successfully",
            "chunk_number": chunk_number,
            "size_bytes": file_size
        }
    
    except Exception as e:
        logger.error(f"❌ Error saving video chunk: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")

# ============================================
# SAVE TRANSCRIPT
# ============================================

@app.post("/api/session/{session_id}/save_transcript")
async def save_transcript(session_id: str, data: SessionData = Body(...)):
    """Save Q&A transcript as CSV"""
    logger.info(f"💾 SAVING TRANSCRIPT:")
    logger.info(f"   Session ID: {session_id}")
    logger.info(f"   Q&A pairs: {len(data.qa_pairs)}")
    
    session_path = os.path.join(BASE_RECORDINGS_DIR, session_id)
    
    if not os.path.exists(session_path):
        logger.error(f"❌ Session not found")
        raise HTTPException(status_code=404, detail="Session not found")
    
    csv_data = []
    for qa in data.qa_pairs:
        csv_data.append({
            "Question_Number": qa.question_number,
            "Question": qa.question,
            "Answer": qa.answer,
            "Timestamp": qa.timestamp,
            "Duration_Seconds": qa.duration_seconds or 0
        })
        logger.info(f"   Q{qa.question_number}: {qa.question[:50]}...")
    
    df = pd.DataFrame(csv_data)
    csv_path = os.path.join(session_path, "transcripts", "qa_data.csv")
    df.to_csv(csv_path, index=False, encoding='utf-8')
    logger.info(f"✅ CSV saved: {csv_path}")
    
    json_path = os.path.join(session_path, "transcripts", "qa_data.json")
    with open(json_path, "w") as f:
        json.dump([qa.model_dump() for qa in data.qa_pairs], f, indent=2)
    logger.info(f"✅ JSON saved: {json_path}")
    
    metadata_path = os.path.join(session_path, "metadata.json")
    with open(metadata_path, "r") as f:
        metadata = json.load(f)
    
    metadata["total_questions"] = len(data.qa_pairs)
    metadata["completed"] = True
    metadata["completed_at"] = datetime.now().isoformat()
    
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)
    
    logger.info(f"✅ Transcript saved successfully")
    
    return {
        "message": "Transcript saved successfully",
        "csv_path": csv_path,
        "json_path": json_path,
        "total_questions": len(data.qa_pairs)
    }

# Global dict to track processing status
processing_status = {}

# ============================================
# COMPLETE SESSION
# ============================================
@app.post("/api/session/{session_id}/complete")
async def complete_session(session_id: str, background_tasks: BackgroundTasks):
    """Mark session as complete and auto-start feedback"""
    logger.info(f"🏁 COMPLETING SESSION: {session_id}")
    
    session_path = os.path.join(BASE_RECORDINGS_DIR, session_id)
    
    if not os.path.exists(session_path):
        raise HTTPException(status_code=404, detail="Session not found")
    
    # Update metadata
    metadata_path = os.path.join(session_path, "metadata.json")
    with open(metadata_path, "r") as f:
        metadata = json.load(f)
    
    metadata["status"] = "completed"
    metadata["completed_at"] = datetime.now().isoformat()
    
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)
    
    logger.info(f"✅ Session marked as completed")
    
    # Auto-start feedback generation with skip_video=True
    logger.info(f"📬 Enqueuing feedback generation for {session_id}")
    background_tasks.add_task(
        run_feedback_pipeline,
        session_id,
        "Candidate",
        "Position",
        None,
        "mid",
        False  
    )
    
    # Mark as processing
    processing_status[session_id] = 'processing'
    
    return {
        "message": "Session completed successfully. Feedback generation started.",
        "session_id": session_id,
        "ready_for_processing": True,
        "feedback_status": "processing"
    }

# ============================================
# GET SESSION INFO
# ============================================

@app.get("/api/session/{session_id}/info")
async def get_session_info(session_id: str):
    """Get detailed session information"""
    session_path = os.path.join(BASE_RECORDINGS_DIR, session_id)

    if not os.path.exists(session_path):
        raise HTTPException(status_code=404, detail="Session not found")

    metadata_path = os.path.join(session_path, "metadata.json")
    with open(metadata_path, "r") as f:
        metadata = json.load(f)

    audio_dir = os.path.join(session_path, "audio")
    video_chunks_dir = os.path.join(session_path, "video", "chunks")
    transcripts_dir = os.path.join(session_path, "transcripts")

    audio_files = os.listdir(audio_dir) if os.path.exists(audio_dir) else []
    video_chunks = os.listdir(video_chunks_dir) if os.path.exists(video_chunks_dir) else []
    transcript_files = os.listdir(transcripts_dir) if os.path.exists(transcripts_dir) else []

    def get_dir_size(path):
        total = 0
        if os.path.exists(path):
            for dirpath, dirnames, filenames in os.walk(path):
                for f in filenames:
                    total += os.path.getsize(os.path.join(dirpath, f))
        return total

    audio_size = get_dir_size(audio_dir)
    video_size = get_dir_size(os.path.join(session_path, "video"))
    transcript_size = get_dir_size(transcripts_dir)

    return {
        "session_id": session_id,
        "metadata": metadata,
        "files": {
            "audio_count": len(audio_files),
            "audio_files": audio_files,
            "video_chunks_count": len(video_chunks),
            "video_chunks": video_chunks,
            "transcript_files": transcript_files,
        },
        "storage": {
            "audio_mb": round(audio_size / (1024 * 1024), 2),
            "video_mb": round(video_size / (1024 * 1024), 2),
            "transcripts_kb": round(transcript_size / 1024, 2),
            "total_mb": round((audio_size + video_size + transcript_size) / (1024 * 1024), 2),
        },
        "paths": {
            "audio": os.path.abspath(audio_dir),
            "video_chunks": os.path.abspath(video_chunks_dir),
            "transcripts": os.path.abspath(transcripts_dir),
        },
    }

# ============================================
# LIST ALL SESSIONS
# ============================================

@app.get("/api/sessions/list")
async def list_sessions():
    """List all sessions"""
    if not os.path.exists(BASE_RECORDINGS_DIR):
        return {"sessions": [], "count": 0}

    sessions = []
    for session_dir in os.listdir(BASE_RECORDINGS_DIR):
        session_path = os.path.join(BASE_RECORDINGS_DIR, session_dir)
        metadata_path = os.path.join(session_path, "metadata.json")

        if os.path.exists(metadata_path):
            with open(metadata_path, "r") as f:
                metadata = json.load(f)
            sessions.append(
                {
                    "session_id": session_dir,
                    "mode": metadata.get("mode"),
                    "status": metadata.get("status"),
                    "created_at": metadata.get("created_at"),
                    "completed": metadata.get("completed", False),
                }
            )

    return {
        "sessions": sorted(sessions, key=lambda x: x.get("created_at", ""), reverse=True),
        "count": len(sessions),
        "recordings_dir": os.path.abspath(BASE_RECORDINGS_DIR),
    }

# ============================================
# FEEDBACK PIPELINE
# ============================================

@app.post("/api/session/{session_id}/generate_feedback")
async def generate_feedback(
    session_id: str,
    background_tasks: BackgroundTasks,
    request_body: dict = {}
):
    """Generate feedback for a completed interview session"""
    
    logger.info(f"📊 Starting feedback generation for session: {session_id}")
    
    # Extract parameters from request body
    candidate_name = request_body.get('candidate_name', 'Candidate')
    position = request_body.get('position', 'Position')
    role = request_body.get('role', None)
    experience = request_body.get('experience', 'mid')
    skip_video = request_body.get('skip_video', True)  # Default to True for stability
    
    # Check if session exists
    session_path = os.path.join(BASE_RECORDINGS_DIR, session_id)
    if not os.path.exists(session_path):
        raise HTTPException(status_code=404, detail="Session not found")
    
    # Check if already processing
    if processing_status.get(session_id) == 'processing':
        return {
            "status": "already_processing",
            "message": "Feedback generation already in progress",
            "session_id": session_id
        }
    
    # Check if already completed
    feedback_path = f"backend/feedback_results/{session_id}/feedback_report.json"
    if os.path.exists(feedback_path):
        return {
            "status": "already_complete",
            "message": "Feedback already generated",
            "session_id": session_id
        }
    
    # Mark as processing
    processing_status[session_id] = 'processing'
    
    # Start processing in background
    background_tasks.add_task(
        run_feedback_pipeline,
        session_id,
        candidate_name,
        position,
        role,
        experience,
        skip_video
    )
    
    estimated_time = "30-45 seconds" if skip_video else "3-6 minutes"
    
    return {
        "status": "processing",
        "message": "Feedback generation started",
        "session_id": session_id,
        "estimated_time": estimated_time,
        "skip_video": skip_video
    }


def run_feedback_pipeline(
    session_id: str,
    candidate_name: str,
    position: str,
    role: str,
    experience: str,
    skip_video: bool
):
    """Background task to run feedback pipeline"""
    
    try:
        logger.info(f"🚀 Running feedback pipeline for {session_id}")
        
        # Get python path
        import sys
        python_path = sys.executable
        
        # Get script path
        script_dir = os.path.dirname(os.path.abspath(__file__))
        process_script = os.path.join(script_dir, "process_session.py")
        
        # Build command
        cmd = [
            python_path,
            process_script,
            session_id,
            "--candidate", candidate_name,
            "--position", position,
            "--experience", experience
        ]
        
        if role:
            cmd.extend(["--role", role])
        
        if skip_video:
            cmd.append("--skip-video")
        
        logger.info(f"🧩 Executing command: {cmd}")
        
        # Run subprocess
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            cwd=script_dir
        )
        
        logger.info(f"STDOUT:\n{result.stdout}")
        
        if result.stderr:
            logger.error(f"STDERR:\n{result.stderr}")
        
        if result.returncode == 0:
            logger.info(f"✅ Feedback generation complete for {session_id}")
            processing_status[session_id] = 'complete'
        else:
            logger.error(f"❌ Feedback generation failed (exit code {result.returncode})")
            processing_status[session_id] = 'failed'
            
    except Exception as e:
        logger.error(f"❌ Exception in feedback pipeline: {e}")
        import traceback
        logger.error(traceback.format_exc())
        processing_status[session_id] = 'failed'

# ============================================
# STATUS, FEEDBACK, SCORES, VISUALS, REPORTS
# ============================================

@app.get("/api/session/{session_id}/processing_status")
async def get_processing_status(session_id: str):
    """Check processing status"""
    
    status = processing_status.get(session_id, 'unknown')
    
    # Double-check if feedback files exist
    feedback_path = f"backend/feedback_results/{session_id}/feedback_report.json"
    
    if os.path.exists(feedback_path):
        status = 'complete'
        processing_status[session_id] = 'complete'
    
    return {
        "session_id": session_id,
        "status": status,
        "ready": status == 'complete'
    }


# Add this endpoint for checking feedback status
@app.get("/api/session/{session_id}/feedback")
async def get_feedback(session_id: str):
    """Get feedback report for a session"""
    feedback_path = os.path.join(BASE_DIR, "feedback_results", session_id, "feedback_report.json")

    if not os.path.exists(feedback_path):
        raise HTTPException(status_code=404, detail=f"Feedback not ready yet: {feedback_path}")

    try:
        with open(feedback_path, 'r', encoding="utf-8") as f:
            feedback = json.load(f)
        return feedback
    except Exception as e:
        logger.error(f"Error loading feedback: {e}")
        raise HTTPException(status_code=500, detail="Error loading feedback")

@app.get("/api/session/{session_id}/weighted_scores")
async def get_weighted_scores(session_id: str):
    """Get weighted scores for a session"""
    scores_path = os.path.join(BASE_DIR, "feedback_results", session_id, "weighted_scores.json")

    if not os.path.exists(scores_path):
        raise HTTPException(status_code=404, detail=f"Scores not available yet: {scores_path}")

    try:
        with open(scores_path, 'r', encoding="utf-8") as f:
            scores = json.load(f)
        return scores
    except Exception as e:
        logger.error(f"Error loading scores: {e}")
        raise HTTPException(status_code=500, detail="Error loading scores")


@app.get("/api/session/{session_id}/processing_status")
async def get_processing_status(session_id: str):
    """Check processing status"""
    status = processing_status.get(session_id, 'unknown')

    feedback_path = os.path.join(BASE_DIR, "feedback_results", session_id, "feedback_report.json")

    if os.path.exists(feedback_path):
        status = 'complete'
        processing_status[session_id] = 'complete'

    return {
        "session_id": session_id,
        "status": status,
        "ready": status == 'complete'
    }


@app.get("/api/session/{session_id}/report/pdf")
async def get_pdf_report(session_id: str):
    pdf_path = os.path.join(BASE_DIR, "feedback_results", session_id, "feedback_report.pdf")

    if not os.path.exists(pdf_path):
        try:
            from visualization.report_builder import ReportBuilder

            builder = ReportBuilder()
            pdf_path = builder.build_report(session_id)
        except Exception as e:
            logger.error(f"Error generating PDF: {e}")
            raise HTTPException(status_code=404, detail="PDF report not available")

    return FileResponse(
        pdf_path,
        media_type="application/pdf",
        filename=f"interview_feedback_{session_id}.pdf",
    )


@app.delete("/api/session/{session_id}/feedback")
async def delete_feedback(session_id: str):
    feedback_dir = os.path.join(BASE_DIR, "feedback_results", session_id)
    features_dir = os.path.join(BASE_DIR, "processed_features", session_id)

    deleted = False

    if os.path.exists(feedback_dir):
        shutil.rmtree(feedback_dir)
        deleted = True

    if os.path.exists(features_dir):
        shutil.rmtree(features_dir)
        deleted = True

    if session_id in processing_status:
        del processing_status[session_id]

    if not deleted:
        raise HTTPException(status_code=404, detail="No feedback data found")

    return {"message": "Feedback data deleted successfully", "session_id": session_id}


# ============================================
# SERVER ENTRY POINT
# ============================================

if __name__ == "__main__":
    print("🚀 Starting AI Mock Interview API server...")
    print("📋 Server available at: http://127.0.0.1:8000")
    print("📖 Docs: http://127.0.0.1:8000/docs")
    print(f"📁 Recordings: {BASE_RECORDINGS_DIR}")
    uvicorn.run(app, host="127.0.0.1", port=8000, log_level="info")