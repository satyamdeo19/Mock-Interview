import os
from dotenv import load_dotenv

# Load .env file variables
load_dotenv()

# --- Gemini API Key ---
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# --- Backend Config ---
BACKEND_HOST = os.getenv("BACKEND_HOST", "127.0.0.1")
BACKEND_PORT = int(os.getenv("BACKEND_PORT", "8000"))

# --- Other Settings ---
MAX_FILE_SIZE = 10 * 1024 * 1024  # 10MB
ALLOWED_EXTENSIONS = [".pdf"]
SESSION_EXPIRY_HOURS = 24


def get_recordings_session_dir(session_id: str) -> str:
    """
    Return the recordings session directory for a given session_id.
    Always uses backend/recordings/sessions consistently.
    """
    # Always use backend/recordings/sessions
    backend_dir = os.path.dirname(os.path.abspath(__file__))
    session_path = os.path.join(backend_dir, 'recordings', 'sessions', session_id)
    return os.path.normpath(session_path)


# ============================================
# CREATE ALL CONFIG JSON FILES
# ============================================

import json

def create_all_config_files():
    """Create all configuration files for the feedback system"""
    
    base_dir = os.path.dirname(os.path.abspath(__file__))
    
    # ============================================
    # 1. FEATURE WEIGHTS CONFIG
    # ============================================
    feature_weights = {
        "dimension_weights": {
            "confidence": 0.15,
            "fluency": 0.15,
            "engagement": 0.15,
            "professionalism": 0.15,
            "emotional_state": 0.10,
            "calmness": 0.10,
            "articulation_quality": 0.08,
            "response_depth": 0.05,
            "cognitive_complexity": 0.04,
            "consistency": 0.03
        },
        "modality_weights": {
            "confidence": {"text": 0.25, "audio": 0.35, "video": 0.40},
            "fluency": {"text": 0.40, "audio": 0.35, "video": 0.25},
            "engagement": {"text": 0.25, "audio": 0.30, "video": 0.45},
            "professionalism": {"text": 0.30, "audio": 0.30, "video": 0.40},
            "emotional_state": {"text": 0.35, "audio": 0.35, "video": 0.30},
            "calmness": {"text": 0.25, "audio": 0.35, "video": 0.40},
            "articulation_quality": {"text": 0.35, "audio": 0.50, "video": 0.15},
            "response_depth": {"text": 0.60, "audio": 0.25, "video": 0.15},
            "cognitive_complexity": {"text": 0.50, "audio": 0.35, "video": 0.15},
            "consistency": {"text": 0.20, "audio": 0.10, "video": 0.70}
        }
    }
    
    with open(os.path.join(base_dir, "feature_weights.json"), 'w') as f:
        json.dump(feature_weights, f, indent=2)
    print("✅ Created: backend/feature_weights.json")
    
    # ============================================
    # 2. FEATURE CONFIG (Scaling & Normalization)
    # ============================================
    feature_config = {
        "text_features": {
            "text_word_count": {
                "scaling": "minmax",
                "range": [0, 200],
                "description": "Number of words in answer"
            },
            "text_sentence_count": {
                "scaling": "minmax",
                "range": [0, 20],
                "description": "Number of sentences"
            },
            "text_avg_word_length": {
                "scaling": "minmax",
                "range": [3, 10],
                "description": "Average word length"
            },
            "text_filler_ratio": {
                "scaling": "none",
                "range": [0, 1],
                "description": "Ratio of filler words (um, uh, like)"
            },
            "emotion_confidence_score": {
                "scaling": "none",
                "range": [0, 1],
                "description": "Sentiment-based confidence"
            }
        },
        "audio_features": {
            "pitch_mean": {
                "scaling": "minmax",
                "range": [80, 300],
                "unit": "Hz",
                "description": "Average fundamental frequency"
            },
            "pitch_std": {
                "scaling": "minmax",
                "range": [0, 50],
                "unit": "Hz",
                "description": "Pitch variation"
            },
            "tempo": {
                "scaling": "minmax",
                "range": [2, 8],
                "unit": "syllables/sec",
                "description": "Speaking rate"
            },
            "energy_mean": {
                "scaling": "minmax",
                "range": [0, 1],
                "description": "Average vocal energy"
            },
            "pause_fraction": {
                "scaling": "none",
                "range": [0, 1],
                "description": "Fraction of time in silence"
            }
        },
        "video_features": {
            "confidence_composite_score": {
                "scaling": "none",
                "range": [0, 1],
                "description": "Overall confidence from video"
            },
            "engagement_composite_score": {
                "scaling": "none",
                "range": [0, 1],
                "description": "Overall engagement from video"
            },
            "professionalism_composite_score": {
                "scaling": "none",
                "range": [0, 1],
                "description": "Overall professionalism from video"
            },
            "nervousness_composite_score": {
                "scaling": "none",
                "range": [0, 1],
                "description": "Nervousness indicators (inverted for calmness)"
            }
        }
    }
    
    with open(os.path.join(base_dir, "feature_config.json"), 'w') as f:
        json.dump(feature_config, f, indent=2)
    print("✅ Created: backend/feature_config.json")
    
    # ============================================
    # 3. BENCHMARK DATA
    # ============================================
    benchmark_data = {
        "industry_averages": {
            "confidence": 0.72,
            "fluency": 0.70,
            "engagement": 0.75,
            "professionalism": 0.73,
            "emotional_state": 0.68,
            "calmness": 0.70,
            "articulation_quality": 0.72,
            "response_depth": 0.65,
            "cognitive_complexity": 0.60,
            "consistency": 0.75
        },
        "top_performer_threshold": 0.82,
        "percentile_ranges": {
            "excellent": [0.85, 1.0],
            "very_good": [0.80, 0.85],
            "good": [0.75, 0.80],
            "above_average": [0.70, 0.75],
            "average": [0.65, 0.70],
            "below_average": [0.60, 0.65],
            "needs_improvement": [0.0, 0.60]
        },
        "role_specific": {
            "software_engineer": {
                "weight_adjustments": {
                    "cognitive_complexity": 1.2,
                    "articulation_quality": 0.9,
                    "professionalism": 1.0
                },
                "benchmarks": {
                    "confidence": 0.70,
                    "fluency": 0.68,
                    "engagement": 0.72,
                    "cognitive_complexity": 0.75
                }
            },
            "sales_representative": {
                "weight_adjustments": {
                    "engagement": 1.3,
                    "confidence": 1.2,
                    "articulation_quality": 1.1
                },
                "benchmarks": {
                    "confidence": 0.80,
                    "engagement": 0.85,
                    "articulation_quality": 0.78
                }
            },
            "manager": {
                "weight_adjustments": {
                    "professionalism": 1.2,
                    "confidence": 1.1,
                    "emotional_state": 1.1
                },
                "benchmarks": {
                    "professionalism": 0.82,
                    "confidence": 0.78,
                    "calmness": 0.76
                }
            }
        },
        "experience_level": {
            "entry": {
                "multiplier": 0.90,
                "expected_ranges": {
                    "confidence": [0.65, 0.75],
                    "fluency": [0.60, 0.70],
                    "professionalism": [0.65, 0.75]
                }
            },
            "mid": {
                "multiplier": 1.0,
                "expected_ranges": {
                    "confidence": [0.70, 0.80],
                    "fluency": [0.68, 0.78],
                    "professionalism": [0.72, 0.82]
                }
            },
            "senior": {
                "multiplier": 1.10,
                "expected_ranges": {
                    "confidence": [0.75, 0.85],
                    "fluency": [0.73, 0.83],
                    "professionalism": [0.78, 0.88]
                }
            }
        }
    }
    
    with open(os.path.join(base_dir, "benchmark_data.json"), 'w') as f:
        json.dump(benchmark_data, f, indent=2)
    print("✅ Created: backend/benchmark_data.json")
    
    print("\n✨ All configuration files created successfully!")


if __name__ == "__main__":
    create_all_config_files()