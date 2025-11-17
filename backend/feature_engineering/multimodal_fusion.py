"""
Multimodal Fusion
Combines text, audio, and video features into final 10-dimensional scores
Input: text_features.csv, audio_features.csv, video_features_aggregated.csv
Output: final_multimodal_features.csv
"""
import os
import pandas as pd
import numpy as np
import logging
import json
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MultimodalFusion:
    """Fuse text, audio, and video features into 10 target dimensions"""
    
    def __init__(self, config_path: str = None):
        """Initialize with configuration"""
        if config_path is None:
            # Get the absolute path to backend/feature_weights.json
            backend_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            config_path = os.path.join(backend_dir, "feature_weights.json")
        
        # Load configuration
        try:
            with open(config_path, 'r') as f:
                config = json.load(f)
            
            # Extract weights from config
            self.modality_weights = config.get('modality_weights', {})
            self.dimension_weights = config.get('dimension_weights', {})
            
            # Set default weights if not in config
            if not self.modality_weights:
                logger.warning("⚠️  No modality_weights in config, using defaults")
                self.modality_weights = self._get_default_modality_weights()
            
            if not self.dimension_weights:
                logger.warning("⚠️  No dimension_weights in config, using defaults")
                self.dimension_weights = self._get_default_dimension_weights()
                
            logger.info(f"✅ Loaded configuration from: {config_path}")
            
        except FileNotFoundError:
            logger.warning(f"⚠️  Config file not found: {config_path}")
            logger.warning("   Using default weights")
            self.modality_weights = self._get_default_modality_weights()
            self.dimension_weights = self._get_default_dimension_weights()
        except Exception as e:
            logger.error(f"❌ Error loading config: {e}")
            logger.warning("   Using default weights")
            self.modality_weights = self._get_default_modality_weights()
            self.dimension_weights = self._get_default_dimension_weights()
    
    def _get_default_modality_weights(self) -> dict:
        """Default modality weights for each dimension"""
        return {
            'confidence': {'text': 0.3, 'audio': 0.3, 'video': 0.4},
            'fluency': {'text': 0.4, 'audio': 0.4, 'video': 0.2},
            'engagement': {'text': 0.3, 'audio': 0.3, 'video': 0.4},
            'professionalism': {'text': 0.4, 'audio': 0.3, 'video': 0.3},
            'emotional_state': {'text': 0.4, 'audio': 0.3, 'video': 0.3},
            'calmness': {'text': 0.3, 'audio': 0.4, 'video': 0.3},
            'articulation_quality': {'text': 0.4, 'audio': 0.4, 'video': 0.2},
            'response_depth': {'text': 0.6, 'audio': 0.3, 'video': 0.1},
            'cognitive_complexity': {'text': 0.6, 'audio': 0.3, 'video': 0.1},
            'consistency': {'text': 0.4, 'audio': 0.2, 'video': 0.4}
        }
    
    def _get_default_dimension_weights(self) -> dict:
        """Default weights for final score calculation"""
        return {
            'confidence': 0.12,
            'fluency': 0.10,
            'engagement': 0.10,
            'professionalism': 0.12,
            'emotional_state': 0.08,
            'calmness': 0.08,
            'articulation_quality': 0.10,
            'response_depth': 0.12,
            'cognitive_complexity': 0.10,
            'consistency': 0.08
        }
    
    def fuse_features(self, text_csv: str, audio_csv: str, video_csv: str, 
                     output_path: str = None) -> pd.DataFrame:
        """
        Main fusion function
        
        Args:
            text_csv: Path to text_features.csv
            audio_csv: Path to audio_features.csv
            video_csv: Path to video_features_aggregated.csv
            output_path: Optional output path for final_multimodal_features.csv
            
        Returns:
            DataFrame with single row containing 10 dimensions + final_score
        """
        logger.info(f"🔄 Starting multimodal feature fusion")
        
        # Load all feature sets
        text_df = pd.read_csv(text_csv)
        audio_df = pd.read_csv(audio_csv)
        video_df = pd.read_csv(video_csv)
        
        logger.info(f"   Text: {len(text_df)} rows, {len(text_df.columns)} features")
        logger.info(f"   Audio: {len(audio_df)} rows, {len(audio_df.columns)} features")
        logger.info(f"   Video: {len(video_df)} rows, {len(video_df.columns)} features")
        
        # Aggregate text features (per-question → session-level)
        text_agg = self._aggregate_text_features(text_df)
        
        # Audio is already session-level (single row)
        audio_features = audio_df.iloc[0].to_dict()
        
        # Video is already session-level (single row)
        video_features = video_df.iloc[0].to_dict()
        
        # Calculate 10 target dimensions
        dimensions = {}
        
        dimensions['confidence'] = self._calculate_confidence(text_agg, audio_features, video_features)
        dimensions['fluency'] = self._calculate_fluency(text_agg, audio_features, video_features)
        dimensions['engagement'] = self._calculate_engagement(text_agg, audio_features, video_features)
        dimensions['professionalism'] = self._calculate_professionalism(text_agg, audio_features, video_features)
        dimensions['emotional_state'] = self._calculate_emotional_state(text_agg, audio_features, video_features)
        dimensions['calmness'] = self._calculate_calmness(text_agg, audio_features, video_features)
        dimensions['articulation_quality'] = self._calculate_articulation(text_agg, audio_features, video_features)
        dimensions['response_depth'] = self._calculate_response_depth(text_agg, audio_features, video_features)
        dimensions['cognitive_complexity'] = self._calculate_cognitive_complexity(text_agg, audio_features, video_features)
        dimensions['consistency'] = self._calculate_consistency(text_agg, audio_features, video_features)
        
        # Calculate weighted final score
        final_score = sum(
            dimensions[dim] * self.dimension_weights.get(dim, 0.1)
            for dim in dimensions.keys()
        ) * 100  # Scale to 0-100
        
        dimensions['final_score'] = round(final_score, 2)
        
        # Create DataFrame
        result_df = pd.DataFrame([dimensions])
        
        logger.info(f"✅ Calculated 10 dimensions + final score")
        logger.info(f"   Final Score: {final_score:.2f}")
        
        # Save if output path provided
        if output_path:
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            result_df.to_csv(output_path, index=False)
            logger.info(f"💾 Saved to: {output_path}")
        
        return result_df
    
    def _aggregate_text_features(self, text_df: pd.DataFrame) -> dict:
        """Aggregate per-question text features to session-level"""
        
        # Meta features are already session-level (same value in all rows)
        meta_features = {
            'meta_total_questions': text_df['meta_total_questions'].iloc[0],
            'meta_avg_answer_length': text_df['meta_avg_answer_length'].iloc[0],
            'meta_answer_length_std': text_df['meta_answer_length_std'].iloc[0],
            'meta_avg_filler_ratio': text_df['meta_avg_filler_ratio'].iloc[0],
            'meta_avg_sentiment': text_df['meta_avg_sentiment'].iloc[0],
            'meta_confidence_trend': text_df['meta_confidence_trend'].iloc[0]
        }
        
        # Aggregate per-question features
        per_question_cols = [col for col in text_df.columns if not col.startswith('meta_') and col != 'question_number']
        
        aggregated = {}
        for col in per_question_cols:
            aggregated[f'{col}_mean'] = text_df[col].mean()
            aggregated[f'{col}_std'] = text_df[col].std()
        
        # Combine
        return {**meta_features, **aggregated}
    
    def _normalize(self, value: float, min_val: float = 0, max_val: float = 1) -> float:
        """Normalize value to 0-1 range"""
        return np.clip((value - min_val) / (max_val - min_val), 0, 1)
    
    def _weighted_mean(self, text_val: float, audio_val: float, video_val: float, dimension: str) -> float:
        """Calculate weighted mean across modalities"""
        weights = self.modality_weights.get(dimension, {'text': 0.33, 'audio': 0.33, 'video': 0.34})
        return (text_val * weights['text'] + 
                audio_val * weights['audio'] + 
                video_val * weights['video'])
    
    # ============================================
    # DIMENSION CALCULATIONS (keeping existing methods)
    # ============================================
    
    def _calculate_confidence(self, text: dict, audio: dict, video: dict) -> float:
        """Calculate confidence score"""
        # Text components
        text_confidence = text.get('emotion_confidence_score_mean', 0.5)
        text_tentative = text.get('emotion_tentative_score_mean', 0.5)
        text_filler = text.get('text_filler_ratio_mean', 0.1)
        text_comp = np.mean([text_confidence, 1 - text_tentative, 1 - text_filler])
        
        # Audio components
        pitch_range = self._normalize(audio.get('audio_pitch_range', 50), 0, 150)
        energy = self._normalize(audio.get('audio_energy_mean', 0), 0, 0.1)
        tempo = self._normalize(audio.get('audio_tempo', 3), 2, 8)
        pause_frac = audio.get('audio_pause_fraction', 0.3)
        audio_comp = np.mean([pitch_range, energy, tempo, 1 - pause_frac])
        
        # Video components
        video_comp = np.mean([
            video.get('confidence_composite_score', 0.7),
            video.get('confidence_eye_contact_stability', 0.7)
        ])
        
        return round(self._weighted_mean(text_comp, audio_comp, video_comp, 'confidence'), 4)
    
    def _calculate_fluency(self, text: dict, audio: dict, video: dict) -> float:
        """Calculate fluency score"""
        # Text components
        filler = text.get('text_filler_ratio_mean', 0.1)
        stopword = text.get('text_stopword_ratio_mean', 0.4)
        sent_len = self._normalize(text.get('text_avg_sentence_length_mean', 10), 5, 20)
        text_comp = np.mean([1 - filler, stopword, sent_len])
        
        # Audio components
        tempo = self._normalize(audio.get('audio_tempo', 3), 2, 8)
        duration = audio.get('audio_duration', 180)
        num_pauses = audio.get('audio_num_pauses', 10)
        pause_rate = min(num_pauses / duration, 1.0) if duration > 0 else 0
        avg_pause = min(audio.get('audio_avg_pause_duration', 0.5), 2.0) / 2.0
        audio_comp = np.mean([tempo, 1 - pause_rate, 1 - avg_pause])
        
        # Video components
        video_comp = np.mean([
            video.get('professionalism_controlled_head_movement', 0.7),
            video.get('temporal_stability_consistency', 0.7)
        ])
        
        return round(self._weighted_mean(text_comp, audio_comp, video_comp, 'fluency'), 4)
    
    def _calculate_engagement(self, text: dict, audio: dict, video: dict) -> float:
        """Calculate engagement score"""
        # Text components
        word_count = self._normalize(text.get('text_word_count_mean', 50), 20, 150)
        avg_ans_len = self._normalize(text.get('meta_avg_answer_length', 50), 20, 150)
        joy = text.get('emotion_joy_score_mean', 0.3)
        text_comp = np.mean([word_count, avg_ans_len, joy])
        
        # Audio components
        energy = self._normalize(audio.get('audio_energy_mean', 0), 0, 0.1)
        pitch_range = self._normalize(audio.get('audio_pitch_range', 50), 0, 150)
        energy_std = self._normalize(audio.get('audio_energy_std', 0), 0, 0.05)
        audio_comp = np.mean([energy, pitch_range, energy_std])
        
        # Video components
        video_comp = np.mean([
            video.get('engagement_composite_score', 0.7),
            video.get('engagement_overall_facial_animation', 0.7)
        ])
        
        return round(self._weighted_mean(text_comp, audio_comp, video_comp, 'engagement'), 4)
    
    def _calculate_professionalism(self, text: dict, audio: dict, video: dict) -> float:
        """Calculate professionalism score"""
        # Text components
        filler = text.get('text_filler_ratio_mean', 0.1)
        word_len = self._normalize(text.get('text_avg_word_length_mean', 4.5), 3, 8)
        analytical = text.get('emotion_analytical_score_mean', 0.5)
        text_comp = np.mean([1 - filler, word_len, analytical])
        
        # Audio components
        tempo = self._normalize(audio.get('audio_tempo', 3), 2, 8)
        pause_frac = audio.get('audio_pause_fraction', 0.3)
        spec_cent = self._normalize(audio.get('audio_spec_cent_mean', 1500), 1000, 3000)
        audio_comp = np.mean([tempo, 1 - pause_frac, spec_cent])
        
        # Video components
        video_comp = np.mean([
            video.get('professionalism_composite_score', 0.7),
            video.get('professionalism_gaze_steadiness', 0.7)
        ])
        
        return round(self._weighted_mean(text_comp, audio_comp, video_comp, 'professionalism'), 4)
    
    def _calculate_emotional_state(self, text: dict, audio: dict, video: dict) -> float:
        """Calculate emotional state score"""
        # Valence dimension
        sentiment = text.get('emotion_sentiment_polarity_mean', 0)
        sentiment_norm = (sentiment + 1) / 2  # Convert -1,1 to 0,1
        pitch = self._normalize(audio.get('audio_pitch_mean', 150), 80, 300)
        expr_range = self._normalize(video.get('engagement_expression_dynamic_range', 0.5), 0, 2)
        valence = np.mean([sentiment_norm, pitch, expr_range])
        
        # Arousal dimension
        joy = text.get('emotion_joy_score_mean', 0.3)
        fear = text.get('emotion_fear_score_mean', 0.1)
        anger = text.get('emotion_anger_score_mean', 0.1)
        emotion_arousal = joy + fear + anger
        
        energy = self._normalize(audio.get('audio_energy_mean', 0), 0, 0.1)
        pitch_std = self._normalize(audio.get('audio_pitch_std', 20), 0, 50)
        audio_arousal = (energy + pitch_std) / 2
        
        nervousness = video.get('nervousness_composite_score', 0.7)
        video_arousal = nervousness
        
        arousal = np.mean([emotion_arousal, audio_arousal, video_arousal])
        
        emotional_state = 0.5 * valence + 0.5 * arousal
        return round(emotional_state, 4)
    
    def _calculate_calmness(self, text: dict, audio: dict, video: dict) -> float:
        """Calculate calmness score"""
        # Text components
        fear = text.get('emotion_fear_score_mean', 0.1)
        analytical = text.get('emotion_analytical_score_mean', 0.5)
        text_comp = np.mean([1 - fear, analytical])
        
        # Audio components
        pitch_mean = audio.get('audio_pitch_mean', 150)
        pitch_std = audio.get('audio_pitch_std', 20)
        pitch_stability = 1 - min(pitch_std / pitch_mean, 1) if pitch_mean > 0 else 0.5
        energy_entropy = self._normalize(audio.get('audio_energy_entropy', 5), 0, 10)
        pause_frac = audio.get('audio_pause_fraction', 0.3)
        audio_comp = np.mean([pitch_stability, 1 - energy_entropy, 1 - pause_frac])
        
        # Video components
        video_comp = np.mean([
            video.get('nervousness_composite_score', 0.7),
            video.get('professionalism_composite_score', 0.7)
        ])
        
        return round(self._weighted_mean(text_comp, audio_comp, video_comp, 'calmness'), 4)
    
    def _calculate_articulation(self, text: dict, audio: dict, video: dict) -> float:
        """Calculate articulation quality score"""
        # Text components
        word_len = self._normalize(text.get('text_avg_word_length_mean', 4.5), 3, 8)
        filler = text.get('text_filler_ratio_mean', 0.1)
        text_comp = np.mean([word_len, 1 - filler])
        
        # Audio components
        spec_cent = self._normalize(audio.get('audio_spec_cent_mean', 1500), 1000, 3000)
        energy = self._normalize(audio.get('audio_energy_mean', 0), 0, 0.1)
        pause_frac = audio.get('audio_pause_fraction', 0.3)
        audio_comp = np.mean([spec_cent, energy, 1 - pause_frac])
        
        # Video components
        mouth_expr = self._normalize(video.get('engagement_mouth_expressiveness', 1), 0, 2)
        video_comp = mouth_expr
        
        return round(self._weighted_mean(text_comp, audio_comp, video_comp, 'articulation_quality'), 4)
    
    def _calculate_response_depth(self, text: dict, audio: dict, video: dict) -> float:
        """Calculate response depth score"""
        # Text components
        word_count = self._normalize(text.get('text_word_count_mean', 50), 20, 150)
        sent_len = self._normalize(text.get('text_avg_sentence_length_mean', 10), 5, 20)
        avg_ans = self._normalize(text.get('meta_avg_answer_length', 50), 20, 150)
        text_comp = np.mean([word_count, sent_len, avg_ans])
        
        # Audio components
        duration = self._normalize(audio.get('audio_duration', 180), 60, 600)
        tempo = audio.get('audio_tempo', 3)
        thoughtfulness = 1 - self._normalize(tempo, 2, 8)
        audio_comp = np.mean([duration, thoughtfulness])
        
        # Video components
        video_comp = video.get('engagement_composite_score', 0.7)
        
        return round(self._weighted_mean(text_comp, audio_comp, video_comp, 'response_depth'), 4)
    
    def _calculate_cognitive_complexity(self, text: dict, audio: dict, video: dict) -> float:
        """Calculate cognitive complexity score"""
        # Text components
        word_len = self._normalize(text.get('text_avg_word_length_mean', 4.5), 3, 8)
        sent_len = self._normalize(text.get('text_avg_sentence_length_mean', 10), 5, 20)
        text_comp = np.mean([word_len, sent_len])
        
        # Audio components
        tempo = audio.get('audio_tempo', 3)
        thoughtfulness = 1 - self._normalize(tempo, 2, 8)
        pause_frac = audio.get('audio_pause_fraction', 0.3)
        audio_comp = np.mean([thoughtfulness, pause_frac])
        
        # Video components
        video_comp = video.get('professionalism_controlled_head_movement', 0.7)
        
        return round(self._weighted_mean(text_comp, audio_comp, video_comp, 'cognitive_complexity'), 4)
    
    def _calculate_consistency(self, text: dict, audio: dict, video: dict) -> float:
        """Calculate consistency score"""
        # Text components
        answer_len_std = text.get('meta_answer_length_std', 20)
        consistency_score = 1 - self._normalize(answer_len_std, 0, 50)
        text_comp = consistency_score
        
        # Audio (already session-level)
        audio_comp = 1.0
        
        # Video components
        video_comp = np.mean([
            video.get('temporal_consistency_composite_score', 0.7),
            video.get('temporal_engagement_consistency', 0.7)
        ])
        
        return round(self._weighted_mean(text_comp, audio_comp, video_comp, 'consistency'), 4)


# ============================================
# CLI USAGE
# ============================================
if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python multimodal_fusion.py <session_id>")
        print("Example: python multimodal_fusion.py session_abc123_1234567890")
        sys.exit(1)
    
    session_id = sys.argv[1]
    
    # Paths
    feature_dir = f"processed_features/{session_id}"
    text_csv = f"{feature_dir}/text_features.csv"
    audio_csv = f"{feature_dir}/audio_features.csv"
    video_csv = f"{feature_dir}/video_features_aggregated.csv"
    output_path = f"{feature_dir}/final_multimodal_features.csv"
    
    # Check if all inputs exist
    missing = []
    for path in [text_csv, audio_csv, video_csv]:
        if not os.path.exists(path):
            missing.append(path)
    
    if missing:
        print(f"❌ Missing input files:")
        for path in missing:
            print(f"   - {path}")
        print("\n   Run feature extractors first!")
        sys.exit(1)
    
    # Fuse features
    fusion = MultimodalFusion()
    
    try:
        result_df = fusion.fuse_features(text_csv, audio_csv, video_csv, output_path)
        
        print("\n" + "="*60)
        print("✅ MULTIMODAL FEATURE FUSION COMPLETE")
        print("="*60)
        print(f"Inputs:")
        print(f"  - Text:  {text_csv}")
        print(f"  - Audio: {audio_csv}")
        print(f"  - Video: {video_csv}")
        print(f"\nOutput: {output_path}")
        print(f"\n📊 10-Dimensional Scores:")
        print(result_df.T)
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)