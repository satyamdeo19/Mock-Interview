"""
Master Processing Script
Runs the complete feedback pipeline for a session
Usage: python process_session.py <session_id> [options]
"""

import sys
import os
import logging
import time
import json
from pathlib import Path
import argparse

# Ensure backend directory is on sys.path so local imports resolve correctly
backend_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, backend_dir)

# Import all modules
from feature_extractors.text_extractor import TextFeatureExtractor
from feature_extractors.audio_extractor import AudioFeatureExtractor
from feature_extractors.video_extractor import VideoFeatureExtractor
from feature_engineering.video_aggregator import VideoAggregator
from feature_engineering.multimodal_fusion import MultimodalFusion
from feedback_engine.weighted_scorer import WeightedScorer
from feedback_engine.feedback_generator import FeedbackGenerator

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class FeedbackPipeline:
    """Complete feedback processing pipeline"""

    def __init__(self, session_id: str, skip_video: bool = False, video_fps: int = 10):
        self.session_id = session_id
        self.skip_video = skip_video
        self.video_fps = video_fps

        # ------------------------------------------------------------------
        # CRITICAL FIX: Always use backend/recordings/sessions
        # ------------------------------------------------------------------
        self.backend_root = os.path.dirname(os.path.abspath(__file__))

        # Use backend/recordings/sessions consistently
        self.session_dir = os.path.join(self.backend_root, "recordings", "sessions", session_id)
        self.session_dir = os.path.normpath(self.session_dir)

        logger.info(f"📂 Session directory: {self.session_dir}")

        # ------------------------------------------------------------------
        # Define output directories inside backend/
        # ------------------------------------------------------------------
        self.features_dir = os.path.join(self.backend_root, "processed_features", session_id)
        self.results_dir = os.path.join(self.backend_root, "feedback_results", session_id)

        # ------------------------------------------------------------------
        # Ensure they exist
        # ------------------------------------------------------------------
        Path(self.features_dir).mkdir(parents=True, exist_ok=True)
        Path(self.results_dir).mkdir(parents=True, exist_ok=True)

        logger.info(f"🚀 Initialized pipeline for session: {session_id}")
        logger.info(f"   Features: {self.features_dir}")
        logger.info(f"   Results: {self.results_dir}")

    def validate_session(self) -> bool:
        """Validate that session has required files"""
        logger.info("🔍 Validating session files...")

        required_files = [
            os.path.join(self.session_dir, "transcripts", "qa_data.csv"),
            os.path.join(self.session_dir, "audio", "full_session.webm"),
        ]

        if not self.skip_video:
            # Check both possible video chunk locations
            video_dir_chunks = os.path.join(self.session_dir, "video", "chunks")
            video_dir_root = os.path.join(self.session_dir, "video")

            logger.info(f"🔍 Checking for video chunks...")
            logger.info(f"   Chunks dir: {video_dir_chunks}")
            logger.info(f"   Root dir: {video_dir_root}")

            # List what's actually in the directories
            if os.path.exists(video_dir_chunks):
                files = os.listdir(video_dir_chunks)
                logger.info(f"   Files in chunks/: {files}")
            else:
                logger.info(f"   chunks/ directory doesn't exist")

            if os.path.exists(video_dir_root):
                files = os.listdir(video_dir_root)
                logger.info(f"   Files in video/: {files}")
            else:
                logger.info(f"   video/ directory doesn't exist")

            has_chunks_in_chunks = (
                os.path.exists(video_dir_chunks)
                and any(
                    f.startswith("chunk") and f.endswith(".webm")
                    for f in os.listdir(video_dir_chunks)
                )
            ) if os.path.exists(video_dir_chunks) else False

            has_chunks_in_root = (
                os.path.exists(video_dir_root)
                and any(
                    f.startswith("chunk") and f.endswith(".webm")
                    for f in os.listdir(video_dir_root)
                )
            ) if os.path.exists(video_dir_root) else False

            logger.info(f"   Has chunks in chunks/: {has_chunks_in_chunks}")
            logger.info(f"   Has chunks in root: {has_chunks_in_root}")

            if not has_chunks_in_chunks and not has_chunks_in_root:
                logger.warning("⚠️  No video chunks found in video/ or video/chunks/, will skip video processing")
                self.skip_video = True
            else:
                logger.info(f"✅ Found video chunks in {'video/chunks/' if has_chunks_in_chunks else 'video/'}")

        missing = []
        for file in required_files:
            if not os.path.exists(file):
                missing.append(file)

        if missing:
            logger.error("❌ Missing required files:")
            for file in missing:
                logger.error(f"   - {file}")
            return False

        logger.info("✅ Session validation passed")
        return True

    def run_phase_1_text(self) -> bool:
        logger.info("\n" + "=" * 60)
        logger.info("📝 PHASE 1: TEXT FEATURE EXTRACTION")
        logger.info("=" * 60)

        try:
            start_time = time.time()
            csv_path = os.path.join(self.session_dir, "transcripts", "qa_data.csv")
            output_path = os.path.join(self.features_dir, "text_features.csv")

            extractor = TextFeatureExtractor()
            features_df = extractor.extract_features(csv_path, output_path)

            elapsed = time.time() - start_time
            logger.info(f"✅ Phase 1 complete in {elapsed:.2f}s")
            logger.info(f"   Extracted {len(features_df.columns)} features from {len(features_df)} Q&A pairs")
            return True

        except Exception as e:
            logger.error(f"❌ Phase 1 failed: {e}")
            import traceback
            traceback.print_exc()
            return False

    def run_phase_2_audio(self) -> bool:
        logger.info("\n" + "=" * 60)
        logger.info("🎵 PHASE 2: AUDIO FEATURE EXTRACTION")
        logger.info("=" * 60)

        try:
            start_time = time.time()
            audio_path = os.path.join(self.session_dir, "audio", "full_session.webm")
            output_path = os.path.join(self.features_dir, "audio_features.csv")

            extractor = AudioFeatureExtractor()
            features_df = extractor.extract_features(audio_path, output_path)

            elapsed = time.time() - start_time
            logger.info(f"✅ Phase 2 complete in {elapsed:.2f}s")
            logger.info(f"   Extracted {len(features_df.columns)} audio features")
            return True

        except Exception as e:
            logger.error(f"❌ Phase 2 failed: {e}")
            import traceback
            traceback.print_exc()
            return False

    def run_phase_3_video(self) -> bool:
        if self.skip_video:
            logger.info("\n⏭️  PHASE 3: VIDEO PROCESSING SKIPPED")
            self._create_default_video_features()
            return True

        logger.info("\n" + "=" * 60)
        logger.info("🎥 PHASE 3: VIDEO FEATURE EXTRACTION")
        logger.info("=" * 60)

        try:
            start_time = time.time()
            video_dir = os.path.join(self.session_dir, "video")
            output_path = os.path.join(self.features_dir, "video_features_raw.csv")

            extractor = VideoFeatureExtractor(fps=self.video_fps)
            features_df = extractor.extract_features(video_dir, output_path)

            elapsed = time.time() - start_time
            logger.info(f"✅ Phase 3 complete in {elapsed:.2f}s")
            logger.info(f"   Extracted {len(features_df.columns)} features from {len(features_df)} frames")
            return True

        except Exception as e:
            logger.error(f"❌ Phase 3 failed: {e}")
            logger.warning("   Creating default video features...")
            self._create_default_video_features()
            return True

    def run_phase_4_video_aggregation(self) -> bool:
        if self.skip_video:
            logger.info("\n⏭️  PHASE 4: VIDEO AGGREGATION SKIPPED (using defaults)")
            return True

        logger.info("\n" + "=" * 60)
        logger.info("📊 PHASE 4: VIDEO FEATURE AGGREGATION")
        logger.info("=" * 60)

        try:
            start_time = time.time()
            input_path = os.path.join(self.features_dir, "video_features_raw.csv")
            output_path = os.path.join(self.features_dir, "video_features_aggregated.csv")

            aggregator = VideoAggregator()
            features_df = aggregator.aggregate_features(input_path, output_path)

            elapsed = time.time() - start_time
            logger.info(f"✅ Phase 4 complete in {elapsed:.2f}s")
            logger.info(f"   Aggregated {len(features_df.columns)} composite features")
            return True

        except Exception as e:
            logger.error(f"❌ Phase 4 failed: {e}")
            import traceback
            traceback.print_exc()
            return False

    def run_phase_5_fusion(self) -> bool:
        logger.info("\n" + "=" * 60)
        logger.info("🔄 PHASE 5: MULTIMODAL FEATURE FUSION")
        logger.info("=" * 60)

        try:
            start_time = time.time()
            text_csv = os.path.join(self.features_dir, "text_features.csv")
            audio_csv = os.path.join(self.features_dir, "audio_features.csv")
            video_csv = os.path.join(self.features_dir, "video_features_aggregated.csv")
            output_path = os.path.join(self.features_dir, "final_multimodal_features.csv")

            fusion = MultimodalFusion()
            result_df = fusion.fuse_features(text_csv, audio_csv, video_csv, output_path)

            elapsed = time.time() - start_time
            logger.info(f"✅ Phase 5 complete in {elapsed:.2f}s")
            logger.info(f"   Generated 10 dimensions + final score")
            return True

        except Exception as e:
            logger.error(f"❌ Phase 5 failed: {e}")
            import traceback
            traceback.print_exc()
            return False

    def run_phase_6_scoring(self, role: str = None, experience: str = "mid") -> bool:
        logger.info("\n" + "=" * 60)
        logger.info("🎯 PHASE 6: WEIGHTED SCORING")
        logger.info("=" * 60)

        try:
            start_time = time.time()
            features_csv = os.path.join(self.features_dir, "final_multimodal_features.csv")
            output_path = os.path.join(self.results_dir, "weighted_scores.json")

            scorer = WeightedScorer()
            result = scorer.generate_scores(features_csv, output_path, role, experience)

            elapsed = time.time() - start_time
            logger.info(f"✅ Phase 6 complete in {elapsed:.2f}s")
            logger.info(f"   Final Score: {result['final_score']:.1f}/100 ({result['overall_grade']})")
            return True

        except Exception as e:
            logger.error(f"❌ Phase 6 failed: {e}")
            import traceback
            traceback.print_exc()
            return False

    def run_phase_7_feedback(self, candidate_name: str = "Candidate", position: str = "Position") -> bool:
        logger.info("\n" + "=" * 60)
        logger.info("📝 PHASE 7: FEEDBACK GENERATION")
        logger.info("=" * 60)

        try:
            start_time = time.time()
            weighted_scores_path = os.path.join(self.results_dir, "weighted_scores.json")
            features_csv_path = os.path.join(self.features_dir, "final_multimodal_features.csv")
            output_path = os.path.join(self.results_dir, "feedback_report.json")

            generator = FeedbackGenerator()
            report = generator.generate_feedback(
                weighted_scores_path,
                features_csv_path,
                output_path,
                candidate_name,
                position
            )

            elapsed = time.time() - start_time
            logger.info(f"✅ Phase 7 complete in {elapsed:.2f}s")
            logger.info(f"   Generated comprehensive feedback report")
            return True

        except Exception as e:
            logger.error(f"❌ Phase 7 failed: {e}")
            import traceback
            traceback.print_exc()
            return False

    def run_complete_pipeline(self, candidate_name="Candidate", position="Position", role=None, experience="mid") -> bool:
        logger.info("\n" + "🎬 " * 20)
        logger.info("STARTING COMPLETE FEEDBACK PIPELINE")
        logger.info("🎬 " * 20)

        pipeline_start = time.time()

        if not self.validate_session():
            return False

        phases = [
            ("Text Extraction", lambda: self.run_phase_1_text()),
            ("Audio Extraction", lambda: self.run_phase_2_audio()),
            ("Video Extraction", lambda: self.run_phase_3_video()),
            ("Video Aggregation", lambda: self.run_phase_4_video_aggregation()),
            ("Multimodal Fusion", lambda: self.run_phase_5_fusion()),
            ("Weighted Scoring", lambda: self.run_phase_6_scoring(role, experience)),
            ("Feedback Generation", lambda: self.run_phase_7_feedback(candidate_name, position))
        ]

        for phase_name, phase_func in phases:
            if not phase_func():
                logger.error(f"\n❌ Pipeline failed at: {phase_name}")
                return False

        total_time = time.time() - pipeline_start
        logger.info("\n" + "🎉 " * 20)
        logger.info("PIPELINE COMPLETE!")
        logger.info("🎉 " * 20)
        logger.info(f"\n⏱️  Total processing time: {total_time:.2f}s ({total_time/60:.1f} min)")
        logger.info(f"\n📁 Results saved to:")
        logger.info(f"   Features: {self.features_dir}")
        logger.info(f"   Feedback: {self.results_dir}")

        self._display_summary()
        return True

    def _create_default_video_features(self):
        import pandas as pd
        default_features = {
            'confidence_composite_score': 0.70,
            'confidence_eye_contact_stability': 0.70,
            'engagement_composite_score': 0.70,
            'engagement_overall_facial_animation': 0.70,
            'professionalism_composite_score': 0.70,
            'professionalism_gaze_steadiness': 0.70,
            'professionalism_controlled_head_movement': 0.70,
            'nervousness_composite_score': 0.70,
            'temporal_consistency_composite_score': 0.70,
            'temporal_engagement_consistency': 0.70,
            'temporal_stability_consistency': 0.70,
            'engagement_mouth_expressiveness': 0.70,
            'engagement_expression_dynamic_range': 0.70
        }
        df = pd.DataFrame([default_features])
        output_path = os.path.join(self.features_dir, "video_features_aggregated.csv")
        df.to_csv(output_path, index=False)
        logger.info(f"   Created default video features at {output_path}")

    def _display_summary(self):
        try:
            with open(os.path.join(self.results_dir, "weighted_scores.json"), 'r') as f:
                scores = json.load(f)
            with open(os.path.join(self.results_dir, "feedback_report.json"), 'r') as f:
                report = json.load(f)

            logger.info("\n" + "=" * 60)
            logger.info("📊 RESULTS SUMMARY")
            logger.info("=" * 60)

            summary = report['executive_summary']
            logger.info(f"\nCandidate: {report['metadata']['candidate_name']}")
            logger.info(f"Position: {report['metadata']['position']}")
            logger.info(f"\nOverall Score: {summary['overall_score']:.1f}/100")
            logger.info(f"Grade: {summary['grade']}")
            logger.info(f"Category: {summary['performance_category']}")

            logger.info(f"\n💪 Top Strengths:")
            for strength in scores['top_strengths']:
                logger.info(f"   • {strength['dimension']}: {strength['score']:.2f}")

            logger.info(f"\n📈 Areas to Improve:")
            for area in scores['improvement_areas']:
                logger.info(f"   • {area['dimension']}: {area['score']:.2f}")

            logger.info(f"\n💡 Recommendation:")
            logger.info(f"   {summary['recommendation']}")

        except Exception as e:
            logger.warning(f"Could not display summary: {e}")


def main():
    parser = argparse.ArgumentParser(description='Process interview session for feedback')
    parser.add_argument('session_id', help='Session ID to process')
    parser.add_argument('--candidate', default='Candidate', help='Candidate name')
    parser.add_argument('--position', default='Position', help='Position title')
    parser.add_argument('--role', default=None, help='Role type (e.g., software_engineer)')
    parser.add_argument('--experience', default='mid', choices=['entry', 'mid', 'senior'], help='Experience level')
    parser.add_argument('--skip-video', action='store_true', help='Skip video processing')
    parser.add_argument('--video-fps', type=int, default=10, help='FPS for video processing')

    args = parser.parse_args()

    pipeline = FeedbackPipeline(
        args.session_id,
        skip_video=args.skip_video,
        video_fps=args.video_fps
    )

    success = pipeline.run_complete_pipeline(
        candidate_name=args.candidate,
        position=args.position,
        role=args.role,
        experience=args.experience
    )

    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
