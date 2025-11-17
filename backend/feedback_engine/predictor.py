"""
Model Predictor
Uses trained ML model for predictions
Input: final_multimodal_features.csv
Output: prediction_results.json
"""

import pandas as pd
import numpy as np
import json
import logging
import joblib
import os
from pathlib import Path
from datetime import datetime

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ModelPredictor:
    """ML model-based predictor for interview performance"""

    def __init__(self,
                 model_path: str = None,
                 scaler_path: str = None):
        """
        Initialize predictor with model and scaler

        Args:
            model_path: Path to trained model (optional)
            scaler_path: Path to feature scaler (optional)
        """
        # ------------------------------------------------------------------
        # Dynamically resolve absolute paths for model and scaler
        # ------------------------------------------------------------------
        backend_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        models_dir = os.path.join(backend_dir, "models")

        if model_path is None:
            model_path = os.path.join(models_dir, "human_aligned_model.joblib")

        if scaler_path is None:
            scaler_path = os.path.join(models_dir, "scaler.joblib")

        self.model_path = model_path
        self.scaler_path = scaler_path

        # ------------------------------------------------------------------
        # Load model and scaler safely
        # ------------------------------------------------------------------
        try:
            if not Path(model_path).exists():
                raise FileNotFoundError(f"Model file not found: {model_path}")

            self.model = joblib.load(model_path)
            logger.info(f"✅ Loaded model from: {model_path}")

            if Path(scaler_path).exists():
                self.scaler = joblib.load(scaler_path)
                logger.info(f"✅ Loaded scaler from: {scaler_path}")
            else:
                self.scaler = None
                logger.warning(f"⚠️  Scaler not found at: {scaler_path}")

        except Exception as e:
            logger.error(f"❌ Error loading model or scaler: {e}")
            raise

    def predict(self, features_csv: str, output_path: str = None) -> dict:
        """
        Generate predictions using ML model
        """
        logger.info(f"🤖 Generating model predictions from: {features_csv}")

        # Load features
        features_df = pd.read_csv(features_csv)
        logger.info(f"   Loaded features: {features_df.shape}")

        # Extract feature columns (exclude final_score if present)
        feature_cols = [col for col in features_df.columns if col != 'final_score']
        X = features_df[feature_cols].values

        # Scale features if scaler exists
        if self.scaler is not None:
            X_scaled = self.scaler.transform(X)
            logger.info(f"   Features scaled")
        else:
            X_scaled = X

        # Generate predictions
        try:
            predictions = self.model.predict(X_scaled)

            # Determine prediction type
            if predictions.ndim == 1:
                prediction_type = "single_output"
                final_score = float(predictions[0])

                # Use feature columns as dimension scores (fallback)
                dimension_scores = {
                    'confidence': float(features_df['confidence'].iloc[0]),
                    'fluency': float(features_df['fluency'].iloc[0]),
                    'engagement': float(features_df['engagement'].iloc[0]),
                    'professionalism': float(features_df['professionalism'].iloc[0]),
                    'emotional_state': float(features_df['emotional_state'].iloc[0]),
                    'calmness': float(features_df['calmness'].iloc[0]),
                    'articulation_quality': float(features_df['articulation_quality'].iloc[0]),
                    'response_depth': float(features_df['response_depth'].iloc[0]),
                    'cognitive_complexity': float(features_df['cognitive_complexity'].iloc[0]),
                    'consistency': float(features_df['consistency'].iloc[0])
                }

            elif predictions.shape[1] == 10:
                prediction_type = "multi_output"
                dimension_names = [
                    'confidence', 'fluency', 'engagement', 'professionalism',
                    'emotional_state', 'calmness', 'articulation_quality',
                    'response_depth', 'cognitive_complexity', 'consistency'
                ]

                dimension_scores = {
                    name: float(predictions[0, i])
                    for i, name in enumerate(dimension_names)
                }

                # Weighted final score
                weights = {
                    'confidence': 0.15, 'fluency': 0.15, 'engagement': 0.15,
                    'professionalism': 0.15, 'emotional_state': 0.10,
                    'calmness': 0.10, 'articulation_quality': 0.08,
                    'response_depth': 0.05, 'cognitive_complexity': 0.04,
                    'consistency': 0.03
                }

                final_score = sum(
                    dimension_scores[dim] * weights[dim]
                    for dim in dimension_names
                ) * 100

            else:
                raise ValueError(f"Unexpected prediction shape: {predictions.shape}")

            logger.info(f"   Prediction type: {prediction_type}")
            logger.info(f"   Final score: {final_score:.2f}")

        except Exception as e:
            logger.error(f"❌ Prediction failed: {e}")
            raise

        # Confidence intervals
        confidence_intervals = self._calculate_confidence_intervals(
            X_scaled, dimension_scores
        )

        # Grade and percentile
        grade = self._calculate_grade(final_score)
        percentile = self._calculate_percentile(final_score)

        # Results
        result = {
            'method': 'model_based',
            'model_path': self.model_path,
            'model_type': type(self.model).__name__,
            'prediction_type': prediction_type,
            'timestamp': datetime.now().isoformat(),
            'predictions': dimension_scores,
            'final_score': round(final_score, 2),
            'confidence_intervals': confidence_intervals,
            'overall_grade': grade,
            'percentile_rank': percentile,
            'feature_importance': self._get_feature_importance()
        }

        logger.info(f"✅ Predictions generated successfully")

        # Save if path provided
        if output_path:
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, 'w') as f:
                json.dump(result, f, indent=2)
            logger.info(f"💾 Saved to: {output_path}")

        return result

    def _calculate_confidence_intervals(self, X: np.ndarray, predictions: dict) -> dict:
        """Calculate confidence intervals for predictions"""
        if hasattr(self.model, 'predict_proba'):
            try:
                proba = self.model.predict_proba(X)
                confidence = float(np.max(proba))
                intervals = {}
                for dim, score in predictions.items():
                    margin = (1 - confidence) * 0.15
                    intervals[dim] = [
                        round(max(score - margin, 0), 3),
                        round(min(score + margin, 1), 3)
                    ]
                return intervals
            except:
                pass

        # Default ±5%
        return {
            dim: [
                round(max(score - 0.05, 0), 3),
                round(min(score + 0.05, 1), 3)
            ]
            for dim, score in predictions.items()
        }

    def _get_feature_importance(self) -> dict:
        """Get feature importance from model if available"""
        try:
            if hasattr(self.model, 'feature_importances_'):
                importances = self.model.feature_importances_
                dimension_names = [
                    'confidence', 'fluency', 'engagement', 'professionalism',
                    'emotional_state', 'calmness', 'articulation_quality',
                    'response_depth', 'cognitive_complexity', 'consistency'
                ]
                return {
                    name: round(float(imp), 4)
                    for name, imp in zip(dimension_names, importances[:10])
                }

            elif hasattr(self.model, 'coef_'):
                coefs = np.abs(self.model.coef_[0])
                dimension_names = [
                    'confidence', 'fluency', 'engagement', 'professionalism',
                    'emotional_state', 'calmness', 'articulation_quality',
                    'response_depth', 'cognitive_complexity', 'consistency'
                ]
                return {
                    name: round(float(coef), 4)
                    for name, coef in zip(dimension_names, coefs[:10])
                }
        except:
            pass
        return {}

    def _calculate_grade(self, score: float) -> str:
        """Convert score to letter grade"""
        if score >= 90:
            return "A+"
        elif score >= 85:
            return "A"
        elif score >= 80:
            return "A-"
        elif score >= 75:
            return "B+"
        elif score >= 70:
            return "B"
        elif score >= 65:
            return "B-"
        elif score >= 60:
            return "C+"
        else:
            return "C"

    def _calculate_percentile(self, score: float) -> float:
        """Estimate percentile rank (simplified)"""
        from scipy import stats
        percentile = stats.norm.cdf(score, loc=70, scale=10) * 100
        return round(percentile, 1)


# ============================================
# CLI USAGE
# ============================================
if __name__ == "__main__":
    import sys
    import os

    if len(sys.argv) < 2:
        print("Usage: python predictor.py <session_id>")
        print("Example: python predictor.py session_abc123_1234567890")
        sys.exit(1)

    session_id = sys.argv[1]

    features_csv = f"processed_features/{session_id}/final_multimodal_features.csv"
    output_path = f"feedback_results/{session_id}/prediction_results.json"

    if not os.path.exists(features_csv):
        print(f"❌ Input file not found: {features_csv}")
        print("   Run multimodal_fusion.py first!")
        sys.exit(1)

    try:
        predictor = ModelPredictor()
        result = predictor.predict(features_csv, output_path)

        print("\n" + "=" * 60)
        print("✅ MODEL PREDICTIONS COMPLETE")
        print("=" * 60)
        print(f"Input:  {features_csv}")
        print(f"Output: {output_path}")
        print(f"\n📊 Results:")
        print(f"  Model Type: {result['model_type']}")
        print(f"  Final Score: {result['final_score']}/100")
        print(f"  Grade: {result['overall_grade']}")
        print(f"  Percentile: {result['percentile_rank']}")

        print(f"\n🎯 Dimension Predictions:")
        for dim, score in result['predictions'].items():
            print(f"  {dim:.<30} {score:.3f}")

    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
