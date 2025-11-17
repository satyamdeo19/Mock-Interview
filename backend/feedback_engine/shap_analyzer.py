"""
SHAP Analyzer
Provides explainable AI insights using SHAP values
Input: final_multimodal_features.csv + model
Output: shap_analysis.json
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


class SHAPAnalyzer:
    """Generate SHAP explanations for model predictions"""

    def __init__(self, model_path: str = None, scaler_path: str = None):
        """Initialize SHAP analyzer with model and optional scaler"""

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
        # Load model
        # ------------------------------------------------------------------
        try:
            if not Path(model_path).exists():
                raise FileNotFoundError(f"Model file not found: {model_path}")

            self.model = joblib.load(model_path)
            logger.info(f"✅ Loaded model for SHAP analysis from: {model_path}")

            # Load scaler if exists
            if Path(scaler_path).exists():
                self.scaler = joblib.load(scaler_path)
                logger.info(f"✅ Loaded scaler from: {scaler_path}")
            else:
                self.scaler = None
                logger.warning(f"⚠️  Scaler not found at: {scaler_path}")

        except Exception as e:
            logger.error(f"❌ Error loading model or scaler: {e}")
            raise

    def analyze(self, features_csv: str, output_path: str = None) -> dict:
        """
        Generate SHAP analysis
        """
        logger.info(f"🔍 Starting SHAP analysis from: {features_csv}")

        # Load features
        features_df = pd.read_csv(features_csv)

        # Extract feature columns
        feature_cols = [col for col in features_df.columns if col != 'final_score']
        X = features_df[feature_cols].values

        # Scale if needed
        if self.scaler is not None:
            X_scaled = self.scaler.transform(X)
        else:
            X_scaled = X

        # Try to import SHAP
        try:
            import shap
            shap_available = True
        except ImportError:
            logger.warning("⚠️  SHAP not installed. Install with: pip install shap")
            shap_available = False

        if shap_available:
            try:
                # Create appropriate explainer
                explainer, shap_values = self._create_shap_explainer(X_scaled)

                # Calculate SHAP analysis
                result = self._calculate_shap_metrics(
                    shap_values,
                    explainer,
                    feature_cols,
                    features_df
                )

            except Exception as e:
                logger.warning(f"⚠️  SHAP analysis failed: {e}")
                logger.info("   Falling back to basic feature importance...")
                result = self._fallback_analysis(features_df)
        else:
            result = self._fallback_analysis(features_df)

        logger.info(f"✅ SHAP analysis complete")

        # Save if output path provided
        if output_path:
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, 'w') as f:
                json.dump(result, f, indent=2)
            logger.info(f"💾 Saved to: {output_path}")

        return result

    def _create_shap_explainer(self, X: np.ndarray):
        """Create appropriate SHAP explainer based on model type"""
        import shap

        model_type = type(self.model).__name__
        logger.info(f"   Model type: {model_type}")

        # Tree-based models
        if 'Forest' in model_type or 'XGB' in model_type or 'LightGBM' in model_type:
            logger.info("   Using TreeExplainer")
            explainer = shap.TreeExplainer(self.model)
            shap_values = explainer.shap_values(X)

        # Linear models
        elif 'Linear' in model_type or 'Logistic' in model_type:
            logger.info("   Using LinearExplainer")
            explainer = shap.LinearExplainer(self.model, X)
            shap_values = explainer.shap_values(X)

        # Default: KernelExplainer (slower but general)
        else:
            logger.info("   Using KernelExplainer (this may take a while...)")
            explainer = shap.KernelExplainer(
                self.model.predict,
                shap.sample(X, 100)
            )
            shap_values = explainer.shap_values(X)

        return explainer, shap_values

    def _calculate_shap_metrics(self, shap_values, explainer,
                                feature_names, features_df) -> dict:
        """Calculate SHAP-based metrics"""

        # Handle multi-output vs single-output
        if isinstance(shap_values, list):
            shap_values_array = shap_values[1] if len(shap_values) > 1 else shap_values[0]
        else:
            shap_values_array = shap_values

        # Get SHAP values for the instance
        instance_shap = shap_values_array[0]

        # Global feature importance
        global_importance = {}
        for i, name in enumerate(feature_names):
            if i < len(instance_shap):
                global_importance[name] = float(np.abs(instance_shap[i]))

        # Sort by importance
        sorted_features = sorted(global_importance.items(), key=lambda x: x[1], reverse=True)

        # Top positive contributors
        top_positive = []
        for name, shap_val in sorted_features[:5]:
            if instance_shap[feature_names.index(name)] > 0:
                top_positive.append({
                    'feature': name,
                    'value': float(features_df[name].iloc[0]),
                    'shap_value': float(instance_shap[feature_names.index(name)]),
                    'impact': 'Positive - Increases score'
                })

        # Top negative contributors
        top_negative = []
        for name, _ in reversed(sorted_features[-5:]):
            if instance_shap[feature_names.index(name)] < 0:
                top_negative.append({
                    'feature': name,
                    'value': float(features_df[name].iloc[0]),
                    'shap_value': float(instance_shap[feature_names.index(name)]),
                    'impact': 'Negative - Decreases score'
                })

        # Base value and prediction
        base_value = float(explainer.expected_value) if hasattr(explainer, 'expected_value') else 0.7
        total_shap_contribution = float(np.sum(instance_shap))
        predicted_value = base_value + total_shap_contribution

        result = {
            'method': 'shap_analysis',
            'timestamp': datetime.now().isoformat(),
            'global_feature_importance': {
                name: round(imp, 4)
                for name, imp in sorted_features
            },
            'local_explanations': {
                'top_positive_features': top_positive,
                'top_negative_features': top_negative
            },
            'base_value': round(base_value * 100, 2),
            'predicted_value': round(predicted_value * 100, 2),
            'total_shap_contribution': round(total_shap_contribution * 100, 2),
            'interpretation': self._generate_interpretation(
                top_positive,
                top_negative
            )
        }

        return result

    def _fallback_analysis(self, features_df: pd.DataFrame) -> dict:
        """Fallback analysis when SHAP is not available"""
        feature_cols = [col for col in features_df.columns if col != 'final_score']

        # Assumed normal values
        normal_values = {
            'confidence': 0.72, 'fluency': 0.70, 'engagement': 0.75,
            'professionalism': 0.73, 'emotional_state': 0.68,
            'calmness': 0.70, 'articulation_quality': 0.72,
            'response_depth': 0.65, 'cognitive_complexity': 0.60,
            'consistency': 0.75
        }

        importance = {}
        positive_features = []
        negative_features = []

        for col in feature_cols:
            actual = float(features_df[col].iloc[0])
            normal = normal_values.get(col, 0.70)
            deviation = actual - normal

            importance[col] = abs(deviation)

            if deviation > 0.05:
                positive_features.append({
                    'feature': col,
                    'value': actual,
                    'shap_value': deviation,
                    'impact': f'Above average by {deviation:.3f}'
                })
            elif deviation < -0.05:
                negative_features.append({
                    'feature': col,
                    'value': actual,
                    'shap_value': deviation,
                    'impact': f'Below average by {abs(deviation):.3f}'
                })

        positive_features.sort(key=lambda x: x['shap_value'], reverse=True)
        negative_features.sort(key=lambda x: x['shap_value'])

        result = {
            'method': 'fallback_analysis',
            'note': 'SHAP not available, using deviation-based analysis',
            'timestamp': datetime.now().isoformat(),
            'global_feature_importance': importance,
            'local_explanations': {
                'top_positive_features': positive_features[:5],
                'top_negative_features': negative_features[:5]
            },
            'base_value': 70.0,
            'predicted_value': float(features_df.get('final_score', [0]).iloc[0])
            if 'final_score' in features_df.columns else 70.0,
            'interpretation': self._generate_interpretation(
                positive_features[:5],
                negative_features[:5]
            )
        }

        return result

    def _generate_interpretation(self, positive_features: list,
                                 negative_features: list) -> str:
        """Generate natural language interpretation"""
        interpretation = []

        if positive_features:
            top_strength = positive_features[0]['feature'].replace('_', ' ').title()
            interpretation.append(
                f"Your strongest area is {top_strength}, which significantly "
                f"boosts your overall performance."
            )

        if negative_features:
            top_weakness = negative_features[0]['feature'].replace('_', ' ').title()
            interpretation.append(
                f"The biggest opportunity for improvement is {top_weakness}, "
                f"which is currently holding back your score."
            )

        if len(positive_features) >= 2:
            second_strength = positive_features[1]['feature'].replace('_', ' ').title()
            interpretation.append(f"You also show strength in {second_strength}.")

        return " ".join(interpretation)


# ============================================
# CLI USAGE
# ============================================
if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python shap_analyzer.py <session_id>")
        print("Example: python shap_analyzer.py session_abc123_1234567890")
        sys.exit(1)

    session_id = sys.argv[1]

    # Paths
    features_csv = f"processed_features/{session_id}/final_multimodal_features.csv"
    output_path = f"feedback_results/{session_id}/shap_analysis.json"

    if not os.path.exists(features_csv):
        print(f"❌ Input file not found: {features_csv}")
        sys.exit(1)

    try:
        analyzer = SHAPAnalyzer()
        result = analyzer.analyze(features_csv, output_path)

        print("\n" + "=" * 60)
        print("✅ SHAP ANALYSIS COMPLETE")
        print("=" * 60)
        print(f"Input:  {features_csv}")
        print(f"Output: {output_path}")

        print(f"\n🔍 Top Positive Contributors:")
        for feat in result['local_explanations']['top_positive_features'][:3]:
            print(f"  • {feat['feature']}: {feat['impact']}")

        print(f"\n📉 Top Negative Contributors:")
        for feat in result['local_explanations']['top_negative_features'][:3]:
            print(f"  • {feat['feature']}: {feat['impact']}")

        print(f"\n💡 Interpretation:")
        print(f"  {result['interpretation']}")

    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
