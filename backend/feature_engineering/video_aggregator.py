"""
Video Aggregator
Aggregates frame-level video features into session-level composite scores
Input: video_features_raw.csv
Output: video_features_aggregated.csv
"""

import pandas as pd
import numpy as np
import logging
from pathlib import Path
from scipy import stats
import os

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class VideoAggregator:
    """Aggregate frame-level video features into composite scores"""
    
    def aggregate_features(self, raw_csv_path: str, output_path: str = None) -> pd.DataFrame:
        """
        Main aggregation function
        
        Args:
            raw_csv_path: Path to video_features_raw.csv
            output_path: Optional output path for video_features_aggregated.csv
            
        Returns:
            DataFrame with single row of aggregated features
        """
        logger.info(f"📊 Starting video feature aggregation from: {raw_csv_path}")
        
        # Load raw features
        df = pd.read_csv(raw_csv_path)
        logger.info(f"   Loaded {len(df)} frames")
        
        # Calculate all composite scores
        confidence_features = self._calculate_confidence_features(df)
        engagement_features = self._calculate_engagement_features(df)
        professionalism_features = self._calculate_professionalism_features(df)
        nervousness_features = self._calculate_nervousness_features(df)
        temporal_features = self._calculate_temporal_features(df)
        
        # Combine all features
        features = {
            **confidence_features,
            **engagement_features,
            **professionalism_features,
            **nervousness_features,
            **temporal_features
        }
        
        # Create DataFrame
        features_df = pd.DataFrame([features])
        
        logger.info(f"✅ Aggregated {len(features)} video features")
        
        # Save if output path provided
        if output_path:
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            features_df.to_csv(output_path, index=False)
            logger.info(f"💾 Saved to: {output_path}")
        
        return features_df
    
    def _calculate_confidence_features(self, df: pd.DataFrame) -> dict:
        """Calculate confidence-related features"""
        
        # Pitch stability: 1 / (1 + σ_Pitch)
        pitch_std = df['head_pitch'].std()
        pitch_stability = 1 / (1 + pitch_std)
        
        # Yaw stability
        yaw_std = df['head_yaw'].std()
        yaw_stability = 1 / (1 + yaw_std)
        
        # Eye contact stability
        eye_contact_stability = (pitch_stability + yaw_stability) / 2
        
        # Gaze consistency: 1 / (1 + Var(Pitch) + Var(Yaw))
        pitch_var = df['head_pitch'].var()
        yaw_var = df['head_yaw'].var()
        gaze_consistency = 1 / (1 + pitch_var + yaw_var)
        
        # Forward gaze tendency: max(0, 1 - |μ_Pitch| / 10)
        pitch_mean = abs(df['head_pitch'].mean())
        forward_gaze_tendency = max(0, 1 - pitch_mean / 10)
        
        # Head posture stability
        roll_std = df['head_roll'].std()
        head_posture_stability = 1 / (1 + roll_std)
        
        # Upright posture score: max(0, 1 - |μ_Roll| / 5)
        roll_mean = abs(df['head_roll'].mean())
        upright_posture_score = max(0, 1 - roll_mean / 5)
        
        # Confidence composite score
        confidence_composite = np.mean([
            pitch_stability, yaw_stability, eye_contact_stability,
            gaze_consistency, forward_gaze_tendency, 
            head_posture_stability, upright_posture_score
        ])
        
        return {
            'confidence_pitch_stability': round(pitch_stability, 4),
            'confidence_yaw_stability': round(yaw_stability, 4),
            'confidence_eye_contact_stability': round(eye_contact_stability, 4),
            'confidence_gaze_consistency': round(gaze_consistency, 4),
            'confidence_forward_gaze_tendency': round(forward_gaze_tendency, 4),
            'confidence_head_posture_stability': round(head_posture_stability, 4),
            'confidence_upright_posture_score': round(upright_posture_score, 4),
            'confidence_composite_score': round(confidence_composite, 4)
        }
    
    def _calculate_engagement_features(self, df: pd.DataFrame) -> dict:
        """Calculate engagement-related features"""
        
        # Eyebrow features
        eyebrow_cols = ['inBrL', 'otBrL', 'inBrR', 'otBrR']
        eyebrow_stds = [df[col].std() for col in eyebrow_cols]
        eyebrow_expressiveness = min(np.mean(eyebrow_stds), 2.0)
        
        # Eyebrow animation frequency
        eyebrow_changes = []
        for col in eyebrow_cols:
            diffs = np.abs(df[col].diff().dropna())
            rapid_changes = (diffs > 0.1).sum() / len(df)
            eyebrow_changes.append(rapid_changes)
        eyebrow_animation_frequency = np.mean(eyebrow_changes)
        
        # Eye expressiveness
        eye_std = (df['EyeOL'].std() + df['EyeOR'].std()) / 2
        eye_expressiveness = min(0.5 * eye_std, 1.5)
        
        # Mouth expressiveness
        mouth_cols = ['oLipH', 'iLipH', 'LipCDt']
        mouth_stds = [df[col].std() for col in mouth_cols]
        mouth_expressiveness = min(np.mean(mouth_stds), 2.0)
        
        # Mouth animation frequency
        mouth_changes = []
        for col in mouth_cols:
            diffs = np.abs(df[col].diff().dropna())
            rapid_changes = (diffs > 0.1).sum() / len(df)
            mouth_changes.append(rapid_changes)
        mouth_animation_frequency = np.mean(mouth_changes)
        
        # Overall facial animation
        all_feature_cols = eyebrow_cols + ['EyeOL', 'EyeOR'] + mouth_cols
        all_stds = [df[col].std() for col in all_feature_cols]
        overall_facial_animation = min(np.mean(all_stds), 1.5)
        
        # Expression dynamic range
        dynamic_ranges = []
        for col in all_feature_cols:
            dynamic_ranges.append(df[col].max() - df[col].min())
        expression_dynamic_range = np.mean(dynamic_ranges)
        
        # Engagement composite score
        engagement_composite = np.mean([
            eyebrow_expressiveness / 2.0,  # Normalize to 0-1
            eyebrow_animation_frequency,
            eye_expressiveness / 1.5,
            mouth_expressiveness / 2.0,
            mouth_animation_frequency,
            overall_facial_animation / 1.5,
            expression_dynamic_range
        ])
        
        return {
            'engagement_eyebrow_expressiveness': round(eyebrow_expressiveness, 4),
            'engagement_eyebrow_animation_frequency': round(eyebrow_animation_frequency, 4),
            'engagement_eye_expressiveness': round(eye_expressiveness, 4),
            'engagement_mouth_expressiveness': round(mouth_expressiveness, 4),
            'engagement_mouth_animation_frequency': round(mouth_animation_frequency, 4),
            'engagement_overall_facial_animation': round(overall_facial_animation, 4),
            'engagement_expression_dynamic_range': round(expression_dynamic_range, 4),
            'engagement_composite_score': round(engagement_composite, 4)
        }
    
    def _calculate_professionalism_features(self, df: pd.DataFrame) -> dict:
        """Calculate professionalism-related features"""
        
        # Pitch smoothness: 1 / (1 + mean(|Δ²Pitch|))
        pitch_second_diff = np.abs(df['head_pitch'].diff().diff().dropna())
        pitch_smoothness = 1 / (1 + pitch_second_diff.mean())
        
        # Yaw smoothness
        yaw_second_diff = np.abs(df['head_yaw'].diff().diff().dropna())
        yaw_smoothness = 1 / (1 + yaw_second_diff.mean())
        
        # Gaze steadiness
        gaze_steadiness = (pitch_smoothness + yaw_smoothness) / 2
        
        # Gaze range
        pitch_range = df['head_pitch'].max() - df['head_pitch'].min()
        yaw_range = df['head_yaw'].max() - df['head_yaw'].min()
        gaze_range = np.sqrt(pitch_range**2 + yaw_range**2)
        
        # Appropriate gaze range: max(0, 1 - gaze_range / 20)
        appropriate_gaze_range = max(0, 1 - gaze_range / 20)
        
        # Controlled head movement: max(0, 1 - σ_Roll / 2)
        roll_std = df['head_roll'].std()
        controlled_head_movement = max(0, 1 - roll_std / 2)
        
        # Professional head position: max(0, 1 - |μ_Roll| / 3)
        roll_mean = abs(df['head_roll'].mean())
        professional_head_position = max(0, 1 - roll_mean / 3)
        
        # Overall posture stability
        pose_cols = ['head_pitch', 'head_yaw', 'head_roll']
        pose_stabilities = [1 / (1 + df[col].std()) for col in pose_cols]
        overall_posture_stability = np.mean(pose_stabilities)
        
        # Expression control (moderate eyebrow movement)
        eyebrow_cols = ['inBrL', 'otBrL', 'inBrR', 'otBrR']
        eyebrow_std = np.mean([df[col].std() for col in eyebrow_cols])
        expression_control = max(0, 1 - abs(eyebrow_std - 0.5) / 0.5)
        
        # Professionalism composite
        professionalism_composite = np.mean([
            pitch_smoothness, yaw_smoothness, gaze_steadiness,
            appropriate_gaze_range, controlled_head_movement,
            professional_head_position, overall_posture_stability,
            expression_control
        ])
        
        return {
            'professionalism_pitch_smoothness': round(pitch_smoothness, 4),
            'professionalism_yaw_smoothness': round(yaw_smoothness, 4),
            'professionalism_gaze_steadiness': round(gaze_steadiness, 4),
            'professionalism_gaze_range': round(gaze_range, 4),
            'professionalism_appropriate_gaze_range': round(appropriate_gaze_range, 4),
            'professionalism_controlled_head_movement': round(controlled_head_movement, 4),
            'professionalism_professional_head_position': round(professional_head_position, 4),
            'professionalism_overall_posture_stability': round(overall_posture_stability, 4),
            'professionalism_expression_control': round(expression_control, 4),
            'professionalism_composite_score': round(professionalism_composite, 4)
        }
    
    def _calculate_nervousness_features(self, df: pd.DataFrame) -> dict:
        """Calculate nervousness-related features (inverted for calmness)"""
        
        # Eyebrow micro-movements
        eyebrow_cols = ['inBrL', 'otBrL', 'inBrR', 'otBrR']
        eyebrow_micro = []
        for col in eyebrow_cols:
            diffs = np.abs(df[col].diff().dropna())
            micro_movements = ((diffs > 0.05) & (diffs < 0.2)).sum() / len(df)
            eyebrow_micro.append(micro_movements)
        eyebrow_micro_movements = np.mean(eyebrow_micro)
        
        # Eyebrow tension level
        eyebrow_tension_diffs = [df[col].diff().std() for col in eyebrow_cols]
        eyebrow_tension_level = min(np.mean(eyebrow_tension_diffs), 1.0)
        
        # Facial asymmetry stress
        facial_asymmetry = np.abs(df['inBrL'] - df['inBrR']).mean()
        
        # Asymmetry variation
        asymmetry_variation = np.abs(df['inBrL'] - df['inBrR']).std()
        
        # Eye asymmetry
        eye_asymmetry = np.abs(df['EyeOL'] - df['EyeOR']).mean()
        
        # Lip tension
        lip_tension = np.abs(df['oLipH'] - df['iLipH']).mean()
        lip_tension_variability = np.abs(df['oLipH'] - df['iLipH']).std()
        
        # Lip micro-movements
        lip_diffs = np.abs(df['oLipH'].diff().dropna()) + np.abs(df['iLipH'].diff().dropna())
        lip_micro_movements = (lip_diffs > 0.1).sum() / len(df)
        
        # Overall nervous frequency
        all_feature_cols = eyebrow_cols + ['EyeOL', 'EyeOR', 'oLipH', 'iLipH']
        total_rapid = 0
        for col in all_feature_cols:
            diffs = np.abs(df[col].diff().dropna())
            total_rapid += (diffs > 0.15).sum()
        overall_nervous_frequency = total_rapid / (len(df) * len(all_feature_cols))
        
        # Nervousness composite (invert: high nervousness = low calmness)
        nervousness_score = np.mean([
            eyebrow_micro_movements,
            eyebrow_tension_level,
            facial_asymmetry * 10,  # Scale up
            asymmetry_variation * 10,
            eye_asymmetry * 10,
            lip_tension * 5,
            lip_tension_variability * 5,
            lip_micro_movements,
            overall_nervous_frequency
        ])
        
        nervousness_composite = 1 - min(nervousness_score, 1.0)  # Invert
        
        return {
            'nervousness_eyebrow_micro_movements': round(eyebrow_micro_movements, 4),
            'nervousness_eyebrow_tension_level': round(eyebrow_tension_level, 4),
            'nervousness_facial_asymmetry_stress': round(facial_asymmetry, 4),
            'nervousness_asymmetry_variation': round(asymmetry_variation, 4),
            'nervousness_eye_asymmetry_stress': round(eye_asymmetry, 4),
            'nervousness_lip_tension_level': round(lip_tension, 4),
            'nervousness_lip_tension_variability': round(lip_tension_variability, 4),
            'nervousness_lip_micro_movements': round(lip_micro_movements, 4),
            'nervousness_overall_nervous_frequency': round(overall_nervous_frequency, 4),
            'nervousness_composite_score': round(nervousness_composite, 4)
        }
    
    def _calculate_temporal_features(self, df: pd.DataFrame) -> dict:
        """Calculate temporal consistency features"""
        
        # Divide into 3 phases
        n = len(df)
        phase1 = df.iloc[:n//3]
        phase2 = df.iloc[n//3:2*n//3]
        phase3 = df.iloc[2*n//3:]
        
        # Calculate engagement for each phase
        def phase_engagement(phase_df):
            eyebrow_cols = ['inBrL', 'otBrL', 'inBrR', 'otBrR']
            stds = [phase_df[col].std() for col in eyebrow_cols]
            return np.mean(stds)
        
        E_opening = phase_engagement(phase1)
        E_middle = phase_engagement(phase2)
        E_closing = phase_engagement(phase3)
        
        # Engagement consistency
        engagement_consistency = 1 / (1 + np.std([E_opening, E_middle, E_closing]))
        
        # Calculate stability for each phase
        def phase_stability(phase_df):
            pose_cols = ['head_pitch', 'head_yaw', 'head_roll']
            stabilities = [1 / (1 + phase_df[col].std()) for col in pose_cols]
            return np.mean(stabilities)
        
        S_opening = phase_stability(phase1)
        S_middle = phase_stability(phase2)
        S_closing = phase_stability(phase3)
        
        # Stability consistency
        stability_consistency = 1 / (1 + np.std([S_opening, S_middle, S_closing]))
        
        # Improvement trends
        engagement_improvement_trend = E_closing - E_opening
        stability_improvement_trend = S_closing - S_opening
        
        # Overall temporal stability
        all_feature_cols = ['inBrL', 'otBrL', 'inBrR', 'otBrR', 
                           'EyeOL', 'EyeOR', 'oLipH', 'iLipH']
        temporal_stds = [df[col].std() for col in all_feature_cols]
        overall_temporal_stability = 1 / (1 + np.mean(temporal_stds))
        
        # Temporal consistency composite
        temporal_composite = np.mean([
            engagement_consistency,
            stability_consistency,
            overall_temporal_stability
        ])
        
        return {
            'temporal_engagement_consistency': round(engagement_consistency, 4),
            'temporal_stability_consistency': round(stability_consistency, 4),
            'temporal_engagement_improvement_trend': round(engagement_improvement_trend, 4),
            'temporal_stability_improvement_trend': round(stability_improvement_trend, 4),
            'temporal_overall_temporal_stability': round(overall_temporal_stability, 4),
            'temporal_consistency_composite_score': round(temporal_composite, 4)
        }


# ============================================
# CLI USAGE
# ============================================
if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python video_aggregator.py <session_id>")
        print("Example: python video_aggregator.py session_abc123_1234567890")
        sys.exit(1)
    
    session_id = sys.argv[1]
    
    # Paths
    input_path = f"processed_features/{session_id}/video_features_raw.csv"
    output_path = f"processed_features/{session_id}/video_features_aggregated.csv"
    
    # Check if input exists
    if not os.path.exists(input_path):
        print(f"❌ Input file not found: {input_path}")
        print("   Run video_extractor.py first!")
        sys.exit(1)
    
    # Aggregate features
    aggregator = VideoAggregator()
    
    try:
        features_df = aggregator.aggregate_features(input_path, output_path)
        
        print("\n" + "="*60)
        print("✅ VIDEO FEATURE AGGREGATION COMPLETE")
        print("="*60)
        print(f"Input:  {input_path}")
        print(f"Output: {output_path}")
        print(f"Features aggregated: {len(features_df.columns)}")
        print("\n📊 Composite scores:")
        composite_cols = [col for col in features_df.columns if 'composite_score' in col]
        print(features_df[composite_cols].T)
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)