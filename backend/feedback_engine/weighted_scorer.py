"""
Weighted Scorer
Generates interpretable scores using predefined weights (rule-based approach)
Input: final_multimodal_features.csv
Output: weighted_scores.json
"""

import os
import pandas as pd
import numpy as np
import json
import logging
from pathlib import Path
from datetime import datetime

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class WeightedScorer:
    """Generate weighted scores and interpretable feedback"""
    
    def __init__(self, weights_path: str = None, benchmark_path: str = None):
        """Load configuration"""
        
        # Get backend directory for absolute paths
        backend_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        
        # Set default paths if not provided
        if weights_path is None:
            weights_path = os.path.join(backend_dir, "feature_weights.json")
        if benchmark_path is None:
            benchmark_path = os.path.join(backend_dir, "benchmark_data.json")
        
        # Load weights
        try:
            with open(weights_path, 'r') as f:
                config = json.load(f)
            self.dimension_weights = config['dimension_weights']
            self.modality_weights = config['modality_weights']
            logger.info(f"✅ Loaded weights from: {weights_path}")
        except FileNotFoundError:
            logger.warning(f"⚠️  Weights file not found: {weights_path}")
            logger.warning("   Using default dimension weights")
            self.dimension_weights = self._get_default_dimension_weights()
            self.modality_weights = {}
        
        # Load benchmarks
        try:
            with open(benchmark_path, 'r') as f:
                self.benchmarks = json.load(f)
            logger.info(f"✅ Loaded benchmarks from: {benchmark_path}")
        except FileNotFoundError:
            logger.warning(f"⚠️  Benchmark file not found: {benchmark_path}")
            logger.warning("   Using default benchmarks")
            self.benchmarks = self._get_default_benchmarks()
    
    def _get_default_dimension_weights(self) -> dict:
        """Default dimension weights"""
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
    
    def _get_default_benchmarks(self) -> dict:
        """Default benchmark data"""
        return {
            'industry_averages': {
                'confidence': 0.70,
                'fluency': 0.65,
                'engagement': 0.68,
                'professionalism': 0.72,
                'emotional_state': 0.65,
                'calmness': 0.70,
                'articulation_quality': 0.68,
                'response_depth': 0.65,
                'cognitive_complexity': 0.62,
                'consistency': 0.70
            }
        }
    
    def generate_scores(self, features_csv: str, output_path: str = None,
                       role: str = None, experience: str = "mid") -> dict:
        """
        Generate weighted scores
        
        Args:
            features_csv: Path to final_multimodal_features.csv
            output_path: Optional output path for weighted_scores.json
            role: Optional role for adjusted benchmarks (e.g., 'software_engineer')
            experience: Experience level ('entry', 'mid', 'senior')
            
        Returns:
            Dictionary with scores and interpretations
        """
        logger.info(f"🎯 Generating weighted scores from: {features_csv}")
        
        # Load features
        features_df = pd.read_csv(features_csv)
        features = features_df.iloc[0].to_dict()
        
        # Calculate scores for each dimension
        dimension_scores = {}
        total_contribution = 0
        
        dimensions = [
            'confidence', 'fluency', 'engagement', 'professionalism',
            'emotional_state', 'calmness', 'articulation_quality',
            'response_depth', 'cognitive_complexity', 'consistency'
        ]
        
        for dim in dimensions:
            score = features.get(dim, 0.5)
            weight = self.dimension_weights.get(dim, 0.1)
            contribution = score * weight * 100
            
            # Get benchmark
            benchmark = self.benchmarks.get('industry_averages', {}).get(dim, 0.7)
            
            # Adjust benchmark for role if specified
            if role and role in self.benchmarks.get('role_specific', {}):
                role_data = self.benchmarks['role_specific'][role]
                if dim in role_data.get('benchmarks', {}):
                    benchmark = role_data['benchmarks'][dim]
            
            # Calculate percentile (simplified)
            if score > benchmark:
                percentile = 50 + (score - benchmark) / (1 - benchmark) * 50
            else:
                percentile = (score / benchmark) * 50
            percentile = min(max(percentile, 0), 100)
            
            dimension_scores[dim] = {
                'score': round(score, 3),
                'weight': weight,
                'contribution': round(contribution, 2),
                'benchmark': round(benchmark, 3),
                'difference': round(score - benchmark, 3),
                'percentile': round(percentile, 1),
                'interpretation': self._interpret_score(score, benchmark)
            }
            
            total_contribution += contribution
        
        # Final score
        final_score = round(total_contribution, 2)
        
        # Overall grade
        grade = self._calculate_grade(final_score)
        
        # Performance category
        performance = self._categorize_performance(final_score)
        
        # Create result
        result = {
            'method': 'weighted_scoring',
            'timestamp': datetime.now().isoformat(),
            'role': role,
            'experience_level': experience,
            'dimension_scores': dimension_scores,
            'final_score': final_score,
            'overall_grade': grade,
            'performance_category': performance,
            'top_strengths': self._identify_strengths(dimension_scores, top_n=3),
            'improvement_areas': self._identify_weaknesses(dimension_scores, top_n=3),
            'summary': self._generate_summary(dimension_scores, final_score)
        }
        
        logger.info(f"✅ Generated scores - Final: {final_score} ({grade})")
        
        # Save if output path provided
        if output_path:
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, 'w') as f:
                json.dump(result, f, indent=2)
            logger.info(f"💾 Saved to: {output_path}")
        
        return result
    
    def _interpret_score(self, score: float, benchmark: float) -> str:
        """Interpret individual dimension score"""
        diff = score - benchmark
        
        if diff >= 0.15:
            return "Excellent - Well above average"
        elif diff >= 0.08:
            return "Very Good - Above average"
        elif diff >= 0:
            return "Good - At or slightly above average"
        elif diff >= -0.08:
            return "Average - Meets expectations"
        elif diff >= -0.15:
            return "Below Average - Room for improvement"
        else:
            return "Needs Improvement - Significant gap"
    
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
        elif score >= 55:
            return "C"
        else:
            return "C-"
    
    def _categorize_performance(self, score: float) -> str:
        """Categorize overall performance"""
        if score >= 85:
            return "Exceptional"
        elif score >= 80:
            return "Excellent"
        elif score >= 75:
            return "Very Good"
        elif score >= 70:
            return "Good"
        elif score >= 65:
            return "Above Average"
        elif score >= 60:
            return "Average"
        else:
            return "Below Average"
    
    def _identify_strengths(self, dimension_scores: dict, top_n: int = 3) -> list:
        """Identify top strengths"""
        # Sort by score
        sorted_dims = sorted(
            dimension_scores.items(),
            key=lambda x: x[1]['score'],
            reverse=True
        )
        
        strengths = []
        for dim, data in sorted_dims[:top_n]:
            strengths.append({
                'dimension': dim.replace('_', ' ').title(),
                'score': data['score'],
                'percentile': data['percentile'],
                'insight': self._get_strength_insight(dim, data['score'])
            })
        
        return strengths
    
    def _identify_weaknesses(self, dimension_scores: dict, top_n: int = 3) -> list:
        """Identify areas for improvement"""
        # Sort by score (ascending)
        sorted_dims = sorted(
            dimension_scores.items(),
            key=lambda x: x[1]['score']
        )
        
        weaknesses = []
        for dim, data in sorted_dims[:top_n]:
            if data['score'] < 0.7:  # Only if actually weak
                weaknesses.append({
                    'dimension': dim.replace('_', ' ').title(),
                    'score': data['score'],
                    'benchmark': data['benchmark'],
                    'gap': data['difference'],
                    'recommendation': self._get_improvement_recommendation(dim, data['score'])
                })
        
        return weaknesses
    
    def _get_strength_insight(self, dimension: str, score: float) -> str:
        """Generate insight for strength"""
        insights = {
            'confidence': f"Strong self-assurance and steady presence (score: {score:.2f})",
            'fluency': f"Smooth and articulate communication without hesitation (score: {score:.2f})",
            'engagement': f"Highly expressive and engaging presentation style (score: {score:.2f})",
            'professionalism': f"Maintains excellent professional demeanor throughout (score: {score:.2f})",
            'emotional_state': f"Positive and balanced emotional tone (score: {score:.2f})",
            'calmness': f"Composed and collected under pressure (score: {score:.2f})",
            'articulation_quality': f"Clear and well-articulated responses (score: {score:.2f})",
            'response_depth': f"Provides comprehensive and thoughtful answers (score: {score:.2f})",
            'cognitive_complexity': f"Demonstrates sophisticated thinking patterns (score: {score:.2f})",
            'consistency': f"Maintains steady performance throughout interview (score: {score:.2f})"
        }
        return insights.get(dimension, f"Performs well in this area (score: {score:.2f})")
    
    def _get_improvement_recommendation(self, dimension: str, score: float) -> str:
        """Generate recommendation for improvement"""
        recommendations = {
            'confidence': "Practice power poses and positive self-talk before interviews. Record yourself to build awareness.",
            'fluency': "Reduce filler words by pausing instead. Practice answering common questions smoothly.",
            'engagement': "Use more facial expressions and gestures. Vary your vocal tone for emphasis.",
            'professionalism': "Maintain steady eye contact and posture. Practice in professional settings.",
            'emotional_state': "Focus on positive framing of experiences. Practice stress management techniques.",
            'calmness': "Try breathing exercises before interviews. Practice mindfulness to reduce nervousness.",
            'articulation_quality': "Slow down your speech slightly. Practice enunciating clearly.",
            'response_depth': "Use the STAR method for behavioral questions. Prepare more detailed examples.",
            'cognitive_complexity': "Expand your vocabulary. Practice explaining complex concepts simply.",
            'consistency': "Maintain energy throughout by taking brief mental breaks. Stay hydrated."
        }
        return recommendations.get(dimension, "Practice and preparation will help improve this area.")
    
    def _generate_summary(self, dimension_scores: dict, final_score: float) -> str:
        """Generate overall summary"""
        # Count dimensions by performance level
        excellent = sum(1 for d in dimension_scores.values() if d['score'] >= 0.85)
        good = sum(1 for d in dimension_scores.values() if 0.70 <= d['score'] < 0.85)
        needs_work = sum(1 for d in dimension_scores.values() if d['score'] < 0.70)
        
        summary = f"Overall score of {final_score:.1f}/100. "
        
        if excellent > 0:
            summary += f"{excellent} dimension{'s' if excellent > 1 else ''} showing excellence. "
        if good > 0:
            summary += f"{good} dimension{'s' if good > 1 else ''} performing well. "
        if needs_work > 0:
            summary += f"{needs_work} dimension{'s' if needs_work > 1 else ''} with room for improvement."
        
        return summary


# ============================================
# CLI USAGE
# ============================================
if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python weighted_scorer.py <session_id> [role] [experience]")
        print("Example: python weighted_scorer.py session_abc123 software_engineer mid")
        sys.exit(1)
    
    session_id = sys.argv[1]
    role = sys.argv[2] if len(sys.argv) > 2 else None
    experience = sys.argv[3] if len(sys.argv) > 3 else "mid"
    
    # Paths
    features_csv = f"processed_features/{session_id}/final_multimodal_features.csv"
    output_path = f"feedback_results/{session_id}/weighted_scores.json"
    
    # Check if input exists
    if not os.path.exists(features_csv):
        print(f"❌ Input file not found: {features_csv}")
        print("   Run multimodal_fusion.py first!")
        sys.exit(1)
    
    # Generate scores
    scorer = WeightedScorer()
    
    try:
        result = scorer.generate_scores(features_csv, output_path, role, experience)
        
        print("\n" + "="*60)
        print("✅ WEIGHTED SCORING COMPLETE")
        print("="*60)
        print(f"Input:  {features_csv}")
        print(f"Output: {output_path}")
        print(f"\n📊 Results:")
        print(f"  Final Score: {result['final_score']}/100")
        print(f"  Grade: {result['overall_grade']}")
        print(f"  Category: {result['performance_category']}")
        
        print(f"\n💪 Top Strengths:")
        for strength in result['top_strengths']:
            print(f"  • {strength['dimension']}: {strength['score']:.3f}")
        
        print(f"\n📈 Areas to Improve:")
        for area in result['improvement_areas']:
            print(f"  • {area['dimension']}: {area['score']:.3f} (gap: {area['gap']:.3f})")
        
        print(f"\n💡 Summary: {result['summary']}")
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)