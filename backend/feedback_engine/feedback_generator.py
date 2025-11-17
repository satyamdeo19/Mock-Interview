"""
Feedback Generator
Generates comprehensive natural language feedback report
Input: weighted_scores.json, final_multimodal_features.csv
Output: feedback_report.json
"""

import json
import pandas as pd
import numpy as np
import logging
import os
from pathlib import Path
from datetime import datetime
from typing import Dict, List

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FeedbackGenerator:
    """Generate comprehensive feedback report with actionable insights"""

    def __init__(self, benchmark_path: str = None):
        """Load benchmark data safely with absolute path resolution"""
        if benchmark_path is None:
            # Get absolute path to backend/benchmark_data.json
            backend_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            benchmark_path = os.path.join(backend_dir, "benchmark_data.json")

        try:
            with open(benchmark_path, 'r') as f:
                self.benchmarks = json.load(f)
            logger.info(f"✅ Loaded benchmarks from: {benchmark_path}")

        except FileNotFoundError:
            logger.warning(f"⚠️ Benchmark file not found: {benchmark_path}")
            self.benchmarks = {}
        except json.JSONDecodeError as e:
            logger.error(f"❌ Error parsing benchmark JSON: {e}")
            self.benchmarks = {}
        except Exception as e:
            logger.error(f"❌ Unexpected error loading benchmarks: {e}")
            self.benchmarks = {}

    def generate_feedback(self, weighted_scores_path: str, features_csv_path: str,
                          output_path: str = None, candidate_name: str = "Candidate",
                          position: str = "Position") -> dict:
        """Generate comprehensive feedback report"""
        logger.info(f"📝 Generating feedback report")

        # Load weighted scores
        with open(weighted_scores_path, 'r') as f:
            scores = json.load(f)

        # Load features
        features_df = pd.read_csv(features_csv_path)
        features = features_df.iloc[0].to_dict()

        # Generate report sections
        report = {
            'metadata': {
                'candidate_name': candidate_name,
                'position': position,
                'interview_date': datetime.now().strftime('%Y-%m-%d'),
                'report_generated': datetime.now().isoformat(),
                'session_id': Path(weighted_scores_path).parent.name
            },

            'executive_summary': self._generate_executive_summary(scores, features),
            'dimension_analysis': self._generate_dimension_analysis(scores),
            'strengths': self._generate_strengths_section(scores),
            'improvement_areas': self._generate_improvement_section(scores),
            'detailed_insights': self._generate_detailed_insights(scores, features),
            'recommendations': self._generate_recommendations(scores),
            'action_plan': self._generate_action_plan(scores),
            'benchmarking': self._generate_benchmarking(scores, features),
            'next_steps': self._generate_next_steps(scores['final_score'])
        }

        logger.info(f"✅ Generated comprehensive feedback report")

        # Save if output path provided
        if output_path:
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, 'w') as f:
                json.dump(report, f, indent=2)
            logger.info(f"💾 Saved to: {output_path}")

        return report

    def _generate_executive_summary(self, scores: dict, features: dict) -> dict:
        """Generate executive summary"""
        final_score = scores['final_score']
        grade = scores['overall_grade']
        performance = scores['performance_category']

        # Overall assessment
        if final_score >= 85:
            assessment = "Outstanding performance demonstrating exceptional readiness."
        elif final_score >= 75:
            assessment = "Strong performance with solid interview skills across most dimensions."
        elif final_score >= 65:
            assessment = "Good performance with some areas showing strength and others needing development."
        else:
            assessment = "Performance shows potential but requires significant improvement in key areas."

        # Key highlights
        highlights = []
        for strength in scores.get('top_strengths', [])[:2]:
            highlights.append(f"Excellent {strength['dimension'].lower()} ({strength['score']:.2f})")

        # Key concerns
        concerns = []
        for area in scores.get('improvement_areas', [])[:2]:
            if area['score'] < 0.65:
                concerns.append(f"{area['dimension']} needs attention ({area['score']:.2f})")

        return {
            'overall_score': final_score,
            'grade': grade,
            'performance_category': performance,
            'assessment': assessment,
            'key_highlights': highlights,
            'key_concerns': concerns if concerns else ["No major concerns identified"],
            'recommendation': self._get_hiring_recommendation(final_score)
        }

    def _get_hiring_recommendation(self, score: float) -> str:
        """Generate hiring recommendation"""
        if score >= 85:
            return "Strong Hire - Candidate demonstrates exceptional interview performance"
        elif score >= 75:
            return "Hire - Candidate shows strong interview skills with minor areas for development"
        elif score >= 65:
            return "Consider - Candidate has potential but may need additional evaluation"
        else:
            return "Develop Skills - Candidate would benefit from interview coaching before reapplying"

    def _generate_dimension_analysis(self, scores: dict) -> list:
        """Generate detailed analysis for each dimension"""
        dimension_analysis = []
        for dim, data in scores.get('dimension_scores', {}).items():
            analysis = {
                'dimension': dim.replace('_', ' ').title(),
                'score': data['score'],
                'percentile': data['percentile'],
                'grade': self._score_to_grade(data['score']),
                'benchmark': data['benchmark'],
                'vs_benchmark': data['difference'],
                'interpretation': data['interpretation']
            }
            dimension_analysis.append(analysis)
        return dimension_analysis

    def _score_to_grade(self, score: float) -> str:
        """Convert dimension score to grade"""
        if score >= 0.9:
            return "A+"
        elif score >= 0.85:
            return "A"
        elif score >= 0.8:
            return "A-"
        elif score >= 0.75:
            return "B+"
        elif score >= 0.7:
            return "B"
        elif score >= 0.65:
            return "B-"
        elif score >= 0.6:
            return "C+"
        else:
            return "C"

    # ✅ Added Missing Function 1
    def _generate_strengths_section(self, scores: dict) -> dict:
        """Generate strengths section"""
        return {
            'overview': f"You demonstrated strong performance in {len(scores.get('top_strengths', []))} key areas.",
            'top_strengths': scores.get('top_strengths', []),
            'advice': "Continue to leverage these strengths while addressing development areas."
        }

    # ✅ Added Missing Function 2
    def _generate_improvement_section(self, scores: dict) -> dict:
        """Generate improvement areas section"""
        return {
            'overview': f"There are {len(scores.get('improvement_areas', []))} areas where focused improvement will enhance your interview performance.",
            'improvement_areas': scores.get('improvement_areas', []),
            'priority': "Focus first on areas with the largest gaps from benchmark."
        }

    # ✅ Added Remaining Helper Functions
    def _generate_detailed_insights(self, scores: dict, features: dict) -> dict:
        return {
            'communication': {
                'fluency_note': "Based on speech patterns and filler word analysis",
                'articulation_note': "Based on audio clarity and vocal characteristics"
            },
            'non_verbal': {
                'body_language': "Based on head pose, posture, and movement stability",
                'facial_expressions': "Based on facial feature animation and expressiveness"
            },
            'emotional': {
                'overall_tone': "Based on sentiment analysis and vocal tone",
                'stress_indicators': "Based on nervousness markers and micro-movements"
            }
        }

    def _generate_recommendations(self, scores: dict) -> list:
        """Generate actionable recommendations"""
        recommendations = []
        for area in scores.get('improvement_areas', []):
            recommendations.append({
                'area': area['dimension'],
                'priority': 'High' if area['score'] < 0.6 else 'Medium',
                'recommendation': area.get('recommendation', 'Focus on improving this competency'),
                'expected_impact': 'High' if area['gap'] > 0.1 else 'Medium'
            })
        return recommendations

    def _generate_action_plan(self, scores: dict) -> dict:
        """Generate 30/60/90 day action plan"""
        improvement_areas = [a['dimension'] for a in scores.get('improvement_areas', [])]
        return {
            'this_week': [
                "Review this feedback report and identify top 2 focus areas",
                "Record yourself answering 3-5 common interview questions"
            ],
            'next_30_days': [
                f"Focus on improving: {', '.join(improvement_areas[:2])}" if improvement_areas else "Focus on overall communication consistency",
                "Do 2-3 mock interviews with feedback"
            ],
            'next_60_days': [
                "Join a speaking or interview practice group",
                "Focus on response structure using the STAR method"
            ],
            'next_90_days': [
                "Reassess progress and do a final practice round",
                "Prepare for a full mock interview session"
            ]
        }

    def _generate_benchmarking(self, scores: dict, features: dict) -> dict:
        """Generate benchmarking comparison"""
        final_score = scores['final_score']
        industry_avg = np.mean(list(self.benchmarks.get('industry_averages', {'default': 0.7}).values())) * 100
        return {
            'your_score': final_score,
            'industry_average': round(industry_avg, 1),
            'difference': round(final_score - industry_avg, 1)
        }

    def _generate_next_steps(self, final_score: float) -> dict:
        """Generate next steps"""
        if final_score >= 85:
            return {'status': 'Ready', 'message': "You're interview-ready! Continue refining your skills."}
        elif final_score >= 70:
            return {'status': 'Nearly Ready', 'message': "You're close. Focus practice on weaker dimensions."}
        else:
            return {'status': 'In Development', 'message': "Focus on fundamentals and repeat practice interviews."}


# ============================================
# CLI USAGE
# ============================================
if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python feedback_generator.py <session_id> [candidate_name] [position]")
        sys.exit(1)

    session_id = sys.argv[1]
    candidate_name = sys.argv[2] if len(sys.argv) > 2 else "Candidate"
    position = sys.argv[3] if len(sys.argv) > 3 else "Position"

    weighted_scores_path = f"feedback_results/{session_id}/weighted_scores.json"
    features_csv_path = f"processed_features/{session_id}/final_multimodal_features.csv"
    output_path = f"feedback_results/{session_id}/feedback_report.json"

    if not os.path.exists(weighted_scores_path) or not os.path.exists(features_csv_path):
        print("❌ Missing input files. Ensure previous phases completed.")
        sys.exit(1)

    generator = FeedbackGenerator()
    try:
        report = generator.generate_feedback(weighted_scores_path, features_csv_path, output_path, candidate_name, position)
        print("\n✅ FEEDBACK REPORT GENERATED")
        print(f"Output: {output_path}")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
