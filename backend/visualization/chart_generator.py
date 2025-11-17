"""
Chart Generator
Creates visualizations for feedback reports
Output: radar_chart.png, comparison_chart.png, etc.
"""

import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import json
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (10, 8)
plt.rcParams['font.size'] = 10


class ChartGenerator:
    """Generate visualization charts for feedback"""
    
    def __init__(self):
        self.colors = {
            'primary': '#3b82f6',
            'secondary': '#10b981',
            'tertiary': '#f59e0b',
            'danger': '#ef4444',
            'muted': '#6b7280'
        }
    
    def generate_all_charts(self, session_id: str, output_dir: str = None):
        """Generate all charts for a session"""
        
        if output_dir is None:
            output_dir = f"backend/feedback_results/{session_id}/visualizations"
        
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        logger.info(f"📊 Generating visualizations for {session_id}")
        
        # Load data
        results_dir = f"backend/feedback_results/{session_id}"
        
        with open(f"{results_dir}/weighted_scores.json", 'r') as f:
            scores = json.load(f)
        
        # Generate charts
        charts_created = []
        
        try:
            # 1. Radar chart
            radar_path = f"{output_dir}/radar_chart.png"
            self.create_radar_chart(scores, radar_path)
            charts_created.append(radar_path)
            
            # 2. Bar chart
            bar_path = f"{output_dir}/dimension_scores.png"
            self.create_dimension_bar_chart(scores, bar_path)
            charts_created.append(bar_path)
            
            # 3. Comparison chart
            comp_path = f"{output_dir}/benchmark_comparison.png"
            self.create_benchmark_comparison(scores, comp_path)
            charts_created.append(comp_path)
            
            # 4. Contribution breakdown
            contrib_path = f"{output_dir}/score_contributions.png"
            self.create_contribution_chart(scores, contrib_path)
            charts_created.append(contrib_path)
            
            logger.info(f"✅ Generated {len(charts_created)} charts")
            
        except Exception as e:
            logger.error(f"❌ Chart generation error: {e}")
            import traceback
            traceback.print_exc()
        
        return charts_created
    
    def create_radar_chart(self, scores: dict, output_path: str):
        """Create radar/spider chart for 10 dimensions"""
        
        logger.info("   Creating radar chart...")
        
        # Extract dimensions
        dimensions = list(scores['dimension_scores'].keys())
        values = [scores['dimension_scores'][dim]['score'] for dim in dimensions]
        benchmarks = [scores['dimension_scores'][dim]['benchmark'] for dim in dimensions]
        
        # Number of variables
        N = len(dimensions)
        
        # Compute angle for each axis
        angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
        values += values[:1]  # Complete the circle
        benchmarks += benchmarks[:1]
        angles += angles[:1]
        
        # Create figure
        fig, ax = plt.subplots(figsize=(12, 12), subplot_kw=dict(projection='polar'))
        
        # Plot candidate scores
        ax.plot(angles, values, 'o-', linewidth=2, label='Your Score', 
                color=self.colors['primary'])
        ax.fill(angles, values, alpha=0.25, color=self.colors['primary'])
        
        # Plot benchmarks
        ax.plot(angles, benchmarks, 'o--', linewidth=2, label='Industry Average',
                color=self.colors['muted'])
        ax.fill(angles, benchmarks, alpha=0.1, color=self.colors['muted'])
        
        # Fix axis to go from 0 to 1
        ax.set_ylim(0, 1)
        
        # Add labels
        labels = [dim.replace('_', ' ').title() for dim in dimensions]
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(labels, size=11)
        
        # Add legend
        ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
        
        # Add title
        plt.title('Interview Performance Profile\n10-Dimension Analysis', 
                 size=16, weight='bold', pad=20)
        
        # Add grid
        ax.grid(True, linewidth=0.5, alpha=0.3)
        
        # Save
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"      Saved: {output_path}")
    
    def create_dimension_bar_chart(self, scores: dict, output_path: str):
        """Create horizontal bar chart for dimensions"""
        
        logger.info("   Creating dimension bar chart...")
        
        # Extract data
        dimensions = list(scores['dimension_scores'].keys())
        values = [scores['dimension_scores'][dim]['score'] for dim in dimensions]
        benchmarks = [scores['dimension_scores'][dim]['benchmark'] for dim in dimensions]
        
        # Sort by score
        sorted_indices = np.argsort(values)
        dimensions = [dimensions[i] for i in sorted_indices]
        values = [values[i] for i in sorted_indices]
        benchmarks = [benchmarks[i] for i in sorted_indices]
        
        # Create figure
        fig, ax = plt.subplots(figsize=(12, 8))
        
        y_pos = np.arange(len(dimensions))
        
        # Plot bars
        bars = ax.barh(y_pos, values, alpha=0.8, label='Your Score')
        
        # Color bars based on performance
        for i, (bar, val, bench) in enumerate(zip(bars, values, benchmarks)):
            if val >= bench + 0.1:
                bar.set_color(self.colors['secondary'])  # Green (good)
            elif val >= bench:
                bar.set_color(self.colors['primary'])  # Blue (average)
            else:
                bar.set_color(self.colors['tertiary'])  # Orange (needs work)
        
        # Add benchmark markers
        ax.scatter(benchmarks, y_pos, color=self.colors['danger'], 
                  s=100, marker='D', label='Industry Average', zorder=5)
        
        # Customize
        ax.set_yticks(y_pos)
        ax.set_yticklabels([d.replace('_', ' ').title() for d in dimensions])
        ax.set_xlabel('Score', fontsize=12, weight='bold')
        ax.set_xlim(0, 1)
        ax.grid(axis='x', alpha=0.3)
        ax.legend()
        
        # Add value labels
        for i, (val, bench) in enumerate(zip(values, benchmarks)):
            ax.text(val + 0.02, i, f'{val:.2f}', va='center', fontsize=10)
        
        plt.title('Dimension Scores vs Industry Benchmarks', 
                 fontsize=16, weight='bold', pad=20)
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"      Saved: {output_path}")
    
    def create_benchmark_comparison(self, scores: dict, output_path: str):
        """Create benchmark comparison chart"""
        
        logger.info("   Creating benchmark comparison...")
        
        # Extract data
        dimensions = list(scores['dimension_scores'].keys())
        your_scores = [scores['dimension_scores'][dim]['score'] for dim in dimensions]
        benchmarks = [scores['dimension_scores'][dim]['benchmark'] for dim in dimensions]
        differences = [scores['dimension_scores'][dim]['difference'] for dim in dimensions]
        
        # Create figure
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        
        # Left plot: Comparison
        x = np.arange(len(dimensions))
        width = 0.35
        
        bars1 = ax1.bar(x - width/2, your_scores, width, label='Your Score',
                       color=self.colors['primary'], alpha=0.8)
        bars2 = ax1.bar(x + width/2, benchmarks, width, label='Benchmark',
                       color=self.colors['muted'], alpha=0.8)
        
        ax1.set_ylabel('Score', fontsize=12, weight='bold')
        ax1.set_title('Score Comparison', fontsize=14, weight='bold')
        ax1.set_xticks(x)
        ax1.set_xticklabels([d.replace('_', '\n').title() for d in dimensions], 
                           rotation=45, ha='right', fontsize=9)
        ax1.legend()
        ax1.grid(axis='y', alpha=0.3)
        ax1.set_ylim(0, 1)
        
        # Right plot: Difference
        colors = [self.colors['secondary'] if d > 0 else self.colors['danger'] 
                 for d in differences]
        
        bars = ax2.barh(dimensions, differences, color=colors, alpha=0.8)
        ax2.axvline(x=0, color='black', linestyle='-', linewidth=0.8)
        ax2.set_xlabel('Difference from Benchmark', fontsize=12, weight='bold')
        ax2.set_title('Gap Analysis', fontsize=14, weight='bold')
        ax2.set_yticklabels([d.replace('_', ' ').title() for d in dimensions])
        ax2.grid(axis='x', alpha=0.3)
        
        # Add value labels
        for bar, val in zip(bars, differences):
            x_pos = val + (0.02 if val > 0 else -0.02)
            ha = 'left' if val > 0 else 'right'
            ax2.text(x_pos, bar.get_y() + bar.get_height()/2, 
                    f'{val:+.2f}', va='center', ha=ha, fontsize=9)
        
        plt.suptitle('Performance vs Industry Benchmarks', 
                    fontsize=16, weight='bold', y=1.02)
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"      Saved: {output_path}")
    
    def create_contribution_chart(self, scores: dict, output_path: str):
        """Create score contribution breakdown"""
        
        logger.info("   Creating contribution chart...")
        
        # Extract data
        dimensions = list(scores['dimension_scores'].keys())
        contributions = [scores['dimension_scores'][dim]['contribution'] 
                        for dim in dimensions]
        weights = [scores['dimension_scores'][dim]['weight'] for dim in dimensions]
        
        # Sort by contribution
        sorted_indices = np.argsort(contributions)[::-1]
        dimensions = [dimensions[i] for i in sorted_indices]
        contributions = [contributions[i] for i in sorted_indices]
        weights = [weights[i] for i in sorted_indices]
        
        # Create figure
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        
        # Left: Pie chart of contributions
        colors = plt.cm.Set3(np.linspace(0, 1, len(dimensions)))
        wedges, texts, autotexts = ax1.pie(contributions, labels=None, autopct='%1.1f%%',
                                            colors=colors, startangle=90)
        ax1.set_title('Score Contribution by Dimension', fontsize=14, weight='bold')
        
        # Legend
        ax1.legend(wedges, [d.replace('_', ' ').title() for d in dimensions],
                  loc="center left", bbox_to_anchor=(1, 0, 0.5, 1), fontsize=9)
        
        # Right: Bar chart
        y_pos = np.arange(len(dimensions))
        bars = ax2.barh(y_pos, contributions, color=colors, alpha=0.8)
        
        ax2.set_yticks(y_pos)
        ax2.set_yticklabels([d.replace('_', ' ').title() for d in dimensions])
        ax2.set_xlabel('Contribution to Final Score', fontsize=12, weight='bold')
        ax2.set_title('Weighted Contributions', fontsize=14, weight='bold')
        ax2.grid(axis='x', alpha=0.3)
        
        # Add value labels
        for i, (contrib, weight) in enumerate(zip(contributions, weights)):
            ax2.text(contrib + 0.5, i, 
                    f'{contrib:.1f} ({weight*100:.0f}%)', 
                    va='center', fontsize=9)
        
        plt.suptitle(f'Final Score Breakdown: {scores["final_score"]:.1f}/100', 
                    fontsize=16, weight='bold', y=1.02)
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"      Saved: {output_path}")


# ============================================
# CLI USAGE
# ============================================
if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python chart_generator.py <session_id>")
        print("Example: python chart_generator.py session_abc123")
        sys.exit(1)
    
    session_id = sys.argv[1]
    
    # Generate charts
    try:
        generator = ChartGenerator()
        charts = generator.generate_all_charts(session_id)
        
        print("\n" + "="*60)
        print("✅ CHART GENERATION COMPLETE")
        print("="*60)
        print(f"Session: {session_id}")
        print(f"Charts created: {len(charts)}")
        for chart in charts:
            print(f"  • {Path(chart).name}")
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)