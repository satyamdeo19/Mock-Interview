"""
Report Builder
Generates PDF reports with feedback and visualizations
Output: feedback_report.pdf
"""

import json
import logging
from pathlib import Path
from datetime import datetime

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ReportBuilder:
    """Build comprehensive PDF feedback reports"""
    
    def __init__(self):
        pass
    
    def build_report(self, session_id: str, output_path: str = None) -> str:
        """
        Build PDF report
        
        Args:
            session_id: Session ID
            output_path: Optional output path for PDF
            
        Returns:
            Path to generated PDF
        """
        logger.info(f"📄 Building PDF report for {session_id}")
        
        if output_path is None:
            output_path = f"backend/feedback_results/{session_id}/feedback_report.pdf"
        
        # Load data
        results_dir = f"backend/feedback_results/{session_id}"
        
        with open(f"{results_dir}/feedback_report.json", 'r') as f:
            report = json.load(f)
        
        with open(f"{results_dir}/weighted_scores.json", 'r') as f:
            scores = json.load(f)
        
        # Try to use reportlab for PDF generation
        try:
            from reportlab.lib.pagesizes import letter, A4
            from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle, PageBreak
            from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
            from reportlab.lib.units import inch
            from reportlab.lib import colors
            from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_JUSTIFY
            
            # Create PDF
            doc = SimpleDocTemplate(output_path, pagesize=letter)
            styles = getSampleStyleSheet()
            story = []
            
            # Title
            title_style = ParagraphStyle(
                'CustomTitle',
                parent=styles['Heading1'],
                fontSize=24,
                textColor=colors.HexColor('#3b82f6'),
                spaceAfter=30,
                alignment=TA_CENTER
            )
            
            story.append(Paragraph("Interview Feedback Report", title_style))
            story.append(Spacer(1, 0.2*inch))
            
            # Metadata
            metadata = report['metadata']
            info_data = [
                ['Candidate:', metadata['candidate_name']],
                ['Position:', metadata['position']],
                ['Interview Date:', metadata['interview_date']],
                ['Report Generated:', datetime.now().strftime('%Y-%m-%d %H:%M')]
            ]
            
            info_table = Table(info_data, colWidths=[2*inch, 4*inch])
            info_table.setStyle(TableStyle([
                ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
                ('FONTNAME', (1, 0), (1, -1), 'Helvetica'),
                ('FONTSIZE', (0, 0), (-1, -1), 11),
                ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
                ('BOTTOMPADDING', (0, 0), (-1, -1), 8),
            ]))
            
            story.append(info_table)
            story.append(Spacer(1, 0.3*inch))
            
            # Executive Summary
            story.append(Paragraph("Executive Summary", styles['Heading1']))
            story.append(Spacer(1, 0.1*inch))
            
            summary = report['executive_summary']
            
            # Score box
            score_data = [
                ['Final Score', f"{summary['overall_score']:.1f}/100"],
                ['Grade', summary['grade']],
                ['Category', summary['performance_category']]
            ]
            
            score_table = Table(score_data, colWidths=[2*inch, 2*inch])
            score_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, -1), colors.HexColor('#f0f9ff')),
                ('TEXTCOLOR', (0, 0), (-1, -1), colors.HexColor('#1e40af')),
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ('FONTNAME', (0, 0), (-1, -1), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, -1), 14),
                ('BOTTOMPADDING', (0, 0), (-1, -1), 12),
                ('GRID', (0, 0), (-1, -1), 1, colors.HexColor('#3b82f6')),
            ]))
            
            story.append(score_table)
            story.append(Spacer(1, 0.2*inch))
            
            story.append(Paragraph(summary['assessment'], styles['Normal']))
            story.append(Spacer(1, 0.2*inch))
            
            # Add radar chart if exists
            viz_dir = f"{results_dir}/visualizations"
            if Path(f"{viz_dir}/radar_chart.png").exists():
                story.append(PageBreak())
                story.append(Paragraph("Performance Profile", styles['Heading1']))
                story.append(Spacer(1, 0.1*inch))
                story.append(Image(f"{viz_dir}/radar_chart.png", width=5*inch, height=5*inch))
            
            # Strengths
            story.append(PageBreak())
            story.append(Paragraph("Key Strengths", styles['Heading1']))
            story.append(Spacer(1, 0.1*inch))
            
            for i, strength in enumerate(scores['top_strengths'], 1):
                story.append(Paragraph(
                    f"<b>{i}. {strength['dimension']}</b> (Score: {strength['score']:.2f})",
                    styles['Normal']
                ))
                story.append(Paragraph(strength['insight'], styles['Normal']))
                story.append(Spacer(1, 0.15*inch))
            
            # Improvement Areas
            story.append(Paragraph("Areas for Improvement", styles['Heading1']))
            story.append(Spacer(1, 0.1*inch))
            
            for i, area in enumerate(scores['improvement_areas'], 1):
                story.append(Paragraph(
                    f"<b>{i}. {area['dimension']}</b> (Score: {area['score']:.2f}, Gap: {area['gap']:.2f})",
                    styles['Normal']
                ))
                story.append(Paragraph(area['recommendation'], styles['Normal']))
                story.append(Spacer(1, 0.15*inch))
            
            # Action Plan
            story.append(PageBreak())
            story.append(Paragraph("Action Plan", styles['Heading1']))
            story.append(Spacer(1, 0.1*inch))
            
            action_plan = report['action_plan']
            
            for phase_key, phase in action_plan.items():
                story.append(Paragraph(f"<b>{phase['title']}</b>", styles['Heading2']))
                
                for action in phase['actions']:
                    story.append(Paragraph(f"• {action}", styles['Normal']))
                
                story.append(Spacer(1, 0.15*inch))
            
            # Build PDF
            doc.build(story)
            
            logger.info(f"✅ PDF report generated: {output_path}")
            return output_path
            
        except ImportError:
            logger.warning("⚠️  reportlab not installed. Install with: pip install reportlab")
            logger.info("   Creating simple text report instead...")
            return self._create_text_report(session_id, report, scores)
        
        except Exception as e:
            logger.error(f"❌ PDF generation failed: {e}")
            return self._create_text_report(session_id, report, scores)
    
    def _create_text_report(self, session_id: str, report: dict, scores: dict) -> str:
        """Fallback: Create simple text report"""
        
        output_path = f"backend/feedback_results/{session_id}/feedback_report.txt"
        
        with open(output_path, 'w') as f:
            f.write("="*60 + "\n")
            f.write("INTERVIEW FEEDBACK REPORT\n")
            f.write("="*60 + "\n\n")
            
            # Metadata
            metadata = report['metadata']
            f.write(f"Candidate: {metadata['candidate_name']}\n")
            f.write(f"Position: {metadata['position']}\n")
            f.write(f"Interview Date: {metadata['interview_date']}\n")
            f.write(f"Report Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}\n\n")
            
            # Executive Summary
            f.write("="*60 + "\n")
            f.write("EXECUTIVE SUMMARY\n")
            f.write("="*60 + "\n\n")
            
            summary = report['executive_summary']
            f.write(f"Final Score: {summary['overall_score']:.1f}/100\n")
            f.write(f"Grade: {summary['grade']}\n")
            f.write(f"Category: {summary['performance_category']}\n\n")
            f.write(f"{summary['assessment']}\n\n")
            
            # Strengths
            f.write("="*60 + "\n")
            f.write("KEY STRENGTHS\n")
            f.write("="*60 + "\n\n")
            
            for i, strength in enumerate(scores['top_strengths'], 1):
                f.write(f"{i}. {strength['dimension']} (Score: {strength['score']:.2f})\n")
                f.write(f"   {strength['insight']}\n\n")
            
            # Improvement Areas
            f.write("="*60 + "\n")
            f.write("AREAS FOR IMPROVEMENT\n")
            f.write("="*60 + "\n\n")
            
            for i, area in enumerate(scores['improvement_areas'], 1):
                f.write(f"{i}. {area['dimension']} (Score: {area['score']:.2f}, Gap: {area['gap']:.2f})\n")
                f.write(f"   {area['recommendation']}\n\n")
            
            # Action Plan
            f.write("="*60 + "\n")
            f.write("ACTION PLAN\n")
            f.write("="*60 + "\n\n")
            
            action_plan = report['action_plan']
            
            for phase_key, phase in action_plan.items():
                f.write(f"{phase['title']}:\n")
                for action in phase['actions']:
                    f.write(f"  • {action}\n")
                f.write("\n")
        
        logger.info(f"✅ Text report generated: {output_path}")
        return output_path


# ============================================
# CLI USAGE
# ============================================
if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python report_builder.py <session_id>")
        print("Example: python report_builder.py session_abc123")
        sys.exit(1)
    
    session_id = sys.argv[1]
    
    # Generate report
    try:
        builder = ReportBuilder()
        report_path = builder.build_report(session_id)
        
        print("\n" + "="*60)
        print("✅ REPORT GENERATION COMPLETE")
        print("="*60)
        print(f"Session: {session_id}")
        print(f"Report: {report_path}")
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)