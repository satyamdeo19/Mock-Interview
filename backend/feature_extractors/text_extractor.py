"""
Text Feature Extractor
Extracts linguistic and sentiment features from Q&A transcripts
Input: qa_data.csv (Question_Number, Question, Answer, Timestamp, Duration_Seconds)
Output: text_features.csv
"""

import pandas as pd
import numpy as np
import re
import logging
from pathlib import Path
from typing import Dict, List, Tuple
import warnings
warnings.filterwarnings('ignore')

# NLP Libraries
import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import textstat

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords', quiet=True)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TextFeatureExtractor:
    """Extract comprehensive text features from interview transcripts"""
    
    def __init__(self):
        self.sentiment_analyzer = SentimentIntensityAnalyzer()
        self.stop_words = set(stopwords.words('english'))
        
        # Common filler words in interviews
        self.filler_words = {
            'um', 'uh', 'er', 'ah', 'like', 'you know', 'sort of', 
            'kind of', 'i mean', 'basically', 'actually', 'literally',
            'so', 'well', 'yeah', 'right', 'okay', 'hmm'
        }
        
    def extract_features(self, csv_path: str, output_path: str = None) -> pd.DataFrame:
        """
        Main extraction function
        
        Args:
            csv_path: Path to qa_data.csv
            output_path: Optional output path for text_features.csv
            
        Returns:
            DataFrame with text features per question
        """
        logger.info(f"📝 Starting text feature extraction from: {csv_path}")
        
        # Load CSV
        try:
            df = pd.read_csv(csv_path)
            logger.info(f"   Loaded {len(df)} Q&A pairs")
        except Exception as e:
            logger.error(f"❌ Error loading CSV: {e}")
            raise
        
        # Validate required columns
        required_cols = ['Question_Number', 'Answer']
        if not all(col in df.columns for col in required_cols):
            raise ValueError(f"CSV must contain columns: {required_cols}")
        
        # Extract features for each answer
        features_list = []
        for idx, row in df.iterrows():
            answer = str(row['Answer'])
            features = self._extract_answer_features(answer, row['Question_Number'])
            features_list.append(features)
        
        # Create features DataFrame
        features_df = pd.DataFrame(features_list)
        
        # Add meta features (session-level aggregates)
        meta_features = self._calculate_meta_features(df, features_df)
        for key, value in meta_features.items():
            features_df[key] = value
        
        logger.info(f"✅ Extracted {len(features_df.columns)} text features")
        
        # Save to CSV if output path provided
        if output_path:
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            features_df.to_csv(output_path, index=False)
            logger.info(f"💾 Saved to: {output_path}")
        
        return features_df
    
    def _extract_answer_features(self, answer: str, question_num: int) -> Dict:
        """Extract all features from a single answer"""
        
        # Basic linguistic features
        linguistic = self._extract_linguistic_features(answer)
        
        # Sentiment and emotion features
        sentiment = self._extract_sentiment_features(answer)
        
        # Combine all features
        features = {
            'question_number': question_num,
            **linguistic,
            **sentiment
        }
        
        return features
    
    def _extract_linguistic_features(self, text: str) -> Dict:
        """Extract linguistic features"""
        
        # Tokenization
        words = word_tokenize(text.lower())
        sentences = sent_tokenize(text)
        
        # Basic counts
        word_count = len([w for w in words if w.isalnum()])
        sentence_count = len(sentences)
        char_count = len(text)
        
        # Word statistics
        word_lengths = [len(w) for w in words if w.isalnum()]
        avg_word_length = np.mean(word_lengths) if word_lengths else 0
        
        # Sentence statistics
        avg_sentence_length = word_count / sentence_count if sentence_count > 0 else 0
        
        # Stopword ratio
        stopword_count = sum(1 for w in words if w in self.stop_words)
        stopword_ratio = stopword_count / word_count if word_count > 0 else 0
        
        # Filler word analysis
        text_lower = text.lower()
        filler_count = sum(text_lower.count(filler) for filler in self.filler_words)
        filler_ratio = filler_count / word_count if word_count > 0 else 0
        
        # Punctuation analysis
        punctuation_count = sum(1 for c in text if c in '.,!?;:')
        punctuation_ratio = punctuation_count / char_count if char_count > 0 else 0
        
        # Case analysis
        uppercase_count = sum(1 for c in text if c.isupper())
        uppercase_ratio = uppercase_count / char_count if char_count > 0 else 0
        
        # Special markers
        question_marks = text.count('?')
        exclamation_marks = text.count('!')
        
        # Readability scores
        try:
            flesch_reading_ease = textstat.flesch_reading_ease(text)
            flesch_kincaid_grade = textstat.flesch_kincaid_grade(text)
        except:
            flesch_reading_ease = 0
            flesch_kincaid_grade = 0
        
        return {
            'text_word_count': word_count,
            'text_sentence_count': sentence_count,
            'text_char_count': char_count,
            'text_avg_word_length': round(avg_word_length, 2),
            'text_avg_sentence_length': round(avg_sentence_length, 2),
            'text_stopword_ratio': round(stopword_ratio, 3),
            'text_filler_ratio': round(filler_ratio, 3),
            'text_punctuation_ratio': round(punctuation_ratio, 3),
            'text_uppercase_ratio': round(uppercase_ratio, 3),
            'text_question_marks': question_marks,
            'text_exclamation_marks': exclamation_marks,
            'text_flesch_reading_ease': round(flesch_reading_ease, 2),
            'text_flesch_kincaid_grade': round(flesch_kincaid_grade, 2)
        }
    
    def _extract_sentiment_features(self, text: str) -> Dict:
        """Extract sentiment and emotion features using VADER"""
        
        # Get VADER sentiment scores
        sentiment_scores = self.sentiment_analyzer.polarity_scores(text)
        
        # Basic sentiment
        sentiment_positive = sentiment_scores['pos']
        sentiment_negative = sentiment_scores['neg']
        sentiment_neutral = sentiment_scores['neu']
        sentiment_compound = sentiment_scores['compound']
        
        # Derived confidence indicators (from text patterns)
        # High confidence: assertive words, strong verbs, definite statements
        confidence_indicators = len(re.findall(
            r'\b(certainly|definitely|absolutely|confident|sure|clearly|obviously)\b', 
            text.lower()
        ))
        
        # Tentative indicators: hedging words
        tentative_indicators = len(re.findall(
            r'\b(maybe|perhaps|possibly|probably|might|could|seems|somewhat|fairly)\b',
            text.lower()
        ))
        
        # Analytical tone: logical connectors
        analytical_indicators = len(re.findall(
            r'\b(because|therefore|thus|however|although|consequently|moreover)\b',
            text.lower()
        ))
        
        # Normalize by word count
        word_count = len(text.split())
        confidence_score = min(confidence_indicators / max(word_count, 1) * 10, 1.0)
        tentative_score = min(tentative_indicators / max(word_count, 1) * 10, 1.0)
        analytical_score = min(analytical_indicators / max(word_count, 1) * 10, 1.0)
        
        # Simple emotion proxies from sentiment
        emotion_joy = max(sentiment_positive - 0.3, 0)  # High positive
        emotion_sadness = max(sentiment_negative - 0.3, 0)  # High negative
        emotion_fear = tentative_score * sentiment_negative  # Tentative + negative
        emotion_anger = sentiment_negative * (1 - tentative_score)  # Negative but assertive
        emotion_surprise = abs(sentiment_compound) if '!' in text else 0
        
        return {
            'emotion_sentiment_positive': round(sentiment_positive, 3),
            'emotion_sentiment_negative': round(sentiment_negative, 3),
            'emotion_sentiment_neutral': round(sentiment_neutral, 3),
            'emotion_sentiment_polarity': round(sentiment_compound, 3),
            'emotion_confidence_score': round(confidence_score, 3),
            'emotion_tentative_score': round(tentative_score, 3),
            'emotion_analytical_score': round(analytical_score, 3),
            'emotion_joy_score': round(emotion_joy, 3),
            'emotion_sadness_score': round(emotion_sadness, 3),
            'emotion_fear_score': round(emotion_fear, 3),
            'emotion_anger_score': round(emotion_anger, 3),
            'emotion_surprise_score': round(emotion_surprise, 3)
        }
    
    def _calculate_meta_features(self, qa_df: pd.DataFrame, features_df: pd.DataFrame) -> Dict:
        """Calculate session-level meta features"""
        
        # Total questions answered
        total_questions = len(qa_df)
        
        # Average answer length across all questions
        avg_answer_length = features_df['text_word_count'].mean()
        
        # Answer length consistency (lower is more consistent)
        answer_length_std = features_df['text_word_count'].std()
        
        # Average filler ratio across session
        avg_filler_ratio = features_df['text_filler_ratio'].mean()
        
        # Average sentiment polarity
        avg_sentiment = features_df['emotion_sentiment_polarity'].mean()
        
        # Confidence trend (slope of confidence scores over questions)
        if len(features_df) > 1:
            question_nums = features_df['question_number'].values
            confidence_scores = features_df['emotion_confidence_score'].values
            confidence_trend = np.polyfit(question_nums, confidence_scores, 1)[0]
        else:
            confidence_trend = 0
        
        return {
            'meta_total_questions': total_questions,
            'meta_avg_answer_length': round(avg_answer_length, 2),
            'meta_answer_length_std': round(answer_length_std, 2),
            'meta_avg_filler_ratio': round(avg_filler_ratio, 3),
            'meta_avg_sentiment': round(avg_sentiment, 3),
            'meta_confidence_trend': round(confidence_trend, 4)
        }


# ============================================
# CLI USAGE
# ============================================
if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python text_extractor.py <session_id>")
        print("Example: python text_extractor.py session_abc123_1234567890")
        sys.exit(1)
    
    session_id = sys.argv[1]
    
    # Paths (resolve using backend.config helper)
    from backend.config import get_recordings_session_dir
    import os

    session_dir = get_recordings_session_dir(session_id)
    csv_path = os.path.join(session_dir, 'transcripts', 'qa_data.csv')
    output_dir = os.path.join('processed_features', session_id)
    output_path = os.path.join(output_dir, 'text_features.csv')
    
    # Extract features
    extractor = TextFeatureExtractor()
    
    try:
        features_df = extractor.extract_features(csv_path, output_path)
        
        print("\n" + "="*60)
        print("✅ TEXT FEATURE EXTRACTION COMPLETE")
        print("="*60)
        print(f"Input:  {csv_path}")
        print(f"Output: {output_path}")
        print(f"Features extracted: {len(features_df.columns)}")
        print(f"Rows processed: {len(features_df)}")
        print("\n📊 Sample features:")
        print(features_df.head())
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        sys.exit(1)