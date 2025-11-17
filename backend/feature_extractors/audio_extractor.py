"""
Audio Feature Extractor
Extracts prosodic, spectral, and acoustic features from audio recordings
Input: full_session.webm (or .wav)
Output: audio_features.csv (single row, session-level features)
"""

import numpy as np
import pandas as pd
import librosa
import logging
from pathlib import Path
import subprocess
import os
import tempfile

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class AudioFeatureExtractor:
    """Extract comprehensive audio features from interview recordings"""

    def __init__(self, sample_rate: int = 22050):
        self.sr = sample_rate

    def extract_features(self, audio_path: str, output_path: str = None) -> pd.DataFrame:
        """
        Main extraction function
        Args:
            audio_path: Path to audio file (webm, wav, mp3, etc.)
            output_path: Optional output path for audio_features.csv
        Returns:
            DataFrame with single row of audio features
        """
        logger.info(f"🎵 Starting audio feature extraction from: {audio_path}")

        wav_path = self._convert_to_wav(audio_path)

        try:
            logger.info(f"   Loading audio file...")
            y, sr = librosa.load(wav_path, sr=self.sr)
            logger.info(f"   Audio loaded: {len(y)} samples, {sr} Hz, {len(y)/sr:.2f} seconds")

            basic_features = self._extract_basic_features(y, sr)
            prosodic_features = self._extract_prosodic_features(y, sr)
            energy_features = self._extract_energy_features(y, sr)
            spectral_features = self._extract_spectral_features(y, sr)
            pause_features = self._extract_pause_features(y, sr)
            mfcc_features = self._extract_mfcc_features(y, sr)

            features = {
                **basic_features,
                **prosodic_features,
                **energy_features,
                **spectral_features,
                **pause_features,
                **mfcc_features
            }

            features_df = pd.DataFrame([features])

            logger.info(f"✅ Extracted {len(features)} audio features")

            if output_path:
                Path(output_path).parent.mkdir(parents=True, exist_ok=True)
                features_df.to_csv(output_path, index=False)
                logger.info(f"💾 Saved to: {output_path}")

            return features_df

        finally:
            if wav_path != audio_path and os.path.exists(wav_path):
                os.remove(wav_path)
                logger.info(f"   Cleaned up temporary file: {wav_path}")

    def _convert_to_wav(self, audio_path: str) -> str:
        """Convert audio to WAV format if needed"""

        if audio_path.lower().endswith('.wav'):
            return audio_path

        logger.info(f"   Converting {Path(audio_path).suffix} to WAV...")

        temp_wav = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)
        temp_wav_path = temp_wav.name
        temp_wav.close()

        try:
            cmd = [
                'ffmpeg', '-i', audio_path,
                '-ar', str(self.sr),
                '-ac', '1',
                '-y',
                temp_wav_path
            ]
            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.returncode != 0:
                logger.error(f"ffmpeg error: {result.stderr}")
                raise RuntimeError(f"Audio conversion failed: {result.stderr}")

            logger.info(f"   Conversion successful")
            return temp_wav_path

        except FileNotFoundError:
            logger.error("ffmpeg not found. Please install ffmpeg.")
            raise RuntimeError("ffmpeg is required but not installed. Install with: sudo apt install ffmpeg")
        except Exception as e:
            if os.path.exists(temp_wav_path):
                os.remove(temp_wav_path)
            raise e

    def _extract_basic_features(self, y: np.ndarray, sr: int) -> dict:
        """Extract basic audio properties"""
        duration = len(y) / sr
        return {
            'audio_sr': sr,
            'audio_duration': round(duration, 2)
        }

    def _extract_prosodic_features(self, y: np.ndarray, sr: int) -> dict:
        """Extract prosodic features (pitch, tempo)"""
        try:
            pitches, magnitudes = librosa.piptrack(y=y, sr=sr)
            pitch_values = []
            for t in range(pitches.shape[1]):
                index = magnitudes[:, t].argmax()
                pitch = pitches[index, t]
                if pitch > 0:
                    pitch_values.append(pitch)

            if pitch_values:
                pitch_mean = np.mean(pitch_values)
                pitch_std = np.std(pitch_values)
                pitch_min = np.min(pitch_values)
                pitch_max = np.max(pitch_values)
                pitch_range = pitch_max - pitch_min
            else:
                pitch_mean = pitch_std = pitch_range = pitch_min = pitch_max = 0
        except Exception:
            logger.warning("   Pitch extraction failed, using defaults")
            pitch_mean = pitch_std = pitch_range = pitch_min = pitch_max = 0

        try:
            tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
            if isinstance(tempo, np.ndarray):
                tempo = float(tempo[0]) if len(tempo) > 0 else 120.0
            else:
                tempo = float(tempo)
            tempo_syllables = tempo / 60 * 2.5
        except Exception as e:
            logger.warning(f"   Tempo estimation failed: {e}, using defaults")
            tempo_syllables = 3.0

        return {
            'audio_pitch_mean': round(float(pitch_mean), 2),
            'audio_pitch_std': round(float(pitch_std), 2),
            'audio_pitch_min': round(float(pitch_min), 2),
            'audio_pitch_max': round(float(pitch_max), 2),
            'audio_pitch_range': round(float(pitch_range), 2),
            'audio_tempo': round(float(tempo_syllables), 2)
        }

    def _extract_energy_features(self, y: np.ndarray, sr: int) -> dict:
        """Extract energy-related features"""
        rms = librosa.feature.rms(y=y)[0]
        rms_mean = np.mean(rms)
        rms_std = np.std(rms)

        energy = rms ** 2
        energy_normalized = energy / (np.sum(energy) + 1e-10)
        energy_entropy = -np.sum(energy_normalized * np.log2(energy_normalized + 1e-10))

        return {
            'audio_energy_mean': round(rms_mean, 4),
            'audio_energy_std': round(rms_std, 4),
            'audio_energy_entropy': round(energy_entropy, 4)
        }

    def _extract_spectral_features(self, y: np.ndarray, sr: int) -> dict:
        """Extract spectral features"""
        zcr = librosa.feature.zero_crossing_rate(y)[0]
        zcr_mean = np.mean(zcr)
        zcr_std = np.std(zcr)

        spectral_centroids = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
        spec_cent_mean = np.mean(spectral_centroids)
        spec_cent_std = np.std(spectral_centroids)

        spectral_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)[0]
        rolloff_mean = np.mean(spectral_rolloff)
        rolloff_std = np.std(spectral_rolloff)

        spec_flux = np.sqrt(np.sum(np.diff(np.abs(librosa.stft(y)))**2, axis=0))
        flux_mean = np.mean(spec_flux)
        flux_std = np.std(spec_flux)

        return {
            'audio_zcr_mean': round(zcr_mean, 4),
            'audio_zcr_std': round(zcr_std, 4),
            'audio_spec_cent_mean': round(spec_cent_mean, 2),
            'audio_spec_cent_std': round(spec_cent_std, 2),
            'audio_rolloff_mean': round(rolloff_mean, 2),
            'audio_rolloff_std': round(rolloff_std, 2),
            'audio_flux_mean': round(flux_mean, 4),
            'audio_flux_std': round(flux_std, 4)
        }

    def _extract_pause_features(self, y: np.ndarray, sr: int) -> dict:
        """Extract pause/silence features"""
        threshold = 0.01
        rms = librosa.feature.rms(y=y)[0]
        is_silent = rms < threshold
        pauses = np.diff(is_silent.astype(int))
        num_pauses = np.sum(pauses == 1)
        silent_frames = np.where(is_silent)[0]

        if len(silent_frames) > 0:
            pause_fraction = len(silent_frames) / len(rms)
            pause_groups = np.split(silent_frames, np.where(np.diff(silent_frames) != 1)[0] + 1)
            pause_durations = [len(group) * 512 / sr for group in pause_groups]
            avg_pause_duration = np.mean(pause_durations) if pause_durations else 0
        else:
            pause_fraction = 0
            avg_pause_duration = 0

        return {
            'audio_num_pauses': int(num_pauses),
            'audio_avg_pause_duration': round(avg_pause_duration, 3),
            'audio_pause_fraction': round(pause_fraction, 3)
        }

    def _extract_mfcc_features(self, y: np.ndarray, sr: int) -> dict:
        """Extract MFCC features"""
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        mfcc_features = {}
        for i in range(13):
            mfcc_features[f'audio_mfcc_mean_{i+1}'] = round(np.mean(mfccs[i]), 4)
            mfcc_features[f'audio_mfcc_std_{i+1}'] = round(np.std(mfccs[i]), 4)
        return mfcc_features


# ============================================
# CLI USAGE
# ============================================
if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python audio_extractor.py <session_id>")
        print("Example: python audio_extractor.py session_abc123_1234567890")
        sys.exit(1)

    session_id = sys.argv[1]
    from backend.config import get_recordings_session_dir

    session_dir = get_recordings_session_dir(session_id)
    audio_path = os.path.join(session_dir, 'audio', 'full_session.webm')
    output_dir = os.path.join('processed_features', session_id)
    output_path = os.path.join(output_dir, 'audio_features.csv')

    if not os.path.exists(audio_path):
        print(f"❌ Audio file not found: {audio_path}")
        sys.exit(1)

    extractor = AudioFeatureExtractor()

    try:
        features_df = extractor.extract_features(audio_path, output_path)

        print("\n" + "=" * 60)
        print("✅ AUDIO FEATURE EXTRACTION COMPLETE")
        print("=" * 60)
        print(f"Input:  {audio_path}")
        print(f"Output: {output_path}")
        print(f"Features extracted: {len(features_df.columns)}")
        print("\n📊 Sample features:")
        print(features_df.T.head(20))
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
