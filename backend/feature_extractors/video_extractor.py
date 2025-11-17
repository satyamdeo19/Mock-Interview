"""
Video Feature Extractor
Extracts facial features from video recordings using MediaPipe
Input: video chunks (chunk_0.webm, chunk_1.webm, ...)
Output: video_features_raw.csv (frame-level features)
"""

import cv2
import numpy as np
import pandas as pd
import logging
from pathlib import Path
import subprocess
import os
import tempfile
import mediapipe as mp

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class VideoFeatureExtractor:
    """Extract facial and head pose features from video recordings"""
    
    def __init__(self, fps: int = 10):
        """
        Args:
            fps: Frames per second to process (reduce for faster processing)
        """
        self.fps = fps
        
        # Initialize MediaPipe Face Mesh
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        logger.info("✅ MediaPipe Face Mesh initialized")
    
    def extract_features(self, video_dir: str, output_path: str = None) -> pd.DataFrame:
        """
        Main extraction function
        
        Args:
            video_dir: Directory containing video chunks
            output_path: Optional output path for video_features_raw.csv
            
        Returns:
            DataFrame with frame-level video features
        """
        logger.info(f"🎥 Starting video feature extraction from: {video_dir}")
        
        # Merge video chunks into single video
        merged_video = self._merge_video_chunks(video_dir)
        
        try:
            # Open video
            cap = cv2.VideoCapture(merged_video)
            if not cap.isOpened():
                raise RuntimeError(f"Cannot open video: {merged_video}")
            
            # Get video properties
            video_fps = cap.get(cv2.CAP_PROP_FPS)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            duration = total_frames / video_fps if video_fps > 0 else 0
            
            logger.info(f"   Video: {total_frames} frames, {video_fps:.2f} FPS, {duration:.2f}s")
            logger.info(f"   Processing at {self.fps} FPS (every {int(video_fps/self.fps)} frames)")
            
            # Extract features frame by frame
            features_list = []
            frame_number = 0
            processed_frames = 0
            frame_skip = max(1, int(video_fps / self.fps))
            
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Process only at specified FPS
                if frame_number % frame_skip == 0:
                    timestamp = frame_number / video_fps
                    features = self._extract_frame_features(frame, processed_frames, timestamp)
                    
                    if features:
                        features_list.append(features)
                        processed_frames += 1
                        
                        if processed_frames % 100 == 0:
                            logger.info(f"   Processed {processed_frames} frames...")
                
                frame_number += 1
            
            cap.release()
            logger.info(f"✅ Processed {processed_frames} frames from {frame_number} total frames")
            
            if not features_list:
                raise RuntimeError("No facial features detected in video")
            
            # Create DataFrame
            features_df = pd.DataFrame(features_list)
            
            logger.info(f"✅ Extracted {len(features_df.columns)} video features per frame")
            
            # Save to CSV if output path provided
            if output_path:
                Path(output_path).parent.mkdir(parents=True, exist_ok=True)
                features_df.to_csv(output_path, index=False)
                logger.info(f"💾 Saved to: {output_path}")
            
            return features_df
            
        finally:
            # Clean up merged video
            if merged_video and os.path.exists(merged_video):
                os.remove(merged_video)
                logger.info(f"   Cleaned up merged video")
    
    def _merge_video_chunks(self, video_dir: str) -> str:
        """Merge video chunks into single video"""
        
        chunks_dir = Path(video_dir) / "chunks"
        
        if not chunks_dir.exists():
            # Try video_dir itself
            chunks_dir = Path(video_dir)
        
        # Find all chunk files
        chunk_files = sorted(chunks_dir.glob("chunk_*.webm"))
        
        if not chunk_files:
            raise FileNotFoundError(f"No video chunks found in {chunks_dir}")
        
        logger.info(f"   Found {len(chunk_files)} video chunks")
        
        # If only one chunk, use it directly (but still copy to temp for cleanup)
        if len(chunk_files) == 1:
            temp_video = tempfile.NamedTemporaryFile(suffix='.webm', delete=False)
            temp_video.close()
            import shutil
            shutil.copy(str(chunk_files[0]), temp_video.name)
            return temp_video.name
        
        # Create file list for ffmpeg
        temp_list = tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False)
        for chunk in chunk_files:
            temp_list.write(f"file '{chunk.absolute()}'\n")
        temp_list.close()
        
        # Create output file
        temp_output = tempfile.NamedTemporaryFile(suffix='.webm', delete=False)
        temp_output.close()
        
        try:
            # Use ffmpeg to concatenate
            cmd = [
                'ffmpeg', '-f', 'concat', '-safe', '0',
                '-i', temp_list.name,
                '-c', 'copy',  # Copy without re-encoding (faster)
                '-y',
                temp_output.name
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode != 0:
                logger.error(f"ffmpeg error: {result.stderr}")
                raise RuntimeError(f"Video merging failed: {result.stderr}")
            
            logger.info(f"   Merged video: {temp_output.name}")
            return temp_output.name
            
        finally:
            # Clean up temp file list
            os.remove(temp_list.name)
    
    def _extract_frame_features(self, frame: np.ndarray, frame_num: int, timestamp: float) -> dict:
        """Extract features from a single frame"""
        
        # Convert BGR to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Process with MediaPipe
        results = self.face_mesh.process(rgb_frame)
        
        if not results.multi_face_landmarks:
            return None  # No face detected
        
        # Get landmarks (468 points)
        landmarks = results.multi_face_landmarks[0]
        
        # Extract head pose
        head_pose = self._calculate_head_pose(landmarks, frame.shape)
        
        # Extract facial action units (eyebrows, eyes, lips)
        action_units = self._calculate_action_units(landmarks)
        
        # Extract facial geometry coefficients (simplified PCA-like features)
        geometry = self._calculate_facial_geometry(landmarks)
        
        # Combine all features
        features = {
            'frame_number': frame_num,
            'timestamp': round(timestamp, 2),
            **head_pose,
            **action_units,
            **geometry
        }
        
        return features
    
    def _calculate_head_pose(self, landmarks, frame_shape) -> dict:
        """Calculate head pose (Pitch, Yaw, Roll) using facial landmarks"""
        
        # Key landmark indices for pose estimation
        # Nose tip, chin, left eye, right eye, left mouth, right mouth
        landmark_points = [1, 152, 33, 263, 61, 291]
        
        # Extract 3D coordinates
        points_3d = np.array([
            [landmarks.landmark[i].x * frame_shape[1],
             landmarks.landmark[i].y * frame_shape[0],
             landmarks.landmark[i].z * frame_shape[1]]
            for i in landmark_points
        ])
        
        # Simplified pose estimation (using first 3 principal directions)
        centered = points_3d - points_3d.mean(axis=0)
        
        # Calculate approximate angles
        # Pitch (up/down): -15 to +15 degrees typical
        pitch = np.arctan2(centered[0, 1], centered[0, 2]) * 180 / np.pi
        
        # Yaw (left/right): -30 to +30 degrees typical
        yaw = np.arctan2(centered[0, 0], centered[0, 2]) * 180 / np.pi
        
        # Roll (head tilt): -10 to +10 degrees typical
        eye_diff = points_3d[2] - points_3d[3]
        roll = np.arctan2(eye_diff[1], eye_diff[0]) * 180 / np.pi
        
        return {
            'head_pitch': round(pitch, 2),
            'head_yaw': round(yaw, 2),
            'head_roll': round(roll, 2)
        }
    
    def _calculate_action_units(self, landmarks) -> dict:
        """Calculate facial action units (eyebrows, eyes, lips)"""
        
        # MediaPipe landmark indices
        # Eyebrows
        left_inner_brow = 70
        left_outer_brow = 105
        right_inner_brow = 300
        right_outer_brow = 334
        
        # Eyes
        left_eye_top = 159
        left_eye_bottom = 145
        right_eye_top = 386
        right_eye_bottom = 374
        
        # Lips
        upper_lip = 13
        lower_lip = 14
        left_mouth = 61
        right_mouth = 291
        
        # Calculate distances/ratios
        def distance(p1, p2):
            return np.sqrt((p1.x - p2.x)**2 + (p1.y - p2.y)**2)
        
        # Eyebrow heights (normalized)
        inBrL = landmarks.landmark[left_inner_brow].y
        otBrL = landmarks.landmark[left_outer_brow].y
        inBrR = landmarks.landmark[right_inner_brow].y
        otBrR = landmarks.landmark[right_outer_brow].y
        
        # Eye openness (aspect ratio)
        left_eye_dist = distance(landmarks.landmark[left_eye_top], 
                                landmarks.landmark[left_eye_bottom])
        right_eye_dist = distance(landmarks.landmark[right_eye_top], 
                                 landmarks.landmark[right_eye_bottom])
        
        # Normalize by face width
        face_width = distance(landmarks.landmark[234], landmarks.landmark[454])
        EyeOL = left_eye_dist / face_width if face_width > 0 else 0
        EyeOR = right_eye_dist / face_width if face_width > 0 else 0
        
        # Lip features
        oLipH = distance(landmarks.landmark[upper_lip], landmarks.landmark[lower_lip])
        iLipH = oLipH * 0.7  # Inner lip (simplified)
        LipCDt = distance(landmarks.landmark[left_mouth], landmarks.landmark[right_mouth])
        
        return {
            'inBrL': round(inBrL, 4),
            'otBrL': round(otBrL, 4),
            'inBrR': round(inBrR, 4),
            'otBrR': round(otBrR, 4),
            'EyeOL': round(EyeOL, 4),
            'EyeOR': round(EyeOR, 4),
            'oLipH': round(oLipH, 4),
            'iLipH': round(iLipH, 4),
            'LipCDt': round(LipCDt, 4)
        }
    
    def _calculate_facial_geometry(self, landmarks) -> dict:
        """Calculate facial geometry coefficients (simplified PCA-like features)"""
        
        # Extract all landmark coordinates
        coords = np.array([[lm.x, lm.y, lm.z] for lm in landmarks.landmark])
        
        # Center and normalize
        centered = coords - coords.mean(axis=0)
        
        # Calculate first 24 principal components (simplified)
        # Using variance along different directions
        variances = []
        
        # Split face into regions and calculate variance
        regions = [
            coords[:100],    # Upper face
            coords[100:200], # Mid face
            coords[200:300], # Lower face
            coords[300:]     # Jaw/chin
        ]
        
        for region in regions:
            if len(region) > 0:
                centered_region = region - region.mean(axis=0)
                # Variance in x, y, z directions
                for axis in range(3):
                    var = np.var(centered_region[:, axis])
                    variances.append(var)
        
        # Pad to 24 coefficients
        while len(variances) < 24:
            variances.append(0.0)
        
        # Create coefficient dict
        geometry = {}
        for i in range(24):
            geometry[f'dicCoeff_local{i}'] = round(variances[i], 6)
        
        return geometry


# ============================================
# CLI USAGE
# ============================================
if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python video_extractor.py <session_id> [fps]")
        print("Example: python video_extractor.py session_abc123_1234567890 10")
        sys.exit(1)
    
    session_id = sys.argv[1]
    fps = int(sys.argv[2]) if len(sys.argv) > 2 else 10
    
    # Paths (resolve using backend.config helper)
    from backend.config import get_recordings_session_dir
    import os

    session_dir = get_recordings_session_dir(session_id)
    video_dir = os.path.join(session_dir, 'video')
    output_dir = os.path.join('processed_features', session_id)
    output_path = os.path.join(output_dir, 'video_features_raw.csv')
    
    # Check if video directory exists
    if not os.path.exists(video_dir):
        print(f"❌ Video directory not found: {video_dir}")
        sys.exit(1)
    
    # Extract features
    extractor = VideoFeatureExtractor(fps=fps)
    
    try:
        features_df = extractor.extract_features(video_dir, output_path)
        
        print("\n" + "="*60)
        print("✅ VIDEO FEATURE EXTRACTION COMPLETE")
        print("="*60)
        print(f"Input:  {video_dir}")
        print(f"Output: {output_path}")
        print(f"Features extracted: {len(features_df.columns)}")
        print(f"Frames processed: {len(features_df)}")
        print("\n📊 Sample features:")
        print(features_df.head())
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)