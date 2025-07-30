import os
import cv2
import shutil
from mtcnn import MTCNN
from tqdm import tqdm

class FaceProcessor:
    """Handles extracting faces from video frames using MTCNN."""
    def __init__(self):
        print("Initializing MTCNN face detector...")
        self.detector = MTCNN()
        print("MTCNN Initialized.")

    def extract_face_frames(self, video_path, output_dir, num_frames, image_size):
        """Extracts a set number of face frames from a single video."""
        print(f"--- Extracting Frames from {os.path.basename(video_path)} ---")
        if os.path.exists(output_dir):
            shutil.rmtree(output_dir)
        os.makedirs(output_dir)

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"Error: Unable to open video {video_path}")
            return []

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_frames == 0:
            cap.release()
            return []

        step = max(1, total_frames // num_frames)
        saved_frame_paths = []

        for i in tqdm(range(num_frames), desc="Extracting Faces"):
            cap.set(cv2.CAP_PROP_POS_FRAMES, i * step)
            ret, frame = cap.read()
            if not ret: break

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.detector.detect_faces(frame_rgb)

            if results:
                x, y, w, h = results[0]['box']
                x, y = max(0, x), max(0, y)
                face = frame[y:y+h, x:x+w]

                if face.size == 0: continue

                face_resized = cv2.resize(face, image_size)
                face_path = os.path.join(output_dir, f"frame_{i}.jpg")
                cv2.imwrite(face_path, face_resized)
                saved_frame_paths.append(face_path)

        cap.release()
        print(f"Successfully extracted {len(saved_frame_paths)} face frames.\n")
        return saved_frame_paths