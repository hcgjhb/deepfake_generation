import cv2
import numpy as np
import os

def ensure_dir(dir_path):
    """Create directory if it doesn't exist"""
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

def detect_face(image):
    """Detect a single face in an image"""
    try:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        faces = face_cascade.detectMultiScale(gray, 1.3, 5, minSize=(30, 30))
        if len(faces) > 0:
            return faces[0]
        return None
    except Exception as e:
        print(f"Error in face detection: {e}")
        return None

def basic_face_swap(source_face, target_image):
    """Basic face swap on a single image"""
    try:
        target_face_rect = detect_face(target_image)
        if target_face_rect is None:
            return target_image
        x, y, w, h = target_face_rect
        result = target_image.copy()
        resized_face = cv2.resize(source_face, (w, h), interpolation=cv2.INTER_NEAREST)

        # Create soft-edged blending mask
        mask = np.zeros((h, w), dtype=np.uint8)
        margin = int(min(w, h) * 0.2)
        cv2.rectangle(mask, (margin, margin), (w-margin, h-margin), 255, -1)
        mask = cv2.GaussianBlur(mask, (margin*2+1, margin*2+1), 0)

        for c in range(3):
            blend = (1 - mask / 255.0) * result[y:y+h, x:x+w, c] + (mask / 255.0) * resized_face[:, :, c]
            result[y:y+h, x:x+w, c] = blend
        return result
    except Exception as e:
        print(f"Error in face swap: {e}")
        return target_image

def extract_source_face(source_path, resize_factor=0.5):
    """Extract face from source image"""
    try:
        source_img = cv2.imread(source_path, cv2.IMREAD_COLOR)
        if source_img is None:
            raise ValueError(f"Could not load source image: {source_path}")
        if resize_factor < 1.0:
            h, w = source_img.shape[:2]
            new_w = int(w * resize_factor)
            new_h = int(h * resize_factor)
            source_img = cv2.resize(source_img, (new_w, new_h), interpolation=cv2.INTER_NEAREST)
        face_rect = detect_face(source_img)
        if face_rect is None:
            raise ValueError("No face detected in source image")
        x, y, w, h = face_rect
        return source_img[y:y+h, x:x+w].copy()
    except Exception as e:
        print(f"Error extracting source face: {e}")
        return None

def extract_frames(video_path, output_dir, frame_skip=5, resize_factor=0.25, max_frames=1000):
    """Extract frames from video in batches"""
    try:
        ensure_dir(output_dir)
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Could not open video: {video_path}")
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        new_width = int(width * resize_factor)
        new_height = int(height * resize_factor)

        frame_count, saved_count = 0, 0
        while saved_count < max_frames:
            ret, frame = cap.read()
            if not ret:
                break
            if frame_count % frame_skip == 0:
                small_frame = cv2.resize(frame, (new_width, new_height), interpolation=cv2.INTER_NEAREST)
                out_path = os.path.join(output_dir, f"frame_{saved_count:04d}.jpg")
                cv2.imwrite(out_path, small_frame)
                saved_count += 1
            frame_count += 1
        cap.release()
        return saved_count, new_width, new_height, fps
    except Exception as e:
        print(f"Error extracting frames: {e}")
        return 0, 0, 0, 0

def process_frames(source_face, frames_dir, output_dir, batch_size=10):
    """Process frames in small batches"""
    try:
        ensure_dir(output_dir)
        frame_files = sorted([f for f in os.listdir(frames_dir) if f.startswith("frame_") and f.endswith(".jpg")])
        total_frames = len(frame_files)

        for i in range(0, total_frames, batch_size):
            for j in range(i, min(i + batch_size, total_frames)):
                frame_path = os.path.join(frames_dir, frame_files[j])
                frame = cv2.imread(frame_path)
                if frame is None:
                    continue
                result = basic_face_swap(source_face, frame)
                out_path = os.path.join(output_dir, f"out_{j:04d}.jpg")
                cv2.imwrite(out_path, result)
        return True
    except Exception as e:
        print(f"Error in batch processing: {e}")
        return False

def create_video_from_frames(frames_dir, output_path, width, height, fps, frame_skip):
    """Create video from processed frames"""
    try:
        frame_files = sorted([f for f in os.listdir(frames_dir) if f.startswith("out_") and f.endswith(".jpg")])
        if not frame_files:
            raise ValueError("No output frames found")
        adjusted_fps = fps / frame_skip
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, adjusted_fps, (width, height))
        for frame_file in frame_files:
            frame_path = os.path.join(frames_dir, frame_file)
            frame = cv2.imread(frame_path)
            if frame is not None:
                out.write(frame)
        out.release()
        return True
    except Exception as e:
        print(f"Error creating video: {e}")
        return False

def process_video_in_batches(source_path, input_path, output_path,
                             resize_factor=0.25, frame_skip=5,
                             batch_size=10, max_frames=300):
    """High-level function to process face swap on video"""
    frames_dir = "frames"
    output_frames_dir = "output_frames"

    try:
        ensure_dir(frames_dir)
        ensure_dir(output_frames_dir)

        source_face = extract_source_face(source_path, resize_factor)
        if source_face is None:
            raise ValueError("No face found in source image")

        num_frames, width, height, fps = extract_frames(
            input_path, frames_dir, frame_skip, resize_factor, max_frames)
        if num_frames == 0:
            raise ValueError("No frames extracted")

        success = process_frames(source_face, frames_dir, output_frames_dir, batch_size)
        if not success:
            raise ValueError("Frame processing failed")

        success = create_video_from_frames(output_frames_dir, output_path, width, height, fps, frame_skip)
        if not success:
            raise ValueError("Video creation failed")

        return output_path
    except Exception as e:
        print(f"Error in batch video processing: {e}")
        return None
