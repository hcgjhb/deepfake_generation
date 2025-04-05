import cv2
import numpy as np
import os
import argparse
import gc
import time
import shutil
import subprocess

def ensure_dir(dir_path):
    """Create directory if it doesn't exist"""
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

def detect_face(image):
    """Detect a single face in an image"""
    try:
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Load face detector
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        
        # Detect faces
        faces = face_cascade.detectMultiScale(gray, 1.3, 5, minSize=(30, 30))
        
        # Free memory
        del gray
        gc.collect()
        
        if len(faces) > 0:
            return faces[0]  # Return first face [x, y, w, h]
        else:
            return None
            
    except Exception as e:
        print(f"Error in face detection: {e}")
        return None

def basic_face_swap(source_face, target_image):
    """Basic face swap on a single image"""
    try:
        # Detect face in target image
        target_face_rect = detect_face(target_image)
        
        if target_face_rect is None:
            return target_image  # No face detected
            
        # Extract coordinates
        x, y, w, h = target_face_rect
        
        # Create a copy of the target
        result = target_image.copy()
        
        # Resize source face to match target face size
        try:
            resized_face = cv2.resize(source_face, (w, h), interpolation=cv2.INTER_NEAREST)
        except Exception as e:
            print(f"Error resizing face: {e}")
            return target_image
        
        # Create a simple mask
        mask = np.zeros((h, w), dtype=np.uint8)
        margin = int(min(w, h) * 0.2)
        cv2.rectangle(mask, (margin, margin), (w-margin, h-margin), 255, -1)
        mask = cv2.GaussianBlur(mask, (margin*2+1, margin*2+1), 0)
        
        # Apply alpha blending
        for c in range(3):
            blend = (1 - mask/255.0) * result[y:y+h, x:x+w, c] + (mask/255.0) * resized_face[:, :, c]
            result[y:y+h, x:x+w, c] = blend
        
        # Clean up
        del resized_face
        del mask
        gc.collect()
        
        return result
        
    except Exception as e:
        print(f"Error in face swap: {e}")
        return target_image

def extract_source_face(source_path, resize_factor=0.5):
    """Extract face from source image"""
    try:
        # Load source image at reduced resolution
        source_img = cv2.imread(source_path, cv2.IMREAD_REDUCED_COLOR_2)
        
        if source_img is None:
            raise ValueError(f"Could not load source image: {source_path}")
            
        # Further resize if needed
        if resize_factor < 0.5:
            h, w = source_img.shape[:2]
            new_w = int(w * resize_factor * 2)  # Adjust for IMREAD_REDUCED_COLOR_2
            new_h = int(h * resize_factor * 2)
            source_img = cv2.resize(source_img, (new_w, new_h), interpolation=cv2.INTER_NEAREST)
        
        # Detect face
        face_rect = detect_face(source_img)
        
        if face_rect is None:
            raise ValueError("No face detected in source image")
            
        # Extract face region
        x, y, w, h = face_rect
        source_face = source_img[y:y+h, x:x+w].copy()
        
        # Clean up
        del source_img
        gc.collect()
        
        return source_face
        
    except Exception as e:
        print(f"Error extracting source face: {e}")
        return None

def extract_frames(video_path, output_dir, frame_skip=5, resize_factor=0.25, max_frames=1000):
    """Extract frames from video in batches"""
    try:
        # Ensure output directory exists
        ensure_dir(output_dir)
        
        # Open video
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Could not open video: {video_path}")
            
        # Get video properties
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Calculate new dimensions
        new_width = int(width * resize_factor)
        new_height = int(height * resize_factor)
        
        print(f"Video info: {width}x{height}, {fps} fps, {total_frames} frames")
        print(f"Output frames: {new_width}x{new_height}")
        print(f"Extracting every {frame_skip}th frame (max: {max_frames})")
        
        # Extract frames
        frame_count = 0
        saved_count = 0
        
        while saved_count < max_frames:
            # Read frame
            ret, frame = cap.read()
            if not ret:
                break
                
            # Process selected frames
            if frame_count % frame_skip == 0:
                try:
                    # Resize frame
                    small_frame = cv2.resize(frame, (new_width, new_height), 
                                          interpolation=cv2.INTER_NEAREST)
                    
                    # Save frame
                    out_path = os.path.join(output_dir, f"frame_{saved_count:04d}.jpg")
                    cv2.imwrite(out_path, small_frame)
                    
                    saved_count += 1
                    
                    if saved_count % 20 == 0:
                        print(f"Extracted {saved_count} frames...")
                        
                    # Clean up
                    del small_frame
                    gc.collect()
                    
                except Exception as e:
                    print(f"Error saving frame {frame_count}: {e}")
            
            # Free memory
            del frame
            gc.collect()
            
            frame_count += 1
            
            # Force GC
            if frame_count % 30 == 0:
                time.sleep(0.1)  # Allow system to clean up memory
        
        cap.release()
        
        print(f"Extracted {saved_count} frames from {frame_count} total frames")
        return saved_count, new_width, new_height, fps
        
    except Exception as e:
        print(f"Error extracting frames: {e}")
        return 0, 0, 0, 0

def process_frames(source_face, frames_dir, output_dir, batch_size=10):
    """Process frames in small batches"""
    try:
        # Ensure output directory exists
        ensure_dir(output_dir)
        
        # Get list of frame files
        frame_files = sorted([f for f in os.listdir(frames_dir) if f.startswith("frame_") and f.endswith(".jpg")])
        total_frames = len(frame_files)
        
        print(f"Processing {total_frames} frames in batches of {batch_size}")
        
        # Process frames in batches
        for i in range(0, total_frames, batch_size):
            batch_end = min(i + batch_size, total_frames)
            print(f"Processing batch {i//batch_size + 1}/{(total_frames+batch_size-1)//batch_size} (frames {i}-{batch_end-1})")
            
            # Process each frame in the batch
            for j in range(i, batch_end):
                try:
                    # Load frame
                    frame_path = os.path.join(frames_dir, frame_files[j])
                    frame = cv2.imread(frame_path)
                    
                    if frame is None:
                        print(f"Could not load frame: {frame_path}")
                        continue
                    
                    # Apply face swap
                    result = basic_face_swap(source_face, frame)
                    
                    # Save result
                    out_path = os.path.join(output_dir, f"out_{j:04d}.jpg")
                    cv2.imwrite(out_path, result)
                    
                    # Clean up
                    del frame
                    del result
                    gc.collect()
                    
                except Exception as e:
                    print(f"Error processing frame {j}: {e}")
            
            # Force GC after each batch
            gc.collect()
            time.sleep(0.2)  # Short pause to let memory clear
            
        print(f"Processed {total_frames} frames")
        return True
        
    except Exception as e:
        print(f"Error in batch processing: {e}")
        return False

def create_video_from_frames(frames_dir, output_path, width, height, fps, frame_skip):
    """Create video from processed frames"""
    try:
        # Get list of output frames
        frame_files = sorted([f for f in os.listdir(frames_dir) if f.startswith("out_") and f.endswith(".jpg")])
        
        if not frame_files:
            raise ValueError("No output frames found")
            
        print(f"Creating video from {len(frame_files)} frames")
        
        # Create video writer
        adjusted_fps = fps / frame_skip  # Adjust FPS for skipped frames
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        out = cv2.VideoWriter(output_path, fourcc, adjusted_fps, (width, height))
        
        if not out.isOpened():
            raise ValueError(f"Could not create output video: {output_path}")
            
        # Add frames to video
        for i, frame_file in enumerate(frame_files):
            try:
                # Load frame
                frame_path = os.path.join(frames_dir, frame_file)
                frame = cv2.imread(frame_path)
                
                if frame is None:
                    print(f"Could not load frame: {frame_path}")
                    continue
                    
                # Add to video
                out.write(frame)
                
                # Progress update
                if i % 20 == 0:
                    print(f"Added {i}/{len(frame_files)} frames to video")
                    
                # Clean up
                del frame
                gc.collect()
                
            except Exception as e:
                print(f"Error adding frame {frame_file}: {e}")
                
        # Finish up
        out.release()
        print(f"Video created: {output_path}")
        return True
        
    except Exception as e:
        print(f"Error creating video: {e}")
        return False

def cleanup_temp_files(temp_dir):
    """Remove temporary files"""
    try:
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)
            print(f"Removed temporary directory: {temp_dir}")
    except Exception as e:
        print(f"Error cleaning up temp files: {e}")

def process_video_in_batches(source_path, input_path, output_path, 
                            resize_factor=0.25, frame_skip=5, 
                            batch_size=10, max_frames=300):
    """Process video in small batches to minimize memory usage"""
    
    # Create temporary directories
    temp_dir = "temp_face_swap"
    frames_dir = os.path.join(temp_dir, "frames")
    output_frames_dir = os.path.join(temp_dir, "output")
    
    try:
        # 1. Create temp directories
        ensure_dir(frames_dir)
        ensure_dir(output_frames_dir)
        
        # 2. Extract source face
        print("Extracting source face...")
        source_face = extract_source_face(source_path, resize_factor)
        if source_face is None:
            raise ValueError("Failed to extract source face")
            
        # Save source face for verification
        cv2.imwrite(os.path.join(temp_dir, "source_face.jpg"), source_face)
        print(f"Source face extracted: {source_face.shape}")
        
        # 3. Extract frames from video
        print("Extracting frames from video...")
        num_frames, width, height, fps = extract_frames(
            input_path, frames_dir, 
            frame_skip=frame_skip, 
            resize_factor=resize_factor,
            max_frames=max_frames
        )
        
        if num_frames == 0:
            raise ValueError("No frames were extracted from the video")
            
        # 4. Process frames in batches
        print("Processing frames...")
        success = process_frames(
            source_face, frames_dir, output_frames_dir, 
            batch_size=batch_size
        )
        
        if not success:
            raise ValueError("Frame processing failed")
            
        # 5. Create output video
        print("Creating output video...")
        success = create_video_from_frames(
            output_frames_dir, output_path, 
            width, height, fps, frame_skip
        )
        
        if not success:
            raise ValueError("Video creation failed")
            
        print(f"Face swap complete. Output: {output_path}")
        return True
        
    except Exception as e:
        print(f"Error in batch processing: {e}")
        return False
        
    finally:
        # Always clean up temp files
        if os.path.exists(temp_dir):
            # Keep files for debugging - comment this out to automatically delete temp files
            print(f"Temporary files saved in: {temp_dir}")
            # Uncomment to delete temp files:
            # cleanup_temp_files(temp_dir)

def main():
    # Parse arguments
    parser = argparse.ArgumentParser(description="Batch-based face swap for memory-constrained systems")
    parser.add_argument("--source", "-s", required=True, help="Source image with face")
    parser.add_argument("--input", "-i", required=True, help="Input video")
    parser.add_argument("--output", "-o", required=True, help="Output video")
    parser.add_argument("--resize", "-r", type=float, default=0.25, help="Processing resolution factor (default: 0.25)")
    parser.add_argument("--skip", "-k", type=int, default=5, help="Process every Nth frame (default: 5)")
    parser.add_argument("--batch", "-b", type=int, default=10, help="Batch size for processing (default: 10)")
    parser.add_argument("--max-frames", "-m", type=int, default=300, help="Maximum frames to process (default: 300)")
    
    args = parser.parse_args()
    
    print("FaceIt Batch - Process Frames In Small Batches")
    print("---------------------------------------------")
    
    # Run the face swap in batches
    process_video_in_batches(
        args.source,
        args.input,
        args.output,
        resize_factor=args.resize,
        frame_skip=args.skip,
        batch_size=args.batch,
        max_frames=args.max_frames
    )
    
if __name__ == "__main__":
    main()