import streamlit as st
from PIL import Image
import io
import time
import cv2
import numpy as np
from face_swap import * 
import os
import tempfile
import subprocess
import shutil
from face_morphing import process_video_in_batches  # Import the face morphing function
import moviepy
# Set page configuration
st.set_page_config(
    page_title="DeepFake Studio",
    page_icon="DF",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .image-box {
        border-radius: 15px;
        padding: 15px;
        background: #1e1e1e;
        box-shadow: 0 4px 8px 0 rgba(0, 0, 0, 0.2);
        transition: 0.3s;
        margin-bottom: 20px;
        height: 100%;
    }
    .image-box:hover {
        box-shadow: 0 8px 16px 0 rgba(0, 0, 0, 0.2);
    }
    .image-box h3 {
        color: #ffffff;
        text-align: center;
        margin-top: 0;
    }
    .stImage>img {
        border-radius: 10px;
        object-fit: contain;
        width: 100%;
        height: 300px;
    }
    </style>
    """, unsafe_allow_html=True)

def resize_image(image, target_size=(300, 300)):
    """Resize image to target size while maintaining aspect ratio"""
    img = Image.fromarray(image)
    img.thumbnail(target_size)
    background = Image.new('RGB', target_size, (0, 0, 0))
    background.paste(img, ((target_size[0] - img.size[0]) // 2, (target_size[1] - img.size[1]) // 2))
    return np.array(background)

def convert_video_to_mp4(input_path, output_path):
    """Convert video to MP4 format with H.264 codec using ffmpeg"""
    try:
        cmd = [
            'ffmpeg',
            '-y',  # overwrite without asking
            '-i', input_path,
            '-c:v', 'libx264',
            '-preset', 'fast',
            '-crf', '22',
            '-c:a', 'copy',
            '-movflags', '+faststart',
            output_path
        ]
        subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        return True
    except subprocess.CalledProcessError as e:
        st.error(f"Video conversion failed: {e}")
        return False
    except Exception as e:
        st.error(f"Error in video conversion: {e}")
        return False


def process_face_morphing(source_img, target_video):
    """Handle the face morphing process with temporary files"""
    with st.spinner("Processing face morphing..."):
        # Create temporary directory
        temp_dir = tempfile.mkdtemp()
        
        try:
            # Save source image
            source_path = os.path.join(temp_dir, "source.jpg")
            source_img.save(source_path)
            
            # Save target video
            target_path = os.path.join(temp_dir, "target.mp4")
            with open(target_path, "wb") as f:
                f.write(target_video.read())
            
            # Output path (use the original video format)
            output_path = os.path.join(temp_dir, "output" + os.path.splitext(target_video.name)[1])
            
            # Validate input files
            if not os.path.exists(source_path):
                raise ValueError("Source image could not be saved")
            
            if not os.path.exists(target_path):
                raise ValueError("Target video could not be saved")
            
            # Process the video
            success = process_video_in_batches(
                source_path=source_path,
                input_path=target_path,
                output_path=output_path,
                resize_factor=0.5,
                frame_skip=5,
                batch_size=10,
                max_frames=300
            )
            
            # Additional validation
            if not success:
                raise ValueError("Video processing failed")
            
            if not os.path.exists(output_path):
                raise ValueError("Output video was not created")
            
            # Check output video file size
            if os.path.getsize(output_path) == 0:
                raise ValueError("Output video is empty")
            
            return output_path
        
        except Exception as e:
            # Log the full error for debugging
            print(f"Detailed error during face morphing: {e}")
            
            # User-friendly error message
            st.error(f"Face morphing failed: {e}. Please check your input files.")
            
            return None
        finally:
            # Optional: Clean up temporary files if processing fails
            try:
                if 'temp_dir' in locals() and os.path.exists(temp_dir):
                    shutil.rmtree(temp_dir, ignore_errors=True)
            except Exception as cleanup_error:
                print(f"Error during cleanup: {cleanup_error}")

# Main app
def main():
    # Sidebar
    with st.sidebar:
        st.title("DeepFake Studio")
        st.markdown("---")
        
        # Navigation
        st.subheader("Navigation")
        page = st.radio("", 
                     ["Face Swap", 
                      "Style Transfer",
                      "Face Morphing",   # Changed from "Face Modification"
                      "Attribute Transfer"
                      ])
        
        st.markdown("---")
        
        # About section
        with st.expander("About"):
            st.write("""
            DeepFake Studio is a powerful computer vision application 
            featuring state-of-the-art deepfake and face manipulation technologies.
            """)
    
    # Main content
    st.title("DeepFake Studio")
    st.markdown("Advanced facial manipulation powered by AI")
    
    if page == "Face Swap":
        st.header("Face Swap")
        st.markdown("Upload two images to swap the face from one to the other.")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### Source Face")
            source_img_file = st.file_uploader("Upload source face", type=['jpg', 'jpeg', 'png'], key="source_face")
            if source_img_file:
                pil_source = Image.open(source_img_file)
                source_image = cv2.cvtColor(np.array(pil_source), cv2.COLOR_RGB2BGR)
                with st.container():
                    st.markdown('<div class="image-box"><h3>Source Image</h3></div>', unsafe_allow_html=True)
                    st.image(pil_source, use_container_width=True)
                
        with col2:
            st.markdown("### Target Image")
            target_img_file = st.file_uploader("Upload target image", type=['jpg', 'jpeg', 'png'], key="target_image")
            if target_img_file:
                pil_target = Image.open(target_img_file)
                target_image = cv2.cvtColor(np.array(pil_target), cv2.COLOR_RGB2BGR)
                with st.container():
                    st.markdown('<div class="image-box"><h3>Target Image</h3></div>', unsafe_allow_html=True)
                    st.image(pil_target, use_container_width=True)
        
        if source_img_file and target_img_file:
            if st.button("Perform Face Swap", key="face_swap_btn"):
                st.markdown("### Processing...")
                
                progress_bar = st.progress(0)
                status_text = st.empty()
                for i in range(101):
                    progress_bar.progress(i)
                    if i < 30:
                        status_text.text("Detecting faces...")
                    elif i < 60:
                        status_text.text("Aligning facial features...")
                    elif i < 90:
                        status_text.text("Applying face swap...")
                    else:
                        status_text.text("Finalizing results...")
                    time.sleep(0.02)

                # Run face swap using proper OpenCV images
                source_landmark_points = get_landmark_points(source_image)
                source_convex_hull = get_convex_hull(source_landmark_points)
                source_bounding_rect = get_bounding_rectangle(source_image, source_convex_hull)
                source_triangles = delunay_triangulation(source_bounding_rect, source_landmark_points)
                source_indexes_triangles = get_index_triangles(source_triangles, source_landmark_points)
                new_face = swap_faces(source_indexes_triangles, source_image, target_image)
                
                status_text.text("Face swap completed!")
                
                # Convert result from BGR to RGB for display
                result_rgb = cv2.cvtColor(new_face, cv2.COLOR_BGR2RGB)
                
                # Display all three images in a row after processing
                col1, col2, col3 = st.columns(3)
                with col1:
                    with st.container():
                        st.markdown('<div class="image-box"><h3>Source Image</h3></div>', unsafe_allow_html=True)
                        st.image(pil_source, use_container_width=True) 
                with col2:
                    with st.container():
                        st.markdown('<div class="image-box"><h3>Target Image</h3></div>', unsafe_allow_html=True)
                        st.image(pil_target, use_container_width=True) 
                with col3:
                    with st.container():
                        st.markdown('<div class="image-box"><h3>Result Image</h3></div>', unsafe_allow_html=True)
                        st.image(result_rgb, use_container_width=True) 
                
                buf = io.BytesIO()
                Image.fromarray(result_rgb).save(buf, format="PNG")
                byte_im = buf.getvalue()
                
                st.download_button(
                    label="Download Result",
                    data=byte_im,
                    file_name="face_swap_result.png",
                    mime="image/png",
                    use_container_width=True
                )


    elif page == "Style Transfer":

        st.header("🎨 Style Transfer")

        st.markdown("Apply a painting or artistic style to your photo using deep learning.")

        col1, col2 = st.columns(2)

        with col1:

            st.markdown("### Content Image")

            content_img_file = st.file_uploader("Upload content image", type=['jpg', 'jpeg', 'png'],
                                                key="content_image")

            if content_img_file:
                pil_content = Image.open(content_img_file)

                with st.container():
                    st.markdown('<div class="image-box"><h3>Content Image</h3></div>', unsafe_allow_html=True)

                    st.image(pil_content, use_container_width=True)

        with col2:

            st.markdown("### Style Image")

            style_img_file = st.file_uploader("Upload style image", type=['jpg', 'jpeg', 'png'], key="style_image")

            if style_img_file:
                pil_style = Image.open(style_img_file)

                with st.container():
                    st.markdown('<div class="image-box"><h3>Style Image</h3></div>', unsafe_allow_html=True)

                    st.image(pil_style, use_container_width=True)

        if content_img_file and style_img_file:

            if st.button("Apply Style Transfer", key="style_transfer_btn"):

                st.markdown("### Processing...")

                # Simulate progress

                progress_bar = st.progress(0)

                status_text = st.empty()

                for i in range(101):

                    progress_bar.progress(i)

                    if i < 25:

                        status_text.text("Loading style model...")

                    elif i < 50:

                        status_text.text("Preparing images...")

                    elif i < 75:

                        status_text.text("Blending content and style...")

                    else:

                        status_text.text("Finalizing output...")

                    time.sleep(0.015)

                from style_transfer import perform_style_transfer  # Import inside the action

                # Run style transfer

                result_image = perform_style_transfer(pil_content, pil_style)

                status_text.text("Style transfer completed!")

                # Display result

                with st.container():

                    st.markdown('<div class="image-box"><h3>Stylized Result</h3></div>', unsafe_allow_html=True)

                    st.image(result_image, use_container_width=True)

                # Download button

                buf = io.BytesIO()

                result_image.save(buf, format="PNG")

                byte_im = buf.getvalue()

                st.download_button(

                    label="Download Stylized Image",

                    data=byte_im,

                    file_name="stylized_result.png",

                    mime="image/png",

                    use_container_width=True

                )




    # Other features abbreviated for simplicity

    elif page == "Face Morphing":

        st.header("🌀 Face Morph")

        st.markdown("Swap a face from an image into every frame of a video.")

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("### Upload Source Image")
            source_image_file = st.file_uploader("Source image (with face)", type=['jpg', 'jpeg', 'png'],
                                                 key="face_image")
            if source_image_file:
                st.image(source_image_file, caption="Source Image", use_container_width=True)

        with col2:
            st.markdown("### Upload Input Video")
            input_video_file = st.file_uploader("Target video", type=['mp4', 'mov', 'avi'], key="input_video")
            if input_video_file:
                st.video(input_video_file)

        if source_image_file and input_video_file:
            if st.button("🔁 Run Face Morph"):
                st.markdown("### Processing...")

                # Progress bar
                progress_bar = st.progress(0)
                status_text = st.empty()

                for i in range(101):
                    progress_bar.progress(i)
                    status_text.text(f"Processing... {i}%")
                    time.sleep(0.01)

                # Save temp files
                from pathlib import Path
                import uuid

                temp_dir = Path("outputs")
                temp_dir.mkdir(exist_ok=True)

                unique_id = uuid.uuid4().hex[:8]
                source_path = temp_dir / f"source_{unique_id}.jpg"
                input_path = temp_dir / f"input_{unique_id}.mp4"
                output_path = temp_dir / f"output_{unique_id}.mp4"

                # Save uploaded files
                with open(source_path, "wb") as f:
                    f.write(source_image_file.read())
                with open(input_path, "wb") as f:
                    f.write(input_video_file.read())

                # Import face swap logic
                from face_morphing import process_video_in_batches

                result_path = process_video_in_batches(
                    source_path=str(source_path),
                    input_path=str(input_path),
                    output_path=str(output_path),
                    resize_factor=0.4,
                    frame_skip=5,
                    batch_size=10,
                    max_frames=300
                )

                from moviepy.video.io.VideoFileClip import VideoFileClip

                def reencode_video(input_path, output_path):
                    clip = VideoFileClip(input_path)
                    clip.write_videofile(output_path, codec='libx264', audio_codec='aac')
                    clip.close()

                # Paths
                original_path = result_path  # the face-swap output path
                fixed_path = original_path.replace(".mp4", "_fixed.mp4")

                # Re-encode for compatibility
                reencode_video(original_path, fixed_path)

                # Now use the re-encoded video
                result_path = fixed_path

                if result_path and os.path.exists(result_path):
                    st.success("✅ Face swap complete!")
                    print(f"Video path: {result_path}")

                    # Read the video as binary
                    with open(result_path, "rb") as file:
                        video_bytes = file.read()
                        st.video(video_bytes)

                    # Optional: download button
                    with open(result_path, "rb") as video_file:
                        st.download_button(
                            label="📥 Download Output Video",
                            data=video_file,
                            file_name="face_swapped_output.mp4",
                            mime="video/mp4"
                        )

                else:
                    st.error("❌ Something went wrong during face swapping.")






    elif page == "Attribute Transfer":

        st.header("😊 Attribute Transfer")

        st.markdown(

            """

            Upload a **source image** with the facial attribute (e.g., a smile) and a **target image** with a neutral face.  

            Click the button to transfer the facial attribute from the source to the target.

            """

        )

        col1, col2 = st.columns(2)

        with col1:

            source_img_file = st.file_uploader("Upload Source Image (with Attribute)", type=["jpg", "jpeg", "png"],
                                               key="attr_source")

            if source_img_file:
                source_img = Image.open(source_img_file).convert("RGB")

                st.image(source_img, caption="Source Image", use_container_width=True)

        with col2:

            target_img_file = st.file_uploader("Upload Target Image (Neutral Face)", type=["jpg", "jpeg", "png"],
                                               key="attr_target")

            if target_img_file:
                target_img = Image.open(target_img_file).convert("RGB")

                st.image(target_img, caption="Target Image", use_container_width=True)

        if source_img_file and target_img_file:

            if st.button("Transfer Attribute"):

                with st.spinner("Transferring attribute..."):

                    try:

                        from attribute_transfer import run_emotion_transfer

                        result = run_emotion_transfer(source_img, target_img)

                        st.success("✅ Attribute transfer complete!")

                        st.image(result, caption="Result Image (Attribute Transferred)", use_container_width=True)

                        # Convert result to byte stream for download

                        img_bytes = io.BytesIO()

                        result.save(img_bytes, format='PNG')

                        img_bytes.seek(0)

                        st.download_button(

                            label="📥 Download Result Image",

                            data=img_bytes,

                            file_name="attribute_transferred_result.png",

                            mime="image/png"

                        )


                    except Exception as e:

                        st.error(f"❌ Something went wrong during transfer: {e}")




    else:
        st.info(f"Selected {page} - Implement this feature by expanding the code.")
    
    # Footer
    st.markdown("---")
    st.markdown("DeepFake Studio 2025 | Built with Streamlit")

if __name__ == "__main__":
    main()
