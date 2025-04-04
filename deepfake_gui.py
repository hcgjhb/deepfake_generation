import streamlit as st
from PIL import Image
import io
import time
import cv2
import numpy as np
from face_swap import * 

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
                      "Face Modification", 
                      "Face Morphing", 
                      "Emotion Transfer", 
                      "Diffusion Generation"])
        
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
                    st.image(pil_source, use_container_width=True)  # CHANGED HERE
                
        with col2:
            st.markdown("### Target Image")
            target_img_file = st.file_uploader("Upload target image", type=['jpg', 'jpeg', 'png'], key="target_image")
            if target_img_file:
                pil_target = Image.open(target_img_file)
                target_image = cv2.cvtColor(np.array(pil_target), cv2.COLOR_RGB2BGR)
                with st.container():
                    st.markdown('<div class="image-box"><h3>Target Image</h3></div>', unsafe_allow_html=True)
                    st.image(pil_target, use_container_width=True)  # CHANGED HERE
        
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
                source_convex_hull = get_convex_hull(source_landmark_points, source_image)
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
    
    # Other features abbreviated for simplicity
    else:
        st.info(f"Selected {page} - Implement this feature by expanding the code.")
    
    # Footer
    st.markdown("---")
    st.markdown("DeepFake Studio 2025 | Built with Streamlit")

if __name__ == "__main__":
    main()