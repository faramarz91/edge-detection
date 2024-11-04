import streamlit as st
import cv2
from edge_detection import apply_edge_detection
from feature_extraction import extract_features
from feature_matching import match_features
from utils import load_image, remove_noise

# Streamlit interface setup
st.title('Edge and Feature-based Object Recognition Project')
st.write('Upload images to perform noise removal, edge detection, feature extraction, and matching.')

uploaded_file1 = st.file_uploader("Upload the first image", type=["jpg", "png", "jpeg"])
uploaded_file2 = st.file_uploader("Upload the second image (for matching)", type=["jpg", "png", "jpeg"])

if uploaded_file1:
    img1 = load_image(uploaded_file1)
    denoised_img1 = remove_noise(img1)

    st.subheader('Original vs. Denoised Image 1')
    compare_switch1 = st.checkbox("Show denoised image for Image 1", key="switch1")

    if compare_switch1:
        st.image(denoised_img1, caption="Denoised Image 1", channels="BGR")
    else:
        st.image(img1, caption="Original Image 1", channels="BGR")

if uploaded_file2:
    img2 = load_image(uploaded_file2)
    denoised_img2 = remove_noise(img2)

    st.subheader('Original vs. Denoised Image 2')
    compare_switch2 = st.checkbox("Show denoised image for Image 2", key="switch2")

    if compare_switch2:
        st.image(denoised_img2, caption="Denoised Image 2", channels="BGR")
    else:
        st.image(img2, caption="Original Image 2", channels="BGR")

if uploaded_file1 and uploaded_file2:
    # Edge Detection and display
    edges1 = apply_edge_detection(denoised_img1 if compare_switch1 else img1)
    edges2 = apply_edge_detection(denoised_img2 if compare_switch2 else img2)
    st.subheader('Edge Detection Results')
    st.image([edges1, edges2], caption=["Edges of Image 1", "Edges of Image 2"], use_column_width=True, channels="GRAY")

    # Feature Extraction
    keypoints1, descriptors1 = extract_features(denoised_img1 if compare_switch1 else img1)
    keypoints2, descriptors2 = extract_features(denoised_img2 if compare_switch2 else img2)

    # Feature Matching
    matched_image = match_features(
        denoised_img1 if compare_switch1 else img1, keypoints1, descriptors1,
        denoised_img2 if compare_switch2 else img2, keypoints2, descriptors2
    )
    st.subheader('Feature Matching Result')
    st.image(matched_image, caption="Feature Matching Result", channels="BGR")
