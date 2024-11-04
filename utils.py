import cv2
import numpy as np

def load_image(uploaded_file):
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    return image

def remove_noise(image):
    # Apply a Gaussian blur to remove noise
    denoised_image = cv2.GaussianBlur(image, (5, 5), 0)
    return denoised_image
