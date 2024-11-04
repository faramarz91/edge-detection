import cv2

def extract_features(image):
    grayscale = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Use ORB for feature extraction
    orb = cv2.ORB_create()
    keypoints, descriptors = orb.detectAndCompute(grayscale, None)
    return keypoints, descriptors
