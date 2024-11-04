import cv2

def apply_edge_detection(image):
    # Convert image to grayscale
    grayscale = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Apply Canny edge detection
    edges = cv2.Canny(grayscale, 100, 200)
    return edges
