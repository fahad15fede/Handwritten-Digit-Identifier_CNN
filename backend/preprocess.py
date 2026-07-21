import cv2
import numpy as np


def read_image(file_bytes):
    """
    Converts uploaded image bytes into an OpenCV image.
    """

    image = np.frombuffer(file_bytes, np.uint8)

    image = cv2.imdecode(image, cv2.IMREAD_COLOR)

    return image

def to_grayscale(image):

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    return gray

def blur_image(gray):

    return cv2.GaussianBlur(gray, (5,5), 0)

def threshold_image(gray):

    _, thresh = cv2.threshold(
        gray,
        0,
        255,
        cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )

    return thresh

def extract_digit(binary_image):
    """
    Finds the largest contour (assumed to be the digit)
    and crops it.
    """

    contours, _ = cv2.findContours(
        binary_image,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    if len(contours) == 0:
        raise ValueError("No digit found.")

    # Largest contour
    largest = max(contours, key=cv2.contourArea)

    x, y, w, h = cv2.boundingRect(largest)

    digit = binary_image[y:y+h, x:x+w]

    return digit

def resize_keep_aspect(image, target_size=20):
    """
    Resize digit while keeping aspect ratio.
    The longest side becomes target_size.
    """

    h, w = image.shape

    if h > w:
        new_h = target_size
        new_w = int(w * target_size / h)
    else:
        new_w = target_size
        new_h = int(h * target_size / w)

    resized = cv2.resize(
        image,
        (new_w, new_h),
        interpolation=cv2.INTER_AREA
    )

    return resized

def pad_image(image, size=28):
    """
    Pad resized digit to 28x28.
    """

    h, w = image.shape

    top = (size - h) // 2
    bottom = size - h - top

    left = (size - w) // 2
    right = size - w - left

    padded = cv2.copyMakeBorder(
        image,
        top,
        bottom,
        left,
        right,
        cv2.BORDER_CONSTANT,
        value=0
    )

    return padded

def normalize(image):

    image = image.astype("float32") / 255.0

    return image

def reshape_image(image):

    image = image.reshape(1, 28, 28, 1)

    return image