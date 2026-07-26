import cv2
import numpy as np


def read_image(file_bytes):
    image = np.frombuffer(file_bytes, np.uint8)
    image = cv2.imdecode(image, cv2.IMREAD_COLOR)
    return image


def to_grayscale(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def blur_image(gray):
    return cv2.GaussianBlur(gray, (5, 5), 0)


def threshold_image(gray):
    _, thresh = cv2.threshold(
        gray,
        0,
        255,
        cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU,
    )
    return thresh


def extract_digits(binary_image):
    """
    Returns every detected digit
    from left to right.
    """

    contours, _ = cv2.findContours(
        binary_image,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    digit_images = []

    for contour in contours:

        area = cv2.contourArea(contour)

        # Ignore tiny noise
        if area < 80:
            continue

        x, y, w, h = cv2.boundingRect(contour)

        digit = binary_image[y:y+h, x:x+w]

        digit_images.append((x, digit))

    # Sort left → right
    digit_images.sort(key=lambda item: item[0])

    # Remove x coordinate
    digit_images = [digit for _, digit in digit_images]

    return digit_images


def resize_keep_aspect(image, target_size=20):

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

    h, w = image.shape

    top = (size-h)//2
    bottom = size-h-top

    left = (size-w)//2
    right = size-w-left

    return cv2.copyMakeBorder(
        image,
        top,
        bottom,
        left,
        right,
        cv2.BORDER_CONSTANT,
        value=0
    )


def normalize(image):
    return image.astype("float32") / 255.0


def reshape_image(image):
    return image.reshape(1, 28, 28, 1)


def preprocess(file_bytes):
    """
    Returns a LIST of processed digits.
    """

    image = read_image(file_bytes)

    gray = to_grayscale(image)

    blur = blur_image(gray)

    thresh = threshold_image(blur)

    digits = extract_digits(thresh)

    processed_digits = []

    for digit in digits:

        digit = resize_keep_aspect(digit)

        digit = pad_image(digit)

        digit = normalize(digit)

        digit = reshape_image(digit)

        processed_digits.append(digit)

    return processed_digits