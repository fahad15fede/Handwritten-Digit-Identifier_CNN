import cv2

from preprocess import (
    read_image,
    to_grayscale,
    blur_image,
    threshold_image,
    extract_digit,
    resize_keep_aspect,
    pad_image,
    normalize,
    reshape_image
)

# Read image as bytes
with open("test_images/digit4_2.png", "rb") as f:
    file_bytes = f.read()

# Process step-by-step
image = read_image(file_bytes)
gray = to_grayscale(image)
blur = blur_image(gray)
thresh = threshold_image(blur)
digit = extract_digit(thresh)
resized = resize_keep_aspect(digit)
padded = pad_image(resized)
normalized = normalize(padded)
final_image = reshape_image(normalized)


# Save outputs
cv2.imwrite("output/original.png", image)
cv2.imwrite("output/gray.png", gray)
cv2.imwrite("output/blur.png", blur)
cv2.imwrite("output/threshold.png", thresh)
cv2.imwrite("output/digit.png", digit)
cv2.imwrite("output/resized.png", resized)
cv2.imwrite("output/padded.png", padded)
print(final_image.shape)


print("Images saved successfully!")