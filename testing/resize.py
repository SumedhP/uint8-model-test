"""
Take in a sample image and try to resize to smaller version using various different opencv resize interpolation methods.
Save each version and display them side by side using matplotlib.
"""

import cv2

POSSIBLE_INTERPOLATIONS = [
    cv2.INTER_NEAREST,
    cv2.INTER_LINEAR,
    cv2.INTER_CUBIC,
    cv2.INTER_AREA,
    cv2.INTER_LANCZOS4,
    cv2.INTER_LINEAR_EXACT,
    cv2.INTER_NEAREST_EXACT,
]

NAME_LUT = {
    cv2.INTER_NEAREST: "INTER_NEAREST",
    cv2.INTER_LINEAR: "INTER_LINEAR",
    cv2.INTER_CUBIC: "INTER_CUBIC",
    cv2.INTER_AREA: "INTER_AREA",
    cv2.INTER_LANCZOS4: "INTER_LANCZOS4",
    cv2.INTER_LINEAR_EXACT: "INTER_LINEAR_EXACT",
    cv2.INTER_NEAREST_EXACT: "INTER_NEAREST_EXACT",
}

img = cv2.imread("image.jpg")
scalaing = 0.25

for interpolation in POSSIBLE_INTERPOLATIONS:
    img_resized = cv2.resize(img, None, fx=scalaing, fy=scalaing, interpolation=interpolation)
    cv2.imwrite(f"image_{NAME_LUT[interpolation]}.jpg", img_resized)
    print(f"Saved image_{NAME_LUT[interpolation]}.jpg")
    
import matplotlib.pyplot as plt

# Display over 2 rows and 4 columns
fig, axs = plt.subplots(2, 4)


# Display original image
axs[0, 0].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
axs[0, 0].set_title("Original")
axs[0, 0].axis("off")

# Display resized images
for i, interpolation in enumerate(POSSIBLE_INTERPOLATIONS):
    i += 1
    img_resized = cv2.resize(img, None, fx=scalaing, fy=scalaing, interpolation=interpolation)
    axs[i // 4, i % 4].imshow(cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB))
    axs[i // 4, i % 4].set_title(NAME_LUT[interpolation])
    axs[i // 4, i % 4].axis("off")


# Tight layout
plt.tight_layout()

plt.show()
