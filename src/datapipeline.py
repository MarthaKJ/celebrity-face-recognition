import cv2
import numpy as np
import requests

# URL of the image you want to use
image_url = "https://sl.bing.net/f9fvg6awYLY"  # Replace with your image URL

# Fetch image from the internet
response = requests.get(image_url, stream=True)
response.raise_for_status()  # Ensure request was successful

# Convert image data into a NumPy array
image_array = np.asarray(bytearray(response.raw.read()), dtype="uint8")

# Decode the image using OpenCV
img = cv2.imdecode(image_array, cv2.IMREAD_COLOR)

# Save the image (optional)
cv2.imwrite("downloaded_image.jpg", img)
