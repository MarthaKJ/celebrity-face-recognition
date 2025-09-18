import cv2
import numpy as np
from insightface.app import FaceAnalysis

# Function to draw on image
def custom_draw_on(img, faces):
    dimg = img.copy()
    for face in faces:
        box = face.bbox.astype(np.int64)
        color = (0, 0, 255)
        cv2.rectangle(dimg, (box[0], box[1]), (box[2], box[3]), color, 2)
        if hasattr(face, "kps"):
            kps = face.kps.astype(np.int64)
            for kp in kps:
                cv2.circle(dimg, (kp[0], kp[1]), 1, (0, 255, 0), 2)
    return dimg

# Initialize FaceAnalysis
app = FaceAnalysis(providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
app.prepare(ctx_id=0, det_size=(640, 640))

# Load new image
img = cv2.imread("my_image.jpg")  # Replace with your image path

# Detect faces
faces = app.get(img)

# Draw bounding boxes
rimg = custom_draw_on(img, faces)

# Save output image
cv2.imwrite("./output.jpg", rimg)
