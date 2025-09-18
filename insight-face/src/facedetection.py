import cv2
import numpy as np
from insightface.app import FaceAnalysis
from insightface.data import get_image as ins_get_image


# Override the draw_on method to use int64 instead of np.int
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
print(app.models)

# Get and process image
img = ins_get_image("t1")
faces = app.get(img)

# Use custom draw_on function instead of app.draw_on
rimg = custom_draw_on(img, faces)
cv2.imwrite("./t1_output.jpg", rimg)
