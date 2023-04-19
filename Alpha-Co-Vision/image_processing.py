import cv2
from PIL import Image

# Converting an input frame to a PIL image
def convert_frame_to_pil_image(frame):
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(rgb_frame)
    return pil_image