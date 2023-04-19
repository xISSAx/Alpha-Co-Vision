import cv2
import time
import threading
from image_processing import convert_frame_to_pil_image
from caption_generation import generate_caption
from response_generation import generate_response

# initialize variables, e.g., cap, last_process_time, etc.
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 25)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 25)

last_process_time = time.time()
last_generation_time = time.time()
previous_caption = ""
previous_response = ""
previous_captions = []
lock = threading.Lock()

# Processing each frame in a separate thread
def process_frame(frame):
    global last_generation_time, previous_captions
    pil_image = convert_frame_to_pil_image(frame)
    caption = generate_caption(pil_image)

    # Generate a response every 3 seconds if a new caption has been generated
    current_time = time.time()
    if current_time - last_generation_time >= 3: # Increase this variable if you want slower LLM outputs
        if caption and caption not in previous_captions:
            previous_captions.append(caption)
            response = generate_response(previous_caption + " " + caption, previous_response)
            print(response)
        last_generation_time = current_time

# Displaying processed frame
def display_frame(frame):
    global previous_captions
    with lock:
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.0
        thickness = 1
        color = (0, 0, 0)
        org = (10, 20)
        previous_captions_str = '\n'.join(previous_captions)
        cv2.putText(frame, previous_captions_str, org, font, font_scale, color, thickness, cv2.LINE_AA)
        cv2.imshow('frame', frame)

# The main loop of the program
def main_loop():
    global last_process_time
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error capturing frame, exiting.")
            break

        current_time = time.time()
        if current_time - last_process_time >= 2: #Tweak this variable should you want more or less less images processed.
            t = threading.Thread(target=process_frame, args=(frame,))
            t.start()
            last_process_time = current_time

        display_frame(frame)

        if cv2.waitKey(1) == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main_loop()