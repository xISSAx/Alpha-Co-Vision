import cv2
import time
import threading
import numpy as np
from image_processing import convert_frame_to_pil_image
from caption_generation import generate_caption
from response_generation import generate_response

# create VideoCapture object for camera feed
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)  # set resolution
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

# initialize variable for tracking processing times
last_process_time = time.time()
last_generation_time = time.time()
previous_caption = ""  # initialize variables for tracking previous captions and responses
previous_response = ""
previous_captions = []
previous_responses = []
lock = threading.Lock()  # initialize lock for threading synchronization


# convert frame to PIL image format and generate caption for a frame
def process_frame(frame):
    global last_generation_time, previous_captions, previous_responses
    pil_image = convert_frame_to_pil_image(frame)
    caption = generate_caption(pil_image)

    current_time = time.time()  # track current time for processing time comparison
    if current_time - last_generation_time >= 3:  # generate response every 3 seconds
        if caption and caption not in previous_captions:
            previous_captions.append(caption)  # add caption to previous captions list
            if len(previous_captions) > 20:  # limit previous captions list to 10 items
                previous_captions.pop(0)

            response = generate_response(previous_caption + " " + caption, previous_response,
                                         previous_responses)  # generate response for caption and previous response

            while response in previous_responses:  # ensure response is unique
                response = generate_response(previous_caption + " " + caption, previous_response)

            previous_responses.append(response)  # add response to previous responses list
            if len(previous_responses) > 20:
                previous_responses.pop(0)

            print(response)  # print response to console
            last_generation_time = current_time  # update last generation time


def draw_chat_bubble(frame, response, y_pos):
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.7
    thickness = 2
    chat_bubble_padding = 10
    chat_bubble_thickness = 1
    chat_bubble_color = (255, 0, 0)
    max_chat_bubble_height = 50
    response_str = response if response else "<No response>"
    chat_bubble_size, _ = cv2.getTextSize(response_str, font, font_scale, thickness)
    chat_bubble_size = (chat_bubble_size[0] + chat_bubble_padding * 2, min(max_chat_bubble_height, chat_bubble_size[1] + chat_bubble_padding * 2))
    chat_bubble_x = int((frame.shape[1] - chat_bubble_size[0]) / 2)
    chat_bubble_y = y_pos - chat_bubble_size[1]
    chat_bubble_background = np.zeros((chat_bubble_size[1], chat_bubble_size[0], 3), dtype=np.uint8)

    cv2.putText(chat_bubble_background, response_str, (chat_bubble_padding, chat_bubble_size[1] - chat_bubble_padding), font, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)

    if chat_bubble_y >= 0:
        cv2.rectangle(frame, (chat_bubble_x, chat_bubble_y), (chat_bubble_x + chat_bubble_size[0], chat_bubble_y - chat_bubble_size[1]), chat_bubble_color, chat_bubble_thickness)
        frame_slice = frame[chat_bubble_y - chat_bubble_size[1]:chat_bubble_y, chat_bubble_x:chat_bubble_x + chat_bubble_size[0]]
        if chat_bubble_background.shape == frame_slice.shape:
            frame_slice[:] = chat_bubble_background
        else:
            print("Error: chat bubble background shape is", chat_bubble_background.shape, "but frame slice shape is", frame_slice.shape)

    return chat_bubble_y - chat_bubble_size[1] - 10


def display_frame(frame):
    global previous_captions, previous_responses
    with lock:
        bottom_y = frame.shape[0] - 10
        for response in previous_responses[-2:][::-1]:  # use the last two generated responses, in reverse order
            bottom_y = draw_chat_bubble(frame, response, bottom_y)

        cv2.imshow('frame', frame)


# The LOOOOOOP
def main_loop():
    global last_process_time
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error capturing frame, exiting.")
            break

        current_time = time.time()
        if current_time - last_process_time >= 1:
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