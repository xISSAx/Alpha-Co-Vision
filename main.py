import argparse
import os.path
import cv2
import time
import threading
import caption_generation
from image_processing import convert_frame_to_pil_image
from caption_generation import generate_caption
from response_generation import generate_response
import config
import edge_tts_playback

# create VideoCapture object for camera feed
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 280)

# initialize variable for tracking processing times
last_process_time = time.time()
last_generation_time = time.time()
previous_captions = []  # initialize variables for tracking previous captions and responses
previous_responses = []
lock = threading.Lock()  # initialize lock for threading synchronization
captions_list = []  # initialize list to hold generated captions


# convert frame to PIL image format and generate caption for a frame
def process_frame(frame):
    global last_generation_time, previous_captions
    pil_image = convert_frame_to_pil_image(frame)
    caption = generate_caption(pil_image)
    current_time = time.time()  # track current time for processing time comparison
    if current_time - last_generation_time >= 2:  # generate captions every 5 seconds
        if caption and caption not in previous_captions and caption not in captions_list:
            print(f"caption: {caption}")
            captions_list.append(caption)  # add caption to list of captions
            if len(captions_list) > 10:  # limit list of captions to 10 items
                captions_list.pop(0)
        last_generation_time = current_time  # update last generation time


def display_frame(frame):
    """
    Function to display a frame on the screen and overlay the previous captions on top of it.
    """
    global captions_list
    with lock:  # synchronize with lock
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.0
        thickness = 1
        color = (0, 0, 0)
        org = (10, 20)
        captions_str = '\n'.join(captions_list)
        cv2.putText(frame, captions_str, org, font, font_scale, color, thickness, cv2.LINE_AA)
        flipped_frame = cv2.flip(frame, 1)  # Flip the frame horizontally
        cv2.imshow('frame', flipped_frame)


def get_user_input():
    while True:
        user_input = input('What do you say now? ')
        if user_input:
            response = generate_response(' '.join(captions_list), previous_response=previous_responses[len(previous_responses)-1], previous_responses=previous_responses,  user_input=user_input)
            # while response in previous_responses:  # ensure response is unique
            #     response = generate_response(' '.join(captions_list), user_input)

            previous_responses.append(response)  # add response to list of previous responses
            if len(previous_responses) > 20:
                previous_responses.pop(0)

            print(response)  # print response to console
            if config.settings.edge_tts_enable:
                edge_tts_playback.playTTS(response, config.settings.edge_tts_voice)


def main_loop():
    global last_process_time
    threading.Thread(target=get_user_input).start()  # start thread to wait for user input
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error capturing frame, exiting.")
            break

        current_time = time.time()
        if current_time - last_process_time >= 10:  # process frame every 5 seconds
            t = threading.Thread(target=process_frame, args=(frame,))
            t.start()
            last_process_time = current_time

        display_frame(frame)

        if cv2.waitKey(1) == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


def setup_config(config_file):
    if not os.path.exists(config_file):
        raise FileNotFoundError(f"Config file {config_file} not found.")

    settings = config.load_config(config_file)

    # print("OpenAI API Key:", settings.openai_api_key)
    print("Enable OpenAI:", settings.enable_openai)
    print("Enable enable_mps:", settings.enable_mps)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Example with config file support.")
    parser.add_argument("--config", dest="config", default="config.json", help="Path to the config file")
    args = parser.parse_args()
    setup_config(args.config)
    caption_generation.init()

    main_loop()
