# Alpha-Co-Vision

![Alpha-Co-Vision](https://user-images.githubusercontent.com/86708276/233063057-44377bf3-8392-42cc-a7f9-9935b8596632.png)

A real-time video-to-text bot that captures frames generates captions and creates conversational responses using a Large Language Models base to create interactive video descriptions.
Powered by Salesforce BLIP (Bootstrapping Language-Image Pre-training) and Cohere AI, this bot is capable of unified vision-language understanding and generation using transformers.

## Description:

Alpha-Co-Vision is the first step in a series of upcoming projects focused on real-time generations to ultimately create a Pet-Toy-Robot capable of understanding its environment to better interact with humans.

The main goal of this project was to efficiently run a VideoFrames-To-Text MultiModal-esk capable of understanding the world while combining it with the power of cutting-edge Large-Language-Models to better interact with the natural environment running BLIP in **half-precision (`float16`)** on MacBook M1 to gain maximum performance.

The project is currently under development and will improve over time with more support for other Chat models, such as GPT-4 and GPT-3.5 Turbo and locally running LLMs like LlaMa and Alpaca.

This was hacked in a couple of nights and maybe optimized incorrectly/poorly. Moreover, this project is for educational purposes only. Future updates, with growing community support, will include ‘Cuda ‘ support, voice input/output support, GPT-3.5 and GPT-4 for extended generations with Chat Support, and much more.

## Requirements

- Python 3.7 or higher
- `cohere`
- `opencv-python`
- `Pillow`
- `torch`
- `transformers`
- `OpenAI (optional)`

## ⬆️ Recent Updates
- Reduced repetition: By maintaining a list of previous responses and checking the similarity between new responses and past responses, the bot is less likely to repeat itself, resulting in a more engaging and natural conversation.
- Improved conversation quality: The updated prompt with more examples and clearer instructions helps the model understand the task better, leading to more relevant and context-aware responses.
- Mirrored video display: Flipping the frame horizontally provides a mirrored display for the user, making it more comfortable for them to view their own video feed without affecting the input to the model.
- Added upto Full-HD/4k support.

You can install the required packages using the following command:

`pip install cohere opencv-python Pillow torch transformers openai`

# Project Structure, Usage, and Customization

## Salesforce BLIP: [🔗](https://github.com/salesforce/BLIP) 

- BLIP on Hugging Face: [🔗](https://huggingface.co/spaces/Salesforce/BLIP)

## Cohere AI: [🔗](https://cohere.ai/) 

- Get Your Cohere AI API Key Here 👉 https://dashboard.cohere.ai/api-keys
- Try Cohere's Playground Here 👉 https://dashboard.cohere.ai/playground/generate
- For Support & More Info, Join The Cohere's Incredible Discord Community: 👉 https://discord.com/invite/co-mmunity

## Project Structure

- `config.py`: Contains API keys and other configurations.
- `image_processing.py`: Contains functions related to image processing.
- `caption_generation.py`: Contains functions related to caption generation using the Blip model.
- `response_generation.py`: Contains functions related to response generation using the Cohere AI API.
- `main.py`: The main file that runs the program.

## Usage

1. Set up your API keys in the `config.py` file: `cohere_api_key = **"YOUR_COHERE_API_KEY"**`
2. `cohere_api_key = **"YOUR_COHERE_API_KEY"**`&  in `config.py
    1. Run the `main.py` file:
        
        `python main.py`
        
    2. Press ‘q’ on the ‘Camera Window’ to quit.
 
- Optional Tweaks:
    - Tweak LLMS outputs: `def process_frame(frame):` `current_time - last_generation_time >= 3` for more or less LLM generations. Optimal ‘captions > 2 .’
    - Tweak Captions outputs: `def main_loop():` `current_time - last_process_time >= 2:` to generate more or less image processing (captions) ‘2’ = optimal, ‘0’ = realtime


Have fun! Make sure to do some activity for the camera for maximum fun! Show your surroundings, more objects, people, or pets! Also, overtime it increases its understanding of your surroundings and would keep generating better & better outputs.

## Use your iPhone as a webcam on Mac: [🔗](https://support.apple.com/en-ca/guide/mac-help/mchl77879b8a/mac)

- On macOS, Ventura 13: Connects to your iPhone first.
    1. Should you not wish to use it, please turn off your Bluetooth either on your iPhone or Mac and disconnect your iPhone from your Mac via cable.
    2. If it fails on your first try, ‘Restart’ `python main.py`

## MacOS CPU/GPU Support:

Tutorials on how to install PT and TF are coming soon. Meanwhile, follow these instructions →

(*Option to switch between Mac CPU & GPU soon.*)

### Install PyTorch For M1

Pre:  **macOS Version** PyTorch is supported on macOS 10.15 (Catalina) or above.

- Visit the link: → [🔗](https://pytorch.org/get-started/locally/)
- Select:  Preview (Nightly) in PyTorch Build.
- Navigate to the ****[macOS Version](https://pytorch.org/get-started/locally/#macos-version)**** section ********& follow the instructions.
- PT (’mps’) is only support for Mac.

### Install TensorFlow For M1:

Tensorflow Model was recently added to Hugging Face. TF update coming soon. Meanwhile:

[🔗](https://developer.apple.com/metal/tensorflow-plugin/) ← Follow the instructions to install TensorFlow on your own. (Currently Optional)

## How it works

1. The program captures webcam frames.
2. Frames are converted to PIL images.
3. Captions are generated using the Salesforce Blip model.
4. Conversational responses are generated based on the captions using the Cohere AI API.
5. Captions and responses are displayed on the webcam feed in real time.

## Example

The bot captures an image of a person working on their computer:

- Caption: "A person working on a computer with code."
- Alpha-Co: "I see you're multitasking while we chat. Keep up the great work! Remember, hard work beats talent when talent doesn't work hard."

## Customizing the bot

You can customize the bot by modifying the `Prompt` in the `response_generation.py` file or adjusting the settings, such as `max_tokens` and `temperature`, when calling the Cohere API.

## Notes

- This bot uses the webcam, so grant permission to access the camera.
- Press 'q' to quit the program while displaying the webcam feed.
- This project was built Mac M1 efficiently running **half-precision (`float16`)**; future updates to include support for **Cuda.**

## Credits

This project utilizes the Salesforce BLIP model for generating image captions. Special thanks Salesforce Research team for their work on BLIP: Bootstrapping Language-Image Pre-training for Unified Vision-Language Understanding and Generation using Transformers. Their research and model have greatly contributed to developing this video caption to interaction bot.

## Special Thanks

Thank you to Cohere AI for their unwavering support and motivation throughout this project. Your encouragement and cutting-edge technology have played a crucial role in our success, and I'm grateful for the opportunity to collaborate and innovate together. Here's to pushing boundaries and shaping the future of AI!

## Future Updates:

1. An API Rate Limiter.
2. GPT-3.5 and GPT-4 for more extended generations and Chat Support.
3. Llama, Alpaca and other LLMs support for running everything locally.
4. Chat Input messages to have a conversation.
5. Voice Input & Output Support
6. Ability to fine-tune BLIP (Caption Model)
7. Ability to fine-tune LLMs
8. CPU & ‘Cuda’ Support
9. Ability to Switch between **Full-precision & Half-precision.**
