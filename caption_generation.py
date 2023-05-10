import torch
from transformers import BlipProcessor, BlipForConditionalGeneration
import config

# Load the image captioning model and processor from the pretrained model

processor = None
model = None
def init():
    global processor
    global model
    blip_model = config.settings.blip_model
    if not blip_model:
        blip_model="Salesforce/blip-image-captioning-base"

    # print(f"blip_model: {blip_model}")
    # "Salesforce/blip-image-captioning-large"
    processor = BlipProcessor.from_pretrained(f"{blip_model}")
    if config.settings.enable_mps:
        model = BlipForConditionalGeneration.from_pretrained(f"{blip_model}", torch_dtype=torch.float16).to("mps")
    else:
        model = BlipForConditionalGeneration.from_pretrained(f"{blip_model}")

# Generate captions for an input image
def generate_caption(pil_image):
    global previous_caption
    try:
        if config.settings.enable_mps:
            inputs = processor(pil_image, return_tensors="pt").to("mps", torch.float16)
        else:
            inputs = processor(pil_image, return_tensors="pt")
        out = model.generate(**inputs, max_new_tokens=50)
        caption = processor.decode(out[0], skip_special_tokens=True)
        previous_caption = caption
        return caption
    except:
        return "Unable to process image."

