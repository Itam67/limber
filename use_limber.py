from image_input import ImageInput
import sys
import os
from limber_gptj import LimberGPTJ
import torch
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import transformer_lens.utils as utils
from transformer_lens import ActivationCache, HookedTransformer



def simple_load_model(config_path, limber_proj_path='auto', device='cuda:0'):
    lm =LimberGPTJ.from_pretrained("EleutherAI/gpt-j-6b", revision="float16", torch_dtype=torch.bfloat16)
    lm.setup_multimodal(config_path, device=device)
    if limber_proj_path == 'auto':
        if config_path.endswith("beit_ft_linear.yml"):
            limber_proj_path = 'limber_weights/beit_ft_linear/proj.ckpt'
        elif config_path.endswith("beit_linear.yml"):
            limber_proj_path = 'limber_weights/beit_linear/proj.ckpt'
        elif config_path.endswith("nfrn50_4096_linear.yml"):
            limber_proj_path = 'limber_weights/nfrn50_4096_linear/proj.ckpt'
        elif config_path.endswith('nfrn50_4096_random_linear.yml'):
            limber_proj_path = 'limber_weights/nfrn50_4096_linear/proj.ckpt'
        elif config_path.endswith('clip_linear.yml'):
            limber_proj_path = 'limber_weights/clip_linear/proj.ckpt'
    proj_ckpt = torch.load(limber_proj_path)
    lm.image_prefix.proj.load_state_dict(proj_ckpt) #Load in the weights for the linear projection
    return lm


def generate_typographic(text_list, save_folder, font_size=80, image_width=384, image_height=384, font_path=None):
    font = ImageFont.truetype(font_path, font_size) if font_path else ImageFont.load_default()

    # Create the folder if it doesn't exist
    os.makedirs(save_folder, exist_ok=True)

    for i, text in enumerate(text_list):
        # Create a white image
        image = Image.new("RGB", (image_width, image_height), "white")
        draw = ImageDraw.Draw(image)

        # Calculate text size and position
        text_width, text_height = draw.textsize(text, font=font)
        text_x = (image_width - text_width) / 2
        text_y = (image_height - text_height) / 2

        # Draw text
        draw.text((text_x, text_y), text, fill="black", font=font)

        # Save the image as JPEG
        image_path = os.path.join(save_folder, f"image_{i}.jpg")
        image.save(image_path)




if __name__ == "__main__":
    config_path = 'configs/clip_linear.yml'

    # model = HookedTransformer.from_pretrained(
    #     "EleutherAI/gpt-j-6b",
    #     center_unembed=True,
    #     center_writing_weights=True,
    #     fold_ln=True,
    #     refactor_factored_attn_matrices=True,
    # )

    model = simple_load_model(config_path)
    print("Loaded model")
    model = model.cuda().half()

    # List of sentences to asses gender directions
    # gender_list = ['King', 'Queen', 'Actor', 'Actress', 'Waiter', 'Waitress', 'Duke', 'Duchess', 'Prince', 'Princess', 'Wizard', 'Witch', 'Hero', 'Heroine', 'Lion', 'Lioness', 'God', 'Goddess', 'Emperor', 'Empress']

    # save_folder = "gender_images"
    # generate_typographic(gender_list, save_folder, font_path='Roboto-Black.ttf')


    #Example image from MAGMA repo:
    imginp = ImageInput('gender_images/image_2.jpg')
    print("Loaded image")
    inputs = model.preprocess_inputs([imginp])
    breakpoint()
    output = model.generate(embeddings=inputs)
    print(output)
    #BEIT linear: [' a traditional house in the village']
    #NFRN50 linear: [' a mountain village in the mountains.']
    #CLIP linear: [' a house in the woods']
