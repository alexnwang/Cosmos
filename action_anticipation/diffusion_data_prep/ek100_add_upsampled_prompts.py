# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
import glob
import json
import os
import random
from typing import Iterable

from math import ceil
import torch
from PIL import Image
import torchvision
from einops import rearrange
from huggingface_hub import snapshot_download
from cosmos1.models.diffusion.inference.world_generation_pipeline import run_chat_completion_vlm
from cosmos1.models.diffusion.prompt_upsampler.video2world_prompt_upsampler_inference import create_vlm_prompt_upsampler
from nemo.collections.diffusion.models.model import DiT7BConfig
from tqdm import tqdm
from transformers import T5EncoderModel, T5TokenizerFast
from transformers import AutoProcessor, LlavaForConditionalGeneration
import transformers

from cosmos1.utils import log
import pandas as pd
from datetime import datetime


def get_parser():
    parser = argparse.ArgumentParser(description="Process some configurations.")
    parser.add_argument(
        "--dataset_path", type=str, default="video_dataset", help="Path to the dataset of already processed examples"
    )
    parser.add_argument(
        "--video_path", type=str, default="video_dataset", help="Path to the dataset (a folder of EK videos)."
    )
    
    parser.add_argument("--pixtral_checkpoint_dir", type=str, default="/home/anw2067/scratch/Cosmos/checkpoints/Pixtral-12B", help="Path to the pixtral checkpoint")
    
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size for encoding")
    return parser


def init_t5(device="cuda"):
    """Initialize and return the T5 tokenizer and text encoder."""
    tokenizer = T5TokenizerFast.from_pretrained("google-t5/t5-11b")
    text_encoder = T5EncoderModel.from_pretrained("google-t5/t5-11b")
    text_encoder.to(device)
    text_encoder.eval()
    return tokenizer, text_encoder

@torch.no_grad()
def encode_for_batch(tokenizer, encoder, prompts: list[str], max_length=512, device="cuda"):
    """
    Encode a batch of text prompts to a batch of T5 embeddings.
    Parameters:
        tokenizer: T5 embedding tokenizer.
        encoder: T5 embedding text encoder.
        prompts: A batch of text prompts.
        max_length: Sequence length of text embedding (defaults to 512).
    """

    batch_encoding = tokenizer.batch_encode_plus(
        prompts,
        return_tensors="pt",
        truncation=True,
        padding="max_length",
        max_length=max_length,
        return_length=True,
        return_offsets_mapping=False,
    )

    # We expect all the processing is done on GPU.
    input_ids = batch_encoding.input_ids.to(device)
    attn_mask = batch_encoding.attention_mask.to(device)
    outputs = encoder(input_ids=input_ids, attention_mask=attn_mask)
    encoded_text = outputs.last_hidden_state

    lengths = attn_mask.sum(dim=1).cpu()
    for batch_id in range(encoded_text.shape[0]):
        encoded_text[batch_id][lengths[batch_id] :] = 0

    return encoded_text
    
def resize_image(image: Image.Image, max_size: int = 1024) -> Image.Image:
    """
    Ensure that the image is no larger than max_size in both dimensions.
    """
    image_width, image_height = image.size
    max_width, max_height = max_size, max_size
    ratio = max(image_width / max_width, image_height / max_height)
    if ratio > 1:
        image = image.resize((ceil(image_width / ratio), ceil(image_height / ratio)))
    return image

PROMPT_TEMPLATE=("Your task is to transform a given prompt into a refined and concise video description, no more than 150 words. "
                 "Focus only on the content and do not describe any actions beyond the provided action: {}. "
                 "Do not use filler words, or descriptions on the style or any extra comments. Never mention things outside the video. ")
# PROMPT_TEMPLATE=("Your task is to transform a given prompt into a refined and concise video description, no more than 150 words."
# "Focus only on the content, no filler words or descriptions on the style. Never mention things outside the video.")

class UpsampleDataset(torch.utils.data.Dataset):
    def __init__(self, dataset_path, video_path):
        self.dataset_path = dataset_path
        self.video_path = video_path
        self.prefixes = self.get_prefixes()
        
    def get_prefixes(self):
        all_files = os.listdir(self.dataset_path)
        prefixes = set()
        for file in all_files:
            prefix = file.split('.')[0]
            prefixes.add(prefix)
        return sorted(list(prefixes))
    
    def __len__(self):
        return len(self.prefixes)
    
    def __getitem__(self, idx):
        rank = int(os.environ.get("RANK", 0))
        worker_info = torch.utils.data.get_worker_info()
        worker_id = worker_info.id if worker_info else 0
        
        prefix = self.prefixes[idx]
        
        # load metadata
        metadata = json.load(open(os.path.join(self.dataset_path, f"{prefix}.info.json")))
        video_name = metadata['video_name']
        start_time = metadata['start_time']
        
        # get video filepath
        video_path = os.path.join(self.video_path, video_name.split('_')[0], 'videos', video_name)
        
        # use ffmpeg to extract the image to a temporary file based on rank and worker_id
        temp_file = os.path.join(os.path.expanduser("~"), f"temp{rank}_{worker_id}.jpg")
        ffmpeg_command = f"ffmpeg -loglevel quiet -nostdin -y -ss {start_time} -i {video_path} -vframes 1 {temp_file}"
        os.system(ffmpeg_command)
        os.system('stty sane')
        
        # read image
        image = Image.open(temp_file)
        image_tensor = torchvision.transforms.ToTensor()(image)
        
        # remove temp file
        if os.path.exists(temp_file):
            os.remove(temp_file)
        
        if "upsampled_prompt" not in metadata:
            upsampled_prompt = []
        else:
            upsampled_prompt = metadata['upsampled_prompt']
        
        return {'images': image_tensor, 'example_names': prefix, "narrations": metadata['narration'], "upsampled_prompt": upsampled_prompt}
        
def load_vlm_upsampler(device="cuda"):
    model_id = "mistral-community/pixtral-12b"
    processor = AutoProcessor.from_pretrained(model_id)
    model = LlavaForConditionalGeneration.from_pretrained(model_id, device_map=device, torch_dtype=torch.bfloat16)
    model = torch.compile(model)
    return model, processor

def process_chat_dialog(processor: transformers.PixtralProcessor, image: torch.Tensor, prompt: str):
    image = torchvision.transforms.ToPILImage()(image)
    image = resize_image(image, max_size=1024)
    prompt = prompt

    chat = [
        {
            "role": "user",
            "content": "[IMG]\n" + prompt,
        }
    ]
    return chat, image
    
        
def main(args):
    rank = int(os.environ.get("RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    
    device = torch.device(f"cuda:{rank}")
    torch.cuda.set_device(device)
    print(f"Rank {rank} using device {device}, {torch.cuda.current_device()}")
    
    # prepare dataset
    dataset = UpsampleDataset(args.dataset_path, args.video_path)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, num_workers=4, prefetch_factor=1,
                                             sampler=torch.utils.data.DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=False))
    
    # Initialize Pixtral
    pixtral12b, processor = load_vlm_upsampler(device)
    processor.tokenizer.add_special_tokens({'pad_token': '<pad>'})
    # pixtral12b = create_vlm_prompt_upsampler(args.pixtral_checkpoint_dir, device=device)

    for i, batch in enumerate(tqdm(dataloader, total=len(dataloader), desc="Upsampling prompts")):
        images = batch['images']
        example_names = batch['example_names']
        narrations = batch['narrations']
        
        # Skip processing if upsampled_prompt already exists
        already_exists = []
        for example_name in example_names:
            with open(os.path.join(args.dataset_path, f"{example_name}.info.json"), "r") as json_file:
                info = json.load(json_file)
                if "upsampled_prompt" in info and info["upsampled_prompt"]:
                    already_exists.append(True)
                else:
                    already_exists.append(False)
        if all(already_exists):
            print(f"Rank {rank} skipping batch {i} as all prompts are already upsampled.")
            continue
        
        templates = []
        images_ = []
        for image, narration in zip(images, narrations):
            template, image = process_chat_dialog(processor, image, PROMPT_TEMPLATE.format(narration))
            templates.append(processor.apply_chat_template(template, add_generation_prompt=True))
            images_.append([image])
        inp = processor(text=templates, images=images_, return_tensors="pt", padding=True).to(device)
        
        ## decoding etc
        with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
            generations = pixtral12b.generate(**inp, max_new_tokens=400, temperature=0.01, top_p=0.9)
        
        # decode
        # Remove input tokens from the generated text
        generations_list = generations.tolist()
        for idx, generation in enumerate(generations_list):
            input_length = len(inp.input_ids[idx])
            generations_list[idx] = generation[input_length:]
        upsampled_prompts = processor.batch_decode(generations_list,  skip_special_tokens=True, clean_up_tokenization_spaces=False)
        
        for j, (upsampled_prompt, example_name) in enumerate(zip(upsampled_prompts, example_names)):
            # add upsampled_prompt to metadata file
            with open(os.path.join(args.dataset_path, f"{example_name}.info.json"), "r") as json_file:
                info= json.load(json_file)
                info['upsampled_prompt'] = upsampled_prompt
                print(f"Rank {rank} processed {example_name} with prompt \n {upsampled_prompt}")
            with open(os.path.join(args.dataset_path, f"{example_name}.info.json"), "w") as json_file:
                json.dump(info, json_file)
    del pixtral12b
    torch.cuda.empty_cache()

    tokenizer, text_encoder = init_t5(device)
    t5_embeding_max_length = 512
    # now, reiterate through the dataset and encode the new 'upsampled_prompt' to T5 embeddings
    for i, batch in enumerate(tqdm(dataloader, total=len(dataloader), desc="encoding prompts with t5")):
        images = batch['images']
        example_names = batch['example_names']
        narrations = batch['narrations']
        upsampled_prompts = batch['upsampled_prompt']
        
        # Initialize T5
        encoded_texts = encode_for_batch(tokenizer, text_encoder, upsampled_prompts, device=device)
        
        for j, (encoded_text, example_name) in enumerate(zip(encoded_texts, example_names)):
            # Pad T5 embedding to t5_embeding_max_length
            L, C_ = encoded_text.shape
            encoded_text = torch.zeros(t5_embeding_max_length, C_, dtype=torch.bfloat16)
            encoded_text[:L] = encoded_text

            torch.save(encoded_text, os.path.join(args.dataset_path, f"{example_name}.upsampled_t5_text_embeddings.pth"))
            torch.save(
                torch.ones(512, dtype=torch.bfloat16), os.path.join(args.dataset_path, f"{example_name}.upsampled_t5_text_mask.pth")
            )
    print(f"Rank {rank} finished processing dataset.")


if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()
    main(args)