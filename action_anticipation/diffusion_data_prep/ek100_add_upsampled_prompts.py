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

def upsample_prompt(model, image, narration):
    """Upsample a prompt using the given model."""
    prompt = PROMPT_TEMPLATE.format(narration)
    dialog = prepare_dialog(image, prompt)
        
    upsampled_prompts = run_chat_completion_vlm(
        model, dialog, max_gen_len=400, temperature=0.01, top_p=0.9, logprobs=False
    )
    return upsampled_prompts

def prepare_dialog(image: torch.Tensor, prompt: str) -> list[dict]:
    image = torchvision.transforms.ToPILImage()(image)
    image = resize_image(image, max_size=1024)
    prompt = prompt.strip()

    return [
        {
            "role": "user",
            "content": "[IMG]\n" + prompt,
            "images": [image],
        }
    ]
    
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

PROMPT_TEMPLATE=("Your task is to transform a given image prompt and action into a refined and concise video description, no more than 150 words."
                 "Focus only on the content of the provied image and do not describe any actions beyond the provided action: {}."
                 "Do not use filler words, or descriptions on the style. Never mention things outside the video.")


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
        ffmpeg_command = f"ffmpeg -loglevel quiet -y -ss {start_time} -i {video_path} -vframes 1 {temp_file}"
        os.system(ffmpeg_command)
        
        # read image
        image = Image.open(temp_file)
        image_tensor = torchvision.transforms.ToTensor()(image)
        
        # remove temp file
        if os.path.exists(temp_file):
            os.remove(temp_file)
        
        return {'images': image_tensor, 'example_names': prefix, "narrations": metadata['narration']}
        
def main(args):

    rank = int(os.environ.get("RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    
    device = torch.device(f"cuda:{rank}")
    torch.cuda.set_device(device)
    print(f"Rank {rank} using device {device}, {torch.cuda.current_device()}")

    # Initialize T5
    tokenizer, text_encoder = init_t5("cpu") # load to CPU and adjust to device at inference

    # Initialize Pixtral 12B
    pixtral12b = create_vlm_prompt_upsampler(args.pixtral_checkpoint_dir).to("cpu")
    
    torch.cuda.empty_cache()

    # Constants
    t5_embeding_max_length = 512
    
    dataset = UpsampleDataset(args.dataset_path, args.video_path)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, num_workers=4, prefetch_factor=1,
                                             sampler=torch.utils.data.DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=False))

    for i, batch in enumerate(tqdm(dataloader, total=len(dataloader))):
        images = batch['images']
        example_names = batch['example_names']
        narrations = batch['narrations']

        pixtral12b = pixtral12b.to(device)
        upsampled_prompts = []
        for image, narration in zip(images, narrations):
            upsampled_prompts.append(upsample_prompt(pixtral12b, image, narration))
        pixtral12b = pixtral12b.to("cpu")

        text_encoder = text_encoder.to(device)
        encoded_texts = encode_for_batch(tokenizer, text_encoder, upsampled_prompts, device=device).cpu()
        text_encoder = text_encoder.to("cpu")

        for j, (encoded_text, example_name) in enumerate(zip(encoded_texts, example_names)):
            
            # Pad T5 embedding to t5_embeding_max_length
            L, C_ = encoded_text.shape
            encoded_text = torch.zeros(t5_embeding_max_length, C_, dtype=torch.bfloat16)
            encoded_text[:L] = encoded_text
            
            torch.save(encoded_text, os.path.join(args.dataset_path, f"{example_name}.upsampled_t5_text_embeddings.pth"))
            torch.save(
                torch.ones(512, dtype=torch.bfloat16), os.path.join(args.dataset_path, f"{example_name}.upsampled_t5_text_mask.pth")
            )
            
            # add upsampled_prompt to metadata file
            with open(os.path.join(args.dataset_path, f"{example_name}.info.json"), "r") as json_file:
                info = json.load(json_file)
                info['upsampled_prompt'] = upsampled_prompts[j]
            with open(os.path.join(args.dataset_path, f"{example_name}.info.json"), "w") as json_file:
                json.dump(info, json_file)


if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()
    main(args)