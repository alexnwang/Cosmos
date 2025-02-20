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

import torch
import torchvision
from einops import rearrange
from huggingface_hub import snapshot_download
from nemo.collections.diffusion.models.model import DiT7BConfig
from tqdm import tqdm
from transformers import T5EncoderModel, T5TokenizerFast

from cosmos1.utils import log
import pandas as pd
from datetime import datetime


def get_parser():
    parser = argparse.ArgumentParser(description="Process some configurations.")
    parser.add_argument("--tokenizer_dir", type=str, default="", help="Path to the VAE model")
    parser.add_argument("--annotations_file", type=str, required=True, help="Path to the annotations file")
    parser.add_argument(
        "--dataset_path", type=str, default="video_dataset", help="Path to the dataset (a folder of videos)"
    )
    
    parser.add_argument("--output_path", type=str, default="video_dataset_cached", help="Path to the output directory")
    parser.add_argument("--height", type=int, default=704, help="Height to resize video")
    parser.add_argument("--width", type=int, default=1280, help="Width to resize video")
    parser.add_argument("--num_frames", type=int, default=121, help="Number of frames per chunk")
    parser.add_argument("--fps", type=int, default=24, help="Frames per second of the video")
    
    parser.add_argument("--temporal_jitter", default=None, required=False, type=str, help="Temporal jitter range in seconds to apply to the pretext clip. Provide as a comma-separated list of two floats.")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size for encoding")
    return parser


def init_t5(device="cuda"):
    """Initialize and return the T5 tokenizer and text encoder."""
    tokenizer = T5TokenizerFast.from_pretrained("google-t5/t5-11b")
    text_encoder = T5EncoderModel.from_pretrained("google-t5/t5-11b")
    text_encoder.to(device)
    text_encoder.eval()
    return tokenizer, text_encoder


def init_video_tokenizer(tokenizer_dir: str):
    """Initialize and return the Cosmos Video tokenizer."""
    dit_config = DiT7BConfig(vae_path=tokenizer_dir)
    vae = dit_config.configure_vae()
    return vae


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

class PreprocessDataset(torch.utils.data.Dataset):
    def __init__(self, annotations_file, dataset_path, output_path, seconds_per_chunk, height, width, num_frames, fps, temporal_jitter=None):
        self.annotations = pd.read_csv(annotations_file)
        self.dataset_path = dataset_path
        self.output_path = output_path
        self.seconds_per_chunk = seconds_per_chunk
        self.height = height
        self.width = width
        self.num_frames = num_frames
        self.fps = fps
        self.temporal_jitter = temporal_jitter
    
    def __len__(self):
        return len(self.annotations)
    
    def __getitem__(self, idx):
        rank = int(os.environ.get("RANK", 0))
        worker_info = torch.utils.data.get_worker_info()
        worker_id = worker_info.id if worker_info else 0
        
        annotation_dict = self.annotations.iloc[idx].to_dict()
        video_path = os.path.join(self.dataset_path, annotation_dict['participant_id'], 'videos', f"{annotation_dict['video_id']}.MP4")
        video_duration = float(os.popen(f"ffprobe -v error -show_entries format=duration -of default=noprint_wrappers=1:nokey=1 {video_path}").read().strip())
        
        if self.temporal_jitter is not None:
            jitter_range = [float(x) for x in self.temporal_jitter.split(',')]
            assert isinstance(jitter_range, Iterable) and len(jitter_range) == 2
            jitter = random.uniform(*jitter_range)
        else:
            jitter = 0
            
        # Convert start and end times to datetime objects
        start_time = datetime.strptime(annotation_dict['start_timestamp'], "%H:%M:%S.%f")
        end_time = datetime.strptime(annotation_dict['stop_timestamp'], "%H:%M:%S.%f")
        # convert datetime objects to seconds
        start_time = start_time.hour*3600 + start_time.minute*60 + start_time.second + start_time.microsecond/1e6
        end_time = end_time.hour*3600 + end_time.minute*60 + end_time.second + end_time.microsecond/1e6
        
        # modify with jitter and make endtime at minimum as long as the seconds_per_chunk
        start_time = max(start_time + jitter, 0)
        end_time = min(max(start_time + self.seconds_per_chunk, end_time), video_duration)
        
        # save the clip to a temp_file
        temp_file = os.path.join(self.output_path, f"temp{rank}_{worker_id}.mp4")
        if not os.path.exists(temp_file):
            os.system(f"ffmpeg -loglevel quiet -y -ss {start_time} -to {end_time} -i {video_path} -vf scale={self.width}:{self.height} -r {self.fps} {temp_file}")
        
        # Read video (T x H x W x C)
        video, _, meta = torchvision.io.read_video(temp_file)
        
        if os.path.exists(temp_file):
            os.remove(temp_file)
        
        return idx, video, meta, annotation_dict, video_path, start_time
        
def main(args):
    rank = int(os.environ.get("RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    
    device = torch.device(f"cuda:{rank}")
    torch.cuda.set_device(device)
    print(f"Rank {rank} using device {device}, {torch.cuda.current_device()}")
    
    # Set up output directory
    os.makedirs(args.output_path, exist_ok=True)

    # Initialize T5
    tokenizer, text_encoder = init_t5(device) # load to CPU and adjust to device at inference

    # Initialize the VAE
    if args.tokenizer_dir == "":
        args.tokenizer_dir = snapshot_download("nvidia/Cosmos-1.0-Tokenizer-CV8x8x8")
    vae = init_video_tokenizer(args.tokenizer_dir).to(device)
    
    torch.cuda.empty_cache()

    # Constants
    t5_embeding_max_length = 512
    if args.num_frames != vae.video_vae.pixel_chunk_duration:
        print(f"Using {args.num_frames} frames per chunk")
        chunk_duration = args.num_frames
    else:
        print(f"Using {vae.video_vae.pixel_chunk_duration} frames per chunk")
        chunk_duration = vae.video_vae.pixel_chunk_duration  # Frames per chunk
        
    dataset = PreprocessDataset(args.annotations_file, args.dataset_path, args.output_path, chunk_duration / args.fps + 0.25, args.height, args.width, args.num_frames, args.fps, args.temporal_jitter)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=None, num_workers=4, prefetch_factor=1,
                                             sampler=torch.utils.data.DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=False))
    
    examples = []
    overflow_examples = []
    for i, video, meta, annotation_dict, video_path, start_time in tqdm(dataloader):
        T, H, W, C = video.shape
        for j, chunk_start in enumerate(range(0, T - chunk_duration, 1 * args.fps)):  # chunks start at 1 second intervals
            chunk = video[chunk_start: chunk_start + chunk_duration]
            chunk = rearrange(chunk, "t h w c -> t c h w")

            # if chunk_duration != vae.video_vae.pixel_chunk_duration, pad it up to the required length
            if chunk_duration != vae.video_vae.pixel_chunk_duration:
                chunk = torch.cat([chunk, torch.zeros(vae.video_vae.pixel_chunk_duration - chunk_duration, C, H, W)], dim=0)

            # Resize to [704, 1280] for each frame
            chunk = torchvision.transforms.functional.resize(chunk, [args.height, args.width])

            # Expand dims: (T, C, H, W) -> (B=1, C, T, H, W)
            chunk = rearrange(chunk, "(b t) c h w -> b c t h w", b=1)

            # Convert to bf16 and normalize from [0, 255] to [-1, 1]
            chunk = chunk.to(dtype=torch.bfloat16, non_blocking=True) / 127.5 - 1.0

            if len(examples) < args.batch_size:
                examples.append((i, j, chunk, annotation_dict, video_path, start_time + chunk_start / args.fps))
            else:
                overflow_examples.append((i, j, chunk, annotation_dict, video_path, start_time + chunk_start / args.fps))

        while len(examples) == args.batch_size:
            # Process the batch of examples
            video_chunks = torch.cat([ex[2] for ex in examples], dim=0)
            # vae = vae.to(device)
            latents = vae.encode(video_chunks.to(device)).cpu()
            # vae = vae.to("cpu")
            
            # Encode text for the batch
            texts = [ex[3]['narration'] for ex in examples]
            encoded_texts = encode_for_batch(tokenizer, text_encoder, texts, device=device).cpu()

            # Process each example in the batch
            for ex, latent, encoded_text in zip(examples, latents, encoded_texts):
                i, j, chunk, annotation_dict, video_path, start_time = ex
                #latent of shape (c, t, h, w)
                #encoded_text of shape (l, d)

                # if chunk_duration != vae.video_vae.pixel_chunk_duration, cut off the padded frames
                if chunk_duration != vae.video_vae.pixel_chunk_duration:
                    latent = latent[:, :(chunk_duration - 1) // 8 + 1, :, :]

                # Pad T5 embedding to t5_embeding_max_length
                L, C_ = encoded_text.shape
                t5_embed = torch.zeros(t5_embeding_max_length, C_, dtype=torch.bfloat16)
                t5_embed[:L] = encoded_text

                # Save data to folder
                torch.save(latent, os.path.join(args.output_path, f"{i}-{j}.video_latent.pth"))
                torch.save(t5_embed, os.path.join(args.output_path, f"{i}-{j}.t5_text_embeddings.pth"))

                # Create a T5 text mask of all ones
                # TODO decide whether or not allowing the model to attend to padding tokens is a good idea
                torch.save(
                    torch.ones(512, dtype=torch.bfloat16), os.path.join(args.output_path, f"{i}-{j}.t5_text_mask.pth")
                )

                # Save metadata
                info = {
                    "height": args.height,
                    "width": args.width,
                    "fps": meta["video_fps"],
                    "num_frames": chunk_duration,
                    "video_name": os.path.basename(video_path),
                    "start_frame": start_time * args.fps,
                    "start_time": start_time,
                    "narration": annotation_dict['narration']
                }
                with open(os.path.join(args.output_path, f"{i}-{j}.info.json"), "w") as json_file:
                    json.dump(info, json_file)
            
            examples = overflow_examples[:args.batch_size]
            overflow_examples = overflow_examples[args.batch_size:]


if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()
    main(args)
