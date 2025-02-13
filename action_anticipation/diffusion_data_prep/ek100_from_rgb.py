import argparse
import datetime
from functools import partial
import glob
import json
import os
import pickle
import random

import pandas as pd
import torch
import torchvision
from einops import rearrange
from huggingface_hub import snapshot_download
from nemo.collections.diffusion.models.model import DiT7BConfig
from tqdm import tqdm
from transformers import T5EncoderModel, T5TokenizerFast

from cosmos1.utils import log
from multiprocessing import Pool

def get_parser():
    parser = argparse.ArgumentParser(description="Process some configurations.")
    parser.add_argument("--tokenizer_dir", type=str, default="", help="Path to the VAE model")
    
    parser.add_argument("--dataset_path", type=str, required=True, help="Path to the epic kitchens dataset")
    parser.add_argument("--output_path", type=str, required=True, help="Path to the output directory")
    
    # parser.add_argument("--prompt", type=str, default="a video of sks.", help="Prompt for the video")
    # parser.add_argument("--num_chunks", type=int, default=5, help="Number of random chunks to sample per video")
    parser.add_argument("--height", type=int, default=704, help="Height to resize video")
    parser.add_argument("--width", type=int, default=1280, help="Width to resize video")
    parser.add_argument("--num_frames", type=int, default=33, help="Number of frames per chunk")
    parser.add_argument("--fps", type=int, default=12, help="Frames per second")
    parser.add_argument("--max_text_length", type=int, default=512, help="Maximum text length")
    return parser

def init_t5(device="cuda"):
    """Initialize and return the T5 tokenizer and text encoder."""
    tokenizer = T5TokenizerFast.from_pretrained("google-t5/t5-11b")
    text_encoder = T5EncoderModel.from_pretrained("google-t5/t5-11b")
    text_encoder.to(device)
    text_encoder.eval()
    return tokenizer, text_encoder


def init_video_tokenizer(tokenizer_dir: str, device="cuda"):
    """Initialize and return the Cosmos Video tokenizer."""
    dit_config = DiT7BConfig(vae_path=tokenizer_dir)
    vae = dit_config.configure_vae()
    vae.to(device)
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

def load_frame(video_dir, frame_number):
    frame_path = os.path.join(video_dir, f"frame_{int(frame_number):010d}.jpg")
    if os.path.exists(frame_path):
        return torchvision.io.read_image(frame_path)
    else:
        return None

def main(args):
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    rank = int(os.environ.get("RANK", 0))
    
    device = torch.device(f"cuda:{rank}")
    torch.cuda.set_device(device)
    
    # Set up output directory
    os.makedirs(args.output_path, exist_ok=True)

    # Initialize T5
    print("Initializing T5")
    tokenizer, text_encoder = init_t5(device)

    # Initialize the VAE
    print("Initializing VAE")
    if args.tokenizer_dir == "":
        args.tokenizer_dir = snapshot_download("nvidia/Cosmos-1.0-Tokenizer-CV8x8x8")
    vae = init_video_tokenizer(args.tokenizer_dir, device)
    
    # setting constants
    t5_embeding_max_length = args.max_text_length
    print(f"Using {t5_embeding_max_length} as the maximum text length")
    if args.num_frames != vae.video_vae.pixel_chunk_duration:
        print(f"Using {args.num_frames} frames per chunk")
        chunk_duration = args.num_frames
    else:
        print(f"Using {vae.video_vae.pixel_chunk_duration} frames per chunk")
        chunk_duration = vae.video_vae.pixel_chunk_duration  # Frames per chunk
    
    # check if annotations exist
    train_annotations = os.path.join(args.dataset_path, "anns/EPIC_100_train.pkl")
    assert os.path.exists(train_annotations), "Training annotations not found"
    
    # load annotations
    with open(train_annotations, "rb") as f:
        train_data = pickle.load(f)
        
    # load video fps details
    fp = "/home/alexnwang/Cosmos/data/epickitchens/EPIC_100_video_info.csv"
    video_info = pd.read_csv(fp)
    
    count = 0  # File index
    with torch.no_grad(), Pool(processes=16) as pool:
        # for each action, read the start time and end time, start frame and end frame
        # train_data is a pandas dataframe
        for count, (_, row) in enumerate(tqdm(train_data.iterrows(), total=len(train_data))):
            if count % world_size != rank:
                continue
            
            participant_id = row['participant_id']
            video_id = row['video_id']
            if video_id in ["P01_109", "P27_103"]:
                continue
            
            
            fps = float(video_info[video_info['video_id'] == video_id]['fps'].values[0])
            
            narration = row['narration']
            start_frame = row['start_frame']
            stop_frame = row['stop_frame']
            
            # print(f"Processing {video_id} with narration: {narration} from frames {start_frame} to {stop_frame}")
            
            # get dir of frames
            video_dir = os.path.join(args.dataset_path, "frames", participant_id, "rgb_frames", video_id)

            frame_ratio = fps / args.fps
            
            stop_frame = max(start_frame + args.num_frames * frame_ratio, stop_frame)
            # TODO add a jitter to the start frame to cover bad start frames
            frame_numbers = torch.arange(start_frame, stop_frame, frame_ratio)
            # round to the nearest frame
            frame_numbers = frame_numbers.round().int()
            
            # if len(frame_indices) > args.num_frames, we have more than 1 candidate start point
            # lets pick one every 2 seconds
            for inner_count, index in enumerate(range(0, len(frame_numbers)-args.num_frames+1, args.fps)):
                chunk = pool.map(partial(load_frame, video_dir), frame_numbers[index:index+args.num_frames])

                try:
                    chunk = torch.stack(chunk)  # (T, C, H, W)
                except Exception as e:
                    print(f"Skipping chunk due to error: {e}")
                    continue
                
                T, C, H, W = chunk.shape
                
                # save video
                if index == 0 and count % 100 == 0:
                    save_path = os.path.join(args.output_path, f"{count}-{inner_count}-{video_id}-{narration}-{frame_numbers[index]}.mp4")
                    torchvision.io.write_video(save_path, chunk.permute(0, 2, 3, 1), args.fps)

                # if chunk_duration != vae.video_vae.pixel_chunk_duration, pad it up to the required length
                if args.num_frames != vae.video_vae.pixel_chunk_duration:
                    chunk = torch.cat([chunk, torch.zeros(vae.video_vae.pixel_chunk_duration - chunk_duration, C, H, W)], dim=0)

                # Resize to [h, w] for each frame
                chunk = torchvision.transforms.functional.resize(chunk, [args.height, args.width])
            
                # Expand dims: (T, C, H, W) -> (B=1, C, T, H, W)
                chunk = rearrange(chunk, "(b t) c h w -> b c t h w", b=1)
                
                # Convert to bf16 and normalize from [0, 255] to [-1, 1]
                chunk = chunk.to(device=device, dtype=torch.bfloat16, non_blocking=True) / 127.5 - 1.0

                # Encode video
                latent = vae.encode(chunk).cpu()  # shape: (1, latent_channels, T//factor, H//factor, W//factor)
                
                # if chunk_duration != vae.video_vae.pixel_chunk_duration, cut off the padded frames
                if chunk_duration != vae.video_vae.pixel_chunk_duration:
                    latent = latent[:, :, :(chunk_duration-1) // 8 + 1, :, :] # hack as it's always temporal stride = 8

                # Encode text
                out = encode_for_batch(tokenizer, text_encoder, [narration], max_length=t5_embeding_max_length, device=device)[0]
                encoded_text = torch.tensor(out, dtype=torch.bfloat16)

                # Pad T5 embedding to t5_embeding_max_length
                L, C_ = encoded_text.shape
                t5_embed = torch.zeros(1, t5_embeding_max_length, C_, dtype=torch.bfloat16)
                t5_embed[0, :L] = encoded_text

                # Save data to folder
                torch.save(latent[0], os.path.join(args.output_path, f"{count}-{inner_count}.video_latent.pth"))
                torch.save(t5_embed[0], os.path.join(args.output_path, f"{count}-{inner_count}.t5_text_embeddings.pth"))

                # Create a T5 text mask of all ones
                torch.save(
                    torch.ones(t5_embeding_max_length, dtype=torch.bfloat16), os.path.join(args.output_path, f"{count}-{inner_count}.t5_text_mask.pth")
                )
                
                # Save metadata
                info = {
                    "height": args.height,
                    "width": args.width,
                    "fps": args.fps,
                    "num_frames": chunk_duration,
                    "video_path": video_dir,
                    "start_frame": int(frame_numbers[index]),
                }
                with open(os.path.join(args.output_path, f"{count}-{inner_count}.info.json"), "w") as json_file:
                    json.dump(info, json_file)

                
                
    
if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()
    main(args)