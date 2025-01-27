import json
import os
from pathlib import Path
from datetime import datetime, timedelta
import random
from typing import Iterable

import click
import pandas as pd
import time
from tqdm import tqdm

LEAD_FRAMES = 8

@click.command()
@click.option('--narration-file', default='/vast/work/public/ml-datasets/ego4d/v1/annotations/narration.json', type=click.Path(exists=True), help='Path to the narration file.')
@click.option('--annotations-file', default='/vast/work/public/ml-datasets/ego4d/2.1-goalstep/ego4d.json', type=click.Path(exists=True), help='Path to the annotations file.')
@click.option('--video-dir', default="/vast/work/public/ml-datasets/ego4d/v1/full_scale", type=click.Path(exists=True), help='Path to the video directory.')
@click.option('--uid', default='bf6e4f01-7891-42fe-9b52-1b5033379b44', help='The uid of the video to process.')
@click.option('--output-dir', default="/home/anw2067/scratch/Cosmos/data/ego4d/splits", type=click.Path(), help='Path to the output directory.')
@click.option('--temporal_jitter', default=None, required=False, type=str, help='Temporal jitter range in seconds to apply to the pretext clip. Provide as a comma-separated list of two floats.')
def main(narration_file, annotations_file, video_dir, uid, output_dir, temporal_jitter):
    output_dir = Path(output_dir)
    if not output_dir.exists():
        output_dir.mkdir(parents=True)
        
    annotations = json.load(open(annotations_file))['videos']
    
    narrations_dict = json.load(open(narration_file))
    
    entries = []
    
    for video_uid in [uid]:
        output_dir = output_dir / video_uid
        output_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"Processing video {video_uid}")
        annotation = next(filter(lambda x: x['video_uid'] == uid, annotations), None)
        
        video_path = os.path.join(video_dir, f"{video_uid}.mp4")
        video_name = video_uid
        
        narrations = narrations_dict[uid]['narration_pass_1']['narrations']
        
        # Write narrations_dict to the output_dir as a jsonl file
        narrations_jsonl_path = output_dir / "narrations.jsonl"
        with open(narrations_jsonl_path, 'w') as narrations_jsonl_file:
            for narration in narrations:
                narrations_jsonl_file.write(json.dumps(narration) + '\n')

        for i, narration in tqdm(enumerate(narrations), total=len(narrations)):
            if temporal_jitter is not None:
                jitter_range = [float(x) for x in temporal_jitter.split(',')]
                assert isinstance(jitter_range, Iterable) and len(jitter_range) == 2
                jitter = random.uniform(*jitter_range)
            else:
                jitter = 0
            
            start_time = narration['timestamp_sec'] + jitter
            end_time = max(start_time+2, min(start_time + 4, narrations[i+1]['timestamp_sec'] if i+1 < len(narrations) else annotation['duration_sec']))
            
            # save the original clip
            clip_path = os.path.join(output_dir, f"{video_name}_{start_time:.2f}_{end_time:.2f}.mp4")
            os.system(f"ffmpeg -loglevel quiet -ss {start_time} -to {end_time} -i {video_path} -r 24 {clip_path}")
            
            # Save the narration to a text file
            narration_file_path = os.path.join(output_dir, f"{video_name}_{start_time:.2f}_{end_time:.2f}.txt")
            with open(narration_file_path, 'w') as narration_file:
                narration_file.write(narration['narration_text'][5:])
            
            # Save the conditioning clip which is the K frames ahead of the original clip
            # Read the frames per second (fps) of the video
            cmd = f"ffprobe -v error -select_streams v:0 -show_entries stream=r_frame_rate -of default=nokey=1:noprint_wrappers=1 {video_path}"
            fps = eval(os.popen(cmd).read().strip())
            
            # compute the start and end time of the conditioning clip
            conditioning_start_time = start_time - LEAD_FRAMES/fps + jitter
            conditioning_end_time = start_time + 1.1/fps + jitter
            if conditioning_start_time < 0:
                print(f"Skipping clip at {start_time} as the {LEAD_FRAMES} frame lead goes before the start of the video.")
                continue
            
            conditioning_clip_path = os.path.join(output_dir, f"{video_name}_{start_time:.2f}_{end_time:.2f}-pretext.mp4")
            os.system(f"ffmpeg -loglevel quiet -ss {conditioning_start_time} -to {conditioning_end_time} -i {video_path} -vf scale=1280:720 -r 24 {conditioning_clip_path}")
            
            # only use in jsonl if the conditioning clip exists
            entries.append({
                "visual_input": conditioning_clip_path,
                "prompt": narration['narration_text'][5:]
            })
        
    # Save a .jsonl file, where each row contains 2 entries. (1) a "visual_input" entry with the path to the conditioning clip, and (2) a "prompt" entry with the narration text
    jsonl_file_path = os.path.join(output_dir, f"video-and-text-prompt.jsonl")
    with open(jsonl_file_path, 'w') as jsonl_file:
        for entry in entries:
            jsonl_file.write(json.dumps(entry) + '\n')
            
    # write another jsonl file without the prompt entry
    jsonl_file_path = os.path.join(output_dir, f"video-prompt.jsonl")
    with open(jsonl_file_path, 'w') as jsonl_file:
        for entry in entries:
            jsonl_file.write(json.dumps({"visual_input": entry["visual_input"]}) + '\n')
            

if __name__ == "__main__":
    main()