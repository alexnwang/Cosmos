import json
import os
from pathlib import Path
from datetime import datetime, timedelta

import click
import pandas as pd
import time
from tqdm import tqdm

LEAD_FRAMES = 8
FRAMEDELTA = timedelta(seconds=LEAD_FRAMES/24)

@click.command()
@click.option('--annotation-file', default="/home/anw2067/scratch/mochi/data/raw/epic-kitchens/EPIC_100_train.csv", type=click.Path(exists=True), help='Path to the annotation file.')
@click.option('--video-dir', default="/home/anw2067/scratch/mochi/data/raw/epic-kitchens/P01/videos", type=click.Path(exists=True), help='Path to the video directory.')
@click.option('--output-dir', default="/home/anw2067/scratch/Cosmos/data", type=click.Path(), help='Path to the output directory.')
def main(annotation_file, video_dir, output_dir):
    output_dir = Path(output_dir)
    if not output_dir.exists():
        output_dir.mkdir(parents=True)
    
    annotations = pd.read_csv(annotation_file)
    
    entries = []
    
    video_files = sorted(os.listdir(video_dir))
    for video_file in video_files:
        print(f"Processing video {video_file}")
        video_path = os.path.join(video_dir, video_file)
        video_name = video_file.split(".")[0]
        
        video_annotations = annotations[annotations['video_id'] == video_name]
        i = 0
        for _, row in tqdm(video_annotations.iterrows(), total=len(video_annotations)):
            start_time = row['start_timestamp'] # string hh:mm:ss.ms
            end_time = row['stop_timestamp'] # string hh:mm:ss.ms
            
            # save the original clip
            clip_path = os.path.join(output_dir, f"{video_name}_{start_time}_{end_time}.mp4")
            os.system(f"ffmpeg -loglevel quiet -ss {start_time} -to {end_time} -i {video_path} -vf scale=848:480 -r 24 {clip_path}")
            
            # Save the narration to a text file
            narration_file_path = os.path.join(output_dir, f"{video_name}_{start_time}_{end_time}.txt")
            with open(narration_file_path, 'w') as narration_file:
                narration_file.write(row['narration'])
            
            # Save the conditioning clip which is the K frames ahead of the original clip
            # Read the frames per second (fps) of the video
            cmd = f"ffprobe -v error -select_streams v:0 -show_entries stream=r_frame_rate -of default=nokey=1:noprint_wrappers=1 {video_path}"
            fps = eval(os.popen(cmd).read().strip())
            
            # compute the start and end time of the conditioning clip
            time_obj = datetime.strptime(start_time, "%H:%M:%S.%f")
            conditioning_start_time = time_obj - FRAMEDELTA
            conditioning_end_time = time_obj + timedelta(seconds=1.1/fps)
            if conditioning_start_time.day != time_obj.day:
                print(f"Skipping clip at {start_time} as the {LEAD_FRAMES} frame lead goes before the start of the video.")
                continue
            
            conditioning_start_time = conditioning_start_time.strftime("%H:%M:%S.%f")[:-3]
            conditioning_end_time = conditioning_end_time.strftime("%H:%M:%S.%f")[:-3]

            conditioning_clip_path = os.path.join(output_dir, f"{video_name}_{start_time}_{end_time}-pretext.mp4")
            os.system(f"ffmpeg -loglevel quiet -ss {conditioning_start_time} -to {conditioning_end_time} -i {video_path} -vf scale=848:480 -r 24 {conditioning_clip_path}")
            
            # only use in jsonl if the conditioning clip exists
            entries.append({
                "visual_input": conditioning_clip_path,
                "prompt": row['narration']
            })
        
        # HACK: only use the first video for now
        break
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