import json
import os
from pathlib import Path
from datetime import datetime, timedelta

if __name__ == "__main__":
    json_path = '/vast/work/public/ml-datasets/ego4d/2.1-goalstep/ego4d.json'
    with open(json_path, 'r') as f:
        data = json.load(f)
    videos =    data['videos']
    print(len(videos))
    
    video_source_map = {}
    for video in videos:
        source = video['video_source']
        scenarios = video['scenarios']

        source = source.strip().lower().replace(' ', '_').replace('\n', '').replace('/', '_').replace('(', '').replace(')', '')
        scenarios = [scenario.strip().lower().replace(' ', '_').replace('\n', '').replace('/', '_').replace('(', '').replace(')', '') for scenario in scenarios]
        
        print(source, scenarios)
        for scenario in scenarios:
            key = (source, scenario)
            if key not in video_source_map:
                video_source_map[key] = []
            video_source_map[key].append(video)

    MAX_TAKE = 10
    SECONDS_PREVIEW = 14
    
    PREVIEW_SAVE_DIR='/home/anw2067/scratch/Cosmos/data/ego4d'
    VIDEO_SOURCE_DIR = "/vast/work/public/ml-datasets/ego4d/v1/full_scale"
    
    for (source, scenario), videos in video_source_map.items():
        print((source, scenario), len(videos))
         
        interval = max(1, len(videos) // MAX_TAKE)
        sampled_videos = videos[::interval][:MAX_TAKE]
        
        output_path = os.path.join(PREVIEW_SAVE_DIR, "10previews", scenario, source)
        os.makedirs(output_path, exist_ok=True)
        
        for i, video in enumerate(sampled_videos):
            video_uid = video['video_uid']
        
            video_path = os.path.join(VIDEO_SOURCE_DIR, f"{video_uid}.mp4")
            video_output_path = os.path.join(output_path, f"{video_uid}_{i}.mp4")
            
            start_time = (video['duration_sec'] // 2) - SECONDS_PREVIEW // 2
            end_time = start_time + SECONDS_PREVIEW
            
            ffmpeg_command = f"ffmpeg -loglevel quiet -ss {start_time} -to {end_time} -i {video_path} -c copy {video_output_path}"
            os.system(ffmpeg_command)
            
    for (source, scenario), videos in video_source_map.items():
        if source == 'bristol' and 'cooking' in scenario:
            print((source, scenario), len(videos))
            
            output_path = os.path.join(PREVIEW_SAVE_DIR, "bristol-cooking-previews")
            os.makedirs(output_path, exist_ok=True)
            
            for i, video in enumerate(videos):
                video_uid = video['video_uid']
            
                video_path = os.path.join(VIDEO_SOURCE_DIR, f"{video_uid}.mp4")
                video_output_path = os.path.join(output_path, f"{video_uid}_{i}.mp4")
                
                start_time = (video['duration_sec'] // 2) - SECONDS_PREVIEW // 2
                end_time = start_time + SECONDS_PREVIEW
                
                ffmpeg_command = f"ffmpeg -loglevel quiet -ss {start_time} -to {end_time} -i {video_path} -c copy {video_output_path}"
                os.system(ffmpeg_command)
        
        
     
    