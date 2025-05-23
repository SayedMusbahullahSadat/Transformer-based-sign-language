
"""
Download and extract all MS-ASL100 clips.
Structure of each entry:
{'url': ..., 'start_time': ..., 'end_time': ..., 'label': int, ...} :contentReference[oaicite:0]{index=0}:contentReference[oaicite:1]{index=1}
"""

import json
import os
import subprocess
from urllib.parse import urlparse, parse_qs

from yt_dlp import YoutubeDL

# 1) Load all splits
splits = ['MSASL_train.json', 'MSASL_val.json', 'MSASL_test.json']
entries = []
for fn in splits:
    with open(fn, 'r', encoding='utf-8') as f:
        entries.extend(json.load(f))

# 2) Filter for MS-ASL100 (label < 100)
subset = [e for e in entries if e.get('label', 1000) < 100]

# 3) Prepare directories
out_root   = 'MS-ASL100'
raw_dir    = os.path.join(out_root, 'raw_videos')
clips_dir  = os.path.join(out_root, 'clips')
os.makedirs(raw_dir, exist_ok=True)
os.makedirs(clips_dir, exist_ok=True)

# 4) yt-dlp options: download highest-quality mp4, naming by video id
ydl_opts = {
    'format': 'mp4',
    'outtmpl': os.path.join(raw_dir, '%(id)s.%(ext)s'),
    'quiet': False,
}

ydl = YoutubeDL(ydl_opts)

def extract_video_id(youtube_url):
    """Normalize URL and extract the 'v' parameter."""
    if youtube_url.startswith('www.'):
        youtube_url = 'https://' + youtube_url
    parsed = urlparse(youtube_url)
    qs = parse_qs(parsed.query)
    vid = qs.get('v')
    if vid:
        return vid[0]
    # fallback: path-based ID (e.g., youtu.be/ID)
    return os.path.basename(parsed.path)

for clip in subset:
    url        = clip['url']
    start_sec  = clip['start_time']
    end_sec    = clip['end_time']
    vid_id     = extract_video_id(url)
    raw_path   = os.path.join(raw_dir,    f'{vid_id}.mp4')
    clip_name  = f'{vid_id}_{start_sec:.2f}_{end_sec:.2f}.mp4'
    clip_path  = os.path.join(clips_dir, clip_name)

    # 4a) Download the raw video if not already present
    if not os.path.exists(raw_path):
        print(f'Downloading {vid_id} …')
        try:
            ydl.download([url])
        except Exception as e:
            print(f'  ERROR downloading {vid_id}: {e}')
            continue

    # 4b) Trim the clip with ffmpeg
    if not os.path.exists(clip_path):
        print(f'Trimming clip {clip_name} …')
        cmd = [
            'ffmpeg', '-y',
            '-ss', str(start_sec),
            '-to', str(end_sec),
            '-i', raw_path,
            '-c', 'copy',
            clip_path
        ]
        subprocess.run(cmd, check=True)
    else:
        print(f'Already have clip {clip_name}, skipping.')
