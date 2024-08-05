# GroundingDINO
Auto labeling is done using GroundingDINO. To install it follow the steps listed in the following repository: \
`https://github.com/IDEA-Research/GroundingDINO`



## Directory structure should be the same as the one below:
.\
├── GroundingDINO \
├── README.md \
├── run_inference.py \
└── weights

# Example of usage

```bash
python3 run_inference.py --video_path /path/to/video.mp4 --output_dir /outputs/dir --start_sec 4 --end_sec 5 --text_prompt car --cpu_only
```

The example above processes video "/path/to/video.mp4", saves output in "/outputs/dir", starts processing from 4th second of video and ends on 5th (exluded), annotates objects recognized as cars and runs model on cpu