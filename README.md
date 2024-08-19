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

# Processing images manually 
For processing images manually I suggest the following procedure:
* Run data labeling tool of your choice (I'm using labelImg) and process all skipped images saving annotation output in directory called "fixed". The proper file structure is shown below.

.\
├── annotated\
├── fixed\
├── images\
├── labels\
├── skipped_images\
└── tags.json

* When you process all the skipped images you can run the following command:
```bash
python3 save_fixed.py --input_dir /path/to/output/dir/ --save_empty True
```
"--save_empty' flag can be omitted if you haven't processed all your files yet. It just assumes that all images that currently are not annotated don't contain the object that should be annotated.  

* Run data labelng tool of your choice, select "images" and "labels" directory and manually check if all annotations are correct. If not correct them.

The code above 