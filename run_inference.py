from groundingdino.util.inference import load_model, predict, annotate
import groundingdino.datasets.transforms as T
import cv2
from PIL import Image
from time import time
import os
import argparse
import utils


# Create argparser
parser = argparse.ArgumentParser(
                    prog='auto_labeling',
                    description='Creates labels automatically using GroundingDINO')
parser.add_argument('--video_path', "-i", required=True, type=str, help="Path to input video")
parser.add_argument('--text_prompt', "-t", required=True, type=str, help="Name of object to label")
parser.add_argument(
    "--output_dir", "-o", type=str, default="outputs", required=True, help="output directory"
)
parser.add_argument('--start_sec', required=False, type=float, default=0, help="Second to start labeling from. Default is 0")
parser.add_argument('--end_sec', required=False, type=float, default=0, help="Second to end labeling on. Default is the last frame of the video")
parser.add_argument("--box_threshold", type=float, default=0.35, help="box threshold")
parser.add_argument("--text_threshold", type=float, default=0.25, help="text threshold")
parser.add_argument("--cpu_only", action="store_true", help="running on cpu only!", default=False)
parser.add_argument("--tag", type=str, default="no_tag")

args = parser.parse_args()

# Set arguments
text_prompt = args.text_prompt
box_threshold = args.box_threshold
text_threshold = args.text_threshold
start_sec = args.start_sec
end_sec = args.end_sec
tag = args.tag
output_dir = args.output_dir
device = "cpu" if args.cpu_only else "cuda"

video_dir = args.video_path[:args.video_path.rfind("/")] # Directory where processed video is stored in
filename_ext = args.video_path[args.video_path.rfind("/")+1:] # Filename with extension
filename = filename_ext[:filename_ext.rfind(".")] # Filename without extension

# Load model weights
model = load_model("GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py", "weights/groundingdino_swint_ogc.pth", device=device)

# Set output dirs and create them if they don't exist
images_output_dir = os.path.join(output_dir, tag, filename, "images")
skipped_images_output_dir = os.path.join(output_dir, tag, filename, "skipped_images")
labels_output_dir = os.path.join(output_dir, tag, filename, "labels")
annotated_output_dir = os.path.join(output_dir, tag, filename, "annotated")

if not os.path.exists(images_output_dir):
    os.makedirs(os.path.join(output_dir, tag, filename, "images"))
if not os.path.exists(skipped_images_output_dir):
    os.makedirs(os.path.join(output_dir, tag, filename, "skipped_images"))
if not os.path.exists(labels_output_dir):
    os.makedirs(os.path.join(output_dir, tag, filename, "labels"))
if not os.path.exists(annotated_output_dir):
    os.makedirs(os.path.join(output_dir, tag, filename, "annotated"))


# Capture video and get fps and number of frames
cap = cv2.VideoCapture(os.path.join(video_dir, filename_ext))
fps = cap.get(cv2.CAP_PROP_FPS)
number_of_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT) 

# Calculate start and end frames given start and end seconds and video fps
start_frame = int(start_sec * fps)
end_frame = end_sec * fps
if end_frame > number_of_frames:
    end_frame = number_of_frames



i = 0
while(cap.isOpened()):
    # Read frame
    ret, frame = cap.read()
    
    # if current frame is before start frame read next one 
    if i < start_frame:
        i += 1
        continue
    
    # Copy frame as numpy ndarray
    image_source = frame.copy() 

    # Preprocess image
    image = utils.preprocess_image(frame) 

    # Detect objects and calculate inference time
    start_time = time()
    boxes, logits, phrases = predict(
        model=model,
        image=image,
        caption=text_prompt,
        box_threshold=box_threshold,
        text_threshold=text_threshold,
        device=device
    )
    
    # Calculate and print inference time
    print(f"Inference time: {time() - start_time}")

    # Draw bounding boxes
    annotated_frame = annotate(image_source=image_source, boxes=boxes, logits=logits, phrases=phrases)

    # Get maximum index in images output dir
    existing_indexes = [int(index[:-4]) for index in os.listdir(images_output_dir)] # Reads existing indexes in images output dir
    existing_indexes = list(set(existing_indexes) | set([int(index[:-4]) for index in os.listdir(skipped_images_output_dir)])) # Adds indexes from skipped images output dir
    if len(existing_indexes) == 0:
        max_index = -1
    else:
        max_index = max(existing_indexes)
    
    # Get current frame index as string 
    curent_index = str(max_index+1)
        
    # Add 0s to the beggining of filename so that all filenames have the same length (COCO like naming)
    file_index = "0"*(12 - len(curent_index)) + curent_index
        
    # Save bboxes
    bboxes = []
    if boxes.size()[0] != 0:
        for box in boxes:
            bbox_x_center = box[0].item()
            bbox_y_center = box[1].item()
            bbox_height = box[2] .item()
            bbox_width = box[3].item()
            bboxes.append([bbox_x_center, bbox_y_center, bbox_height, bbox_width])
    else:
        cv2.imwrite(os.path.join(skipped_images_output_dir, f"{file_index}.jpg"), image_source)
        continue
    
    # Save outputs
    cv2.imwrite(os.path.join(annotated_output_dir, f"{file_index}.jpg"), annotated_frame)
    cv2.imwrite(os.path.join(images_output_dir, f"{file_index}.jpg"), image_source)
    with open(os.path.join(labels_output_dir, f"{file_index}.txt"), "w+") as f:
        for bbox in bboxes:
            f.write(str(bbox)[1:-1].replace(',', ''))

    # Print index of processed frame
    print("Processed frame index:", i)
    i += 1

    # If frame is equal to end frame brake the loop 
    if i > end_frame:
        break