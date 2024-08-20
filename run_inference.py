from groundingdino.util.inference import annotate
import cv2
from time import time
import os
import argparse
import utils
from tracker import Tracker
from model import GroundingDINOModel
import filetagging
import logging

def process_video(text_prompt: str, box_threshold: float, text_threshold: float, start_sec: float, end_sec: float, tags: list, output_dir: str, device: str, reinitialize_tracker_every: int, video_path: str, model) -> None:
    video_dir = video_path[:video_path.rfind("/")] # Directory where processed video is stored in
    filename_ext = video_path[video_path.rfind("/")+1:] # Filename with extension
    filename = filename_ext[:filename_ext.rfind(".")] # Filename without extension

    tracker = Tracker(text_prompt)
    
    output_path_with_filename = os.path.join(output_dir, filename)
    if not os.path.exists(output_path_with_filename):
        os.makedirs(output_path_with_filename)
    for tag in tags:
        filetagging.add_tag(tag, output_path_with_filename)
    # Prepare dirs for outputs
    images_output_dir, skipped_images_output_dir, labels_output_dir, annotated_output_dir = utils.prepare_output_dirs(output_dir, filename)
    
    with open(os.path.join(labels_output_dir, "classes.txt"), "w+") as f:
        f.write("drone")

    # Capture video and get fps and number of frames
    cap = cv2.VideoCapture(os.path.join(video_dir, filename_ext))
    fps = cap.get(cv2.CAP_PROP_FPS)
    number_of_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Calculate start and end frames given start and end seconds and video fps
    start_frame = int(start_sec * fps)
    if start_frame > number_of_frames: # Process incorrect start frame
        start_frame = 0
        
    end_frame = int(end_sec * fps)    
    if end_frame > number_of_frames or end_frame == 0:
        end_frame = number_of_frames

    i = end_frame - 1    
    
    while(cap.isOpened()):
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        # Read frame
        ret, frame = cap.read()
                
        if frame is None:
            i -= 1
            continue
        
        # if current frame is before start frame read next one 
        if i > end_frame:
            i -= 1
            logging.info(f"Frame {i} skipped")
            continue
        
        # Copy frame as numpy ndarray
        image_source = frame.copy() 

        # Preprocess image
        image = utils.preprocess_image(frame) 

        # Detect objects and calculate inference time
        start_time = time()
        
        # Find object on new frame
        if tracker.initialized and i % reinitialize_tracker_every != 0:
            logging.debug(f"Processing frame {i} with tracker")
            bboxes, logits, phrases = tracker.predict(image_source)
        else:
            logging.debug(f"Processing frame {i} with GroundingDINO")
            bboxes, logits, phrases = model.predict(image)
            if utils.validate_bboxes(bboxes):
                tracker.initialize_tracker(image_source, bboxes[0].tolist())
                tracker.initialized = True
        logging.debug(f"Logits: {logits}")
        
        # Calculate and print inference time
        logging.debug(f"Inference time: {time() - start_time}")

        # Draw bounding boxes
        annotated_frame = annotate(image_source=image_source[:,:,::-1], boxes=bboxes, logits=logits, phrases=phrases)

        # Get maximum index in images output dir
        existing_indexes = [int(index[:-4]) for index in os.listdir(images_output_dir)] # Reads existing indexes in images output dir
        existing_indexes = list(set(existing_indexes) | set([int(index[:-4]) for index in os.listdir(skipped_images_output_dir)])) # Adds indexes from skipped images output dir
        
        # Get maximum existing index
        if len(existing_indexes) == 0:
            max_index = -1
        else:
            max_index = max(existing_indexes)
        
        # Get current frame index as string 
        curent_index = str(max_index+1)
            
        # Add 0s to the beggining of filename so that all filenames have the same length (COCO like naming)
        file_index = "0"*(12 - len(curent_index)) + curent_index
            
        # Save bboxes
        bboxes_list = []
        if bboxes.size()[0] != 0:
            for box in bboxes:
                bbox_x_center = box[0].item()
                bbox_y_center = box[1].item()
                bbox_height = box[2] .item()
                bbox_width = box[3].item()
                bboxes_list.append([0, bbox_x_center, bbox_y_center, bbox_height, bbox_width])
        else:
            cv2.imwrite(os.path.join(skipped_images_output_dir, f"{file_index}.jpg"), image_source)
            i-=1
            continue
        
        # Save outputs
        cv2.imwrite(os.path.join(annotated_output_dir, f"{file_index}.jpg"), annotated_frame)
        cv2.imwrite(os.path.join(images_output_dir, f"{file_index}.jpg"), image_source)
        with open(os.path.join(labels_output_dir, f"{file_index}.txt"), "w+") as f:
            for bbox in bboxes_list:
                f.write(str(bbox)[1:-1].replace(',', ''))

        # Print index of processed frame
        logging.info(f"Frame {i} processed")
        i -= 1

        # If frame is equal to end frame brake the loop 
        if i < start_frame:
            logging.info("Processing finished")
            break
    

def main():

    # Create argparser
    parser = argparse.ArgumentParser(
                        prog='auto_labeling',
                        description='Creates labels automatically using GroundingDINO')
    parser.add_argument('--video_path', "-i", required=True, type=str, help="Path to input video")
    parser.add_argument('--text_prompt', "-t", required=True, type=str, help="Name of object to label")
    parser.add_argument(
        "--output_dir", "-o", type=str, default="outputs", required=True, help="output directory")
    parser.add_argument('--start_sec', required=False, type=float, default=0, help="Second to start labeling from. Default is 0")
    parser.add_argument('--end_sec', required=False, type=float, default=0, help="Second to end labeling on. Default is the last frame of the video")
    parser.add_argument("--box_threshold", type=float, default=0.4, help="box threshold")
    parser.add_argument("--text_threshold", type=float, default=0.4, help="text threshold")
    parser.add_argument("--cpu_only", action="store_true", help="running on cpu only!", default=False)
    parser.add_argument("--tag", type=str, default="no_tag")
    parser.add_argument("--check_every", type=int, default=10, help="Number of frames after which Grounding DINO initializes tracker with new detection")
    parser.add_argument("--debug", action="store_true", help="Prining additional messages", default=False)

    args = parser.parse_args()

    # Set arguments
    text_prompt = args.text_prompt
    box_threshold = args.box_threshold
    text_threshold = args.text_threshold
    start_sec = args.start_sec
    end_sec = args.end_sec
    tags = args.tag
    tags = tags.split(' ')
    output_dir = args.output_dir
    device = "cpu" if args.cpu_only else "cuda"
    reinitialize_tracker_every = args.check_every # After object was detected tracker will continue to track it. This value defines after how many frames Grounding DINO will initialize trakcer with new detection
    video_path = args.video_path
    debug = args.debug
        
    if debug == True:
        logging_level = logging.DEBUG
    else:
        logging_level = logging.INFO
        
    logging.basicConfig(format="%(levelname)s | %(asctime)s | %(message)s", level=logging_level) # Make sure appropraite logs are printed
    
    model_config_path = "GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py"
    model_weights_path = "weights/groundingdino_swint_ogc.pth"

    # Create Grounding DINO model
    model = GroundingDINOModel(model_config_path, model_weights_path, device, text_prompt, box_threshold, text_threshold)
    
    if video_path.endswith('.mp4'):
        process_video(text_prompt, box_threshold, text_threshold, start_sec, end_sec, tags, output_dir, device, reinitialize_tracker_every, video_path, model)
    else:
        for filename in os.listdir(video_path):
            logging.info(f"Processing video: {filename}")
            single_video_path = os.path.join(video_path, filename)
            process_video(text_prompt, box_threshold, text_threshold, start_sec, end_sec, tags, output_dir, device, reinitialize_tracker_every, single_video_path, model)

if __name__ == '__main__':
    main()