from groundingdino.util.inference import load_model, predict, annotate
import groundingdino.datasets.transforms as T
import cv2
from PIL import Image
from time import time
import os


# Set constants
TEXT_PROMPT = "drone"
BOX_TRESHOLD = 0.35
TEXT_TRESHOLD = 0.25
START_SEC = 4
END_SEC = 5
TAG = "cloudy"
VIDEO_PATH = "/mnt/nas-data/HardKill/hardkill_videos/data_validation_videos/rgb_1080p"
FILENAME = 'dsv_rgb_1080p_nalot_air_9s_cloudy'

# Load model weights
model = load_model("GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py", "weights/groundingdino_swint_ogc.pth", device="cpu")

# Set output dirs and create them if they don't exist
images_output_dir = os.path.join("outputs", TAG, FILENAME, "images")
labels_output_dir = os.path.join("outputs", TAG, FILENAME, "labels")
annotated_output_dir = os.path.join("outputs", TAG, FILENAME, "annotated")

if not os.path.exists(images_output_dir):
    os.makedirs(os.path.join("outputs", TAG, FILENAME, "images"))
if not os.path.exists(labels_output_dir):
    os.makedirs(os.path.join("outputs", TAG, FILENAME, "labels"))
if not os.path.exists(annotated_output_dir):
    os.makedirs(os.path.join("outputs", TAG, FILENAME, "annotated"))


# Capture video and get fps and number of frames
full_name = FILENAME + ".mp4"
cap = cv2.VideoCapture(os.path.join(VIDEO_PATH, full_name))
fps = cap.get(cv2.CAP_PROP_FPS)
number_of_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT) 

# Calculate start and end frames given start and end seconds and video fps
start_frame = int(START_SEC * fps)
end_frame = END_SEC * fps
if end_frame > number_of_frames:
    end_frame = number_of_frames


# Add image preprocessing
transform = T.Compose(
    [
        T.RandomResize([800], max_size=1333),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]
)

i = 0
while(cap.isOpened()):
    ret, frame = cap.read()
    
    # if current frame is before start frame read next one 
    if i < start_frame:
        i += 1
        continue
    
    # Copy frame as numpy ndarray
    image_source = frame.copy() 

    # Convert frame to PIL Image
    image = Image.fromarray(frame).convert("RGB")

    # Preprocess image
    image, _ = transform(image, None)

    # Detect objects and calculate inference time
    start_time = time()
    boxes, logits, phrases = predict(
        model=model,
        image=image,
        caption=TEXT_PROMPT,
        box_threshold=BOX_TRESHOLD,
        text_threshold=TEXT_TRESHOLD,
        device="cpu"
    )
    print(f"Inference time: {time() - start_time}")

    # Draw bounding boxes
    annotated_frame = annotate(image_source=image_source, boxes=boxes, logits=logits, phrases=phrases)

    # Get maximum index in images output dir
    existing_indexes = [int(index[:-4]) for index in os.listdir(images_output_dir)]
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
        cv2.imwrite(os.path.join(images_output_dir, f"{file_index}.jpg"), image_source)
        continue
    
    # Save outputs
    cv2.imwrite(os.path.join(annotated_output_dir, f"{file_index}.jpg"), annotated_frame)
    cv2.imwrite(os.path.join(images_output_dir, f"{file_index}.jpg"), image_source)
    with open(os.path.join(labels_output_dir, f"{file_index}.txt"), "w+") as f:
        for bbox in bboxes:
            f.write(str(bbox)[1:-1].replace(',', ''))

    i += 1

    print("Processed frame index:", i)

    # If frame is equal to end frame brake the loop 
    if i > end_frame:
        break