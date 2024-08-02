from groundingdino.util.inference import load_model, load_image, predict, annotate
import groundingdino.datasets.transforms as T
import cv2
from PIL import Image
from time import time
import os



model = load_model("GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py", "weights/groundingdino_swint_ogc.pth", device="cpu")
TEXT_PROMPT = "drone"
BOX_TRESHOLD = 0.35
TEXT_TRESHOLD = 0.25
START_SEC = 4
END_SEC = 5
TAG = "cloudy"
video_path = "/mnt/nas-data/HardKill/hardkill_videos/data_validation_videos/rgb_1080p"
file_name = 'dsv_rgb_1080p_nalot_air_9s_cloudy'

if os.path.exists(os.path.join("outputs", TAG, file_name, "images")):
    os.mkdir(os.path.join("outputs", TAG, file_name, "images"))
if os.path.exists(os.path.join("outputs", TAG, file_name, "labels")):
    os.mkdir(os.path.join("outputs", TAG, file_name, "labels"))


full_name = file_name + ".mp4"
cap = cv2.VideoCapture(os.path.join(video_path, full_name))
fps = cap.get(cv2.CAP_PROP_FPS)
length = cap.get(cv2.CAP_PROP_FRAME_COUNT) 

start_sec = 4
end_sec = 5
start_frame = int(START_SEC * fps)
end_frame = END_SEC * fps
if end_frame > length:
    end_frame = length



transform = T.Compose(
    [
        T.RandomResize([800], max_size=1333),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]
)

i = 0

# image_source, image = load_image(IMAGE_PATH)
while(cap.isOpened()):
    ret, frame = cap.read()
    if i < start_frame:
        i += 1
        continue
    image_source = frame.copy()

    width, height, _ = image_source.shape

    image = Image.fromarray(frame).convert("RGB")

    image, _ = transform(image, None)

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


    bboxes = []
    if boxes.size()[0] != 0:
        for box in boxes:
            bbox_x_center = box[0] * height
            bbox_y_center = box[1] * width
            bbox_height = box[2] * height
            bbox_width = box[3] * width

            bboxes.append([bbox_x_center, bbox_y_center, bbox_height, bbox_width])

    annotated_frame = annotate(image_source=image_source, boxes=boxes, logits=logits, phrases=phrases)

    annotation = {"filename": full_name, "image_name": f"frame_{i}", "bboxes": bboxes}
    cv2.imwrite(os.path.join("outputs", TAG, file_name, "annotated", f"annotated_{i}.jpg"), annotated_frame)
    cv2.imwrite(os.path.join("outputs", TAG, file_name, "images", f"frame_{i}.jpg"), annotated_frame)
    with open(os.path.join("outputs", TAG, file_name, "labels", f"frame_{i}.txt"), "w+") as f:
        for bbox in bboxes:
            f.write(bbox)

    i += 1

    print("Processed frame index:", i)

    if i > end_frame:
        break