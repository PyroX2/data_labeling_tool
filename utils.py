import numpy as np
from PIL import Image
import groundingdino.datasets.transforms as T
import os
import torch
import cv2

# Add image preprocessing
TRANSFORM = T.Compose(
    [
        T.RandomResize([800], max_size=1333),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]
)

def preprocess_image(frame: np.ndarray) -> Image:
    # Convert frame to PIL Image
    image = Image.fromarray(frame).convert("RGB")

    # Preprocess image
    image, _ = TRANSFORM(image, None)
        
    return image

def prepare_output_dirs(output_dir: str, filename: str) -> tuple[str]:
    # Set output dirs and create them if they don't exist
    images_output_dir = os.path.join(output_dir, filename, "images")
    skipped_images_output_dir = os.path.join(output_dir, filename, "skipped_images")
    labels_output_dir = os.path.join(output_dir, filename, "labels")
    annotated_output_dir = os.path.join(output_dir, filename, "annotated")

    # If dirs don't exist create them
    if not os.path.exists(images_output_dir):
        os.makedirs(os.path.join(output_dir, filename, "images"))
    if not os.path.exists(skipped_images_output_dir):
        os.makedirs(os.path.join(output_dir, filename, "skipped_images"))
    if not os.path.exists(labels_output_dir):
        os.makedirs(os.path.join(output_dir, filename, "labels"))
    if not os.path.exists(annotated_output_dir):
        os.makedirs(os.path.join(output_dir, filename, "annotated"))
        
    return images_output_dir, skipped_images_output_dir, labels_output_dir, annotated_output_dir

def validate_bboxes(bboxes: torch.Tensor) -> list[bool]:
    if bboxes.size()[0] == 0:
        return False
    elif bboxes[0, 2].item() > 0.95 or bboxes[0, 3].item() > 0.95:
        return False
    return True

def draw_bboxes(bbox: list[float], image: np.ndarray) -> np.ndarray:
    height, width, _ = image.shape
    print(height, width)
    print(bbox)
    x1 = bbox[0]*width - bbox[2]*width/2
    y1 = bbox[1]*height - bbox[3]*height/2
    x2 = bbox[0]*width + bbox[2]*width/2
    y2 = bbox[1]*height + bbox[3]*height/2
    pt1 = [int(x1), int(y1)]
    pt2 = [int(x2), int(y2)]
    annotated_image = cv2.rectangle(image, pt1, pt2, (255, 0, 0), 2)
    return annotated_image