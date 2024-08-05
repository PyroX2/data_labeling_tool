import numpy as np
from PIL import Image
import groundingdino.datasets.transforms as T

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