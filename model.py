from groundingdino.util.inference import load_model, predict, annotate
from PIL import Image
import torch


class GroundingDINOModel:
    def __init__(self, config_path, weights_path, device, text_prompt, box_threshold, text_threshold):
        self.device = device
        self.model = load_model(config_path, weights_path, device=self.device)
        self.text_prompt = text_prompt
        self.box_threshold = box_threshold
        self.text_threshold = text_threshold
        
    def predict(self, frame: Image) -> tuple[torch.Tensor, torch.Tensor, str]:
        bboxes, logits, phrases = predict(
            model=self.model,
            image=frame,
            caption=self.text_prompt,
            box_threshold=self.box_threshold,
            text_threshold=self.text_threshold,
            device=self.device
        )
        
        return bboxes, logits, phrases
        
