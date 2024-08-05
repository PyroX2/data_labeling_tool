import cv2
import numpy as np
import torch
import matplotlib.pyplot as plt


class Tracker:
    def __init__(self, text_prompt: str) -> None:
        # Create cv2 tracker
        nano_params = cv2.TrackerNano.Params()
        nano_params.backbone = 'nano_tracker_weights/nanotrack_backbone_sim.onnx'
        nano_params.neckhead = 'nano_tracker_weights/nanotrack_head_sim.onnx'
        self.tracker = cv2.TrackerNano.create(nano_params)
        self.initialized = False
        self.text_prompt = text_prompt
        
    # Initialize tracker with detected bbox
    def initialize_tracker(self, frame: np.ndarray, bbox: list)  -> None:
        height, width, _ = frame.shape
        x_center = bbox[0] * width
        y_center = bbox[1] * height
        bbox_width = bbox[2] * width
        bbox_height = bbox[3] * height
        
        bbox = [int(x_center - bbox_width / 2), int(y_center - bbox_height / 2), int(bbox_width), int(bbox_height)]
        
        self.tracker.init(frame, bbox)
        self.initialized = True
    
    # Predict bbox on next frame
    def predict(self, frame: np.ndarray) -> tuple[torch.Tensor, torch.Tensor, str]:
        ok, bbox = self.tracker.update(frame)
        
        height, width, _ = frame.shape
        bbox_width = bbox[2] / width
        bbox_height = bbox[3] / height
        x_center = (bbox[0]) / width + bbox_width / 2
        y_center = (bbox[1]) / height + bbox_height / 2
        
        bbox = [x_center, y_center, bbox_width, bbox_height]
        
        bbox = torch.Tensor(bbox)
        bboxes = bbox.unsqueeze(0)
        logits = torch.Tensor([0.0])
        phrase = [self.text_prompt]
            
        return bboxes, logits, phrase