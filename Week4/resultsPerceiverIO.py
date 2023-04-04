from perceiver.model.vision.optical_flow import convert_config, OpticalFlow
from perceiver.data.vision.optical_flow import OpticalFlowProcessor
from transformers import AutoConfig
import numpy as np
import torch


class PercIOOptFlow:
    def __init__(self):
        # Load pretrained model configuration from the Hugging Face Hub
        self.config = AutoConfig.from_pretrained("deepmind/optical-flow-perceiver")
        self.device = "cuda"#"cuda"
        
        # Convert configuration, instantiate model and load weights
        self.model = OpticalFlow(convert_config(self.config)).eval().to(self.device)
        
        # Create optical flow processor
        self.processor = OpticalFlowProcessor(patch_size=tuple(self.config.train_size))
    
    def inference(self, im1, im2, grayscale = True):
        
        # Eval mode
        self.model.eval()
        
        with torch.no_grad():
        
            # Grayscale
            if grayscale:
                im1 = np.tile(im1, (1, 1, 3))
                im2 = np.tile(im2, (1, 1, 3))
            
            frame_pair = (im1, im2)
        
            optical_flow = self.processor.process(self.model, image_pairs=[frame_pair], batch_size=1, device=self.device).numpy()
            optical_flow = optical_flow[0]
        
        return optical_flow
    