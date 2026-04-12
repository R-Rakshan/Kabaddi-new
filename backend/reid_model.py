import torch
import torchvision.models as models
import torchvision.transforms as T
import numpy as np
import cv2

class ReIDExtractor:
    def __init__(self, device=None):
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = device
            
        # Use a lightweight pretrained model (ResNet18) for feature extraction
        self.model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        # Remove the classification head (fc layer)
        self.model = torch.nn.Sequential(*list(self.model.children())[:-1])
        self.model.eval()
        self.model.to(self.device)
        
        self.transform = T.Compose([
            T.ToPILImage(),
            T.Resize((256, 128)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    @torch.no_grad()
    def extract(self, frame, boxes):
        """
        Extract features for multiple bounding boxes.
        boxes: list of (x1, y1, x2, y2)
        Returns a numpy array of shape (N, feature_dim)
        """
        if len(boxes) == 0:
            return np.zeros((0, 512), dtype=np.float32)
            
        crops = []
        for (x1, y1, x2, y2) in boxes:
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(frame.shape[1], x2), min(frame.shape[0], y2)
            crop = frame[y1:y2, x1:x2]
            
            # Handle edge cases where box is empty or outside
            if crop.size == 0 or x2 <= x1 or y2 <= y1:
                crop = np.zeros((128, 64, 3), dtype=np.uint8)
                
            crop_tensor = self.transform(crop)
            crops.append(crop_tensor)
            
        batch = torch.stack(crops).to(self.device)
        features = self.model(batch)
        features = features.view(features.size(0), -1) # Flatten (N, 512, 1, 1) -> (N, 512)
        
        # Normalize features
        features = torch.nn.functional.normalize(features, p=2, dim=1)
        return features.cpu().numpy()

# Singleton instance
_extractor = None

def get_reid_extractor():
    global _extractor
    if _extractor is None:
        _extractor = ReIDExtractor()
    return _extractor
