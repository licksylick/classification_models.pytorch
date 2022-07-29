from torchvision.transforms import Compose, ToTensor, Normalize, Resize
from config import IMAGE_SIZE


def transform():
    return Compose([
        Resize((IMAGE_SIZE, IMAGE_SIZE)),
        ToTensor(),
        Normalize(mean=[0.5977, 0.6098, 0.6091],
                  std=[0.3130, 0.3054, 0.3175])
    ])
