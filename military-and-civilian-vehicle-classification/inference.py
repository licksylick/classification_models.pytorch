import os
import warnings
import cv2
import torch
from PIL import Image
from argparse import ArgumentParser
from utils import image_resize
from transforms import transform
import torch.nn.functional as F
from models.efficientnet import EfficientNetModel
from config import NUM_CLASSES


warnings.filterwarnings('ignore')


def predict(model, image):
    img_tensor = transform()(image).unsqueeze_(0)
    prediction = torch.argmax(F.softmax(model(img_tensor))).item()
    return prediction


def main():
    parser = ArgumentParser()
    parser.add_argument('--model', type=str, required=True, help='Model name (resnet18|resnet34|resnet50|efficientnet')
    parser.add_argument('--model_path', required=True, type=str, help='Path to model checkpoint')
    parser.add_argument('--image', type=str, required=True, help='Path to input image')
    args = parser.parse_args()

    idx_to_label = {0: "military tank", 1: "civilian car",
                    2: "military helicopter", 3: "military aircraft", 4: "military truck", 5: "civilian aircraft"}

    checkpoint = torch.load(args.model_path)
    model = EfficientNetModel(args.model, NUM_CLASSES)
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()

    image = Image.open(os.path.join(args.image)).convert('RGB')
    pred_label = idx_to_label[predict(model, image)]

    cv2.imshow(f'Prediction is: {pred_label}', image_resize(cv2.imread(args.image), width=500))
    cv2.waitKey(0)


if __name__ == '__main__':
    main()
