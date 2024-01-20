import os
import warnings
import torch
import cv2
from argparse import ArgumentParser
from models.model import ClassificationModel
import torch.nn.functional as F
from omegaconf import OmegaConf
from transforms import Transforms


warnings.filterwarnings('ignore')


def predict(model, image, config):
    transforms = Transforms(config['dataset']['img_size'])
    test_transform = transforms.val_test_transform()
    img_tensor = test_transform(image).unsqueeze_(0)
    prediction = torch.argmax(F.softmax(model(img_tensor))).item()
    return prediction


def main():
    parser = ArgumentParser()
    parser.add_argument('--config', type=str, required=True, help='Model name (resnet18|resnet34|resnet50|efficientnet')
    parser.add_argument('--model_path', required=True, type=str, help='Path to model checkpoint')
    parser.add_argument('--image', type=str, required=True, help='Path to input image')
    args = parser.parse_args()

    checkpoint = torch.load(args.model_path)
    config = OmegaConf.load(args.config)
    model = ClassificationModel(config)
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()

    image = cv2.imread(os.path.join(args.image))
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    print(f'Prediction is: {predict(model, image, config)}')


if __name__ == '__main__':
    main()
