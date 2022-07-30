import pandas as pd
import cv2
import torch
from torchvision import transforms
from models.resnet import Resnet
from models.efficientnet import EfficientNetModel


def preprocess_csv(df):
    df = df.drop(['width', 'height', 'xmin', 'ymin', 'xmax', 'ymax'], axis=1)
    df['class'] = df['class'].replace(['military tank', 'civilian car', 'military helicopter', 'military aircraft',
                                       'military truck', 'civilian aircraft'], ['0', '1', '2', '3', '4', '5'])
    return df


def get_model(model_name, num_classes):
    model = None
    if 'resnet' in model_name:
        model = Resnet(model_name, num_classes)
    elif 'efficientnet' in model_name:
        model = EfficientNetModel(model_name, num_classes)
    else:
        raise ValueError(f'Undefined model name: {model_name}')
    return model


def get_mean_and_std(dataloader):
    channels_sum, channels_squared_sum, num_batches = 0, 0, 0
    for data, _ in dataloader:
        convert_tensor = transforms.ToTensor()
        data = convert_tensor(data)
        data = data.unsqueeze(0)
        channels_sum += torch.mean(data, dim=[0, 2, 3])
        channels_squared_sum += torch.mean(data ** 2, dim=[0, 2, 3])
        num_batches += 1

    mean = channels_sum / num_batches
    std = (channels_squared_sum / num_batches - mean ** 2) ** 0.5

    return mean, std


def balance_dataframe(data):
    data = data.groupby('class')
    data = pd.DataFrame(data.apply(lambda x: x.sample(data.size().min()).reset_index(drop=True)))

    return data


def get_dataset_counts_info(X_train, X_test):
    print(f'Length of X_train: {len(X_train)}')
    print(X_train['class'].value_counts())
    print('--------------------')
    print(f'Length of X_test: {len(X_test)}')
    print(X_test['class'].value_counts())


def image_resize(image, width=None, height=None, inter=cv2.INTER_AREA):
    dim = None
    (h, w) = image.shape[:2]
    if width is None and height is None:
        return image
    if width is None:
        r = height / float(h)
        dim = (int(w * r), height)
    else:
        r = width / float(w)
        dim = (width, int(h * r))

    resized = cv2.resize(image, dim, interpolation=inter)
    return resized
