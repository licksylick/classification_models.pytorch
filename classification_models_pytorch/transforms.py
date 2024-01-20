from torchvision import transforms


class Transforms:
    def __init__(self, image_size):
        self.image_size = image_size

    def train_transform(self):
        train_transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((self.image_size, self.image_size)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])
        return train_transform

    def val_test_transform(self):
        val_test_transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((self.image_size, self.image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])
        return val_test_transform
