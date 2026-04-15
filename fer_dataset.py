from pathlib import Path
from torch.utils.data import Dataset
from PIL import Image


class FERDataset(Dataset):
    """
    Wrap a torch.utils.data.Dataset around the raw FER2013 data.
    Allows indexing to access an image, label pair. Intended to be used with a DataLoader.
    """
    def __init__(self, fer_path: str, split: str, transforms=None) -> None:
        """
        Args:
            fer_path (str): Path up to and including the fer2013 dataset.
                Contained in the path directory should be "train" and "test"
            split (str): Which split of the data. Either "train" or "test"
            transforms: PyTorch transforms to apply to the images.
        """
        split = split.lower()
        if split not in ["train", "test"]:
            raise ValueError('split argument must be one of "train" or "test"')
            
        dataset_path = Path(fer_path)
        if not dataset_path.exists():
            raise ValueError(f'The path {fer_path} does not exist')
            
        self.transforms = transforms
        split_path = dataset_path / split
        
        self.images = []
        self.labels = []
        
        self.classes = sorted([d.name for d in split_path.iterdir() if d.is_dir()])
        
        self.class_to_idx = {cls_name: i for i, cls_name in enumerate(self.classes)}
        
        for emotion in self.classes:
            emotion_dir = split_path / emotion
            images_for_emotion = list(emotion_dir.glob('*.jpg'))
            
            label_idx = self.class_to_idx[emotion]
            
            self.images.extend(images_for_emotion)
            self.labels.extend([label_idx] * len(images_for_emotion))
            
        assert len(self.images) == len(self.labels), f"There are {len(self.images)} images but {len(self.labels)} labels"


    def __getitem__(self, index):
        img_path = self.images[index]
        label = self.labels[index]
        
        image = Image.open(img_path).convert('L')
        
        if self.transforms is not None:
            image = self.transforms(image)
            
        return image, label


    def __len__(self) -> int:
        return len(self.images)
