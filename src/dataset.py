import random

from PIL import Image
from torch.utils.data import Dataset


class wheatDataset(Dataset):
    def __init__(self, df, image_dir, image_size, mode="train"):
        super().__init__()
        self.df = df
        self.image_dir = image_dir
        self.image_ids = df["image_id"].unique().to_list()
        self.image_size = image_size
        self.mode = mode
        if self.mode == "train":
            random.shuffle(self.image_ids)

    def __len__(self):
        return len(self.image_ids)

    # Use for Pytorch DataLoader which passes indices to retrieve image samples
    def __getitem__(self, index):
        image_id = self.image_ids[index]
        image_id_path = f"{self.image_dir}{image_id}.jpg"
        self.load_image_and_bbox(image_id_path)

    def load_image_and_bbox(self, image_id_path):
        tmp_bbox_df = self.df.loc[self.df.loc[self.df["image_path"] == image_id_path]]

        with Image.open(image_id_path) as image:
            return image
