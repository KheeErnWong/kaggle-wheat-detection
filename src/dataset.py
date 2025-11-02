import ast
import random

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset


class wheatDataset(Dataset):
    def __init__(self, df, image_dir, image_size, mode="train"):
        super().__init__()
        self.df = df.copy()
        self.df["bbox"] = self.df["bbox"].apply(ast.literal_eval)
        self.image_dir = image_dir
        self.image_ids = df["image_id"].unique().tolist()
        self.image_size = image_size
        self.mode = mode
        if self.mode == "train":
            random.shuffle(self.image_ids)

    def __len__(self):
        return len(self.image_ids)

    # returns image and bbox tensors for model
    def __getitem__(self, index):
        image_id = self.image_ids[index]

        # Get image and boxes in numpy array format
        image, boxes = self.load_image_and_bbox(image_id)

        # Normalize image and convert to torch tensor, then  convert to C,H,W dimensions
        image = image.astype(np.float32)
        image /= 255.0
        image = torch.from_numpy(image).permute(2, 0, 1)

        # Get area of bbox (COCO: xmin, ymin, width, height)
        area = boxes[:, 2] * boxes[:, 3]

        # Convert bounding boxes from COCO to PASCAL VOC format
        # PASCAL VOC: xmin, ymin, xmax, ymax
        boxes[:, 2] = boxes[:, 0] + boxes[:, 2]
        boxes[:, 3] = boxes[:, 1] + boxes[:, 3]

        # Get labels for each bounding box (i.e class = 1 for wheat)
        labels = np.ones(boxes.shape[0], dtype=np.int64)

        # Set iscrowd
        iscrowd = np.zeros(boxes.shape[0], dtype=np.int64)

        target_dict = {
            "boxes": torch.as_tensor(boxes, dtype=torch.float32),
            "area": torch.as_tensor(area, dtype=torch.float32),
            "labels": torch.as_tensor(labels, dtype=torch.int64),
            "iscrowd": torch.as_tensor(iscrowd, dtype=torch.int64),
        }

        return image, target_dict

    def load_image_and_bbox(self, image_id):
        # get image and convert to tensor
        image_path = f"{self.image_dir}/{image_id}.jpg"
        image = Image.open(image_path)
        image = image.convert("RGB")
        image = np.array(image, dtype=np.uint8)

        # get bbox of image and convert to tensor
        tmp_df = self.df.loc[self.df["image_id"] == image_id]
        boxes = tmp_df["bbox"].tolist()
        boxes = np.array(boxes)

        return image, boxes
