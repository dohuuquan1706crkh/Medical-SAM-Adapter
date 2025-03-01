from typing import Optional, Callable, Dict
from pathlib import Path
import numpy as np
import cv2
import torch
from torch.utils.data import Dataset
from .utils import *

__all__ = [
    "FLARE22"
]


class FLARE22(Dataset):
    """
    FLARE Dataset ver. 2016 for Organ segmentation (3D).
    Link:
    """
    def __init__(
        self,
        path: str,
        prompt: str = "click",
        image_size: int = 1024,
        num_classes: int = 13,
        transform: Optional[Callable] = None,
        transform_mask: Optional[Callable] = None
    ) -> None:
        """
        Args:
            path(str): data path.
            prompt (str): prompt types, including
                "none": no applying prompt.
                "click": applying click prompt.
                "box": applying bbox prompt.
            image_size (int): input image size.
            num_classes (int): number of classes.
            transform (Callable): transform functions for image.
            transform_mask: transform function for mask.
        """
        super().__init__()
        self.path_data = path
        self.names = [f.name for f in sorted(Path(path, "images").iterdir()) if f.is_file()]
        self.prompt = prompt
        self.image_size = image_size
        self.num_classes = num_classes
        self.transform = transform
        self.transform_mask = transform_mask

    def __len__(self) -> int:
        return len(self.names)

    def __getitem__(self, idx: int) -> Dict:
        """
        Args:
            idx (int): input index.
        Returns:
            (Dict): image, ground truth (mask), prompt data and related metadata.
        """
        # read image and label (mask)
        image = cv2.imread(f"{self.path_data}/images_liver/{self.names[idx]}", cv2.IMREAD_COLOR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        label = cv2.imread(f"{self.path_data}/labels_l/{self.names[idx]}", cv2.IMREAD_GRAYSCALE)
        # resize the label's resolution as same as image's
        label = cv2.resize(label, (self.image_size, self.image_size))
        masks = []
        unique = np.unique(label)
        # if len(unique) == 2:
        mask = np.expand_dims(label/255,0)
        # else:
        #     masks = []
        #     # print(unique)
        #     for i in range(1, self.num_classes + 1):
        #         mask = np.zeros((self.image_size, self.image_size), dtype=np.uint8)
        #         if i in unique:
        #             mask[label == i] = 255
        #         masks.append(mask)
        #     mask = np.stack(masks, axis=-1)
        # print(mask.shape)
        # get click points
        point_label, point_coord = 1, np.array([0, 0], np.int32)
        if self.prompt == "click":
            point_label, point_coord = random_click(mask[..., 0] / 255., point_labels=1)
        # transform the input
        # if self.transform:
        #     # save the current random number generate for reproducibility
        #     state = torch.get_rng_state()
        #     image = self.transform(image)
        #     torch.set_rng_state(state)
        # if self.transform_mask:
        #     state = torch.get_rng_state()
        #     mask = self.transform_mask(mask).int()
        #     torch.set_rng_state(state)
        # breakpoint()
        # print(type(image))
        # breakpoint()
        # print(image.shape, mask.shape)
        image = torch.from_numpy(image/255).float().permute(2, 0, 1)
        mask = torch.from_numpy(mask).float()
        if mask.shape[0] != 1:
            mask = mask.permute(2,0,1)
        image = torch.flip(image,dims= [1,2])
        mask = torch.flip(mask,dims= [1,2])
        
        # print(image.shape, mask.shape)
        # image = image.permute(0,3,1,2)
        # mask = mask.permute(0,3,1,2)
        name = self.names[idx][:-4]
        image_meta_dict = {'filename_or_obj':name}
        return {
            "image": image,
            "label": mask,
            "p_label": point_label,
            "pt": point_coord,
            "image_meta_dict": image_meta_dict
        }
