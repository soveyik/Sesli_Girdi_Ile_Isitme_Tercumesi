import json
import os
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

class SignLanguageDataset(Dataset):
    def _init_(self, json_path, image_folder, image_size=(64, 64)):
        with open(json_path, "r", encoding="utf-8") as f:
            self.data = json.load(f)

        self.image_paths = []
        self.labels = []
        self.label_map = {word: idx for idx, word in enumerate(self.data.keys())}

        for word, frames in self.data.items():
            for frame in frames:
                img_path = os.path.normpath(os.path.join(image_folder, os.path.basename(frame["image_path"])))
                if os.path.exists(img_path):
                    img = cv2.imread(img_path)
                    if img is not None:
                        img = cv2.resize(img, image_size)
                        img = img / 255.0
                        self.image_paths.append(img)
                        self.labels.append(self.label_map[word])

        self.image_paths = np.array(self.image_paths, dtype=np.float32)
        self.labels = np.array(self.labels, dtype=np.int64)

    def _len_(self):
        return len(self.image_paths)

    def _getitem_(self, idx):
        img = self.image_paths[idx]
        label = self.labels[idx]
        img = np.transpose(img, (2, 0, 1))
        return torch.tensor(img, dtype=torch.float32), torch.tensor(label, dtype=torch.long)