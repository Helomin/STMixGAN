import os
import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.utils.data import Dataset


class RainDataset(Dataset):
    def __init__(self, radar_map: np.memmap = None):
        assert radar_map is not None, "radar_map should not be None"
        self.radar_map = radar_map

    def __len__(self):
        return self.radar_map.shape[0]

    def __getitem__(self, index):
        batch_imgs = self.radar_map[index]
        return torch.as_tensor(batch_imgs)


def get_memmap(file_path):
    radar_map = np.load(file_path)
    radar_map = np.expand_dims(radar_map, axis=2)
    os.system(f"echo Total length of radar-map : {radar_map.shape[0]}")
    return radar_map


def make_dataset(file_path):
    radar_map = get_memmap(file_path)
    return RainDataset(radar_map=radar_map)


def create_rainloader(hkl_path, batch_size, num_workers, shuffle=False):
    rain_dataset = make_dataset(hkl_path)
    return DataLoader(
        rain_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=shuffle
    )
