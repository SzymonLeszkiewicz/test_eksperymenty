import os
import torch
import numpy as np
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import glob


def scale_sparse_to_dense(sparse_depth, dense_shape):
    """
    Skaluje rzadką mapę głębi do wymiarów gęstej mapy głębi bez interpolacji.
    Tylko oryginalne punkty głębi zostają przeniesione, reszta to zera.

    Args:
        sparse_depth (np.ndarray): Rzadka mapa głębi (H_sparse, W_sparse).
                                  Zakłada się, że zera oznaczają brak danych.
        dense_shape (tuple): Kształt gęstej mapy głębi (H_dense, W_dense).

    Returns:
        np.ndarray: Nowa mapa głębi o wymiarach dense_shape, zawierająca
                    przeskalowane punkty z sparse_depth i zera w pozostałych miejscach.
    """
    h_sparse, w_sparse = sparse_depth.shape
    h_dense, w_dense = dense_shape

    scaled_sparse = np.zeros(dense_shape, dtype=sparse_depth.dtype)
    scale_y = h_dense / h_sparse
    scale_x = w_dense / w_sparse

    sparse_rows, sparse_cols = np.nonzero(sparse_depth)

    for r_sparse, c_sparse in zip(sparse_rows, sparse_cols):
        depth_value = sparse_depth[r_sparse, c_sparse]
        r_dense = int(r_sparse * scale_y)
        c_dense = int(c_sparse * scale_x)

        if 0 <= r_dense < h_dense and 0 <= c_dense < w_dense:
            scaled_sparse[r_dense, c_dense] = depth_value

    return scaled_sparse


class ARKitScenesDepthDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        """
        Args:
            root_dir (str): Katalog główny datasetu (np. /mnt/datasets/airkitdata/upsampling/Validation).
            transform (callable, optional): Opcjonalne transformacje do zastosowania na obrazach.
        """
        self.root_dir = root_dir
        self.transform = transform
        self.scenes = sorted([d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))])

        self.data_paths = []
        for scene_id in self.scenes:
            scene_path = os.path.join(root_dir, scene_id)
            rgb_files = sorted(glob.glob(os.path.join(scene_path, 'wide', '*.png')))
            lowres_depth_files = sorted(glob.glob(os.path.join(scene_path, 'lowres_depth', '*.png')))
            highres_depth_files = sorted(glob.glob(os.path.join(scene_path, 'highres_depth', '*.png')))

            if len(rgb_files) == len(lowres_depth_files) and len(rgb_files) == len(highres_depth_files):
                for i in range(len(rgb_files)):
                    self.data_paths.append({
                        'image': rgb_files[i],
                        'sparse_depth': lowres_depth_files[i],
                        'ground_truth': highres_depth_files[i]
                    })
            else:
                print(f"Ostrzeżenie: Niezgodność liczby plików w scenie {scene_id}. Pomijam.")

    def __len__(self):
        return len(self.data_paths)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        data_sample = self.data_paths[idx]

        rgb_image = Image.open(data_sample['image']).convert('RGB')
        lowres_depth = np.array(Image.open(data_sample['sparse_depth']), dtype=np.float32) / 1000.0
        highres_depth = np.array(Image.open(data_sample['ground_truth']), dtype=np.float32) / 1000.0

        h_dense, w_dense = highres_depth.shape
        scaled_lowres_depth = scale_sparse_to_dense(lowres_depth, (h_dense, w_dense))

        if self.transform:
            rgb_image = self.transform(rgb_image)
        else:
            rgb_image = transforms.ToTensor()(rgb_image)
            rgb_image = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])(rgb_image)

        highres_depth_tensor = torch.from_numpy(highres_depth).unsqueeze(0)
        scaled_lowres_depth_tensor = torch.from_numpy(scaled_lowres_depth).unsqueeze(0)

        sample = {
            'image': rgb_image,
            'sparse_depth': scaled_lowres_depth_tensor,
            'ground_truth': highres_depth_tensor
        }

        return sample


def create_dataloader(root_dir, batch_size=1, shuffle=True, num_workers=4, image_transforms=None):
    """
    Tworzy PyTorch DataLoader dla datasetu ARKitScenes.

    Args:
        root_dir (str): Katalog główny datasetu (np. /mnt/datasets/airkitdata/upsampling/Validation).
        batch_size (int, optional): Rozmiar partii danych. Domyślnie 1.
        shuffle (bool, optional): Czy tasować dane. Domyślnie True.
        num_workers (int, optional): Liczba procesów roboczych do ładowania danych. Domyślnie 4.
        image_transforms (callable, optional): Opcjonalne transformacje torchvision dla obrazów RGB.
                                              Jeśli None, zostaną użyte domyślne (ToTensor + Normalize).

    Returns:
        torch.utils.data.DataLoader: Skonfigurowany DataLoader.
    """
    if image_transforms is None:
        image_transforms = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    arkit_dataset = ARKitScenesDepthDataset(root_dir=root_dir, transform=image_transforms)
    dataloader = DataLoader(
        arkit_dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True  # Użyj pin_memory dla szybszego transferu do GPU
    )
    print(f"DataLoader utworzony dla {len(arkit_dataset)} próbek.")
    return dataloader
