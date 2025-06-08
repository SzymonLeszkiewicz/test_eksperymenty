import os
import torch
import numpy as np
from PIL import Image
import cv2
from torch.utils.data import Dataset, DataLoader
import glob
from typing import Dict, Any


class KittiDepthDataset(Dataset):
    """
    Ulepszony dataset dla KITTI, który poprawnie dopasowuje nazwy plików
    dla obrazu, rzadkiej głębi (velodyne) i prawdy objawowej (ground truth).
    """

    def __init__(self, root_dir: str, split: str = 'val_selection_cropped'):
        self.root_dir = root_dir
        self.split = split

        # Ścieżka do katalogu z obrazami
        self.image_dir = os.path.join(root_dir, split, 'image')
        # ZMIANA: Bardziej elastyczne wyszukiwanie plików (jpg i png)
        self.image_files = sorted(glob.glob(os.path.join(self.image_dir, '*.png')))
        self.image_files.extend(sorted(glob.glob(os.path.join(self.image_dir, '*.jpg'))))

        if not self.image_files:
            raise FileNotFoundError(f"Nie znaleziono żadnych obrazów w katalogu: {self.image_dir}")

        print(f"Znaleziono {len(self.image_files)} obrazów w splicie '{split}'.")

    def __len__(self) -> int:
        return len(self.image_files)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        image_path = self.image_files[idx]
        image_name = os.path.basename(image_path)

        # --- 1. Wczytaj obraz RGB ---
        image = cv2.imread(image_path)
        if image is None:
            raise IOError(f"Nie można wczytać obrazu: {image_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = image.astype(np.float32) / 255.0
        image_tensor = torch.from_numpy(image.transpose(2, 0, 1))  # [3, H, W]

        # --- 2. Wczytaj rzadką mapę głębi (Velodyne) ---
        # ZMIANA: Bezpośrednie tworzenie ścieżki zamiast używania glob
        velodyne_name = image_name.replace('_image_', '_velodyne_raw_', 1)
        velodyne_path = os.path.join(self.root_dir, self.split, 'velodyne_raw', velodyne_name)

        if os.path.exists(velodyne_path):
            sparse_depth = np.array(Image.open(velodyne_path), dtype=np.float32) / 256.0
            sparse_tensor = torch.from_numpy(sparse_depth).unsqueeze(0)  # [1, H, W]
        else:
            # Pusty tensor, jeśli plik nie istnieje
            sparse_tensor = torch.zeros((1, image.shape[0], image.shape[1]), dtype=torch.float32)

        # --- 3. Wczytaj prawdę objawową (Ground Truth) - tylko dla walidacji ---
        gt_tensor = torch.empty(0)  # Pusty tensor domyślnie
        if 'val' in self.split:
            # ZMIANA: Bezpośrednie tworzenie ścieżki zamiast używania glob
            gt_name = image_name.replace('_image_', '_groundtruth_depth_', 1)
            gt_path = os.path.join(self.root_dir, self.split, 'groundtruth_depth', gt_name)
            if os.path.exists(gt_path):
                gt_depth = np.array(Image.open(gt_path), dtype=np.float32) / 256.0
                gt_tensor = torch.from_numpy(gt_depth).unsqueeze(0)  # [1, H, W]
            else:
                # ZMIANA: Lepsza obsługa braku pliku - tworzymy pusty tensor
                print(f"Warning: Nie znaleziono ground truth dla {image_name} w {os.path.dirname(gt_path)}")
                gt_tensor = torch.zeros((1, image.shape[0], image.shape[1]), dtype=torch.float32)

        return {
            'image': image_tensor,
            'sparse_depth': sparse_tensor,
            'ground_truth': gt_tensor,
            'image_path': image_path
        }


def create_dataloader(root_dir: str, split: str = 'val_selection_cropped',
                      batch_size: int = 1, shuffle: bool = False,
                      num_workers: int = 2) -> DataLoader:
    """Tworzy DataLoader dla KITTI"""
    dataset = KittiDepthDataset(root_dir, split)
    return DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=False
    )


# --- Przykład użycia (bez zmian) ---
if __name__ == "__main__":
    # Załóżmy, że dane są w katalogu ../data/kitti_depth
    # Utworzymy tymczasową strukturę katalogów do testów
    if not os.path.exists("../data/kitti_depth/val_selection_cropped/image"):
        print("Tworzenie tymczasowej struktury katalogów i plików do testów...")
        os.makedirs("../data/kitti_depth/val_selection_cropped/image", exist_ok=True)
        os.makedirs("../data/kitti_depth/val_selection_cropped/velodyne_raw", exist_ok=True)
        os.makedirs("../data/kitti_depth/val_selection_cropped/groundtruth_depth", exist_ok=True)

        # Tworzenie przykładowych pustych plików
        img = Image.new('RGB', (1216, 352))
        depth = Image.new('L', (1216, 352))  # 'L' for 8-bit grayscale

        img.save(
            "../data/kitti_depth/val_selection_cropped/image/2011_09_26_drive_0002_sync_image_0000000005_image_02.png")
        depth.save(
            "../data/kitti_depth/val_selection_cropped/velodyne_raw/2011_09_26_drive_0002_sync_velodyne_raw_0000000005_image_02.png")
        depth.save(
            "../data/kitti_depth/val_selection_cropped/groundtruth_depth/2011_09_26_drive_0002_sync_groundtruth_depth_0000000005_image_02.png")

    try:
        val_loader = create_dataloader(
            root_dir="../data/kitti_depth",
            split='val_selection_cropped',
            batch_size=1
        )

        print("\n--- Test dla 'val_selection_cropped' ---")
        for i, batch in enumerate(val_loader):
            print(f"Batch {i + 1}:")
            print(f"  Image shape: {batch['image'].shape}")
            print(f"  Sparse depth shape: {batch['sparse_depth'].shape}")
            print(f"  Ground truth shape: {batch['ground_truth'].shape}")
            print(f"  Image path: {batch['image_path'][0]}")
            # Przerwij po pierwszym batchu dla zwięzłości
            break

    except FileNotFoundError as e:
        print(e)
    except Exception as e:
        print(f"Wystąpił nieoczekiwany błąd: {e}")