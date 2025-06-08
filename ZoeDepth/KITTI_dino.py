import os
import torch
import numpy as np
from PIL import Image
import cv2
from torch.utils.data import Dataset, DataLoader
import glob
from torchvision.transforms import Compose
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, Tuple

# Importy dla różnych modeli
from util.transform import Resize, NormalizeImage, PrepareForNet


class BaseRawDataset(Dataset):
    """
    Bazowa klasa dla surowych danych - tylko wczytuje dane z dysku,
    bez żadnych transformacji specyficznych dla modeli.

    Ta klasa implementuje wzorzec Template Method - definiuje ogólną strukturę
    wczytywania danych, ale pozwala podklasom dostosować szczegóły.
    """

    def __init__(self, root_dir: str, split: str = 'val_selection_cropped'):
        self.root_dir = root_dir
        self.split = split
        self._validate_paths()
        self._collect_files()

    def _validate_paths(self):
        """Sprawdza czy wymagane ścieżki istnieją"""
        self.image_dir = os.path.join(self.root_dir, self.split, 'image')

        # Różne struktury dla różnych splitów
        if self.split == 'test_depth_completion_anonymous':
            self.depth_dir = os.path.join(self.root_dir, self.split, 'velodyne_raw')
        else:
            self.depth_dir = os.path.join(self.root_dir, self.split, 'groundtruth_depth')

        if not os.path.exists(self.image_dir):
            raise ValueError(f"Folder {self.image_dir} nie istnieje")
        if not os.path.exists(self.depth_dir):
            raise ValueError(f"Folder {self.depth_dir} nie istnieje")

    def _collect_files(self):
        """Zbiera listę plików do przetworzenia"""
        self.image_files = sorted(glob.glob(os.path.join(self.image_dir, '*.png')))
        if not self.image_files:
            raise ValueError(f"Nie znaleziono obrazów w {self.image_dir}")

    def __len__(self):
        return len(self.image_files)

    def get_depth_filename(self, image_path: str) -> str:
        """Konwertuje nazwę pliku obrazu na odpowiadającą nazwę pliku depth"""
        image_name = os.path.basename(image_path)

        if self.split == 'test_depth_completion_anonymous':
            # Dla velodyne_raw nazwy plików są takie same
            return image_name
        else:
            # KITTI specyficzna konwersja nazw
            return image_name.replace('_image_', '_groundtruth_depth_', 1)

    def load_raw_data(self, idx: int) -> Dict[str, Any]:
        """Wczytuje surowe dane bez żadnych transformacji."""
        if torch.is_tensor(idx):
            idx = idx.tolist()

        # Wczytaj surowy obraz RGB
        image_path = self.image_files[idx]
        raw_image = cv2.imread(image_path)
        if raw_image is None:
            raise ValueError(f"Nie można wczytać obrazu: {image_path}")

        image_rgb = cv2.cvtColor(raw_image, cv2.COLOR_BGR2RGB)

        # Wczytaj dane depth
        depth_filename = self.get_depth_filename(image_path)
        depth_path = os.path.join(self.depth_dir, depth_filename)

        if not os.path.exists(depth_path):
            raise FileNotFoundError(f"Nie znaleziono pliku depth: {depth_path}")

        raw_depth = np.array(Image.open(depth_path), dtype=np.float32)

        # Opcjonalnie wczytaj velodyne_raw dla val_selection_cropped
        velodyne_data = None
        if self.split == 'val_selection_cropped':
            velodyne_dir = os.path.join(self.root_dir, self.split, 'velodyne_raw')
            if os.path.exists(velodyne_dir):
                # Dla val_selection_cropped velodyne ma inne nazwy plików
                image_name = os.path.basename(image_path)
                # Konwersja: image_02.png -> sync_velodyne_raw_0000000005_image_02.png
                velodyne_files = glob.glob(os.path.join(velodyne_dir, f'*{image_name}'))
                if velodyne_files:
                    velodyne_path = velodyne_files[0]
                    velodyne_data = np.array(Image.open(velodyne_path), dtype=np.float32)

        return {
            'image_rgb': image_rgb,
            'depth_raw': raw_depth,
            'velodyne_raw': velodyne_data,  # Może być None
            'image_path': image_path,
            'depth_path': depth_path,
            'original_shape': image_rgb.shape[:2],
            'is_sparse': self.split == 'test_depth_completion_anonymous'
        }


class ModelTransformConfig(ABC):
    """
    Abstrakcyjna klasa konfiguracji transformacji dla konkretnego modelu.

    Każdy model dziedziczy z tej klasy i implementuje swoje specyficzne wymagania.
    To jest wzorzec Strategy - enkapsuluje różne algorytmy transformacji.
    """

    @abstractmethod
    def get_input_size(self) -> int:
        """Zwraca preferowany rozmiar wejściowy dla modelu"""
        pass

    @abstractmethod
    def create_image_transform(self):
        """Tworzy pipeline transformacji dla obrazów RGB"""
        pass

    @abstractmethod
    def create_depth_transform(self):
        """Tworzy pipeline transformacji dla map głębi"""
        pass

    @abstractmethod
    def postprocess_depth(self, raw_depth: np.ndarray) -> np.ndarray:
        """Postprocesing surowej głębi (np. dzielenie przez 256 dla KITTI)"""
        pass


class RawTransformConfig(ModelTransformConfig):
    """
    Minimalna konfiguracja transformacji - tylko konwersja na tensory.

    Ta konfiguracja jest idealna do eksperymentów i gdy chcesz mieć pełną
    kontrolę nad preprocessing-em. Jedyne co robi to:
    1. Konwertuje obrazy RGB na tensory z normalizacją do [0,1]
    2. Konwertuje depth na tensory z podstawowym postprocessingiem KITTI
    3. Nie stosuje żadnego skalowania ani innych transformacji

    To jest jak otrzymanie surowych składników do gotowania - masz pełną
    swobodę w decydowaniu, co z nimi zrobić dalej.
    """

    def __init__(self, input_size: int = None):
        # W trybie raw, input_size jest ignorowany - zachowujemy oryginalne rozmiary
        self.input_size = input_size

    def get_input_size(self) -> int:
        # Zwracamy None bo nie narzucamy konkretnego rozmiaru
        return self.input_size if self.input_size is not None else "original"

    def create_image_transform(self):
        """
        Brak transformacji - zwracamy funkcję identity.
        Obraz pozostaje w oryginalnej rozdzielczości i formacie.
        """

        def identity_transform(sample):
            # Przekazujemy obraz bez zmian
            return sample

        return identity_transform

    def create_depth_transform(self):
        """
        Brak transformacji depth - zachowujemy oryginalną rozdzielczość.
        """

        def identity_transform(sample):
            return sample

        return identity_transform

    def postprocess_depth(self, raw_depth: np.ndarray) -> np.ndarray:
        """
        Podstawowy postprocessing KITTI - tylko dzielenie przez 256.
        Nie robimy żadnego dodatkowego skalowania czy filtrowania.
        """
        return raw_depth / 256.0


class ZoeDepthTransformConfig(ModelTransformConfig):
    """
    Konfiguracja transformacji specyficzna dla ZoeDepth.

    ZoeDepth jest elastyczny co do rozdzielczości wejściowej,
    ale preferuje rozmiary będące wielokrotnością 32.
    """

    def __init__(self, input_size: int = 518):
        self.input_size = input_size

    def get_input_size(self) -> int:
        return self.input_size

    def create_image_transform(self):
        """
        ZoeDepth używa standardowej normalizacji ImageNet
        i może obsługiwać różne rozdzielczości dzięki swojej architekturze.
        """
        return Compose([
            Resize(
                width=self.input_size,
                height=self.input_size,
                resize_target=False,
                keep_aspect_ratio=True,
                ensure_multiple_of=32,  # ZoeDepth preferuje wielokrotności 32
                resize_method='lower_bound',
                image_interpolation_method=cv2.INTER_CUBIC,
            ),
            NormalizeImage(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            PrepareForNet(),
        ])

    def create_depth_transform(self):
        """Transformacja głębi dla ZoeDepth - zachowuje proporcje i precyzję"""
        return Resize(
            width=self.input_size,
            height=self.input_size,
            resize_target=True,
            keep_aspect_ratio=True,
            ensure_multiple_of=32,
            resize_method='lower_bound',
            image_interpolation_method=cv2.INTER_NEAREST,  # Zachowaj dokładne wartości
        )

    def postprocess_depth(self, raw_depth: np.ndarray) -> np.ndarray:
        """KITTI przechowuje głębię jako wartości * 256"""
        return raw_depth / 256.0


class DepthAnythingTransformConfig(ModelTransformConfig):
    """
    Przykład konfiguracji dla innego modelu - Depth Anything.

    Depth Anything ma sztywne wymagania co do rozdzielczości
    i używa innej normalizacji.
    """

    def __init__(self, input_size: int = 518):
        self.input_size = input_size

    def get_input_size(self) -> int:
        return self.input_size

    def create_image_transform(self):
        """Depth Anything może wymagać innych transformacji"""
        return Compose([
            Resize(
                width=self.input_size,
                height=self.input_size,
                resize_target=False,
                keep_aspect_ratio=True,
                ensure_multiple_of=14,  # Depth Anything używa ViT z patch_size=14
                resize_method='lower_bound',
                image_interpolation_method=cv2.INTER_CUBIC,
            ),
            NormalizeImage(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            PrepareForNet(),
        ])

    def create_depth_transform(self):
        return Resize(
            width=self.input_size,
            height=self.input_size,
            resize_target=True,
            keep_aspect_ratio=True,
            ensure_multiple_of=14,
            resize_method='lower_bound',
            image_interpolation_method=cv2.INTER_NEAREST,
        )

    def postprocess_depth(self, raw_depth: np.ndarray) -> np.ndarray:
        return raw_depth / 256.0


class UniversalDepthDataset(BaseRawDataset):
    """
    Uniwersalny dataset, który może pracować z dowolnym modelem
    dzięki wzorcowi Strategy (ModelTransformConfig).

    Ta klasa łączy surowe dane z transformacjami specyficznymi dla modelu.
    Obsługuje również tryb 'raw' dla maksymalnej kontroli nad danymi.
    """

    def __init__(self, root_dir: str, transform_config: ModelTransformConfig,
                 split: str = 'val_selection_cropped'):
        super().__init__(root_dir, split)
        self.transform_config = transform_config

        # Sprawdź czy to jest tryb raw
        self.is_raw_mode = isinstance(transform_config, RawTransformConfig)

        if not self.is_raw_mode:
            # Inicjalizuj transformacje na podstawie konfiguracji (tylko dla nie-raw modeli)
            self.image_transform = transform_config.create_image_transform()
            self.depth_transform = transform_config.create_depth_transform()

        print(f"Zainicjalizowano dataset dla {transform_config.__class__.__name__}")
        if self.is_raw_mode:
            print("Tryb RAW: Dane będą zwracane w oryginalnej rozdzielczości bez transformacji")
        else:
            print(f"Rozmiar wejściowy: {transform_config.get_input_size()}")
        print(f"Liczba próbek: {len(self)}")

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Zwraca przetworzone dane gotowe dla konkretnego modelu lub surowe dane w trybie raw.

        W trybie raw proces jest uproszczony:
        1. Wczytaj surowe dane
        2. Zastosuj podstawowy postprocessing depth (KITTI /256)
        3. Konwertuj na tensory bez dodatkowych transformacji
        4. Zachowaj oryginalne rozdzielczości

        W trybie normalnym proces jest pełny jak wcześniej.
        """
        # Wczytaj surowe dane (ten krok jest zawsze taki sam)
        raw_data = self.load_raw_data(idx)

        # Postprocesing głębi specyficzny dla datasetu (zawsze robimy dzielenie przez 256 dla KITTI)
        depth_values = self.transform_config.postprocess_depth(raw_data['depth_raw'])

        if raw_data.get('velodyne_raw') is not None:
            velodyne_values = self.transform_config.postprocess_depth(raw_data['velodyne_raw'])
            velodyne_tensor = torch.from_numpy(velodyne_values).unsqueeze(0)
        else:
            # Zwróć pusty tensor zamiast None
            velodyne_tensor = torch.zeros(1, 1, 1)  # Placeholder


        if self.is_raw_mode:
            # Tryb RAW: minimalne przetwarzanie
            # Konwertuj obraz RGB na tensor [0,1] bez dodatkowych transformacji
            image_normalized = raw_data['image_rgb'].astype(np.float32) / 255.0

            # Konwersja na tensory PyTorch z zachowaniem oryginalnych wymiarów
            # Przestawiamy osie z (H, W, C) na (C, H, W) dla zgodności z PyTorch
            image_tensor = torch.from_numpy(image_normalized.transpose(2, 0, 1))  # [3, H, W]
            depth_tensor = torch.from_numpy(depth_values).unsqueeze(0)  # [1, H, W]

            # print(f"Raw mode - Image shape: {image_tensor.shape}, Depth shape: {depth_tensor.shape}")

        else:
            # Tryb normalny: pełne transformacje specyficzne dla modelu
            image_normalized = raw_data['image_rgb'].astype(np.float32) / 255.0

            # Zastosuj transformacje modelu
            image_transformed = self.image_transform({'image': image_normalized})['image']
            depth_transformed = self.depth_transform({'image': depth_values})['image']

            # Konwersja na tensory
            image_tensor = torch.from_numpy(image_transformed)  # [3, H, W]
            depth_tensor = torch.from_numpy(depth_transformed).unsqueeze(0)  # [1, H, W]

        return {
            'image': image_tensor,
            'ground_truth': depth_tensor,
            'sparse_depth': velodyne_tensor,
            'has_sparse': raw_data.get('velodyne_raw') is not None,  # Flag informujący o dostępności
            'metadata': {
                'image_path': raw_data['image_path'],
                'original_shape': raw_data['original_shape'],
                'is_raw_mode': self.is_raw_mode
            }
        }


def create_universal_dataloader(root_dir: str, model_name: str,
                                split: str = 'val_selection_cropped',
                                batch_size: int = 4, input_size: int = 518,
                                shuffle: bool = False, num_workers: int = 2) -> DataLoader:
    """
    Factory function tworząca DataLoader dla konkretnego modelu.

    To jest wzorzec Factory - enkapsuluje logikę tworzenia obiektów
    i udostępnia prosty interfejs użytkownikowi.

    NOWA FUNKCJONALNOŚĆ: Dodano tryb 'raw' dla surowych danych.

    Args:
        root_dir: Ścieżka do danych
        model_name: Nazwa modelu ('zoedepth', 'depth_anything', 'raw', etc.)
        split: Podzbiór danych
        batch_size: Rozmiar batcha
        input_size: Rozmiar wejściowy (ignorowany w trybie 'raw')
        shuffle: Czy mieszać dane
        num_workers: Liczba procesów do wczytywania

    Returns:
        DataLoader skonfigurowany dla konkretnego modelu

    Nowe użycie:
        # Dla surowych danych bez transformacji
        raw_loader = create_universal_dataloader(
            root_dir='/path/to/data',
            model_name='raw'
        )
    """

    # Mapowanie nazw modeli na konfiguracje transformacji
    transform_configs = {
        'raw': RawTransformConfig(input_size),  # Nowa opcja!
        'zoedepth': ZoeDepthTransformConfig(input_size),
        'depth_anything': DepthAnythingTransformConfig(input_size),
        # Łatwo dodać nowe modele tutaj
    }

    if model_name not in transform_configs:
        available_models = list(transform_configs.keys())
        raise ValueError(f"Nieznany model: {model_name}. "
                         f"Dostępne modele: {available_models}")

    # Utwórz dataset z odpowiednią konfiguracją
    dataset = UniversalDepthDataset(
        root_dir=root_dir,
        transform_config=transform_configs[model_name],
        split=split
    )

    return DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),  # Automatyczna optymalizacja dla GPU
        drop_last=False
    )


# Przykład użycia dla różnych modeli, włączając nowy tryb raw
if __name__ == "__main__":
    # DataLoader dla surowych danych - NOWA FUNKCJONALNOŚĆ
    print("=== Testowanie trybu RAW ===")
    raw_loader = create_universal_dataloader(
        root_dir='/path/to/kitti',
        model_name='raw',  # Nowy tryb!
        batch_size=2,
        input_size=518,  # Ten parametr jest ignorowany w trybie raw
        shuffle=True
    )

    print("Testing Raw DataLoader:")
    for batch in raw_loader:
        images = batch['image']
        depths = batch['ground_truth']
        metadata = batch['metadata']
        print(f"Images shape: {images.shape}")
        print(f"Depths shape: {depths.shape}")
        print(f"Image range: [{images.min():.3f}, {images.max():.3f}]")
        print(f"Depth range: [{depths.min():.3f}, {depths.max():.3f}]")
        print(f"Is raw mode: {metadata['is_raw_mode'][0]}")
        print(f"Original shape: {metadata['original_shape'][0]}")
        break

    print("\n=== Porównanie z ZoeDepth ===")
    # DataLoader dla ZoeDepth dla porównania
    zoedepth_loader = create_universal_dataloader(
        root_dir='/path/to/kitti',
        model_name='zoedepth',
        batch_size=2,
        input_size=518,
        shuffle=True
    )

    print("Testing ZoeDepth DataLoader:")
    for batch in zoedepth_loader:
        images = batch['image']
        depths = batch['ground_truth']
        metadata = batch['metadata']
        print(f"Images shape: {images.shape}")
        print(f"Depths shape: {depths.shape}")
        print(f"Image range: [{images.min():.3f}, {images.max():.3f}]")
        print(f"Depth range: [{depths.min():.3f}, {depths.max():.3f}]")
        print(f"Is raw mode: {metadata['is_raw_mode'][0]}")
        break

    print("\n=== Przykład eksperymentalnego użycia trybu raw ===")
    # Pokaż jak można użyć trybu raw do własnych eksperymentów
    for batch in raw_loader:
        raw_images = batch['image']  # [B, 3, H_orig, W_orig]
        raw_depths = batch['ground_truth']  # [B, 1, H_orig, W_orig]

        print(f"Otrzymane surowe dane:")
        print(f"  - Obraz: {raw_images.shape}, zakres: [{raw_images.min():.3f}, {raw_images.max():.3f}]")
        print(f"  - Głębia: {raw_depths.shape}, zakres: [{raw_depths.min():.3f}, {raw_depths.max():.3f}]")

        # Teraz możesz zastosować własne transformacje
        # Przykład: resize do konkretnego rozmiaru
        import torch.nn.functional as F

        custom_size = (384, 512)

        resized_images = F.interpolate(raw_images, size=custom_size, mode='bilinear', align_corners=False)
        resized_depths = F.interpolate(raw_depths, size=custom_size, mode='bilinear', align_corners=False)

        print(f"Po własnym resizing:")
        print(f"  - Obraz: {resized_images.shape}")
        print(f"  - Głębia: {resized_depths.shape}")

        # Lub zastosuj własną normalizację
        custom_normalized = (raw_images - raw_images.mean()) / raw_images.std()
        print(f"Po własnej normalizacji: zakres [{custom_normalized.min():.3f}, {custom_normalized.max():.3f}]")

        break