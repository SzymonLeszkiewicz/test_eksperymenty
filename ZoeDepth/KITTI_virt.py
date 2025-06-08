import os
import torch
import numpy as np
from PIL import Image
import cv2
from torch.utils.data import Dataset, DataLoader
import glob
from torchvision.transforms import Compose
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, Tuple, List
from pathlib import Path


# Importy dla różnych modeli (zakładając, że masz dostęp do util.transform)
# from util.transform import Resize, NormalizeImage, PrepareForNet


class VirtualKITTIBaseDataset(Dataset):
    """
    Bazowa klasa dla Virtual KITTI dataset - obsługuje specyficzną strukturę tego datasetu.

    Virtual KITTI ma unikalną strukturę z "światami" (worlds) i "wariantami" (variations),
    co wymaga specjalnego podejścia do organizacji danych. Ta klasa implementuje wzorzec
    Template Method, definiując ogólną strukturę pracy z Virtual KITTI.
    """

    # Mapowanie nazw światów Virtual KITTI na odpowiednie sekwencje KITTI
    WORLD_MAPPING = {
        '0001': 'clone_of_kitti_sequence_0001',
        '0002': 'clone_of_kitti_sequence_0002',
        '0006': 'clone_of_kitti_sequence_0006',
        '0018': 'clone_of_kitti_sequence_0018',
        '0020': 'clone_of_kitti_sequence_0020'
    }

    # Dostępne warianty renderowania w Virtual KITTI
    AVAILABLE_VARIATIONS = [
        'clone',  # Renderowanie maksymalnie zbliżone do oryginalnego KITTI
        '15-deg-right',  # Rotacja kamery 15 stopni w prawo
        '15-deg-left',  # Rotacja kamery 15 stopni w lewo
        '30-deg-right',  # Rotacja kamery 30 stopni w prawo
        '30-deg-left',  # Rotacja kamery 30 stopni w lewo
        'morning',  # Typowe oświetlenie poranne w słoneczny dzień
        'sunset',  # Oświetlenie tuż przed zachodem słońca
        'overcast',  # Typowa pochmurna pogoda z rozproszonym światłem
        'fog',  # Efekt mgły implementowany poprzez wzór wolumetryczny
        'rain'  # Prosty efekt deszczu (bez refrakcji kropel na kamerze)
    ]

    def __init__(self, root_dir: str, worlds: Optional[List[str]] = None,
                 variations: Optional[List[str]] = None, max_frames: Optional[int] = None):
        """
        Inicjalizuje dataset Virtual KITTI.

        Args:
            root_dir: Ścieżka do głównego katalogu z danymi Virtual KITTI
            worlds: Lista światów do załadowania (domyślnie wszystkie)
            variations: Lista wariantów do załadowania (domyślnie tylko 'clone')
            max_frames: Maksymalna liczba klatek na świat-wariant (dla debugowania)
        """
        self.root_dir = Path(root_dir)
        self.max_frames = max_frames

        # Jeśli nie podano światów, użyj wszystkich dostępnych
        self.worlds = worlds if worlds else list(self.WORLD_MAPPING.keys())

        # Jeśli nie podano wariantów, użyj tylko 'clone' (najbliższy oryginalnemu KITTI)
        self.variations = variations if variations else ['clone']

        # Sprawdź poprawność podanych parametrów
        self._validate_parameters()

        # Sprawdź czy katalogi istnieją
        self._validate_paths()

        # Zbierz wszystkie dostępne próbki
        self._collect_samples()

        print(f"Załadowano Virtual KITTI dataset:")
        print(f"  - Światy: {self.worlds}")
        print(f"  - Warianty: {self.variations}")
        print(f"  - Łączna liczba próbek: {len(self.samples)}")

    def _validate_parameters(self):
        """Sprawdza czy podane parametry są poprawne"""
        # Sprawdź czy wszystkie światy istnieją
        invalid_worlds = [w for w in self.worlds if w not in self.WORLD_MAPPING]
        if invalid_worlds:
            raise ValueError(f"Nieznane światy: {invalid_worlds}. "
                             f"Dostępne: {list(self.WORLD_MAPPING.keys())}")

        # Sprawdź czy wszystkie warianty są poprawne
        invalid_variations = [v for v in self.variations if v not in self.AVAILABLE_VARIATIONS]
        if invalid_variations:
            raise ValueError(f"Nieznane warianty: {invalid_variations}. "
                             f"Dostępne: {self.AVAILABLE_VARIATIONS}")

    def _validate_paths(self):
        """Sprawdza czy wymagane ścieżki istnieją"""
        self.rgb_dir = self.root_dir / 'vkitti_1.3.1_rgb'
        self.depth_dir = self.root_dir / 'vkitti_1.3.1_depthgt'

        if not self.rgb_dir.exists():
            raise ValueError(f"Katalog RGB nie istnieje: {self.rgb_dir}")
        if not self.depth_dir.exists():
            raise ValueError(f"Katalog depth nie istnieje: {self.depth_dir}")

        # Sprawdź czy wszystkie wymagane światy istnieją
        for world in self.worlds:
            world_rgb = self.rgb_dir / world
            world_depth = self.depth_dir / world

            if not world_rgb.exists():
                raise ValueError(f"Katalog RGB dla świata {world} nie istnieje: {world_rgb}")
            if not world_depth.exists():
                raise ValueError(f"Katalog depth dla świata {world} nie istnieje: {world_depth}")

    def _collect_samples(self):
        """
        Zbiera wszystkie dostępne próbki z określonych światów i wariantów.

        Tworzy listę słowników, gdzie każdy słownik zawiera informacje o jednej próbce:
        ścieżki do plików RGB i depth, metadane o świecie, wariancie i numerze klatki.
        """
        self.samples = []

        for world in self.worlds:
            for variation in self.variations:
                # Zbuduj ścieżki do katalogu dla tego świata i wariantu
                world_variation_rgb = self.rgb_dir / world / variation
                world_variation_depth = self.depth_dir / world / variation

                # Sprawdź czy ten wariant istnieje dla tego świata
                if not world_variation_rgb.exists():
                    print(f"Ostrzeżenie: Brak wariantu {variation} dla świata {world} (RGB)")
                    continue
                if not world_variation_depth.exists():
                    print(f"Ostrzeżenie: Brak wariantu {variation} dla świata {world} (depth)")
                    continue

                # Zbierz wszystkie pliki RGB dla tego świata-wariantu
                rgb_pattern = str(world_variation_rgb / "*.png")
                rgb_files = sorted(glob.glob(rgb_pattern))

                # Ogranicz liczbę klatek jeśli określono
                if self.max_frames:
                    rgb_files = rgb_files[:self.max_frames]

                for rgb_file in rgb_files:
                    # Wyciągnij numer klatki z nazwy pliku
                    rgb_filename = Path(rgb_file).name
                    frame_number = rgb_filename.replace('.png', '')

                    # Zbuduj odpowiadającą ścieżkę do pliku depth
                    depth_file = world_variation_depth / f"{frame_number}.png"

                    # Sprawdź czy odpowiadający plik depth istnieje
                    if depth_file.exists():
                        self.samples.append({
                            'rgb_path': Path(rgb_file),
                            'depth_path': depth_file,
                            'world': world,
                            'variation': variation,
                            'frame_number': frame_number,
                            'sample_id': f"{world}_{variation}_{frame_number}"
                        })
                    else:
                        print(f"Ostrzeżenie: Brak pliku depth dla {rgb_file}")

    def __len__(self):
        return len(self.samples)

    def load_raw_data(self, idx: int) -> Dict[str, Any]:
        """
        Wczytuje surowe dane dla określonej próbki.

        Virtual KITTI ma specyficzny format przechowywania głębi - wartości są bezpośrednio
        w centymetrach, więc nie wymagają dzielenia przez 256 jak w oryginalnym KITTI.
        """
        if torch.is_tensor(idx):
            idx = idx.tolist()

        sample = self.samples[idx]

        # Wczytaj obraz RGB
        rgb_image = cv2.imread(str(sample['rgb_path']))
        if rgb_image is None:
            raise ValueError(f"Nie można wczytać obrazu RGB: {sample['rgb_path']}")

        # Konwersja BGR -> RGB
        image_rgb = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2RGB)

        # Wczytaj głębię (Virtual KITTI używa 16-bit PNG)
        # Zgodnie z dokumentacją: 1 jednostka intensywności = 1cm odległości
        depth_raw = cv2.imread(str(sample['depth_path']),
                               cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)

        if depth_raw is None:
            raise ValueError(f"Nie można wczytać obrazu głębi: {sample['depth_path']}")

        # Virtual KITTI przechowuje głębię już w centymetrach
        # Konwertujemy na float32 dla dalszego przetwarzania
        depth_raw = depth_raw.astype(np.float32)

        return {
            'image_rgb': image_rgb,  # uint8, shape (H, W, 3)
            'depth_raw': depth_raw,  # float32, shape (H, W), wartości w centymetrach
            'rgb_path': str(sample['rgb_path']),
            'depth_path': str(sample['depth_path']),
            'world': sample['world'],
            'variation': sample['variation'],
            'frame_number': sample['frame_number'],
            'sample_id': sample['sample_id'],
            'original_shape': image_rgb.shape[:2]  # (H, W)
        }


class VirtualKITTITransformConfig(ABC):
    """
    Abstrakcyjna klasa konfiguracji transformacji dla Virtual KITTI.

    Podobnie jak w oryginalnym systemie, enkapsuluje różne strategie
    przetwarzania danych, ale dostosowane do specyfiki Virtual KITTI.
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
        """Postprocessing głębi specyficzny dla Virtual KITTI"""
        pass


class VirtualKITTIRawConfig(VirtualKITTITransformConfig):
    """
    Minimalna konfiguracja dla Virtual KITTI - tylko konwersja na tensory.

    Idealna do eksperymentów gdzie chcesz mieć pełną kontrolę nad preprocessing-iem.
    Zachowuje oryginalne rozdzielczości i formatowanie danych Virtual KITTI.
    """

    def __init__(self, input_size: int = None):
        self.input_size = input_size

    def get_input_size(self) -> int:
        return self.input_size if self.input_size is not None else "original"

    def create_image_transform(self):
        """Zwraca funkcję identity - brak transformacji"""

        def identity_transform(sample):
            return sample

        return identity_transform

    def create_depth_transform(self):
        """Zwraca funkcję identity - brak transformacji"""

        def identity_transform(sample):
            return sample

        return identity_transform

    def postprocess_depth(self, raw_depth: np.ndarray) -> np.ndarray:
        """
        Virtual KITTI przechowuje głębię w centymetrach.
        Konwertujemy na metry i obsługujemy wartości specjalne.
        """
        # Konwersja z centymetrów na metry
        depth_meters = raw_depth / 100.0

        # Virtual KITTI używa 655.35m jako maksymalną odległość (niebo, itp.)
        # Wartości większe lub równe tej granicy oznaczają "nieskończoność"
        max_depth = 655.35
        depth_meters[depth_meters >= max_depth] = 0.0  # Ustaw na 0 (brak głębi)

        return depth_meters


class VirtualKITTIZoeDepthConfig(VirtualKITTITransformConfig):
    """
    Konfiguracja Virtual KITTI dostosowana do wymagań ZoeDepth.

    ZoeDepth może obsługiwać różne rozdzielczości, ale preferuje wielokrotności 32.
    Ta konfiguracja adaptuje to do specyfiki Virtual KITTI.
    """

    def __init__(self, input_size: int = 518):
        self.input_size = input_size

    def get_input_size(self) -> int:
        return self.input_size

    def create_image_transform(self):
        """
        Tworzy transformacje obrazu dla ZoeDepth z Virtual KITTI.
        Używamy uproszczonej wersji bez zewnętrznych zależności.
        """

        def zoedepth_image_transform(sample):
            image = sample['image']

            # Tutaj możesz dodać własne transformacje resize i normalizacji
            # Na razie zwracamy obraz bez zmian, ale w rzeczywistej implementacji
            # dodałbyś odpowiednie skalowanie i normalizację ImageNet

            # Przykład prostej normalizacji (możesz to rozwinąć):
            # image = (image - [0.485, 0.456, 0.406]) / [0.229, 0.224, 0.225]

            return {'image': image}

        return zoedepth_image_transform

    def create_depth_transform(self):
        """Transformacje głębi dla ZoeDepth"""

        def zoedepth_depth_transform(sample):
            # Tutaj możesz dodać skalowanie głębi do rozdzielczości modelu
            return sample

        return zoedepth_depth_transform

    def postprocess_depth(self, raw_depth: np.ndarray) -> np.ndarray:
        """Postprocessing głębi dla ZoeDepth"""
        # Konwersja z centymetrów na metry
        depth_meters = raw_depth / 100.0

        # Obsłuż wartości specjalne Virtual KITTI
        max_depth = 655.35
        depth_meters[depth_meters >= max_depth] = 0.0

        return depth_meters


class UniversalVirtualKITTIDataset(VirtualKITTIBaseDataset):
    """
    Uniwersalny dataset Virtual KITTI z podporą różnych konfiguracji modeli.

    Łączy elastyczność bazowej klasy Virtual KITTI z konfigurowalnymi transformacjami
    dla różnych modeli. Zachowuje wszystkie metadane Virtual KITTI (świat, wariant, etc.)
    """

    def __init__(self, root_dir: str, transform_config: VirtualKITTITransformConfig,
                 worlds: Optional[List[str]] = None, variations: Optional[List[str]] = None,
                 max_frames: Optional[int] = None):

        # Inicjalizuj bazową klasę
        super().__init__(root_dir, worlds, variations, max_frames)

        self.transform_config = transform_config
        self.is_raw_mode = isinstance(transform_config, VirtualKITTIRawConfig)

        # Inicjalizuj transformacje jeśli to nie tryb raw
        if not self.is_raw_mode:
            self.image_transform = transform_config.create_image_transform()
            self.depth_transform = transform_config.create_depth_transform()

        print(f"Dataset skonfigurowany dla {transform_config.__class__.__name__}")
        if self.is_raw_mode:
            print("Tryb RAW: Zachowane oryginalne rozdzielczości Virtual KITTI")
        else:
            print(f"Rozmiar docelowy: {transform_config.get_input_size()}")

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """
        Zwraca przetworzoną próbkę z Virtual KITTI.

        Zachowuje bogate metadane Virtual KITTI (świat, wariant) dla późniejszej analizy.
        """
        # Wczytaj surowe dane
        raw_data = self.load_raw_data(idx)

        # Zastosuj postprocessing głębi specyficzny dla Virtual KITTI
        depth_values = self.transform_config.postprocess_depth(raw_data['depth_raw'])

        if self.is_raw_mode:
            # Tryb RAW: minimalne przetwarzanie
            image_normalized = raw_data['image_rgb'].astype(np.float32) / 255.0

            # Konwersja na tensory z zachowaniem oryginalnych wymiarów
            image_tensor = torch.from_numpy(image_normalized.transpose(2, 0, 1))  # [3, H, W]
            depth_tensor = torch.from_numpy(depth_values).unsqueeze(0)  # [1, H, W]

        else:
            # Tryb normalny: pełne transformacje
            image_normalized = raw_data['image_rgb'].astype(np.float32) / 255.0

            # Zastosuj transformacje modelu
            image_transformed = self.image_transform({'image': image_normalized})['image']
            depth_transformed = self.depth_transform({'image': depth_values})['image']

            # Konwersja na tensory
            image_tensor = torch.from_numpy(image_transformed)
            depth_tensor = torch.from_numpy(depth_transformed).unsqueeze(0)

        return {
            'image': image_tensor,
            'ground_truth': depth_tensor,
            'metadata': {
                'rgb_path': raw_data['rgb_path'],
                'depth_path': raw_data['depth_path'],
                'world': raw_data['world'],
                'variation': raw_data['variation'],
                'frame_number': raw_data['frame_number'],
                'sample_id': raw_data['sample_id'],
                'original_shape': raw_data['original_shape'],
                'is_raw_mode': self.is_raw_mode,
                'dataset_type': 'virtual_kitti'
            }
        }


def create_virtual_kitti_dataloader(root_dir: str, model_name: str,
                                    worlds: Optional[List[str]] = None,
                                    variations: Optional[List[str]] = None,
                                    batch_size: int = 4, input_size: int = 518,
                                    shuffle: bool = False, num_workers: int = 2,
                                    max_frames: Optional[int] = None) -> DataLoader:
    """
    Factory function dla tworzenia DataLoader-ów Virtual KITTI.

    Ta funkcja enkapsuluje złożoność konfiguracji Virtual KITTI i udostępnia
    prosty interfejs podobny do oryginalnego systemu.

    Args:
        root_dir: Ścieżka do katalogu głównego Virtual KITTI
        model_name: Nazwa modelu ('raw', 'zoedepth', etc.)
        worlds: Lista światów do załadowania (None = wszystkie)
        variations: Lista wariantów do załadowania (None = tylko 'clone')
        batch_size: Rozmiar batcha
        input_size: Rozmiar wejściowy (ignorowany w trybie raw)
        shuffle: Czy mieszać dane
        num_workers: Liczba procesów do wczytywania
        max_frames: Maksymalna liczba klatek na świat-wariant (do debugowania)

    Returns:
        DataLoader skonfigurowany dla Virtual KITTI

    Przykłady użycia:
        # Wszystkie dane w trybie raw
        loader = create_virtual_kitti_dataloader(
            root_dir='./data',
            model_name='raw'
        )

        # Tylko określone światy i warianty pogodowe
        loader = create_virtual_kitti_dataloader(
            root_dir='./data',
            model_name='zoedepth',
            worlds=['0001', '0002'],
            variations=['clone', 'fog', 'rain']
        )
    """

    # Mapowanie nazw modeli na konfiguracje
    transform_configs = {
        'raw': VirtualKITTIRawConfig(input_size),
        'zoedepth': VirtualKITTIZoeDepthConfig(input_size),
        # Łatwo dodać kolejne modele...
    }

    if model_name not in transform_configs:
        available_models = list(transform_configs.keys())
        raise ValueError(f"Nieznany model: {model_name}. "
                         f"Dostępne: {available_models}")

    # Utwórz dataset
    dataset = UniversalVirtualKITTIDataset(
        root_dir=root_dir,
        transform_config=transform_configs[model_name],
        worlds=worlds,
        variations=variations,
        max_frames=max_frames
    )

    return DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        drop_last=False
    )


# Przykłady użycia i testowania
if __name__ == "__main__":
    print("=== Testowanie Virtual KITTI DataLoader ===")



    # Test 2: Określone światy i warianty pogodowe
    print("\n2. Test zaawansowany - określone światy i warianty:")
    try:
        weather_loader = create_virtual_kitti_dataloader(
            root_dir='./data',
            model_name='raw',
            worlds=['0001'],  # Tylko pierwsze dwa światy
            variations=['clone'],  # Tylko wybrane warianty
            batch_size=1,
            max_frames=3
        )

        sample_count = 0
        for batch in weather_loader:
            metadata = batch['metadata']
            print(f"   Próbka {sample_count}: {metadata['world'][0]} - {metadata['variation'][0]}")
            sample_count += 1
            if sample_count >= 5:  # Pokaż tylko pierwsze 5 próbek
                break

    except Exception as e:
        print(f"   Błąd: {e}")

