"""Dataset PyTorch WSTS -> STConvS2S (janelas diárias, recortes de tamanho fixo).

Contrato por amostra (`P = patch`):

- `x`:      float32 `[C, T_in, P, P]` — canais de `wsts_core.input_channel_names`, dias t−T_in+1..t;
- `y`:      float32 `[1, 1, P, P]`    — fogo ativo binário em t+1;
- `mask`:   float32 `[1, 1, P, P]`    — 1 em pixels avaliados (negativos incluídos), 0 em padding/halo;
- `fire_t`: float32 `[1, 1, P, P]`    — fogo ativo em t, todas as detecções (persistência oficial
  `use_all_detections=True`; define pixels "novos vs t");
- `fire_last_t`: idem, só pixels com a hora máxima de t na imagem inteira (persistência oficial
  `use_all_detections=False`, "última passagem");
- `fire_window`: união das detecções dos dias de entrada t−T_in+1..t (pixels "novos vs janela");
- `crop32`: 1 dentro da região `center_crop_x32` da imagem (vista comparável ao artigo);
- `event`: índice do evento (histogramas por evento para IC por bootstrap de eventos).

Treino: um recorte por janela e por época, escolhido como no oficial (10 tentativas; escore =
média da hora de detecção nos dias de entrada + 1000 × fração de fogo em t+1), sem augmentation
geométrica (desvio registrado: exigiria transformar ângulos e vetores). Avaliação: tiles
determinísticos com núcleo `core` e halo `halo`; cada pixel da imagem cai no núcleo de exatamente
um tile, então as métricas cobrem a imagem inteira uma única vez.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.core import (
    ACTIVE_FIRE_CHANNEL,
    DEFAULT_T_IN,
    EventRef,
    fire_mask,
    input_channel_names,
    open_h5,
    parse_dates,
    transform_frames,
    valid_window_starts,
)

try:
    from torch.utils.data import Dataset
except ImportError:  # pragma: no cover - torch é dependência da venv WSTS
    Dataset = object  # type: ignore[assignment,misc]


DEFAULT_PATCH = 128
DEFAULT_HALO = 8
DEFAULT_CACHE_BYTES = 128 * 2**20
MAX_OPEN_FILES = 2
CROP_TRIALS = 10


@dataclass(frozen=True)
class Sample:
    event: int
    start: int
    row: int
    col: int


def window_index(events: Sequence[EventRef], *, t_in: int, skip_initial: int = 0) -> list[tuple[int, int]]:
    """(evento, início) de todas as janelas contíguas `t_in + 1` dias, sem atravessar eventos."""
    index = []
    for position, event in enumerate(events):
        dates = parse_dates(event.dates)
        index += [(position, start) for start in valid_window_starts(dates, t_in=t_in, t_out=1, skip_initial=skip_initial)]
    return index


def center_crop32_bounds(size: int) -> tuple[int, int]:
    """Limites `[início, fim)` do `center_crop` do torchvision para o maior múltiplo de 32 <= size."""
    new = size // 32 * 32
    start = int(round((size - new) / 2.0))
    return start, start + new


def last_pass_mask(af_hours: np.ndarray) -> np.ndarray:
    """Pixels com a hora de detecção máxima da imagem (0 em tudo se não houver fogo)."""
    hours = np.nan_to_num(af_hours, nan=0.0)
    peak = hours.max()
    if peak <= 0:
        return np.zeros(hours.shape, dtype=np.float32)
    return (hours == peak).astype(np.float32)


def tile_origins(size: int, core: int) -> list[int]:
    """Origens dos núcleos de tiles que cobrem `[0, size)` sem sobreposição."""
    return list(range(0, size, core))


def read_region(data: Any, start: int, stop: int, rows: tuple[int, int], cols: tuple[int, int]) -> np.ndarray:
    """Lê `[stop−start, 23, h, w]` da região pedida de um dataset h5py, recortada aos limites da imagem."""
    r0, r1 = max(rows[0], 0), min(rows[1], data.shape[2])
    c0, c1 = max(cols[0], 0), min(cols[1], data.shape[3])
    return np.asarray(data[start:stop, :, r0:r1, c0:c1], dtype=np.float32)


def choose_training_crop(
    af_hours_in: np.ndarray, fire_next: np.ndarray, patch: int, rng: np.random.Generator
) -> tuple[int, int]:
    """Escolha oficial do recorte: maximiza `mean(hora de fogo) + 1000 * mean(y)` em 10 sorteios."""
    height, width = fire_next.shape
    best, best_score = (0, 0), -np.inf
    for _ in range(CROP_TRIALS):
        row = int(rng.integers(0, max(height - patch, 0) + 1))
        col = int(rng.integers(0, max(width - patch, 0) + 1))
        score = float(af_hours_in[:, row : row + patch, col : col + patch].mean()) + 1000.0 * float(
            fire_next[row : row + patch, col : col + patch].mean()
        )
        if score > best_score:
            best, best_score = (row, col), score
    return best


class WSTSWindowDataset(Dataset):
    def __init__(
        self,
        events: Sequence[EventRef],
        normalization: dict[str, Any],
        *,
        mode: str,
        t_in: int = DEFAULT_T_IN,
        patch: int = DEFAULT_PATCH,
        halo: int = DEFAULT_HALO,
        skip_initial: int = 0,
        include_forecast_precip: bool = False,
        drop_groups: Sequence[str] = (),
        seed: int = 0,
        max_windows: int | None = None,
        cache_bytes: int = DEFAULT_CACHE_BYTES,
    ) -> None:
        if mode not in {"train", "eval"}:
            raise ValueError("mode deve ser 'train' ou 'eval'.")
        self.events = list(events)
        self.normalization = normalization
        self.mode = mode
        self.t_in = t_in
        self.patch = patch
        self.halo = halo
        self.include_forecast_precip = include_forecast_precip
        self.drop_groups = tuple(drop_groups)
        self.seed = seed
        self.epoch = 0
        self.channel_names = input_channel_names(
            include_forecast_precip=include_forecast_precip, drop_groups=self.drop_groups
        )
        windows = window_index(self.events, t_in=t_in, skip_initial=skip_initial)
        if max_windows is not None:
            windows = windows[:max_windows]
        self.windows = windows
        self.cache_bytes = cache_bytes
        self._handles: dict[Path, Any] = {}
        if mode == "train":
            self.samples = [Sample(e, s, -1, -1) for e, s in windows]
        else:
            core = patch - 2 * halo
            if core <= 0:
                raise ValueError("patch deve ser maior que 2 * halo.")
            self.samples = [
                Sample(e, s, r, c)
                for e, s in windows
                for r in tile_origins(self.events[e].height, core)
                for c in tile_origins(self.events[e].width, core)
            ]

    def __len__(self) -> int:
        return len(self.samples)

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def _data(self, path: Path) -> Any:
        """Dataset h5py aberto e reaproveitado; tiles da mesma janela reutilizam o cache de chunks."""
        # O cache de chunks pertence ao objeto Dataset aberto: guardar o Dataset, não só o arquivo.
        if path not in self._handles:
            while len(self._handles) >= MAX_OPEN_FILES:
                oldest = next(iter(self._handles))
                self._handles.pop(oldest)[0].close()
            handle = open_h5(path, cache_bytes=self.cache_bytes)
            self._handles[path] = (handle, handle["data"])
        return self._handles[path][1]

    def close(self) -> None:
        for handle, _ in self._handles.values():
            handle.close()
        self._handles.clear()

    def __getstate__(self) -> dict[str, Any]:
        state = self.__dict__.copy()
        state["_handles"] = {}
        return state

    def _rng(self, index: int) -> np.random.Generator:
        return np.random.default_rng([self.seed, self.epoch, index])

    def __getitem__(self, index: int) -> dict[str, Any]:
        import torch

        sample = self.samples[index]
        event = self.events[sample.event]
        stop = sample.start + self.t_in + 1
        data = self._data(event.path)
        # Canal de fogo (horas) da imagem inteira nos dias da janela: escolha do recorte e "última passagem".
        af = np.asarray(data[sample.start : stop, ACTIVE_FIRE_CHANNEL], dtype=np.float32)
        if self.mode == "train":
            row, col = choose_training_crop(af[: self.t_in], fire_mask(af[-1]), self.patch, self._rng(index))
            rows, cols = (row, row + self.patch), (col, col + self.patch)
            core_rows, core_cols = rows, cols
        else:
            core = self.patch - 2 * self.halo
            core_rows, core_cols = (sample.row, sample.row + core), (sample.col, sample.col + core)
            rows = (sample.row - self.halo, sample.row - self.halo + self.patch)
            cols = (sample.col - self.halo, sample.col - self.halo + self.patch)
        raw = read_region(data, sample.start, stop, rows, cols)
        x_valid = transform_frames(
            raw[: self.t_in],
            self.normalization,
            include_forecast_precip=self.include_forecast_precip,
            drop_groups=self.drop_groups,
        )
        r0, c0 = max(rows[0], 0), max(cols[0], 0)
        h, w = raw.shape[-2:]
        crop_rows, crop_cols = center_crop32_bounds(event.height), center_crop32_bounds(event.width)
        region_rows = np.arange(r0, r0 + h)[:, None]
        region_cols = np.arange(c0, c0 + w)[None, :]
        fields = {
            "y": fire_mask(raw[self.t_in]),
            "fire_t": fire_mask(raw[self.t_in - 1]),
            "fire_last_t": last_pass_mask(af[self.t_in - 1])[r0 : r0 + h, c0 : c0 + w],
            "fire_window": np.nan_to_num(raw[: self.t_in, ACTIVE_FIRE_CHANNEL], nan=0.0).max(axis=0) > 0,
            "crop32": (
                (region_rows >= crop_rows[0]) & (region_rows < crop_rows[1])
                & (region_cols >= crop_cols[0]) & (region_cols < crop_cols[1])
            ),
        }

        # Posição da região lida dentro do patch (padding quando a janela sai da imagem).
        pr0, pc0 = max(-rows[0], 0), max(-cols[0], 0)
        x = np.zeros((len(self.channel_names), self.t_in, self.patch, self.patch), dtype=np.float32)
        x[:, :, pr0 : pr0 + h, pc0 : pc0 + w] = x_valid
        item: dict[str, Any] = {"x": torch.from_numpy(x)}
        for name, values in fields.items():
            padded = np.zeros((1, 1, self.patch, self.patch), dtype=np.float32)
            padded[0, 0, pr0 : pr0 + h, pc0 : pc0 + w] = values
            item[name] = torch.from_numpy(padded)
        # Máscara: pixels dentro da imagem e dentro do núcleo avaliado.
        mask = np.zeros((1, 1, self.patch, self.patch), dtype=np.float32)
        mr0, mr1 = max(core_rows[0], 0) - rows[0], min(core_rows[1], event.height) - rows[0]
        mc0, mc1 = max(core_cols[0], 0) - cols[0], min(core_cols[1], event.width) - cols[0]
        mask[0, 0, mr0:mr1, mc0:mc1] = 1.0
        item["mask"] = torch.from_numpy(mask)
        item["event"] = sample.event
        item["index"] = index
        return item


def receptive_radius(num_layers: int, kernel_size: int = 3) -> int:
    """Raio espacial do STConvS2S_C: `num_layers` SpatialBlocks 3x3 + `conv_final` 3x3x3."""
    return (num_layers + 1) * (kernel_size // 2)


def worker_init_fn(worker_id: int) -> None:
    """Semente por worker (o oficial usa `np.random` global sem semear workers)."""
    import torch

    np.random.seed((torch.initial_seed() + worker_id) % 2**32)
