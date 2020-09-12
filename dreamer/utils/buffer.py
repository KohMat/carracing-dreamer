import datetime
from io import BytesIO
from pathlib import Path
from typing import Dict, Tuple
from uuid import uuid4

import numpy as np
import torch


class Buffer:
    def __init__(self):
        self.data = dict()

    @classmethod
    def from_dict(self, dict_data: Dict[str, np.ndarray]):
        buffer = Buffer()
        buffer.data = dict_data
        return buffer

    @classmethod
    def load(self, filename: str):
        filename = Path(filename)
        try:
            with filename.open("rb") as f:
                data = np.load(f)
                data = {k: data[k] for k in data.keys()}
                buffer = Buffer.from_dict(data)
                return buffer
        except Exception as e:
            print(f"Could not load episode: {e}")

    def dump(self, filename: str):
        filename = Path(filename)
        with BytesIO() as f1:
            np.savez_compressed(f1, **self.data)
            f1.seek(0)
            with filename.open("wb") as f2:
                f2.write(f1.read())

    def dump_each_episode(self, directory: str):
        directory = Path(directory).expanduser()
        directory.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.datetime.now().strftime("%Y%m%dT%H%M%S")

        for i in range(self.num_episodes()):
            identifier = str(uuid4().hex)
            filename = (
                directory
                / f"{timestamp}-{identifier}-{self.episode_length()}.npz"
            )
            self[:, i][:, np.newaxis].dump(filename)

    def __getattr__(self, key):
        return self.data[key]

    def __getitem__(self, *args):
        sliced = {k: v.__getitem__(*args) for k, v in self.data.items()}
        return Buffer.from_dict(sliced)

    def add(self, *args, **kwargs):
        for i, arg in enumerate(args):
            self._add(str(i), np.array(arg))
        for k, v in kwargs.items():
            self._add(k, np.array(v))

    def _add(self, key: str, data: np.ndarray):
        # add the dims. (time, batch, ...)
        data = data[np.newaxis]
        if key not in self.data.keys():
            self.data[key] = data
        else:
            self.data[key] = np.concatenate((self.data[key], data), axis=0)

    def adjust_episode_length(self, length: int):
        def impl(data: np.ndarray):
            assert length <= data.shape[0]
            num_episodes, deleted_frames = divmod(data.shape[0], length)
            # to contains the last(event) frames, delete from the first
            data = np.delete(data, np.arange(deleted_frames), axis=0)
            data = np.split(data, num_episodes)
            data = np.concatenate(data, axis=1)
            return data

        for k in self.data.keys():
            self.data[k] = impl(self.data[k])

    def reduce_to(self, num_episodes: int):
        num_remove = max(self.num_episodes() - num_episodes, 0)
        remove_idx = np.random.randint(0, self.num_episodes(), num_remove)

        if len(remove_idx) == 0:
            return

        def remove(data):
            return np.delete(data, remove_idx, axis=1)

        for k in self.data.keys():
            self.data[k] = remove(self.data[k])

    def merge(self, other):
        for k in other.data.keys():
            if k not in self.data.keys():
                self.data[k] = other.data[k]
            else:
                self.data[k] = np.concatenate(
                    (self.data[k], other.data[k]), axis=1
                )

    def shape(self) -> Tuple[int, ...]:
        return {k: v.shape for k, v in self.data.items()}

    def _data_shape(self):
        if len(self.data.keys()) == 0:
            return 0

        data = self.data[next(iter(self.data.keys()))]
        return data.shape

    def num_episodes(self) -> int:
        shape = self._data_shape()
        if shape == 0:
            return 0
        return shape[1]

    def episode_length(self) -> int:
        shape = self._data_shape()
        if shape == 0:
            return 0
        return shape[0]

    def num_frames(self) -> int:
        shape = self._data_shape()
        if shape == 0:
            return 0
        return shape[0] * shape[1]


class Dataset(torch.utils.data.Dataset):
    def __init__(self, directory: str):
        self.directory = Path(directory).expanduser()
        self.filenames = list(self.directory.glob("*.npz"))

        if len(self.filenames) == 0:
            raise ValueError(f"There is no data in {directory}")

    def __len__(self) -> int:
        return len(self.filenames)

    def __getitem__(self, idx):
        filename = self.filenames[idx]
        return Buffer.load(filename)

    @classmethod
    def collate_fn(self, batch):
        buffer = Buffer()
        for datum in batch:
            buffer.merge(datum)

        return {key: torch.as_tensor(buffer.data[key]) for key in buffer.data}

    def loader(self, **kwargs):
        return torch.utils.data.DataLoader(
            self, collate_fn=self.collate_fn, **kwargs
        )
