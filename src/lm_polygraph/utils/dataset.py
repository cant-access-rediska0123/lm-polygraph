import os
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from datasets import load_dataset, Dataset as hf_dataset

from typing import Iterable, Tuple, List, Union, Optional


class Dataset:
    """
    Seq2seq dataset for calculating quality of uncertainty estimation method.
    """

    def __init__(self, x: List[str], h: List[str], y: List[str], batch_size: int, metainfo: List | None = None):
        """
        Parameters:
            x (List[str]): a list of input texts.
            h (List[str]): a list of text to force proxy model to generate
            y (List[str]): a list of output (target) texts. Must have the same length as `x`.
            batch_size (int): the size of the texts batch.
        """
        self.x = x
        self.h = h
        self.y = y
        self.metainfo = metainfo
        assert len(x) == len(y) == len(h)
        if metainfo is not None:
            assert len(metainfo) == len(x)
        self.batch_size = batch_size

    def __iter__(self) -> Iterable[Tuple[List[str], List[str], List[str], List]]:
        """
        Returns:
            Iterable[Tuple[List[str], List[str]]]: iterates over batches in dataset,
                returns list of input texts and list of corresponding output texts.
        """
        for i in range(0, len(self.x), self.batch_size):
            yield (
                self.x[i : i + self.batch_size],
                self.h[i : i + self.batch_size],
                self.y[i : i + self.batch_size],
                self.metainfo[i : i + self.batch_size] if self.metainfo is not None else None,
            )

    def __len__(self) -> int:
        """
        Returns:
            int: number of batches in the dataset.
        """
        return (len(self.x) + self.batch_size - 1) // self.batch_size

    def select(self, indices: List[int]):
        """
        Shrinks the dataset down to only texts with the specified index.

        Parameters:
            indices (List[int]): indices to left in the dataset.Must have the same length as input texts.
        """
        self.x = [self.x[i] for i in indices]
        self.h = [self.h[i] for i in indices]
        self.y = [self.y[i] for i in indices]
        return self

    def subsample(self, size: int, seed: int):
        """
        Subsamples the dataset to the provided size.

        Parameters:
            size (int): size of the resulting dataset,
            seed (int): seed to perform random subsampling with.
        """
        np.random.seed(seed)
        if len(self.x) < size:
            indices = list(range(len(self.x)))
        else:
            if size < 1:
                size = int(size * len(self.x))
            indices = np.random.choice(len(self.x), size, replace=False)
        self.select(indices)
