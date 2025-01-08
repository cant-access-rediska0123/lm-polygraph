import numpy as np

import logging
from typing import List, Dict, Tuple
from lm_polygraph.estimators.estimator import Estimator

log = logging.getLogger(__name__)


class Processor:
    """
    Abstract class to perform actions after processing new texts batch.
    """

    def on_batch(
        self,
        batch_stats: Dict[str, np.ndarray],
        batch_gen_metrics: Dict[Tuple[str, str], List[float]],
        batch_estimations: Dict[Tuple[str, str], List[float]],
    ):
        """
        Processes new batch.

        Parameters:
            batch_stats (Dict[str, np.ndarray]): Dictionary of statistics calculated with `stat_calculators`.
            batch_gen_metrics (Dict[Tuple[str, str], List[float]]): Dictionary of generation metrics calculated
                for the batch. Dictionary keys consist of UE level (`sequence` or `token`) and generation metrics
                name.
            batch_estimations (Dict[Tuple[str, str], List[float]]): Dictionary of UE estimations calculated
                for the batch. Dictionary keys consist of UE level (`sequence` or `token`) and UE estimator name.
        """
        pass

    def on_eval(self, metrics: Dict[Tuple[str, str, str, str], float]):
        """
        Processes newly calculated evaluation metrics.

        Parameters:
            metrics (Dict[Tuple[str, str, str, str], float]: metrics calculated using `ue_metrics` on the batch which
                was considered at the last `on_batch` call. Dictionary keys consist of UE level,
                estimator name, generation metrics name and `ue_metrics` name which was used to calculate quality
                metrics between this estimator's uncertainty estimations and generation metric outputs.
        """
        pass


class Logger(Processor):
    """
    Processor logging batch information to stdout.
    """

    def __init__(self, print_fn=log.info):
        self.print_fn = print_fn

    def on_batch(
        self,
        batch_stats: Dict[str, np.ndarray],
        batch_gen_metrics: Dict[Tuple[str, str], List[float]],
        batch_estimations: Dict[Tuple[str, str], List[float]],
    ):
        """
        Outputs statistics from `batch_stats`, `batch_gen_metrics` and `batch_estimations` to stdout.
        """
        self.print_fn("=" * 50 + " NEW BATCH " + "=" * 50)
        self.print_fn("Statistics:")
        self.print_fn("")
        for key, val in batch_stats.items():
            str_repr = str(val)
            # to skip large outputs
            if len(str_repr) < 10000 and str_repr.count("\n") < 10:
                self.print_fn(f"{key}: {val}")
                self.print_fn("")
        self.print_fn("-" * 100)
        self.print_fn("Estimations:")
        self.print_fn("")
        for key, val in batch_estimations.items():
            self.print_fn(f"{key}: {val}")
            self.print_fn("")
        self.print_fn("-" * 100)
        self.print_fn("Generation metrics:")
        self.print_fn("")
        for key, val in batch_gen_metrics.items():
            self.print_fn(f"{key}: {val}")
            self.print_fn("")

    def on_eval(
        self,
        metrics: Dict[Tuple[str, str, str, str], float],
        bad_estimators: Dict[Estimator, int],
    ):
        """
        Outputs statistics from `metrics` and failed estimators to stdout.
        """
        self.print_fn("=" * 50 + " METRICS " + "=" * 50)
        self.print_fn("Metrics:")
        self.print_fn("")
        for key, val in metrics.items():
            self.print_fn(f"{key}: {val}")
            self.print_fn("")
        if len(bad_estimators) > 0:
            self.print_fn("=" * 45 + " FAILED ESTIMATORS " + "=" * 45)
            for bad_estimator, batch_i in bad_estimators.items():
                self.print_fn(str(bad_estimator) + " on batch " + str(batch_i))
