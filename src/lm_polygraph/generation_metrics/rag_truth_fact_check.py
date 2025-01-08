import numpy as np

from typing import List, Dict
from .generation_metric import GenerationMetric


class RAGTruthFactCheck(GenerationMetric):
    def __init__(self):
        super().__init__(["input_texts"], "claim")

    def __str__(self):
        return "RAGTruthFactCheck"

    def __call__(
            self,
            stats: Dict[str, np.ndarray],
            target_texts: List[str],
    ) -> np.ndarray:
        labels = []
        for inp_text, sample_claims in zip(stats["input_texts"], stats["claims"]):
            labels.append([])
            for claim in sample_claims:
                labels[-1].append(0 if claim.implicit_true else 1)
        return labels
