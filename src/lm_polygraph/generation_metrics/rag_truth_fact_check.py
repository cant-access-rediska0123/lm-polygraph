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
                if claim.implicit_true is None:
                    labels[-1].append(np.nan)
                elif claim.implicit_true:
                    labels[-1].append(0)
                else:
                    labels[-1].append(1)
        return labels
