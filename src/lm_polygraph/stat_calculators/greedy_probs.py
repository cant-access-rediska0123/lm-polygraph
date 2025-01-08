import torch
import numpy as np

from typing import Dict, List, Tuple

from .embeddings import get_embeddings_from_output
from .stat_calculator import StatCalculator
from lm_polygraph.utils.model import WhiteboxModel, BlackboxModel


class BlackboxGreedyTextsCalculator(StatCalculator):
    """
    Calculates generation texts for Blackbox model (lm_polygraph.BlackboxModel).
    """

    @staticmethod
    def meta_info() -> Tuple[List[str], List[str]]:
        """
        Returns the statistics and dependencies for the calculator.
        """

        return ["greedy_texts"], []

    def __init__(self):
        super().__init__()

    def __call__(
            self,
            dependencies: Dict[str, np.array],
            texts: List[str],
            model: BlackboxModel,
            max_new_tokens: int = 100,
    ) -> Dict[str, np.ndarray]:
        """
        Calculates generation texts for Blackbox model on the input batch.

        Parameters:
            dependencies (Dict[str, np.ndarray]): input statistics, can be empty (not used).
            texts (List[str]): Input texts batch used for model generation.
            model (Model): Model used for generation.
            max_new_tokens (int): Maximum number of new tokens at model generation. Default: 100.
        Returns:
            Dict[str, np.ndarray]: dictionary with List[List[float]] generation texts at 'greedy_texts' key.
        """
        with torch.no_grad():
            sequences = model.generate_texts(
                input_texts=texts,
                max_new_tokens=max_new_tokens,
                n=1,
            )

        return {"greedy_texts": sequences}


class GreedyProbsCalculator(StatCalculator):
    """
    For Whitebox model (lm_polygraph.WhiteboxModel), at input texts batch calculates:
    * generation texts
    * tokens of the generation texts
    * probabilities distribution of the generated tokens
    * attention masks across the model (if applicable)
    * embeddings from the model
    """

    @staticmethod
    def meta_info() -> Tuple[List[str], List[str]]:
        """
        Returns the statistics and dependencies for the calculator.
        """

        return [
            "input_texts",
            "input_tokens",
            "greedy_log_probs",
            "greedy_tokens",
            "greedy_tokens_alternatives",
            "greedy_texts",
            "greedy_log_likelihoods",
        ], []

    def __init__(self, n_alternatives: int = 10):
        super().__init__()
        self.n_alternatives = n_alternatives

    def __call__(
            self,
            dependencies: Dict[str, np.array],
            texts: List[str],
            model: WhiteboxModel,
            max_new_tokens: int = 100,
    ) -> Dict[str, np.ndarray]:
        """
        Calculates the statistics of probabilities at each token position in the generation.

        Parameters:
            dependencies (Dict[str, np.ndarray]): input statistics, can be empty (not used).
            texts (List[str]): Input texts batch used for model generation.
            model (Model): Model used for generation.
            max_new_tokens (int): Maximum number of new tokens at model generation. Default: 100.
        Returns:
            Dict[str, np.ndarray]: dictionary with the following items:
                - 'input_tokens' (List[List[int]]): tokenized input texts,
                - 'greedy_log_probs' (List[List[np.array]]): logarithms of autoregressive
                        probability distributions at each token,
                - 'greedy_texts' (List[str]): model generations corresponding to the inputs,
                - 'greedy_tokens' (List[List[int]]): tokenized model generations,
                - 'attention' (List[List[np.array]]): attention maps at each token, if applicable to the model,
                - 'greedy_log_likelihoods' (List[List[float]]): log-probabilities of the generated tokens.
        """

        if 'hyp_texts' not in dependencies.keys():
            raise Exception(
                "No 'hyp_texts' found in depencendies. "
                "Only proxy-model generations are supported in is LM-Polygraph version."
            )
        hyp_texts = dependencies['hyp_texts']
        assert len(texts) == len(hyp_texts)

        input_tokens = [model.tokenizer(t)["input_ids"] for t in texts]

        # Tokenizer hyp_texts but make sure tokens begin with input_batch tokens
        hyp_tokens = [
            model.tokenizer(h, add_special_tokens=False)["input_ids"] for h in hyp_texts
        ]
        combined_tokens = [it + ht for it, ht in zip(input_tokens, hyp_tokens)]
        combined_batch = model.tokenizer.pad(
            {"input_ids": combined_tokens},
            padding=True,
            return_tensors="pt",
        )
        combined_batch = {k: v.to(model.device()) for k, v in combined_batch.items()}


        with torch.no_grad():
            out = model(**combined_batch)
            logits = out.logits.log_softmax(-1)

        cut_logits = []
        cut_sequences = []
        cut_texts = []
        cut_alternatives = []
        for i in range(len(texts)):
            begin_pos = len(input_tokens[i])
            end_pos = begin_pos + len(hyp_tokens[i])
            cut_sequences.append(hyp_tokens[i])
            cut_texts.append(hyp_texts[i])
            cut_logits.append(logits[i][begin_pos - 1:end_pos - 1].cpu().numpy())
            cut_alternatives.append([[] for _ in range(begin_pos, end_pos)])

            for j in range(begin_pos, end_pos):
                lt = logits[i, j - 1, :].cpu().numpy()
                best_tokens = np.argpartition(lt, -self.n_alternatives)[-self.n_alternatives:]
                best_tokens = best_tokens[np.argsort(-lt[best_tokens])].tolist()

                # as hyp_texts are not necessarily greedy, so
                # need to make sure that first token is from hyp_texts
                cur_token = hyp_tokens[i][j - begin_pos]
                if cur_token not in best_tokens:
                    best_tokens = [cur_token] + best_tokens[:-1]
                else:
                    best_tokens = [cur_token] + [t for t in best_tokens if t != cur_token]

                for t in best_tokens:
                    cut_alternatives[-1][j - begin_pos].append((t, lt[t].item()))

        ll = []
        for i in range(len(texts)):
            log_probs = cut_logits[i]
            tokens = cut_sequences[i]
            assert len(tokens) == len(log_probs)
            ll.append([log_probs[j, tokens[j]] for j in range(len(log_probs))])

        result_dict = {
            "input_tokens": input_tokens,
            "greedy_log_probs": cut_logits,
            "greedy_tokens": cut_sequences,
            "greedy_tokens_alternatives": cut_alternatives,
            "greedy_texts": cut_texts,
            "greedy_log_likelihoods": ll,
        }

        return result_dict
