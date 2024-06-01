from typing import List

import torch
from torch import Tensor

from hw_asr.base.base_metric import BaseMetric
from hw_asr.base.base_text_encoder import BaseTextEncoder
from hw_asr.metric.utils import calc_cer


class ArgmaxCERMetric(BaseMetric):
    def __init__(self, text_encoder: BaseTextEncoder, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.text_encoder = text_encoder

    def __call__(self, log_probs: Tensor, text: List[str], *args, **kwargs):
        cers = []
        predictions = torch.argmax(log_probs.cpu(), dim=-1)
        for log_prob_vec, target_text, log_prob_length in zip(predictions, text, kwargs['log_probs_length']):
            if hasattr(self.text_encoder, "ctc_decode"):
                pred_text = self.text_encoder.ctc_decode(log_prob_vec[:log_prob_length.item()])
            else:
                pred_text = self.text_encoder.decode(log_prob_vec)
            cers.append(calc_cer(target_text, pred_text))
        return sum(cers) / len(cers)


class BeamSearchCERMetric(BaseMetric):
    def __init__(self, text_encoder: BaseTextEncoder, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.text_encoder = text_encoder

    def __call__(self, log_probs: Tensor, text: List[str], *args, **kwargs):
        cers = []

        if hasattr(self.text_encoder, "ctc_beam_search"):
            predictions = log_probs.cpu()
        else:
            predictions = torch.argmax(log_probs.cpu(), dim=-1)

        for log_prob_length, log_prob_vec, target_text in zip(kwargs['log_probs_length'], predictions, text):
            if hasattr(self.text_encoder, "ctc_beam_search"):
                pred_text = self.text_encoder.ctc_beam_search(log_prob_vec[:log_prob_length.item(), :].unsqueeze(0))
            elif hasattr(self.text_encoder, "ctc_decode"):
                pred_text = self.text_encoder.ctc_decode(log_prob_vec)
            else:
                pred_text = self.text_encoder.decode(log_prob_vec)
            cers.append(calc_cer(target_text, pred_text))
        return sum(cers) / len(cers)
