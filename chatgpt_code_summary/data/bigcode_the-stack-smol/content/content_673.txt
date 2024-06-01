# Import dependencies

# Math/Torch
import numpy as np
import torch.nn as nn

# Typing
from typing import List

# Instantiate class
class MRR(nn.Module):
    """Compute MRR metric (Mean reciprocal rank)"""
    def __init__(self, max_rank = 10):

        super(MRR, self).__init__()

        # Set max mrr rank
        self.max_rank = max_rank

    def _calculate_reciprocal_rank(self, hypothesis_ids: np.ndarray, reference_id: int) -> float:
        """Calculate the reciprocal rank for a given hypothesis and reference
            
            Params:
                hypothesis_ids: Iterator of hypothesis ids (as numpy array) ordered by its relevance
                reference_id: Reference id (as a integer) of the correct id of response
            Returns:
                reciprocal rank
        """

        # Assure hypothesis_ids is a numpy array
        hypothesis_ids = np.asarray(hypothesis_ids)

        # Calculate rank
        try:
            rank = np.where(hypothesis_ids == reference_id)[0][0] + 1
        except IndexError:
            rank = self.max_rank + 1

        # Rank grater then max_rank is set to zero
        if rank > self.max_rank:
            reciprocal_rank = 0.0

        else:
            # Calculate reciprocal rank
            reciprocal_rank = 1. / rank
        
        return reciprocal_rank

    def forward(self, batch_hypothesis_ids: List[np.ndarray], batch_reference_id: List[int]) -> float:
        """Score the mean reciprocal rank for the batch
            
            Example from http://en.wikipedia.org/wiki/Mean_reciprocal_rank
            
            >>> batch_hypothesis_ids = [[1, 0, 2], [0, 2, 1], [1, 0, 2]]
            >>> batch_reference_id = [2, 2, 1]
            >>> mrr = MRR()
            >>> mrr(batch_hypothesis_ids, batch_reference_id)
                0.61111111111111105

            Args:
                batch_hypothesis_ids: Batch of hypothesis ids (as numpy array) ordered by its relevance
                reference_id: Batch of reference id (as a integer) of the correct id of response
            Returns:
                Mean reciprocal rank (MRR)
        """

        # Assure batches have same length
        assert len(batch_hypothesis_ids) == len(batch_reference_id), "Hypothesis batch and reference batch must have same length."

        # Size of batch
        batch_size = len(batch_hypothesis_ids)
        
        # MRR to be calculated
        mrr = 0

        for hypothesis_ids, reference_id in zip(batch_hypothesis_ids, batch_reference_id):

            # Calculate reciprocal rank
            reciprocal_rank = self._calculate_reciprocal_rank(hypothesis_ids, reference_id)

            # Add to MRR
            mrr += reciprocal_rank/batch_size

        return mrr
