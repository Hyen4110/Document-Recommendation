import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor as T
from torch import nn

def dot_product_scores(sent1_vectors, sent2_vectors):
    """
    calculates q->ctx scores for every row in ctx_vector
    :param q_vector:
    :param ctx_vector:
    :return:
    """
    # q_vector: n1 x D, sent2_vectors: n2 x D, result n1 x n2
    r = torch.matmul(sent1_vectors, torch.transpose(sent2_vectors, 0, 1))
    return r

class BiEncoderNllLoss(object):
    def calc(self, sent1_vectors, sent2_vectors):
        positive_idx_per_question = list(range(sent1_vectors.size()[0]))
        # hard_negative_idx_per_question = None
        scores = self.get_scores(sent1_vectors, sent2_vectors)

        if len(sent1_vectors.size()) > 1:
            q_num = sent1_vectors.size(0)
            scores = scores.view(q_num, -1)

        softmax_scores = F.log_softmax(scores, dim=1)

        loss = F.nll_loss(
            softmax_scores,
            torch.tensor(positive_idx_per_question).to(softmax_scores.device),
            reduction="mean",
        )
        print(l)
        max_score, max_idxs = torch.max(softmax_scores, 1)
        # correct_predictions_count = (max_idxs == torch.tensor(positive_idx_per_question).to(max_idxs.device)).sum()

        return loss, max_idxs

    def get_scores(self,sent1_vectors, sent2_vectors):
        f = BiEncoderNllLoss.get_similarity_function()
        return f(sent1_vectors, sent2_vectors)

    def get_similarity_function():
        return dot_product_scores