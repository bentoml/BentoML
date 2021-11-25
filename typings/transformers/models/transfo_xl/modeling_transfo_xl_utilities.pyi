

from torch import nn

"""
 Utilities for PyTorch Transformer XL model. Directly adapted from https://github.com/kimiyoung/transformer-xl.
"""
class ProjectedAdaptiveLogSoftmax(nn.Module):
    def __init__(self, n_token, d_embed, d_proj, cutoffs, div_val=..., keep_order=...) -> None:
        ...
    
    def forward(self, hidden, labels=..., keep_order=...):
        """
        Params:
            hidden :: [len*bsz x d_proj]
            labels :: [len*bsz

        Return:
            if labels is None: out :: [len*bsz x n_tokens] log probabilities of tokens over the vocabulary else: out ::
            [(len-1)*bsz] Negative log likelihood. We could replace this implementation by the native PyTorch one if
            theirs had an option to set bias on all clusters in the native one. here:
            https://github.com/pytorch/pytorch/blob/dbe6a7a9ff1a364a8706bf5df58a1ca96d2fd9da/torch/nn/modules/adaptive.py#L138
        """
        ...
    
    def log_prob(self, hidden):
        r"""
        Computes log probabilities for all :math:`n\_classes` From:
        https://github.com/pytorch/pytorch/blob/master/torch/nn/modules/adaptive.p

        Args:
            hidden (Tensor): a minibatch of example

        Returns:
            log-probabilities of for each class :math:`c` in range :math:`0 <= c <= n\_classes`, where
            :math:`n\_classes` is a parameter passed to ``AdaptiveLogSoftmaxWithLoss`` constructor. Shape:

            - Input: :math:`(N, in\_features)`
            - Output: :math:`(N, n\_classes)`
        """
        ...
    


