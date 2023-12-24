import torch 
import torch.nn as nn
from torch import Tensor 
import math

def make_pad_mask(lengths: torch.Tensor, max_len: int = 0) -> torch.Tensor:
    # Copied from https://github.com/k2-fsa/icefall/blob/master/icefall/utils.py
    """
    Args:
      lengths:
        A 1-D tensor containing sentence lengths.
      max_len:
        The length of masks.
    Returns:
      Return a 2-D bool tensor, where masked positions
      are filled with `True` and non-masked positions are
      filled with `False`.

    >>> lengths = torch.tensor([1, 3, 2, 5])
    >>> make_pad_mask(lengths)
    tensor([[False,  True,  True,  True,  True],
            [False, False, False,  True,  True],
            [False, False,  True,  True,  True],
            [False, False, False, False, False]])
    """
    assert lengths.ndim == 1, lengths.ndim
    max_len = max(max_len, lengths.max())
    n = lengths.size(0)
    seq_range = torch.arange(0, max_len, device=lengths.device)
    expaned_lengths = seq_range.unsqueeze(0).expand(n, max_len)

    return expaned_lengths >= lengths.unsqueeze(-1)


class PositionalEncoding(nn.Module):

    def __init__(self, d_model: int, dropout: float = 0.1, num_heads = 8, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe : torch.Tensor = torch.zeros(1, max_len, d_model)
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        pe = pe * (d_model ** -0.5)
        self.pe = pe.view(1, max_len, num_heads, -1).permute(0, 2, 1, 3)


    def forward(self, x: Tensor) -> Tensor:
        """
        Arguments:
            x: Tensor, shape ``[B, h, S, C]``
        """
        return self.dropout(self.pe[..., :x.size(2), :])


class AttnPhi(nn.Module):
    def __init__(
        self,
        d_model,
        dropout = 0.1,
        num_heads = 8,
        is_key_transform = False, 
        device= 'cuda:0',
    ):
        """Attention pooling via query

        Args:
            d_model (int): Dimension of the model backbone
            dropout (float, optional): Dropout probability for attention matrix and positional embedding. Defaults to 0.1.
            num_heads (int, optional): Number of heads. Defaults to 8.
        """
        super(AttnPhi, self).__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.dropout = nn.Dropout(p=dropout)
        assert not (d_model % self.num_heads), (
            f"d_model {d_model} must be divisible by num_heads {num_heads}."
        )
        self.d_att = d_model // self.num_heads 
        self.query = nn.Parameter(torch.randn(self.num_heads, self.d_att) * self.d_att ** -0.5)
        # Afterthought: Might want to test if xavier initialisation improves performance. 
        # nn.init.xavier_uniform_(self.query)
        if is_key_transform:
            self.key_transform = nn.Linear(d_model, d_model)  # Linear layer for key transformation
        
        self.pos = PositionalEncoding(d_model, dropout, num_heads) 
        
        self.device = device

    
    def forward(self, src : Tensor, src_key_padding_mask : Tensor, is_key_transform=False):
        """_summary_

        Args:
            src (Tensor): Input with shape (Batch, Sequence, Channel)
            src_key_padding_mask (Tensor): Mask with pad as True. Shape (B, S)
        """
        
        B, S, C = src.shape

        # print(src.size())
        # print(src)
        if is_key_transform:
            src = self.key_transform(src)
        # print(src.size())
        # print(src)
    
        key = src.reshape(B, S, self.num_heads, self.d_att)
        key = key.permute(0, 2, 1, 3) # (B, S, h, d_att) -> (B, h, S, d_att)

        val = key.to(device=self.device) + self.pos(key).to(device=self.device)
        # val = key + self.pos(key)

        scores = (self.query[None, :, None, :] * key).sum(dim=-1).to(device=self.device) # (B, h, S)
        # print(scores)
        scores.masked_fill_(src_key_padding_mask[:, None, :], float("-inf"))
        weights = scores.softmax(dim=-1)
        # print(weights)
        weights = self.dropout(weights)

        ret : Tensor = val * weights[..., None].to(device=self.device)
        # ret : Tensor = val * weights[..., None]
        ret = ret.permute(0, 2, 1, 3).reshape(B, S, -1) # (B, h, S, d_att) -> (B, S, d_model)
        ret = ret.sum(dim=1) # (B, d_model)
        
        return ret


def test():
    B = 1
    S = 2
    C = 16
    
    src = torch.randn(B, S, C)
    lengths = torch.tensor([S])
    src_key_padding_mask = make_pad_mask(lengths)
    
    pooler = AttnPhi(d_model=C)
    out : Tensor = pooler(src, src_key_padding_mask)
    print(out.shape) # Expect shape of (4, 32)
    
if __name__ == "__main__":
    test()    

    