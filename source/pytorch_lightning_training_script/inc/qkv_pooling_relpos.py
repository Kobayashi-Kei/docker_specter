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
    # 各シーケンスの実際の長さより大きいインデックス位置をTrue（パディングされるべき位置）とし、それ以外をFalseとして、2次元のブールテンソルを返します。
    return expaned_lengths >= lengths.unsqueeze(-1)

class RelPositionalEncoding(torch.nn.Module):
    # Modified from https://github.com/k2-fsa/icefall/blob/master/egs/librispeech/ASR/pruned_transducer_stateless7_streaming/zipformer.py
    """Relative positional encoding module.

    See : Appendix B in "Transformer-XL: Attentive Language Models Beyond a Fixed-Length Context"
    Modified from https://github.com/espnet/espnet/blob/master/espnet/nets/pytorch_backend/transformer/embedding.py

    Args:
        d_model: Embedding dimension.
        dropout_rate: Dropout rate.
        max_len: Maximum input length.

    """

    def __init__(
        self,
        d_model: int,
        dropout_rate: float,
        num_heads: int = 8,
        max_len: int = 5000,
    ) -> None:
        """Construct a PositionalEncoding object."""
        super(RelPositionalEncoding, self).__init__()
        self.d_model = d_model
        self.dropout = torch.nn.Dropout(dropout_rate)
        self.pe = None
        self.num_heads = num_heads
        # if batch_first:
        self.T_of_x = 2
        self.extend_pe(torch.tensor(0.0).expand(1, 1, max_len))
        # else:
        #     self.T_of_x = 1
        #     self.extend_pe(torch.tensor(0.0).expand(1, max_len))


    def extend_pe(self, x: Tensor) -> None:
        """Reset the positional encodings."""
        x_size = x.size(self.T_of_x)
        if self.pe is not None:
            # self.pe contains both positive and negative parts
            # the length of self.pe is 2 * input_len - 1
            if self.pe.size(1) >= x_size * 2 - 1:
                # Note: TorchScript doesn't implement operator== for torch.Device
                if self.pe.dtype != x.dtype or str(self.pe.device) != str(x.device):
                    self.pe = self.pe.to(dtype=x.dtype, device=x.device)
                return
        # Suppose `i` means to the position of query vector and `j` means the
        # position of key vector. We use positive relative positions when keys
        # are to the left (i>j) and negative relative positions otherwise (i<j).
        pe_positive = torch.zeros(x_size, self.d_model)
        pe_negative = torch.zeros(x_size, self.d_model)
        position = torch.arange(0, x_size, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, self.d_model, 2, dtype=torch.float32)
            * -(math.log(10000.0) / self.d_model)
        )
        pe_positive[:, 0::2] = torch.sin(position * div_term)
        pe_positive[:, 1::2] = torch.cos(position * div_term)
        pe_negative[:, 0::2] = torch.sin(-1 * position * div_term)
        pe_negative[:, 1::2] = torch.cos(-1 * position * div_term)

        # Reserve the order of positive indices and concat both positive and
        # negative indices. This is used to support the shifting trick
        # as in "Transformer-XL: Attentive Language Models Beyond a Fixed-Length Context"
        pe_positive2 = torch.flip(pe_positive, [0])
        pe_negative = pe_negative[1:]
        pe = torch.cat([pe_positive2, pe_negative], dim=0)
        self.pe = pe.to(device=x.device, dtype=x.dtype)
        S = self.pe.size(0)
        self.pe = self.pe.reshape(1, S, self.num_heads, -1).permute(0, 2, 1, 3)

    def forward(self, x: Tensor) -> Tensor:
        """Add positional encoding.

        Args:
            x (Tensor): Input tensor (time, batch, `*`) or (batch, time, `*`)

        Returns:
            Tensor: Encoded tensor (batch, num_heads, left_context_len + 2*time-1, `*`).

        """
        self.extend_pe(x)
        x_size_left = x.size(self.T_of_x)
        pos_emb = self.pe[
            :,
            :,
            self.pe.size(2) // 2
            - x_size_left
            + 1 : self.pe.size(2) // 2  # noqa E203
            + x.size(self.T_of_x),
        ]
        return self.dropout(pos_emb)


class AttnPhi(nn.Module):
    def __init__(
        self,
        d_model,
        dropout = 0.1,
        num_heads = 8,
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
        
        self.pos = RelPositionalEncoding(d_model, dropout, num_heads)#, batch_first=True)
        
    def _get_pos(self, src : Tensor):
        B, h, S, C = src.shape
        pos_emb : Tensor = self.pos(src) * (self.d_model ** -0.5) # (1, h, 2S-1, C)
        
        pos_emb = pos_emb.as_strided(
            (1, h, S, C),
            (
                pos_emb.stride(0),
                pos_emb.stride(1),
                pos_emb.stride(2) - pos_emb.stride(3),
                pos_emb.stride(3),
            ),
            storage_offset = pos_emb.stride(3) * (S - 1)
        ) # (B, h, 2S-1, C) -> (B, h, S, C)
        
        return pos_emb
    
    def forward(self, src : Tensor, src_key_padding_mask : Tensor):
        """_summary_

        Args:
            src (Tensor): Input with shape (Batch, Sequence, Channel)
            src_key_padding_mask (Tensor): Mask with pad as True. Shape (B, S)
        """
        
        B, S, C = src.shape
        key = src.reshape(B, S, self.num_heads, self.d_att)
        key = key.permute(0, 2, 1, 3) # (B, S, h, d_att) -> (B, h, S, d_att)
        
        val = key + self._get_pos(key)
        
        scores = (self.query[None, :, None, :] * key).sum(dim=-1) # (B, h, S)
        scores.masked_fill_(src_key_padding_mask[:, None, :], float("-inf"))
        weights = scores.softmax(dim=-1)
        weights = self.dropout(weights)
        
        ret : Tensor = val * weights[..., None]
        ret = ret.permute(0, 2, 1, 3).reshape(B, S, -1) # (B, h, S, d_att) -> (B, S, d_model)
        ret = ret.sum(dim=1) # (B, d_model)
        
        return ret


def test():
    B = 1
    S = 50
    C = 768
    
    src = torch.randn(B, S, C)
    lengths = torch.tensor([S])
    src_key_padding_mask = make_pad_mask(lengths)
    
    pooler = AttnPhi(d_model=C)
    out : Tensor = pooler(src, src_key_padding_mask)
    print(out.shape) # Expect shape of (4, 32)
    
if __name__ == "__main__":
    test()    

    