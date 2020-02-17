import torch
from reformer_pytorch import Reformer, Autopadder
from reformer_pytorch.reformer_pytorch import FixedPositionEmbedding
from torch import nn


class ReformerTimeSeriesModel(nn.Module):

    """
        Reformer: The Efficient Transformer

        https://ai.googleblog.com/2020/01/reformer-efficient-transformer.html

        https://openreview.net/pdf?id=rkgNKkHtvB
    """

    def __init__(self,
                 input_size: int,
                 hidden_size: int,
                 out_size: int,
                 max_seq_len: int,
                 num_layers: int = 12,
                 dropout: float = 0.1,
                 num_heads: int = 8):

        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.out_size = out_size

        self.pos_emb = FixedPositionEmbedding(hidden_size)

        # self.reformer = Reformer(dim=hidden_size,
        #                          depth=num_layers,
        #                          max_seq_len=max_seq_len,
        #                          heads=num_heads,
        #                          lsh_dropout=dropout,
        #                          causal=True)
        #
        # self.padder = Autopadder(self.reformer)

        self.padder = nn.Transformer(d_model=hidden_size,
                                     nhead=num_heads,
                                     num_encoder_layers=num_layers,
                                     )

        self.to_hidden_size = nn.Linear(input_size, hidden_size)
        self.to_num_classes = nn.Linear(hidden_size, out_size)

    def forward(self, features, valid_seq_len):

        t_index = torch.arange(features.shape[1], device=features.device)
        pos_emb = self.pos_emb(t_index).type(features.type())

        output = pos_emb + self.to_hidden_size(features)

        # TODO: reduce load by only attending the valid time steps with the past
        output = self.padder(output)

        output = self.to_num_classes(output)

        output = output[:, -valid_seq_len:, :].contiguous()

        return output
