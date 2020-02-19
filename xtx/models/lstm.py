import torch
from torch import nn


class LstmTimeSeriesModel(nn.Module):

    def __init__(self, input_size: int, hidden_size: int, out_size: int, num_layers: int = 1, dropout: float = 0.0):

        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.out_size = out_size

        self.encoder = nn.LSTM(input_size=input_size,
                               hidden_size=hidden_size,
                               num_layers=num_layers,
                               dropout=dropout,
                               batch_first=True)

        self.to_num_classes = nn.Linear(hidden_size, self.out_size)

        self.init_weights()

    def init_weights(self):
        # encoder / decoder
        # nn.init.xavier_uniform_(self.to_hidden_size.weight)
        # nn.init.constant_(self.to_hidden_size.bias, 0)

        nn.init.xavier_uniform_(self.to_num_classes.weight)
        nn.init.constant_(self.to_num_classes.bias, 0)

        # LSTM
        self.init_lstm(self.encoder)
        #self.init_lstm(self.decoder)

    @staticmethod
    def init_lstm(lstm_mod):
        for name, param in lstm_mod.named_parameters():
            if 'weight' in name:
                nn.init.orthogonal_(param)
            elif 'bias' in name:
                nn.init.constant_(param, 0)
                # LSTM remember gate bias should be initialised to 1
                # https://github.com/pytorch/pytorch/issues/750
                r_gate = param[int(0.25 * len(param)):int(0.5 * len(param))]
                nn.init.constant_(r_gate, 1)

    def forward(self, features, valid_seq_len):

        hidden = self.init_hidden(features.size(0), features.device)

        encoder_features = features[:, :-valid_seq_len]
        decoder_features = features[:, -valid_seq_len:]

        output, hidden = self.encoder(encoder_features, hidden)
        output, hidden = self.encoder(decoder_features, hidden)
        #output, hidden = self.decoder(decoder_features, hidden)

        output = self.to_num_classes(output)

        return output

    def init_hidden(self, bsz, device):
        return (torch.zeros(self.num_layers, bsz, self.hidden_size).to(device),
                torch.zeros(self.num_layers, bsz, self.hidden_size).to(device))
