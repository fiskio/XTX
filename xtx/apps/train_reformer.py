from argparse import ArgumentParser
from typing import List

from torch import nn

from xtx.apps.base import TimeSeriesRecipe
from xtx.models.lstm import LstmTimeSeriesModel
from xtx.models.reformer import ReformerTimeSeriesModel


class ReformerTimeSeriesRecipe(TimeSeriesRecipe):

    @staticmethod
    def add_parser_arguments(parser: ArgumentParser) -> ArgumentParser:
        """
            Adds LSTM specific options to the default parser
        """
        parser.add_argument('--hidden_size',
                            default=512,
                            type=int,
                            help='Hidden size of MLPs')
        parser.add_argument('--num_layers',
                            default=12,
                            type=int,
                            help='Number of layers')
        parser.add_argument('--num_heads',
                            default=8,
                            type=int,
                            help='Number of attention heads')

        return parser

    def get_model(self, args) -> nn.Module:
        return ReformerTimeSeriesModel(input_size=args.dataset.num_features,
                                       out_size=args.dataset.num_labels,
                                       max_seq_len=args.dataset.tot_size,
                                       hidden_size=args.hidden_size,
                                       num_layers=args.num_layers,
                                       num_heads=args.num_heads,
                                       dropout=args.dropout)


if __name__ == '__main__':
    ReformerTimeSeriesRecipe().run()