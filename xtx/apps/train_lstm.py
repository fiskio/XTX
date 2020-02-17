from argparse import ArgumentParser
from typing import List

from torch import nn

from xtx.apps.base import TimeSeriesRecipe
from xtx.models.lstm import LstmTimeSeriesModel


class LstmTimeSeriesRecipe(TimeSeriesRecipe):

    @staticmethod
    def add_parser_arguments(parser: ArgumentParser) -> ArgumentParser:
        """
            Adds LSTM specific options to the default parser
        """
        parser.add_argument('--hidden_size',
                            default=64,
                            type=int,
                            help='Hidden size for LSTM')
        parser.add_argument('--num_layers',
                            default=2,
                            type=int,
                            help='Number of layers for LSTM')
        return parser

    def get_model(self, args) -> nn.Module:
        return LstmTimeSeriesModel(input_size=args.dataset.num_features,
                                   out_size=args.dataset.num_labels,
                                   hidden_size=args.hidden_size,
                                   num_layers=args.num_layers,
                                   dropout=args.dropout)


if __name__ == '__main__':
    LstmTimeSeriesRecipe().run()
