import pytest
import torch

from torch import nn
from torch.utils.data import DataLoader

from xtx.dataset.dataset import TimeSeriesDataSet
from xtx.models.lstm import LstmTimeSeriesModel


@pytest.mark.parametrize('features, labels, train_size, skip_size, valid_size', [
    # (torch.arange(40).view(4, 10), torch.rand(10), 4, 3, 2),
    (torch.arange(40).view(2, 20), torch.rand(20), 7, 1, 2),
    # (torch.arange(80).view(4, 20), torch.rand(20), 17, 2, 1),
    # (torch.arange(80).view(4, 20), torch.rand(20), 5, 4, 3),
])
def test_timeseries_dataset(features, labels, train_size, skip_size, valid_size):
    dataset = TimeSeriesDataSet(features=features.float(),
                                labels=labels,
                                train_size=train_size,
                                skip_size=skip_size,
                                valid_size=valid_size)

    tsm = LstmTimeSeriesModel(input_size=dataset.num_features,
                              out_size=dataset.num_labels,
                              hidden_size=16,
                              num_layers=2,
                              dropout=0.5)

    optim = torch.optim.Adam(lr=0.1, params=tsm.parameters())

    orig_params = {k: v.clone() for k, v in tsm.named_parameters()}

    for batch in DataLoader(dataset, shuffle=True, batch_size=2):

        optim.zero_grad()

        # show bias vectors
        # print([(n, p) for n, p in tsm.named_parameters() if 'bias' in n])

        for k, v in batch.items():
            print(k.upper())
            print(v)

        # check batch sizes
        assert batch['features'].size(-2) == dataset.tot_size
        assert batch['labels'].size(-1) == dataset.valid_size
        assert train_size + skip_size + valid_size == dataset.tot_size

        hidden = tsm.init_hidden(bsz=batch['features'].size(0))

        out, hidden = tsm(batch['features'], hidden, dataset.offset)

        # check output seq_len
        assert out.size(1) == dataset.valid_size
        print('OUT', out.size())

        loss = nn.CrossEntropyLoss()(out.view(-1, dataset.num_labels),
                                     batch['cat_labels'].view(-1))

        loss.backward()
        optim.step()

    # weights should have changed due to training
    for name, param in tsm.named_parameters():
        assert not torch.all(torch.eq(param, orig_params[name])), name
