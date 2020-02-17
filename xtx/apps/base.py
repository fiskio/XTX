import os
import logging
import time
from abc import ABC, abstractmethod
from argparse import ArgumentParser, Namespace
from pprint import pformat
from typing import List
import torch
import numpy as np
from torch import nn
from torch.optim import AdamW
from torch.utils.data import RandomSampler, DistributedSampler, DataLoader

from xtx.utils.args import get_default_parser
from xtx.dataset.dataset import TimeSeriesDataSet
from xtx.utils.utils import set_random_seed, get_safe_output_dir, create_directory

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

try:
    # only import apex if available (to be able to test it locally)
    import apex
    from apex import amp
except ModuleNotFoundError as e:
    logging.warning(e)


class TimeSeriesRecipe(ABC):

    def __init__(self, raw_args=None):

        args = self.parse_args(raw_args)

        set_random_seed(args.seed)

        args = self.setup_output_dir(args)

        args = self.load_datasets(args)

        args = self.get_sampler(args)

        args = self.setup_device(args)

        args = self.setup_gradient_accumulation(args)

        args.model = self.get_model(args)

        # args.output_ids = self.get_output_ids(args)

        # args.criteria = self.get_criteria(args)

        # args.evaluation_metrics = self.get_evaluation_metrics(args)

        args.optimizer = self.get_optimizer(args)

        # args.lr_scheduler = self.get_lr_scheduler(args)

        args = self.finalize_model_setup(args)

        args = self.log_model_size(args)

        self.log_args(args)

        self.args = args

    def parse_args(self, raw_args: List[str]) -> Namespace:
        """
            Parse command line arguments
        """
        parser = get_default_parser()
        parser = self.add_parser_arguments(parser)
        return parser.parse_args(raw_args)

    @staticmethod
    def add_parser_arguments(parser: ArgumentParser) -> ArgumentParser:
        """
            Adds model specific options to the default parser
        """
        return parser

    @staticmethod
    def get_sampler(args: Namespace) -> Namespace:
        """
            Returns: the appropriate Sampler
        """
        args.sampler = RandomSampler if args.local_rank == -1 else DistributedSampler
        return args

    @staticmethod
    def load_datasets(args: Namespace) -> Namespace:
        """
        Logic to load `train` and/or `validation` dataset(s)

        Args:
            args: context

        Returns: train_dataset, valid_dataset, vocab_size
        """
        args.dataset = TimeSeriesDataSet.from_matlab(mat_path=args.mat_file,
                                                     train_size=args.train_seq_len,
                                                     skip_size=args.skip_seq_len,
                                                     valid_size=args.valid_seq_len,
                                                     size=args.dataset_seq_len)
        return args

    @staticmethod
    def setup_output_dir(args: Namespace) -> Namespace:
        """
            Get a directory if not supplied, and create it if needed
            Handle normal and polyaxon case, check if it's empty
        """
        # only save models if local_rank == 0
        if args.local_rank > 0 or args.mat_file is None:
            args.output_dir = None
        else:
            args.output_dir = create_directory(get_safe_output_dir(args.output_dir))
            if os.path.exists(args.output_dir) and os.listdir(args.output_dir):
                raise ValueError(f'Output directory {args.output_dir} already exists and is not empty!')
        return args

    @staticmethod
    def setup_device(args: Namespace) -> Namespace:
        """
            Setup GPU and distributed environment if needed
        """
        if args.local_rank == -1:
            if args.no_cuda is True:
                if torch.cuda.is_available():
                    logger.warning('GPU is available but NOT being used!')
                args.device = 'cpu'
                args.n_gpus = 0
            else:
                args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                args.n_gpus = torch.cuda.device_count()
        else:
            if torch.cuda.is_available() is False:
                logger.warning('GPU is NOT available!')
                args.device = 'cpu'
                args.n_gpus = 0
                raise ValueError(f'local_rank > -1 when cuda is *not* available is not supported')

            torch.cuda.set_device(args.local_rank)
            args.device = torch.device('cuda', args.local_rank)
            args.n_gpus = 1  # we've launched a distributed process and each process should have 1 GPU
            logger.info(f'GPU: {args.device}')

        # deterministic?
        logger.info(f'Deterministic run: {args.deterministic}')
        torch.backends.cudnn.benchmark = not args.deterministic  # type: ignore

        # FOR DISTRIBUTED:  If we are running under torch.distributed.launch,
        # the 'WORLD_SIZE' environment variable will also be set automatically.
        args.distributed = False
        if 'WORLD_SIZE' in os.environ:
            logger.info(f"WORLD_SIZE {os.environ['WORLD_SIZE']}")
            args.distributed = int(os.environ['WORLD_SIZE']) > 1

        if args.distributed:
            # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
            torch.distributed.init_process_group(backend='nccl', init_method='env://')  # type: ignore

            # divide the workers between the processes
            args.num_workers = max(0, args.num_workers // int(os.environ['WORLD_SIZE']))

        # summary report
        logger.info(f'device: {args.device} | '
                    f'n_gpu: {args.n_gpus} | '
                    f'local_rank: {args.local_rank} | '
                    f'distributed training: {bool(args.local_rank != -1)} | '
                    f'fp16_opt_level: {args.fp16_opt_level}')

        return args

    @staticmethod
    def setup_gradient_accumulation(args) -> Namespace:
        if args.gradient_accumulation_steps < 1:
            raise ValueError(f'Invalid gradient_accumulation_steps '
                             f'parameter: {args.gradient_accumulation_steps}, should be >= 1')

        args.train_batch_size = args.train_batch_size // args.gradient_accumulation_steps
        return args

    @staticmethod
    def log_model_size(args):
        args.total_params = sum(p.numel() for p in args.model.parameters() if p.requires_grad)
        logger.info(f'Total number of trainable parameters: {args.total_params}')
        return args

    @staticmethod
    def get_optimizer(args):
        param_optimizer = list(args.model.named_parameters())
        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
             'weight_decay': args.weight_decay},
            {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)],
             'weight_decay': 0}
        ]
        return AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)

    # @staticmethod
    # def get_lr_scheduler(args):
    #     num_batches = args.dataset_size // args.train_batch_size
    #     t_total = num_batches // args.gradient_accumulation_steps * args.num_train_epochs
    #     warmup_steps = t_total * args.warmup_proportion
    #
    #     if args.distributed:
    #         t_total = t_total // int(os.environ['WORLD_SIZE'])
    #         warmup_steps = warmup_steps // int(os.environ['WORLD_SIZE'])
    #
    #     return WarmupLinearSchedule(args.optimizer, warmup_steps=warmup_steps, t_total=t_total)

    @staticmethod
    def finalize_model_setup(args):
        # FP16
        if args.fp16_opt_level != 'None':
            args.model = args.model.to(args.device)
            args.model = apex.parallel.convert_syncbn_model(args.model)
            args.model, args.optimizer = amp.initialize(args.model, args.optimizer,
                                                        opt_level=args.fp16_opt_level,
                                                        keep_batchnorm_fp32=False)
        if args.distributed:
            logger.info(f'Distributed Training! 1 process -> 1 GPU, {args.num_workers} CPUs')
            args.model = args.model.to(args.device)
            args.model = apex.parallel.DistributedDataParallel(args.model)

        elif args.n_gpus > 1:
            logger.info(
                f'DataParallel Training! 1 process -> {torch.cuda.device_count()} GPUs, {args.num_workers} CPUs')
            args.model = torch.nn.DataParallel(args.model)

        return args

    @staticmethod
    def log_args(args) -> None:

        args_buf = pformat(vars(args), indent=2, width=1)

        logger.info(f'{args_buf}')

        if args.output_dir is not None:
            with open(os.path.join(args.output_dir, 'args.json'), 'w') as jf:
                jf.write(args_buf)


    @abstractmethod
    def get_model(self, args) -> nn.Module:
        raise NotImplementedError

    def run(self):

        args = self.args

        dataloader = DataLoader(dataset=args.dataset,
                                batch_size=args.train_batch_size,
                                sampler=args.sampler(args.dataset),
                                pin_memory=True)

        for epoch in range(args.num_train_epochs):

            metrics = dict(ce=[], mse=[], n_wrong=[])

            for batch_idx, batch in enumerate(dataloader):

                start_time = time.time()

                args.optimizer.zero_grad()

                # show bias vectors
                # print([(n, p) for n, p in tsm.named_parameters() if 'bias' in n])

                # for k, v in batch.items():
                #     print(k.upper())
                #     print(v)

                # TODO LSTM
                # hidden = args.model.init_hidden(bsz=batch['features'].size(0))
                # hidden = [h.to(dtype=batch['features'].dtype, device=args.device) for h in hidden]
                #
                # out, hidden = args.model(batch['features'], hidden, args.dataset.offset)

                out = args.model(batch['features'], args.dataset.valid_size)

                # check output seq_len
                assert out.size(1) == args.dataset.valid_size
                # print('OUT', out.size())

                ce = nn.CrossEntropyLoss()(out.view(-1, args.dataset.num_labels), batch['cat_labels'].view(-1))

                predicted_indices = np.argmax(out.detach().numpy(), axis=-1)
                predicted_labels = np.vectorize(args.dataset.id_to_value.__getitem__)(predicted_indices)
                predicted_labels = torch.from_numpy(predicted_labels)
                # print(predicted_indices)
                # print(predicted_labels)
                # print(predicted_labels.size(), batch['labels'].size())

                # predicted_labels = [args.dataset.id_to_value[i] for i in out.view(-1)]

                mse = nn.MSELoss()(predicted_labels, batch['labels'])

                print(predicted_labels.size())
                print('DIFF', torch.abs(batch['labels'] - predicted_labels))
                wrong_mask = predicted_labels != batch['labels']
                n_wrong = int(torch.sum(wrong_mask).item())
                print(wrong_mask)
                print(n_wrong)

                metrics['ce'].append(ce.item())
                metrics['mse'].append(mse.item())
                metrics['n_wrong'].append(n_wrong)

                ce.backward()
                args.optimizer.step()

                if batch_idx % args.print_every == 0:
                    iter_time = time.time() - start_time
                    i = epoch + float(batch_idx) / len(dataloader)
                    m_str = ' | '.join(f'{k}: {np.mean(v):.2f}' for (k, v) in metrics.items())
                    out = f'[Epoch {i:.2f}] {m_str} | Time: {iter_time:.2f}'
                    print(out)
                    metrics = dict(ce=[], mse=[], n_wrong=[])
