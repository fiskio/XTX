import argparse


def get_default_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument('--mat_file',
                        default=None,
                        type=str,
                        required=True,
                        help='The input train corpus.')
    parser.add_argument('--train_seq_len',
                        default=64,
                        type=int,
                        help='Steps to use for the encoder')
    parser.add_argument('--valid_seq_len',
                        default=24,
                        type=int,
                        help='Steps to use for the decoder (num_labels)')
    parser.add_argument('--skip_seq_len',
                        default=8,
                        type=int,
                        help='Steps to skip between encoder and decoder')
    parser.add_argument('--dataset_seq_len',
                        default=None,
                        type=int,
                        help='Maximum number of steps to truncate the dataset')

    parser.add_argument('--output_dir',
                        default=None,
                        type=str,
                        help='The output directory where the model checkpoints will be written.')

    parser.add_argument('--train_batch_size',
                        default=32,
                        type=int,
                        help='Total batch size for training.')
    parser.add_argument('--learning_rate',
                        default=3e-5,
                        type=float,
                        help='The initial learning rate for Adam.')
    parser.add_argument('--num_train_epochs',
                        default=3,
                        type=int,
                        help='Total number of training epochs to perform.')
    parser.add_argument('--warmup_proportion',
                        default=0.1,
                        type=float,
                        help='Proportion of training to perform to_num_classes learning rate warmup for. '
                             'E.g., 0.1 = 10%% of training.')
    parser.add_argument('--adam_epsilon',
                        default=1e-8,
                        type=float,
                        help='Epsilon for Adam optimizer')
    parser.add_argument('--no_cuda',
                        action='store_true',
                        help='Whether not to use CUDA when available')
    parser.add_argument("--weight_decay",
                        default=0.0,
                        type=float,
                        help='Weight decay if we apply some')
    parser.add_argument('--local_rank',
                        type=int,
                        default=-1,
                        help='local_rank for distributed training on GPUs')
    parser.add_argument('--seed',
                        type=int,
                        default=42,
                        help='random seed for initialization')
    parser.add_argument('--max_gradients',
                        type=int,
                        default=1.0,
                        help='Maximum magnitude for gradient clipping')
    parser.add_argument('--fp16_opt_level',
                        type=str,
                        default='None',
                        help="For fp16: Apex AMP optimization level selected in ['None', 'O0', 'O1', 'O2', and 'O3']."
                             "See details at https://nvidia.github.io/apex/amp.html")
    parser.add_argument('--deterministic',
                        action='store_true',
                        help='Turn off some stochastic optimisations in cuDNN')
    parser.add_argument('--loss_scale',
                        type=float, default=0,
                        help='Loss scaling to improve fp16 numeric stability. Only used when fp16 set to True.\n'
                             '0 (default value): dynamic loss scaling.\n'
                             'Positive power of 2: static loss scaling value.\n')
    parser.add_argument('--tiny',
                        action='store_true',
                        help='Tiny model for debugging')
    parser.add_argument('--print_every',
                        default=1,
                        type=int,
                        help='Training epoch print rate')
    parser.add_argument('--valid_every',
                        default=3,
                        type=int,
                        help='Validation epoch print rate')
    parser.add_argument('--min_epochs',
                        type=int,
                        default=1,
                        help='Minimum number of epochs')
    parser.add_argument('--num_workers',
                        default=0,
                        type=int,
                        help='Number of processes to use for DataLoader')
    parser.add_argument('--dropout',
                        default=0,
                        type=float,
                        help='Amount of regularisation [0, 1]')
    return parser
