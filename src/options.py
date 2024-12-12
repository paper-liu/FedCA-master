# -*- coding: utf-8 -*-
# Python version: 3.11

import argparse


def args_parser():
    parser = argparse.ArgumentParser()

    # fedca arguments
    parser.add_argument('--pre_round', type=int, default=10,
                        help="number of rounds of pre-training")
    parser.add_argument('--opt', type=str, default='normal',
                        help="type of specical_processing (normal, noise, mislabel)")
    parser.add_argument('--get_gradients', type=int, default=0,
                        help="Default set to gradients. Set to 0 for weights.")
    parser.add_argument('--loss', type=str, default='NLLLoss',
                        help="Default set to loss function. (NLLLoss, CrossEntropyLoss)")
    parser.add_argument('--baseline', type=int, default=0,
                        help="Whether to conduct the baseline experiment, 0 means no")

    # federated arguments (Notation for the arguments followed from paper)
    parser.add_argument('--comm_round', type=int, default=10,
                        help="number of rounds of training")
    parser.add_argument('--num_users', type=int, default=10,
                        help="number of users: K")
    parser.add_argument('--frac', type=float, default=0.1,
                        help='the fraction of clients: C')
    parser.add_argument('--local_ep', type=int, default=10,
                        help="the number of local epochs: E")
    parser.add_argument('--local_bs', type=int, default=10,
                        help="local batch size: B")
    parser.add_argument('--lr', type=float, default=0.01,
                        help='learning rate')
    parser.add_argument('--momentum', type=float, default=0.5,
                        help='SGD momentum (default: 0.5)')

    # model arguments
    parser.add_argument('--model', type=str, default='mlp',
                        help='model name')
    parser.add_argument('--kernel_num', type=int, default=9,
                        help='number of each kind of kernel')
    parser.add_argument('--kernel_sizes', type=str, default='3,4,5',
                        help='comma-separated kernel size to use for convolution')
    parser.add_argument('--num_channels', type=int, default=1,
                        help="number of channels of imgs")
    parser.add_argument('--norm', type=str, default='batch_norm',
                        help="batch_norm, layer_norm, or None")
    parser.add_argument('--num_filters', type=int, default=32,
                        help="number of filters for conv nets -- 32 for mini-imagenet, 64 for omiglot.")
    parser.add_argument('--max_pool', type=str, default='True',
                        help="Whether use max pooling rather than strided convolutions")

    # other arguments
    parser.add_argument('--logdir', type=str, required=False, default="./logs",
                        help='Log directory path')
    parser.add_argument('--log_file_name', type=str, default=None,
                        help='The log file name')
    parser.add_argument('--dataset', type=str, default='mnist',
                        help="name of dataset. (mnist, fmnist, cifar, seeships)")
    parser.add_argument('--num_classes', type=int, default=10,
                        help="number of classes")
    parser.add_argument('--gpu', default=None,
                        help="To use cuda, set to a specific GPU ID. Default set to use CPU.")
    parser.add_argument('--gpu_id', default=None,
                        help="gpu_id.")
    parser.add_argument('--optimizer', type=str, default='sgd',
                        help="type of optimizer")
    parser.add_argument('--iid', type=int, default=1,
                        help='Default set to IID. Set to 0 for non-IID.')
    parser.add_argument('--unequal', type=int, default=0,
                        help='whether to use unequal data splits for non-iid setting (use 0 for equal splits)')
    # parser.add_argument('--stopping_rounds', type=int, default=10,
    #                     help='rounds of early stopping')
    parser.add_argument('--verbose', type=int, default=1,
                        help='verbose')
    parser.add_argument('--init_seed', type=int, default=1,
                        help='random seed')

    args = parser.parse_args()

    return args
