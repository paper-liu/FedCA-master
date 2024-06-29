# -*- coding: utf-8 -*-
# Python version: 3.11

import os
import copy
import torch
import torch.nn.functional as F
from torchvision import datasets, transforms
from sampling import mnist_iid, mnist_noniid, mnist_noniid_unequal
from sampling import cifar_iid, cifar_noniid
import matplotlib
import matplotlib.pyplot as plt

import logging

logging.basicConfig()
logger = logging.getLogger()
logger.setLevel(logging.INFO)


def mkdirs(dirpath):
    try:
        os.makedirs(dirpath)
    except Exception as _:
        pass


def get_dataset(args):
    """
        Returns train and test datasets and a user group,
        user group is a dict where
        the keys are the user index and
        the values are the corresponding data for each of those users.
    """

    if args.dataset == 'cifar':
        data_dir = './data/cifar/'
        apply_transform = transforms.Compose(
            [transforms.ToTensor(),
             transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        train_dataset = datasets.CIFAR10(data_dir,
                                         train=True,
                                         download=True,
                                         transform=apply_transform)
        test_dataset = datasets.CIFAR10(data_dir,
                                        train=False,
                                        download=True,
                                        transform=apply_transform)

        # sample training data amongst users
        if args.iid:
            # Sample IID user data from CIFAR10
            user_groups, modified_user_groups, modified_dataset = cifar_iid(train_dataset, args.num_users, args.opt)
        else:
            if args.unequal:
                # Chose uneuqal splits for every user
                raise NotImplementedError()
            else:
                # Chose euqal splits for every user
                user_groups = cifar_noniid(train_dataset, args.num_users)

    elif args.dataset == 'fmnist':
        data_dir = './data/fashion_mnist/'
        apply_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.286], std=[0.352])])
        train_dataset = datasets.FashionMNIST(data_dir,
                                       train=True,
                                       download=True,
                                       transform=apply_transform)
        test_dataset = datasets.FashionMNIST(data_dir,
                                      train=False,
                                      download=True,
                                      transform=apply_transform)

        # sample training data amongst users
        if args.iid:
            # Sample IID user data from fmnist
            user_groups, modified_user_groups, modified_dataset = mnist_iid(train_dataset, args.num_users, args.opt)
        else:
            if args.unequal:
                # Chose uneuqal splits for every user
                user_groups = mnist_noniid_unequal(train_dataset, args.num_users)
            else:
                # Chose euqal splits for every user
                user_groups = mnist_noniid(train_dataset, args.num_users)

    elif args.dataset == 'mnist':
        data_dir = './data/mnist/'
        apply_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))])
        train_dataset = datasets.MNIST(data_dir,
                                       train=True,
                                       download=True,
                                       transform=apply_transform)
        test_dataset = datasets.MNIST(data_dir,
                                      train=False,
                                      download=True,
                                      transform=apply_transform)

        # sample training data amongst users
        if args.iid:
            # Sample IID user data from Mnist
            user_groups, modified_user_groups, modified_dataset = mnist_iid(train_dataset, args.num_users, args.opt)
        else:
            # Sample Non-IID user data from Mnist
            if args.unequal:
                # Chose uneuqal splits for every user
                user_groups = mnist_noniid_unequal(train_dataset, args.num_users)
            else:
                # Chose euqal splits for every user
                user_groups = mnist_noniid(train_dataset, args.num_users)

    return train_dataset, test_dataset, user_groups, modified_user_groups, modified_dataset


def average_weights(w):
    """
    Returns the average of the weights.
    """
    w_avg = copy.deepcopy(w[0])
    for key in w_avg.keys():
        for i in range(1, len(w)):
            w_avg[key] += w[i][key]
        w_avg[key] = torch.div(w_avg[key], len(w))
    return w_avg


def average_gradients(gradients_list):
    """
    Returns the average of the gradients.
    """
    # Compute the average of the gradients of the clients
    average_gradients = {}
    for grad in gradients_list:
        for name, param in grad.items():
            if name in average_gradients:
                average_gradients[name] += param
            else:
                average_gradients[name] = param.clone().detach()
    for name in average_gradients.keys():
        average_gradients[name] /= len(gradients_list)

    return average_gradients


def update_global_weights(args, global_model, gradients_list):
    """
    Update the global model weights
    :param global_model: The current global model
    :param gradients_list: A list of gradients collected from individual clients
    :return: The updated global model
    """

    # The average of all client gradients is calculated
    average_gradients = {}
    for grad in gradients_list:
        for name, param in grad.items():
            if name in average_gradients:
                average_gradients[name] += param
            else:
                average_gradients[name] = param.clone().detach()

    for name in average_gradients.keys():
        average_gradients[name] /= len(gradients_list)

    # The weights of the global model are updated using the average gradient
    for name, param in global_model.named_parameters():
        if name in average_gradients:
            lr = args.lr
            param.data -= lr * average_gradients[name]
    return global_model


# Show experiment details
def exp_details(args):
    print('\nExperimental details:')
    print(f'    Model     : {args.model}')
    print(f'    Optimizer : {args.optimizer}')
    print(f'    Learning  : {args.lr}')
    print(f'    Global Rounds   : {args.comm_round}\n')

    print('    Federated parameters:')
    if args.iid:
        print('    IID')
    else:
        print('    Non-IID')
    print(f'    Fraction of users  : {args.frac}')
    print(f'    Local Batch size   : {args.local_bs}')
    print(f'    Local Epochs       : {args.local_ep}\n')

    print('    FedCON parameters:')
    print(f'    Pre_training       : {args.pre_round}')
    print(f'    Get_gradients      : {args.get_gradients}')
    print(f'    Specical_process   : {args.opt}\n')

    return


# Check the gpu device
def gpu_is_available():
    print('\nGPU details:')
    print(f'    gpu_is_available      : ', torch.cuda.is_available())
    print(f'    cuda_device_count     : ', torch.cuda.device_count())
    print(f'    cuda_device_name      : ', torch.cuda.get_device_name())
    print(f'    cuda_device_capability: ', torch.cuda.get_device_capability(0))
    print('\n')

    return


# Output training state graph (training loss, training accuracy)
def show_training_state_diagram(args, train_loss, train_accuracy):

    matplotlib.use('Agg')

    # Plot Loss curve
    plt.figure()
    plt.title('Training Loss vs Communication rounds')
    plt.plot(range(len(train_loss)), train_loss, color='g')
    plt.ylabel('Training loss')
    plt.xlabel('Communication Rounds')
    plt.savefig('./save/fed_{}_{}_{}_C[{}]_iid[{}]_P[{}]_O[{}]_loss.png'.
                format(args.dataset,
                       args.model,
                       args.comm_round,
                       args.frac,
                       args.iid,
                       args.pre_round,
                       args.opt))

    # Plot Accuracy curve
    plt.figure()
    plt.title('Average Accuracy vs Communication rounds')
    plt.plot(range(len(train_accuracy)), train_accuracy, color='r')
    plt.ylabel('Average Accuracy')
    plt.xlabel('Communication Rounds')
    plt.savefig('./save/fed_{}_{}_{}_C[{}]_iid[{}]_P[{}]_O[{}]_acc.png'.
                format(args.dataset,
                       args.model,
                       args.comm_round,
                       args.frac,
                       args.iid,
                       args.pre_round,
                       args.opt))

    return


# Output the contribution graph
def show_final_contribution(args, contri_final):

    matplotlib.use('Agg')
    users_list, contri_list = list(contri_final.keys()), list(contri_final.values())

    plt.figure(figsize=(20, 10), dpi=200)
    plt.grid(axis="y", c='#d2c9eb', linestyle='--', zorder=0)
    plt.bar(range(len(users_list)), contri_list, label='Contribution', color='#ff9999', width=0.8,
            edgecolor='black', linewidth=2.0, zorder=10)
    plt.xticks(range(len(users_list)), users_list, fontproperties='Times New Roman', fontsize=20)
    plt.yticks(fontproperties='Times New Roman', fontsize=20)

    for i in range(len(users_list)):
        plt.text(x=i-0.05, y=contri_list[i]+0.002, s='%.6f'%contri_list[i],
                 ha='center', fontproperties='Times New Roman', fontsize=20, zorder=10)

    plt.ylim(0, 0.2)
    plt.title('Contributions Of Federated Users', fontproperties='Times New Roman', fontsize=40)
    plt.xlabel('Local Users', fontproperties='Times New Roman', fontsize=40)
    plt.ylabel('Contributions', fontproperties='Times New Roman', fontsize=40)

    plt.legend(prop={'family': 'Times New Roman', 'size': 25}, ncol=1)

    plt.savefig('./save/fed_{}_{}_{}_C[{}]_iid[{}]_P[{}]_O[{}]_contri.png'.
                format(args.dataset,
                       args.model,
                       args.comm_round,
                       args.frac,
                       args.iid,
                       args.pre_round,
                       args.opt))

    return


def calculate_similarity_distance(global_weights, local_weights, alpha=0.5):
    """
    Compute a combined similarity score for each layer between two sets of weights (global and local),
    considering both cosine similarity and Euclidean distance.

    :param global_weights: a dictionary of global model weights
    :param local_weights: a dictionary of local model weights
    :param alpha: the weight given to cosine similarity in the combined score, where (1 - alpha) is the weight for Euclidean distance
    :return: combined similarity score
    """
    cos_sim_dict = {}
    euc_dist_dict = {}

    for (global_name, global_weight), (local_name, local_weight) in zip(
            global_weights.items(), local_weights.items()):
        assert global_name == local_name  # Ensure the layers match

        # Flatten the weights
        global_weight_flat = global_weight.view(-1)
        local_weight_flat = local_weight.view(-1)

        # Calculate cosine similarity
        cos_sim = F.cosine_similarity(global_weight_flat.unsqueeze(0), local_weight_flat.unsqueeze(0)).item()
        cos_sim_dict[global_name] = cos_sim

        # Calculate Euclidean distance and normalize it
        euc_dist = torch.norm(global_weight_flat - local_weight_flat).item()
        # Normalize the Euclidean distance to be in range [0, 1] for consistency with cosine similarity
        max_dist = torch.norm(
            torch.ones_like(global_weight_flat) * torch.max(global_weight_flat.abs(), local_weight_flat.abs())).item()
        norm_euc_dist = euc_dist / max_dist
        euc_dist_dict[global_name] = norm_euc_dist

    # Average the cosine similarity and Euclidean distances across all layers
    cos_sim_avg = sum(cos_sim_dict.values()) / len(cos_sim_dict)
    euc_dist_avg = sum(euc_dist_dict.values()) / len(euc_dist_dict)

    # Combine the two metrics into a single similarity score
    combined_similarity = (alpha * cos_sim_avg) + (
                (1 - alpha) * (1 - euc_dist_avg))   # (1 - euc_dist_avg) to convert distance to similarity

    logger.info('cos_sim_dict: {}'.format(cos_sim_dict))
    logger.info('euc_dist_dict: {}'.format(euc_dist_dict))
    logger.info('cos_sim_avg: {}, euc_dist_avg: {}'.format(cos_sim_avg, euc_dist_avg))
    print('combined_similarity: {}'.format(combined_similarity))
    logger.info('combined_similarity: {}'.format(combined_similarity))

    return combined_similarity


# Calculate  model distance
def calculate_distance(global_weights, local_weights):

    euc_dist_dict = {}

    for (global_name, global_weight), (local_name, local_weight) in zip(
            global_weights.items(), local_weights.items()):
        assert global_name == local_name

        # Flatten the weights
        global_weight_flat = global_weight.view(-1)
        local_weight_flat = local_weight.view(-1)

        # Calculate Euclidean distance and normalize it
        euc_dist = torch.norm(global_weight_flat - local_weight_flat).item()
        max_dist = torch.norm(
            torch.ones_like(global_weight_flat) * torch.max(global_weight_flat.abs(), local_weight_flat.abs())).item()
        norm_euc_dist = euc_dist / max_dist
        euc_dist_dict[global_name] = norm_euc_dist

    euc_dist_avg = 1 - sum(euc_dist_dict.values()) / len(euc_dist_dict)

    logger.info('euc_dist_dict: {}'.format(euc_dist_dict))
    logger.info('euc_dist_avg: {}'.format(euc_dist_avg))

    return euc_dist_avg


# Calculate model similarity
def calculate_similarity(global_weights, local_weights):
    cos_sim_dict = {}

    for (global_name, global_weight), (local_name, local_weight) in zip(
            global_weights.items(), local_weights.items()):
        assert global_name == local_name

        # Flatten the weights
        global_weight_flat = global_weight.view(-1)
        local_weight_flat = local_weight.view(-1)

        # Calculate cosine similarity
        cos_sim = F.cosine_similarity(global_weight_flat.unsqueeze(0), local_weight_flat.unsqueeze(0)).item()
        cos_sim_dict[global_name] = cos_sim

    cos_sim_avg = sum(cos_sim_dict.values()) / len(cos_sim_dict)

    # sim_dict = {}
    # for (global_name, global_weight), (local_name, local_weight) in zip(
    #         global_weights.items(), local_weights.items()):
    #     # Make sure the layer names match
    #     assert global_name == local_name
    #     global_weight_flat = global_weight.view(-1)
    #     local_weight_flat = local_weight.view(-1)
    #     # Calculating similarity
    #     similarity = F.cosine_similarity(global_weight_flat.unsqueeze(0), local_weight_flat.unsqueeze(0))
    #     sim_dict[global_name] = similarity.item()
    # # print('\nsim_dict: {}'.format(sim_dict))
    # logger.info('sim_dict: {}'.format(sim_dict))
    # # Average the similarity of each layer
    # sim_avg = sum(sim_dict.values()) / len(sim_dict.items())
    # # print('sim_avg: {}'.format(sim_avg))
    # logger.info('sim_avg: {}'.format(sim_avg))

    logger.info('cos_sim_dict: {}'.format(cos_sim_dict))
    logger.info('cos_sim_avg: {}'.format(cos_sim_avg))

    return cos_sim_avg


# Calculate the contribution of the model
def calculate_contribution(global_weights, local_weights, users):

    # Similarity[]: Similarity list
    # Contribution: Contribution list
    Similarity = []
    # Contribution = []

    for i in range(len(local_weights)):
        logger.info('local user: {}'.format(users[i]))
        simi_local = calculate_similarity_distance(global_weights, local_weights[i])
        # Add the returned local model Similarity to the similarity list.
        Similarity.append(simi_local)

    ## # Normalize the similarity list # ##

    # simi_freqs: List of normalized weights for model similarity
    # if sum(Similarity) != 0:
    #     simi_freqs = [item / sum(Similarity) for item in Similarity]
    # else:
    #     simi_freqs = [1.0 for i in range(len(Similarity))]
    #
    # for j in range(len(Similarity)):
    #     Similarity[j] *= simi_freqs[j]

    # if sum(Similarity) != 0:
    #     Contribution = [item / sum(Similarity) for item in Similarity]
    # else:
    #     Contribution = [0.0 for i in range(len(Similarity))]

    print('Similarity: {}'.format(Similarity))
    Contribution = normalize_list_values(Similarity, 1)
    print('Contribution: {}'.format(Contribution))

    return Contribution


# Calculating final contribution
def calculate_final_contribution(contri_dict, contri_global_list, pre_round):

    contri_final_dict = {}
    round = 0

    # Iterate over the contribution dictionary
    for comm, contribution in contri_dict.items():
        if round < pre_round:
            round += 1
            logger.info('in pre_{}'.format(comm))
            continue

        logger.info('in {}'.format(comm))
        # Iterate over the warmed up user contribution dictionary
        # print('comm: {}, contribution: {}'.format(comm, contribution))
        for user, contri in contribution.items():
            # The user contribution is modified according to the corresponding weight in the global model weight list
            weighted_contri = contri * contri_global_list[0]
            # print('| user: {} | contri: {:.6f} | weighted_contri: {:.6f}'.format(user, contri, weighted_contri))
            logger.info('| user: {} | contri: {:.6f} | weighted_contri: {:.6f}'.format(user, contri, weighted_contri))
            # Sum up each client's contribution
            if user not in contri_final_dict:
                contri_final_dict[user] = 0.0
            contri_final_dict[user] += weighted_contri

        contri_global_list.pop(0)
        # print('contri_global_list: {}'.format(contri_global_list))

    # print('contri_final_dict: {}'.format(contri_final_dict))
    logger.info('contri_final_dict: {}'.format(contri_final_dict))
    normalized_contri_dict = normalize_dict_values(contri_final_dict)
    # print('normalized_contri_dict: {}'.format(normalized_contri_dict))
    logger.info('normalized_contri_dict: {}'.format(normalized_contri_dict))

    return normalized_contri_dict


# Regularize the values in the list
# def normalize_list_values(list_orig):
#
#     if sum(list_orig) != 0:
#         list_normal = [item / sum(list_orig) for item in list_orig]
#     else:
#         list_normal = [0.0 for i in range(len(list_orig))]
#
#     return list_normal


# Regularize the values in the list
def normalize_list_values(list_orig, max_value=1):

    # Make sure the list is not empty
    if not list_orig:
        return []

    min_orig = min(list_orig) / 1.1
    max_orig = max(list_orig)

    # Normalized to [0, 1]
    normalized = [(item - min_orig) / (max_orig - min_orig) if max_orig != min_orig else 1.0 for item in list_orig]

    # Calculate the adjustment factor such that the minimum is greater than 0
    min_adjustment = 1.0 / len(list_orig)
    adjusted = [item + min_adjustment for item in normalized]

    # Adjust the sum to max_value
    sum_adjusted = sum(adjusted)
    final_list = [item * max_value / sum_adjusted for item in adjusted]

    return final_list


# Normalize the list of global model similarities
def normalize_global_list_values(list_orig, max_value=100):

    if not list_orig:
        return []

    min_orig = min(list_orig) / 1.1
    max_orig = max(list_orig)
    normalized = [(item - min_orig) / (max_orig - min_orig) if max_orig != min_orig else 1.0 for item in list_orig]
    min_adjustment = 1.0 / len(list_orig)
    adjusted = [item + min_adjustment for item in normalized]
    sum_adjusted = sum(adjusted)
    final_list = [item * max_value / sum_adjusted for item in adjusted]

    return final_list


# Regularize the values in the dictionary
def normalize_dict_values(dict_orig):

    total_sum = sum(dict_orig.values())
    normalized_dict = {}

    for key, value in dict_orig.items():
        normalized_value = value / total_sum
        normalized_dict[key] = normalized_value

    return normalized_dict


# Calculate the average accuracy and loss of the user
def calculate_avg_acc_loss(acc_loss_dict):

    acc_loss_avg = {}
    for user, acc_loss_list in acc_loss_dict.items():
        acc_loss_avg[user] = sum(acc_loss_list) / len(acc_loss_list)

    return acc_loss_avg
