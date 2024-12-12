# -*- coding: utf-8 -*-
# Python version: 3.11


import numpy as np

import skimage
import torchvision
import random
import torch
from torchvision import datasets, transforms


def specical_processing(dict_users, dataset, num_users, opt):

    # Create a new list to store the modified dataset
    modified_dataset = []
    # Create a new dictionary to store the modified users
    modified_dict_users = {}

    # Reduce the amount of data for the last two users to 50% of the original
    if opt == 'less':
        # ll = len(dict_users[0]) * 0.2
        # for i in range(num_users - 2, num_users):
        #     while len(dict_users[i]) > ll:
        #         dict_users[i].pop()

        for user_id in [num_users - 2, num_users - 1]:
            for idx in dict_users[user_id]:
                modified_dataset.append(dataset[idx])

        # Redistributing data
        modified_num_items = int(len(dict_users[0]) * 0.5)
        modified_dict_users, modified_all_idxs = {}, [i for i in range(len(modified_dataset))]
        for user_id in [num_users - 2, num_users - 1]:
            modified_dict_users[user_id] = set(
                np.random.choice(modified_all_idxs, modified_num_items, replace=False))
            modified_all_idxs = list(set(modified_all_idxs) - modified_dict_users[user_id])

    # elif opt == 'less_rank':
    #     ll = len(dict_users[0]) * 0.50
    #     for i in range(num_users - 4, num_users - 2):
    #         while len(dict_users[i]) > ll:
    #             dict_users[i].pop()
    #     ll = len(dict_users[0]) * 0.30
    #     for i in range(num_users - 2, num_users):
    #         while len(dict_users[i]) > ll:
    #             dict_users[i].pop()

    # Random noise is applied to the data of the last two users
    elif opt == 'noise':

        # Add salt and pepper noise to the image
        def add_salt_pepper_noise(item, salt_prob=0.01, pepper_prob=0.01):
            """
            :param item: A tuple containing an image x and a label y
            :param salt_prob: Probability of salt (white dot) noise
            :param pepper_prob: Pepper (black dot) noise probability
            :return: Image and original labels after adding salt and pepper noise
            """
            x, y = item
            # Create a random matrix of the same size as the original image
            noise = torch.rand_like(x)

            # Adding salt noise
            salt_mask = noise < salt_prob
            noisy_x = torch.where(salt_mask, torch.ones_like(x), x)

            # Add pepper noise
            pepper_mask = noise < pepper_prob
            noisy_x = torch.where(pepper_mask, torch.zeros_like(x), noisy_x)

            return noisy_x, y

        # Add Gaussian noise to the image
        def add_gaussian_noise(item, mean=0.0, std=1.0):
            x, y = item
            # Gaussian noise with mean and standard deviation std is generated
            noise = torch.randn_like(x) * std + mean
            noisy_x = x + noise
            return noisy_x, y

        # Add noise to user-specific data (last two users)
        for user_id in [num_users - 2, num_users - 1]:
            for idx in dict_users[user_id]:
                original_item = dataset[idx]
                # Gaussian noise
                noisy_item = add_gaussian_noise(original_item, mean=0.5, std=1.5)
                # salt and pepper noise
                # noisy_item = add_salt_pepper_noise(original_item, salt_prob=0.30, pepper_prob=0.30)
                modified_dataset.append(noisy_item)

        # Redistribute the noisy data
        modified_num_items = int(len(modified_dataset) / 2)
        modified_dict_users, modified_all_idxs = {}, [i for i in range(len(modified_dataset))]
        for user_id in [num_users - 2, num_users - 1]:
            modified_dict_users[user_id] = set(np.random.choice(modified_all_idxs, modified_num_items, replace=False))
            modified_all_idxs = list(set(modified_all_idxs) - modified_dict_users[user_id])

    elif opt == 'gaussian_noise':
        def add_gaussian_noise(item, mean=0.0, std=1.0):
            x, y = item
            noise = torch.randn_like(x) * std + mean
            noisy_x = x + noise
            return noisy_x, y

        for user_id in [num_users - 2, num_users - 1]:
            for idx in dict_users[user_id]:
                original_item = dataset[idx]
                noisy_item = add_gaussian_noise(original_item, mean=0.5, std=1.5)
                modified_dataset.append(noisy_item)
        modified_num_items = int(len(modified_dataset) / 2)
        modified_dict_users, modified_all_idxs = {}, [i for i in range(len(modified_dataset))]
        for user_id in [num_users - 2, num_users - 1]:
            modified_dict_users[user_id] = set(np.random.choice(modified_all_idxs, modified_num_items, replace=False))
            modified_all_idxs = list(set(modified_all_idxs) - modified_dict_users[user_id])

    elif opt == 'salt_pepper_noise':
        def add_salt_pepper_noise(item, salt_prob=0.01, pepper_prob=0.01):
            x, y = item
            noise = torch.rand_like(x)
            salt_mask = noise < salt_prob
            noisy_x = torch.where(salt_mask, torch.ones_like(x), x)
            pepper_mask = noise < pepper_prob
            noisy_x = torch.where(pepper_mask, torch.zeros_like(x), noisy_x)
            return noisy_x, y

        for user_id in [num_users - 2, num_users - 1]:
            for idx in dict_users[user_id]:
                original_item = dataset[idx]
                noisy_item = add_salt_pepper_noise(original_item, salt_prob=0.30, pepper_prob=0.30)
                modified_dataset.append(noisy_item)
        modified_num_items = int(len(modified_dataset) / 2)
        modified_dict_users, modified_all_idxs = {}, [i for i in range(len(modified_dataset))]
        for user_id in [num_users - 2, num_users - 1]:
            modified_dict_users[user_id] = set(np.random.choice(modified_all_idxs, modified_num_items, replace=False))
            modified_all_idxs = list(set(modified_all_idxs) - modified_dict_users[user_id])

    # The labels of the last two user data are randomly changed to introduce label errors
    elif opt == 'mislabel':

        def mislabel(item):
            x, y = item
            random_label = random.randint(0, 9)
            return x, random_label

        # The label is randomly changed for each data item in the data set of the last two users
        for user_id in [num_users - 2, num_users - 1]:
            for idx in dict_users[user_id]:
                original_item = dataset[idx]
                mislabeled_item = mislabel(original_item)
                modified_dataset.append(mislabeled_item)

        # Redistribute the noisy data
        modified_num_items = int(len(modified_dataset) / 2)
        modified_dict_users, modified_all_idxs = {}, [i for i in range(len(modified_dataset))]
        for user_id in [num_users - 2, num_users - 1]:
            modified_dict_users[user_id] = set(np.random.choice(modified_all_idxs, modified_num_items, replace=False))
            modified_all_idxs = list(set(modified_all_idxs) - modified_dict_users[user_id])

    # Random noise is applied to the data of the last three or four users,
    # and the labels of the data of the last two users are randomly changed
    elif opt == 'noise_mislabel':

        def add_gaussian_noise(item, mean=0.0, std=1.0):
            x, y = item
            noise = torch.randn_like(x) * std + mean
            noisy_x = x + noise
            return noisy_x, y

        def mislabel(item):
            x, y = item
            random_label = random.randint(0, 9)
            return x, random_label

        for user_id in [num_users-4, num_users-3, num_users-2, num_users-1]:  # 6 7 8 9
            if user_id < num_users-2:  # 6 7
                for idx in dict_users[user_id]:
                    original_item = dataset[idx]
                    noisy_item = add_gaussian_noise(original_item, mean=0.5, std=1.5)
                    modified_dataset.append(noisy_item)
            else:  # 8 9
                for idx in dict_users[user_id]:
                    original_item = dataset[idx]
                    mislabeled_item = mislabel(original_item)
                    modified_dataset.append(mislabeled_item)

        modified_num_items = int(len(modified_dataset) / 4)
        modified_noise_idxs = [i for i in range(2*modified_num_items)]
        modified_mislabel_idxs = [j+2*modified_num_items for j in modified_noise_idxs]

        for user_id in [num_users-4, num_users-3, num_users-2, num_users-1]:
            if user_id < num_users-2:  # 6-7
                modified_dict_users[user_id] = set(
                    np.random.choice(modified_noise_idxs, modified_num_items, replace=False))
                modified_noise_idxs = list(set(modified_noise_idxs) - modified_dict_users[user_id])
            else:  # 8-9
                modified_dict_users[user_id] = set(
                    np.random.choice(modified_mislabel_idxs, modified_num_items, replace=False))
                modified_mislabel_idxs = list(set(modified_mislabel_idxs) - modified_dict_users[user_id])

    # No special data processing is performed
    elif opt == 'normal':
        pass
    else:
        print('No such option')
        exit(1)

    return modified_dict_users, modified_dataset


def mnist_iid(dataset, num_users, opt='normal'):
    """
    Sample I.I.D. client data from MNIST dataset
    :param dataset:
    :param num_users:
    :return: dict of image index
    """
    num_items = int(len(dataset)/num_users)
    dict_users, all_idxs = {}, [i for i in range(len(dataset))]
    for i in range(num_users):
        dict_users[i] = set(np.random.choice(all_idxs, num_items, replace=False))
        all_idxs = list(set(all_idxs) - dict_users[i])

    # modified_dataset: A list to store the modified dataset
    # modified_dict_users: A dictionary to keep track of modified users and user data
    modified_dict_users, modified_dataset = specical_processing(dict_users, dataset, num_users, opt)

    return dict_users, modified_dict_users, modified_dataset


def mnist_noniid(dataset, num_users):
    """
    Sample non-I.I.D client data from MNIST dataset
    :param dataset:
    :param num_users:
    :return:
    """
    # 60,000 training imgs -->  200 imgs/shard X 300 shards
    num_shards, num_imgs = 200, 300
    idx_shard = [i for i in range(num_shards)]
    dict_users = {i: np.array([]) for i in range(num_users)}
    idxs = np.arange(num_shards*num_imgs)
    labels = dataset.train_labels.numpy()

    # sort labels
    idxs_labels = np.vstack((idxs, labels))
    idxs_labels = idxs_labels[:, idxs_labels[1, :].argsort()]
    idxs = idxs_labels[0, :]

    # divide and assign 2 shards/client
    for i in range(num_users):
        rand_set = set(np.random.choice(idx_shard, 2, replace=False))
        idx_shard = list(set(idx_shard) - rand_set)
        for rand in rand_set:
            dict_users[i] = np.concatenate(
                (dict_users[i], idxs[rand*num_imgs:(rand+1)*num_imgs]), axis=0)
    return dict_users


def mnist_noniid_unequal(dataset, num_users):
    """
    Sample non-I.I.D client data from MNIST dataset s.t clients
    have unequal amount of data
    :param dataset:
    :param num_users:
    :returns a dict of clients with each clients assigned certain
    number of training imgs
    """
    # 60,000 training imgs --> 50 imgs/shard X 1200 shards
    num_shards, num_imgs = 1200, 50
    idx_shard = [i for i in range(num_shards)]
    dict_users = {i: np.array([]) for i in range(num_users)}
    idxs = np.arange(num_shards*num_imgs)
    labels = dataset.train_labels.numpy()

    # sort labels
    idxs_labels = np.vstack((idxs, labels))
    idxs_labels = idxs_labels[:, idxs_labels[1, :].argsort()]
    idxs = idxs_labels[0, :]

    # Minimum and maximum shards assigned per client:
    min_shard = 1
    max_shard = 30

    # Divide the shards into random chunks for every client
    # s.t the sum of these chunks = num_shards
    random_shard_size = np.random.randint(min_shard, max_shard+1,
                                          size=num_users)
    random_shard_size = np.around(random_shard_size /
                                  sum(random_shard_size) * num_shards)
    random_shard_size = random_shard_size.astype(int)

    # Assign the shards randomly to each client
    if sum(random_shard_size) > num_shards:

        for i in range(num_users):
            # First assign each client 1 shard to ensure every client has
            # atleast one shard of data
            rand_set = set(np.random.choice(idx_shard, 1, replace=False))
            idx_shard = list(set(idx_shard) - rand_set)
            for rand in rand_set:
                dict_users[i] = np.concatenate(
                    (dict_users[i], idxs[rand*num_imgs:(rand+1)*num_imgs]),
                    axis=0)

        random_shard_size = random_shard_size-1

        # Next, randomly assign the remaining shards
        for i in range(num_users):
            if len(idx_shard) == 0:
                continue
            shard_size = random_shard_size[i]
            if shard_size > len(idx_shard):
                shard_size = len(idx_shard)
            rand_set = set(np.random.choice(idx_shard, shard_size,
                                            replace=False))
            idx_shard = list(set(idx_shard) - rand_set)
            for rand in rand_set:
                dict_users[i] = np.concatenate(
                    (dict_users[i], idxs[rand*num_imgs:(rand+1)*num_imgs]),
                    axis=0)
    else:

        for i in range(num_users):
            shard_size = random_shard_size[i]
            rand_set = set(np.random.choice(idx_shard, shard_size,
                                            replace=False))
            idx_shard = list(set(idx_shard) - rand_set)
            for rand in rand_set:
                dict_users[i] = np.concatenate(
                    (dict_users[i], idxs[rand*num_imgs:(rand+1)*num_imgs]),
                    axis=0)

        if len(idx_shard) > 0:
            # Add the leftover shards to the client with minimum images:
            shard_size = len(idx_shard)
            # Add the remaining shard to the client with lowest data
            k = min(dict_users, key=lambda x: len(dict_users.get(x)))
            rand_set = set(np.random.choice(idx_shard, shard_size,
                                            replace=False))
            idx_shard = list(set(idx_shard) - rand_set)
            for rand in rand_set:
                dict_users[k] = np.concatenate(
                    (dict_users[k], idxs[rand*num_imgs:(rand+1)*num_imgs]),
                    axis=0)

    return dict_users


def cifar_iid(dataset, num_users, opt='normal'):
    """
    Sample I.I.D. client data from CIFAR10 dataset
    :param dataset:
    :param num_users:
    :return: dict of image index
    """
    num_items = int(len(dataset)/num_users)
    # dict_users: a dictionary to store users and their corresponding image indexes
    dict_users, all_idxs = {}, [i for i in range(len(dataset))]
    for i in range(num_users):
        dict_users[i] = set(np.random.choice(all_idxs,
                                             num_items,
                                             replace=False))
        all_idxs = list(set(all_idxs) - dict_users[i])

    modified_dict_users, modified_dataset = specical_processing(dict_users, dataset, num_users, opt)

    return dict_users, modified_dict_users, modified_dataset


def cifar_noniid(dataset, num_users):
    """
    Sample non-I.I.D client data from CIFAR10 dataset
    :param dataset:
    :param num_users:
    :return:
    """

    num_shards, num_imgs = 200, 250
    idx_shard = [i for i in range(num_shards)]
    dict_users = {i: np.array([]) for i in range(num_users)}

    idxs = np.arange(num_shards*num_imgs)
    labels = np.array(dataset.train_labels)

    # sort labels
    idxs_labels = np.vstack((idxs, labels))
    idxs_labels = idxs_labels[:, idxs_labels[1, :].argsort()]
    idxs = idxs_labels[0, :]

    # divide and assign
    for i in range(num_users):
        rand_set = set(np.random.choice(idx_shard, 2, replace=False))
        idx_shard = list(set(idx_shard) - rand_set)
        for rand in rand_set:
            dict_users[i] = np.concatenate(
                (dict_users[i], idxs[rand*num_imgs:(rand+1)*num_imgs]), axis=0)
    return dict_users
