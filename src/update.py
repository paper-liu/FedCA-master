# -*- coding: utf-8 -*-
# Python version: 3.11

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

import logging

logging.basicConfig()
logger = logging.getLogger()
logger.setLevel(logging.INFO)

class DatasetSplit(Dataset):
    """
    An abstract Dataset class wrapped around Pytorch Dataset class.
    """

    def __init__(self, dataset, idxs):
        self.dataset = dataset
        self.idxs = [int(i) for i in idxs]

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        image, label = self.dataset[self.idxs[item]]
        return torch.tensor(image), torch.tensor(label)


# Manage training and inference for local clients
class LocalUpdate(object):
    # Initialize an instance of the LocalUpdate class
    def __init__(self, args, dataset, idxs, logger):
        self.args = args
        self.logger = logger
        self.trainloader, self.validloader, self.testloader = self.train_val_test(dataset, list(idxs))
        self.device = 'cuda' if args.gpu else 'cpu'
        if args.loss == 'NLLLoss':
            # Default criterion set to NLL loss function 默认使用负对数似然损失
            self.criterion = nn.NLLLoss().to(self.device)
            # self.logger.info('Using NLLLoss.')
        else:
            # 当数据集为 cifar 时使用交叉熵损失函数
            self.criterion = nn.CrossEntropyLoss().to(self.device)
            # self.logger.info('Using CrossEntropyLoss.')
        # self.logger.info(f'Loss function set to: {self.criterion}')

    def train_val_test(self, dataset, idxs):
        """
        Returns train, validation and test dataloaders for a given dataset
        and user indexes.
        """
        # split indexes for train, validation, and test (80, 10, 10)
        idxs_train = idxs[:int(0.8*len(idxs))]
        idxs_val = idxs[int(0.8*len(idxs)):int(0.9*len(idxs))]
        idxs_test = idxs[int(0.9*len(idxs)):]

        trainloader = DataLoader(DatasetSplit(dataset, idxs_train),
                                 batch_size=self.args.local_bs,
                                 shuffle=True)
        validloader = DataLoader(DatasetSplit(dataset, idxs_val),
                                 batch_size=int(len(idxs_val)/10),
                                 shuffle=False)
        testloader = DataLoader(DatasetSplit(dataset, idxs_test),
                                batch_size=int(len(idxs_test)/10),
                                shuffle=False)

        return trainloader, validloader, testloader

    def get_gradients(self, model):
        gradients = {}
        for name, parameter in model.named_parameters():
            # 确保只获取需要梯度的参数
            if parameter.requires_grad:
                # 使用.clone()来确保梯度是复制品
                gradients[name] = parameter.grad.clone().detach()
        return gradients

    # Perform model training on the local client
    def update_weights(self, model, global_round):
        # Set mode to train model
        model.train()
        epoch_loss = []

        # Set optimizer for the local updates
        if self.args.optimizer == 'sgd':
            optimizer = torch.optim.SGD(model.parameters(),
                                        lr=self.args.lr,
                                        momentum=0.5)
        elif self.args.optimizer == 'adam':
            optimizer = torch.optim.Adam(model.parameters(),
                                         lr=self.args.lr,
                                         weight_decay=1e-4)

        ## # Training locally on the client # ##

        for iter in range(self.args.local_ep):
            batch_loss = []
            # Iterate over each batch in the training data loader for the local client
            for batch_idx, (images, labels) in enumerate(self.trainloader):
                images, labels = images.to(self.device), labels.to(self.device)

                model.zero_grad()
                log_probs = model(images)
                loss = self.criterion(log_probs, labels)
                loss.backward()
                optimizer.step()

                # if self.args.verbose and (batch_idx % 10 == 0):
                #     print('| Global Round : {} | Local Epoch : {} | [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                #         global_round,
                #         iter,
                #         batch_idx * len(images),
                #         len(self.trainloader.dataset),
                #         100. * batch_idx / len(self.trainloader),
                #         loss.item()))

                # self.logger.add_scalar('loss', loss.item())
                batch_loss.append(loss.item())
            epoch_loss.append(sum(batch_loss)/len(batch_loss))
            print('| Global Round : {} | Local Epoch : {} | Loss: {:.6f}'.format(global_round, iter, epoch_loss[-1]))
            logger.info('| Global Round : {} | Local Epoch : {} | Loss: {:.6f}'.format(global_round, iter, epoch_loss[-1]))

        # 在最后一个epoch结束时获取梯度
        local_gradients = self.get_gradients(model)

        return model, model.state_dict(), local_gradients, sum(epoch_loss) / len(epoch_loss)

    # Perform inference operations on the model
    def inference(self, model):
        """
        Returns the inference accuracy and loss.
        """
        model.eval()
        # loss: cumulative loss value
        # total: total number of samples
        # correct: the number of correctly classified samples
        loss, total, correct = 0.0, 0.0, 0.0

        for batch_idx, (images, labels) in enumerate(self.testloader):
            images, labels = images.to(self.device), labels.to(self.device)

            # Inference

            outputs = model(images)
            batch_loss = self.criterion(outputs, labels)
            loss += batch_loss.item()

            # Prediction

            _, pred_labels = torch.max(outputs, 1)
            pred_labels = pred_labels.view(-1)
            correct += torch.sum(torch.eq(pred_labels, labels)).item()
            total += len(labels)

        accuracy = correct/total

        return accuracy, loss


# Perform the inference operations of the model on the test data
def test_inference(args, model, test_dataset):
    """
    Returns the test accuracy and loss.
    """
    model.eval()
    # loss: cumulative loss value
    # total: total number of samples
    # correct: the number of correctly classified samples
    loss, total, correct = 0.0, 0.0, 0.0

    device = 'cuda' if args.gpu else 'cpu'
    if args.loss == 'NLLLoss':
        criterion = nn.NLLLoss().to(device)
    else:
        criterion = nn.CrossEntropyLoss().to(device)
    # criterion = nn.NLLLoss().to(device)
    testloader = DataLoader(test_dataset,
                            batch_size=128,
                            shuffle=False)

    for batch_idx, (images, labels) in enumerate(testloader):
        images, labels = images.to(device), labels.to(device)

        # Inference

        outputs = model(images)
        batch_loss = criterion(outputs, labels)
        loss += batch_loss.item()

        # Prediction

        _, pred_labels = torch.max(outputs, 1)
        pred_labels = pred_labels.view(-1)
        correct += torch.sum(torch.eq(pred_labels, labels)).item()
        total += len(labels)

    accuracy = correct/total

    return accuracy, loss
