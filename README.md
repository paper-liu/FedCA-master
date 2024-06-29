# FedCA (PyTorch)

This experiment is based on the thesis: [FedCA]()

The experiments were conducted on MNIST, Fashion MNIST and CIFAR10 (both IID and non-IID) datasets, 
and in the non-IID case, the data can be split evenly or unevenly between users, 
as the purpose of the experiments was to demonstrate the effectiveness of measuring user contributions with FedCON, 
so only a simple model was used as an example.


## Requirments
Install all the packages from requirments.txt
* Python3
* Pytorch
* Torchvision

## Data
* Download train and test datasets manually or they will be automatically downloaded from torchvision datasets.
* Experiments are run on Mnist, Fashion Mnist and Cifar.
* To use your own dataset: Move your dataset to data directory and write a wrapper on pytorch dataset class.

## Running the experiments
The baseline experiment trains the model in the conventional way.

* To run the baseline experiment with MNIST on MLP using CPU:
```
python src/baseline_main.py --model=mlp --dataset=mnist --epochs=10
```
* Or to run it on GPU (eg: if gpu:0 is available):
```
python src/baseline_main.py --model=mlp --dataset=mnist --gpu=0 --epochs=10
```

-----

Federated experiment involves training a global model using many local models.

* To run the federated experiment with CIFAR on CNN (IID):
```
python src/federated_main.py --model=cnn --dataset=mnist --gpu=0 --iid=1 --epochs=10
```
* To run the same experiment under non-IID condition:
```
python src/federated_main.py --model=cnn --dataset=mnist --gpu=0 --iid=0 --epochs=10
```
-----

You can change the default values of other parameters to simulate different conditions. Refer to the options section.

* cnn / mnist / gpu / non-iid / comm_round=10 / num_users=10 / frac=0.5 / local_ep=10
```
python src/federated_main.py --model=cnn --dataset=mnist --gpu=0 --iid=0 --comm_round=10 --num_users=10 --frac=0.5 --local_ep=10
```
