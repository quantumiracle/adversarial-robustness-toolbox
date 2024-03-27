"""
This is an example of how to use ART for adversarial training of a model with Fast is better than free protocol
"""
import math
from PIL import Image
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
import argparse

from art.estimators.classification import PyTorchClassifier
from art.data_generators import PyTorchDataGenerator
from art.defences.trainer import AdversarialTrainerBAPyTorch
from art.utils import load_mnist
from art.attacks.evasion import ProjectedGradientDescent, CarliniL2Method, Wasserstein, AutoProjectedGradientDescent, BayesianAdversary
from adversarial_training_BA_mnist import PreActResNet18, initialize_weights, MNIST_dataset



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--delta_coeff", default=1.25, type=float)
    parser.add_argument('--model_dir', type=str, default='log/', help='save result logs')
    parser.add_argument('--method', type=str, default='fbf', help='save result logs')
    parser.add_argument("--attack_type", default='pgd', type=str)  # ['pgd', 'carlini', 'wasserstein', 'auto_pgd']
    parser.add_argument("--attack_eps", default=0.1, type=float)
    args = parser.parse_args()

    # Step 1: Load the MNIST dataset
    (x_train, y_train), (x_test, y_test), min_pixel_value, max_pixel_value = load_mnist()

    x_train = x_train.transpose(0, 3, 1, 2).astype("float32")
    x_test = x_test.transpose(0, 3, 1, 2).astype("float32")

    transform = transforms.Compose(
        # [transforms.RandomCrop(28, padding=4), transforms.RandomHorizontalFlip(), transforms.ToTensor()]
        [transforms.ToTensor()]
    )

    dataset = MNIST_dataset(x_train, y_train, transform=transform)
    dataloader = DataLoader(dataset, batch_size=128, shuffle=True)

    # Step 2: create the PyTorch model
    model = PreActResNet18(in_channel=1)
    # For running on GPU replace the model with the
    # model = PreActResNet18().cuda()

    model.apply(initialize_weights)
    model.eval()

    opt = torch.optim.SGD(model.parameters(), lr=0.21, momentum=0.9, weight_decay=5e-4)

    # if you have apex installed, the following line should be uncommented for faster processing
    # import apex.amp as amp
    # model, opt = amp.initialize(model, opt, opt_level="O2", loss_scale=1.0, master_weights=False)

    criterion = nn.CrossEntropyLoss(reduction='none')
    # Step 3: Create the ART classifier


    mnist_mu = np.ones((1, 28, 28))
    mnist_mu[0, :, :] = 0.1307

    mnist_std = np.ones((1, 28, 28))
    mnist_std[0, :, :] = 0.3081


    classifier = PyTorchClassifier(
        model=model,
        clip_values=(0.0, 1.0),
        preprocessing=(mnist_mu, mnist_std),
        loss=criterion,
        optimizer=opt,
        input_shape=(1, 28, 28),
        nb_classes=10,
    )

    classifier.load(filename=f'{args.method}_mnist', path=args.model_dir)
    print(f"Load model from {args.model_dir}/{args.method}_mnist")

    x_test_pred = np.argmax(classifier.predict(x_test), axis=1)
    log_entry = ""
    prt1 = f"Accuracy on benign test samples after adversarial training: \
             {(np.sum(x_test_pred == np.argmax(y_test, axis=1)) / x_test.shape[0] * 100):.2f}"
    print(prt1)
    log_entry += prt1

    attack_eps = 0.1
    
    # attack_types = ['pgd', 'carlini', 'wasserstein', 'auto_pgd', 'ba']
    # for attack_type in attack_types:
    if args.attack_type == 'pgd':
        attack = ProjectedGradientDescent(
            classifier,
            norm=np.inf,
            eps=attack_eps,
            eps_step=0.01,
            max_iter=40,
            targeted=False,
            num_random_init=5,
            batch_size=32,
        )
        
    elif args.attack_type == 'carlini':
        attack = CarliniL2Method(
            classifier, 
            confidence=0.0, 
            targeted=False, 
            learning_rate=0.01, 
            max_iter=10, 
            binary_search_steps=10, 
            initial_const=0.01, 
            max_halving=5, 
            max_doubling=5, 
            batch_size=32)

    elif args.attack_type == 'wasserstein':
        attack = Wasserstein(
            classifier, 
            eps=attack_eps,
            eps_step=0.01,
            max_iter=40,
            conjugate_sinkhorn_max_iter=40,
            projected_sinkhorn_max_iter=40,
            batch_size=32
            )
    elif args.attack_type == 'auto_pgd':
        attack = AutoProjectedGradientDescent(
            classifier,
            norm=np.inf,
            eps=attack_eps,
            eps_step=0.01,
            max_iter=40,
            targeted=False,
            nb_random_init=5,
            batch_size=32,
        )
    elif args.attack_type == 'ba':
        attack = BayesianAdversary(
            classifier,
            norm=2,  # inf norm takes sign of grad
            eps=args.attack_eps,
            eps_step=0.01,
            max_iter=40,
            targeted=False,
            num_random_init=5,
            batch_size=32,
            mean_attack=False,
        )
    elif args.attack_type == 'ba_mean':
        attack = BayesianAdversary(
            classifier,
            norm=2,  # inf norm takes sign of grad
            eps=args.attack_eps,
            eps_step=0.01,
            max_iter=40,
            targeted=False,
            num_random_init=5,
            batch_size=32,
            mean_attack=True, # if True, take mean of inner loop as attacking perturbation
        )

    x_test_attack = attack.generate(x_test)
    x_test_attack_pred = np.argmax(classifier.predict(x_test_attack), axis=1)
    prt2 = f"Accuracy on {args.attack_type} adversarial samples after adversarial training: \
        {(np.sum(x_test_attack_pred == np.argmax(y_test, axis=1)) / x_test.shape[0] * 100):.2f}"
    print(prt2)
    log_entry += '\n' + prt2
        
    # log result
    log_file_path = os.path.join(args.model_dir, f'attack_type={args.attack_type}_attack_eps={attack_eps}_delta_coeff={args.delta_coeff}.log')

    # Append the log entry to the file
    with open(log_file_path, "a") as log_file:
        log_file.write(f"{log_entry}\n")

