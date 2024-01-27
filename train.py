################################################################################
# MIT License
#
# Copyright (c) 2022 University of Amsterdam
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to conditions.
#
# Author: Deep Learning Course (UvA) | Fall 2022
# Date Created: 2022-11-14
################################################################################

import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data
import torchvision.models as models

from cifar100_utils import get_train_validation_set, get_test_set


def set_seed(seed):
    """
    Function for setting the seed for reproducibility.
    """
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_model(num_classes=100):
    """
    Returns a pretrained ResNet18 on ImageNet with the last layer
    replaced by a linear layer with num_classes outputs.
    Args:
        num_classes: Number of classes for the final layer (for CIFAR100 by default 100)
    Returns:
        model: nn.Module object representing the model architecture.
    """

    # Get the pretrained ResNet18 model on ImageNet from torchvision.models
    model = models.resnet18(weights="IMAGENET1K_V1")

    if not model.fc:
        model.params.requires_grad = False
    
      
    # Randomly initialize and modify the model's last layer for CIFAR100.
    ## as the only fc layer in resnet is the last one we can just replace it by calling model.fc.
    ## If we did not know that that was the only fc we would have to inspect the model structure to ensure that was infact the right layer.
    
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)

    ## randomly initialize weights with mean = 0 and sigma = 0.01 and bias to zero
    nn.init.normal_(model.fc.weight, mean=0, std=0.01)
    nn.init.constant_(model.fc.bias, 0)


    return model


def train_model(model, lr, batch_size, epochs, data_dir, checkpoint_name, device, augmentation_name=None):
    """
    Trains a given model architecture for the specified hyperparameters.

    Args:
        model: Model to train.
        lr: Learning rate to use in the optimizer.
        batch_size: Batch size to train the model with.
        epochs: Number of epochs to train the model for.
        data_dir: Directory where the dataset should be loaded from or downloaded to.
        checkpoint_name: Filename to save the best model on validation.
        device: Device to use.
        augmentation_name: Augmentation to use for training.
    Returns:
        model: Model that has performed best on the validation set.
    """

    # Load the datasets
    print(augmentation_name, "augmentation name")
    train_dataset, val_dataset = get_train_validation_set(data_dir, validation_size=5000, augmentation_name=augmentation_name)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)


    # Initialize the optimizer (Adam) to train the last layer of the model.
    optimizer = torch.optim.Adam(model.fc.parameters() , lr)
    criterion = nn.CrossEntropyLoss()
    model.to(device)

    print(f"Training model on: {device}")

    # Training loop with validation after each epoch. Save the best model.
    train_loss = np.zeros(epochs)
    ## if you want to plot val_loss, then uncomment the next line and then add some more lines to store the loss at each epoch
    # val_loss = np.zeros(epochs)

    val_accuracies = np.zeros(epochs) 

    ## initialize the best model and its best accuracy for later testing
    best_val_accuracy = 0

    for epoch in range(epochs):

        import time
        start_time = time.time()

        train_iter_loss=np.zeros(len(train_dataset))

        model.train()
        
        print(f"epoch: {epoch}")
        
        for batch_idx, data in enumerate(train_loader):

            inputs, labels = data

            inputs = inputs.to(device)
            labels = labels.to(device)

            # zero out the gradients in order to update them iteration by iteration. This is done to avoid summing up gradients in different iterations
            optimizer.zero_grad()

            predictions = model(inputs)
            
            # calculate the loss
            loss = criterion(predictions, labels)

            # backpropagate the loss to calculate gradients
            loss.backward()

            train_iter_loss[batch_idx] = loss.item()
            
            # update the weights
            optimizer.step()
        

        train_loss[epoch] = train_iter_loss.mean()



        val_accuracies[epoch] = evaluate_model(model, val_loader, device)
        

        if val_accuracies[epoch] > best_val_accuracy:
          best_val_accuracy = val_accuracies[epoch]
          torch.save(model.state_dict(), checkpoint_name)

        
        print(time.time() - start_time)
        print("Training epoch: {} \t Training loss: {} \t Validation accuracy: {}".format(epoch, train_loss[epoch], val_accuracies[epoch]))


    # Load the best model on val accuracy and return it.
    model.load_state_dict(torch.load(checkpoint_name))

    print(f"Best validation accuracy: {best_val_accuracy}")
    

    return model


def evaluate_model(model, data_loader, device):
    """
    Evaluates a trained model on a given dataset.

    Args:
        model: Model architecture to evaluate.
        data_loader: The data loader of the dataset to evaluate on.
        device: Device to use for training.
    Returns:
        accuracy: The accuracy on the dataset.

    """

    # Set model to evaluation mode (Remember to set it back to training mode in the training loop)
    model.eval()
    model.to(device)
    
    # Loop over the dataset and compute the accuracy. Return the accuracy
    # Remember to use torch.no_grad().
    
    ## if you would like to store loss to plot uncomment these lines and change the function structure so it returns valid_iter_loss 
    # criterion = nn.CrossEntropyLoss()
    # valid_iter_loss=np.zeros(len(data_loader))

    correct_preds = 0 
    total_preds = 0


    with torch.no_grad():
        for batch_idx, data in enumerate(data_loader):

            inputs, labels = data

            inputs = inputs.to(device)
            labels = labels.to(device)

            predictions = model.forward(inputs)

            
            ## if you would like to store loss to plot uncomment these lines and change the function structure so it returns valid_iter_loss 
            # loss = criterion.forward(predictions, labels)

            # valid_iter_loss[batch_idx] = loss.item()

            _, preds = torch.max(predictions, dim=1)
            total_preds += labels.shape[0]

            correct_preds += preds.eq(labels).sum().item()

    accuracy = correct_preds / total_preds

    return accuracy


def main(lr, batch_size, epochs, data_dir, seed, augmentation_name, test_noise):
    """
    Main function for training and testing the model.

    Args:
        lr: Learning rate to use in the optimizer.
        batch_size: Batch size to train the model with.
        epochs: Number of epochs to train the model for.
        data_dir: Directory where the CIFAR10 dataset should be loaded from or downloaded to.
        seed: Seed for reproducibility.
        augmentation_name: Name of the augmentation to use.
    """

    # Set the seed for reproducibility
    set_seed(seed)
    
    # Set the device to use for training
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    # Load the model
    model = get_model(num_classes=100)
    model.to(device)

    # Get the augmentation to use

    # Train the model
    checkpoint_name='best_model.pth'
    trained_model = train_model(model, lr, batch_size, epochs, data_dir, checkpoint_name, device, augmentation_name = augmentation_name) 
    ## look chekpoint name up, there should be something in training iter we are not doing

    # Evaluate the model on the test set
    test_set = get_test_set(data_dir, test_noise) ## test_noise look it up in cifar100 utils, you have to put something there
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)
    test_accuracy = evaluate_model(trained_model, test_loader, device)

    print(f"Test accuracy: {test_accuracy}")
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Feel free to add more arguments or change the setup

    parser.add_argument('--lr', default=0.001, type=float,
                        help='Learning rate to use')
    parser.add_argument('--batch_size', default=128, type=int,
                        help='Minibatch size')
    parser.add_argument('--epochs', default=30, type=int,
                        help='Max number of epochs')
    parser.add_argument('--seed', default=42, type=int,
                        help='Seed to use for reproducing results')
    parser.add_argument('--data_dir', default='data/', type=str,
                        help='Data directory where to store/find the CIFAR100 dataset.')
    parser.add_argument('--augmentation_name', default=None, type=str,
                        help='Augmentation to use.')
    parser.add_argument('--test_noise', default=False, action="store_true",
                        help='Whether to test the model on noisy images or not.')

    args = parser.parse_args()
    kwargs = vars(args)
    main(**kwargs)
