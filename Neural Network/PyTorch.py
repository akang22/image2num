#Disclaimer, if you find yourself asking "what does this mean" or "what does this do" ask literally anyone else please

import numpy as np
import torch
import torchvision
import matplotlib.pyplot as plt
from time import time
from torchvision import datasets, transforms
from torch import nn, optim

# simply returns a data loader for MNIST
def get_loader(train: bool, transform):
    return torch.utils.data.DataLoader(
                datasets.MNIST(
                    './data', 
                    download=True, 
                    train=train, 
                    transform=transform), 
                batch_size=64,
                shuffle=True) 

def main():
    # transformation being performed on the dataset
    transform = transforms.Compose(
        [transforms.ToTensor(),
        transforms.Normalize((0.5),(0.5)) ])

    # download datasets and load them to DataLoader
    train_loader = get_loader(True, transform)
    val_loader = get_loader(False, transform)

    # constants for our model (see below explanation)
    input_size = 784
    hidden_sizes = [128,64]
    output_size = 10

    # simple explanation:
    # nn.Sequential takes in a varargs of 'layers' to apply to the input data.
    # the first later is a linear operation from the input size (784, which is the amount of pixels in the image), and turns it into 128 values
    # those 128 values are then made non-negative (negatives all go to zero)
    # this process is then done to decrease it to 64 value (attributes?), then finally 10
    # and then, LogSoftMax allows it to quantify the goodness of our fit
    # and negative log-likelihood loss then somehow distills that into a single number 
    # these last two aren't really important to understand, just know that they take the final probabilities
    # and convert them to a measure of our model's capability
    model = nn.Sequential(
        nn.Linear(input_size, hidden_sizes[0]), 
        nn.ReLU(),
        nn.Linear(hidden_sizes[0], hidden_sizes[1]), 
        nn.ReLU(), 
        nn.Linear(hidden_sizes[1], output_size),
        nn.LogSoftmax(dim=1))

    criterion = nn.NLLLoss()

    # core training process

    optimizer = optim.SGD(
        model.parameters(),
        lr=0.003,
        momentum=0.9)
    time0=time()
    epochs = 50
    for e in range (epochs):
        running_loss = 0
        for images, labels in train_loader:
            #Flattens the images into a 784 dimensional vector
            images = images.view(images.shape[0],-1)

            #The training pass
            optimizer.zero_grad()
            output = model(images)
            loss = criterion(output, labels)

            loss.backward()

            optimizer.step()

            running_loss += loss.item()
        else:
            print("Epoch {} - Training loss: {}".format(e, running_loss/len(train_loader)))
    print("\n Training Time (minutes) is", (time()-time0)/60)        

    correct_count, all_count = 0, 0
    for images,labels in val_loader:
        for i in range(len(labels)):
            img = images[i].view(1, 784)
            with torch.no_grad():
                logps = model(img)

            ps = torch.exp(logps)
            probab = list(ps.numpy()[0])
            pred_label = probab.index(max(probab))
            true_label = labels.numpy()[i]
            if(true_label == pred_label):
                correct_count += 1
            all_count += 1
    print("Numbers of Images Tested: ", all_count)
    print("\nModel Accuracy: ", (correct_count/all_count))   

    torch.save(model, './my_mnist_model.pt')

    #figure = plt.figure()
    #num_of_images = 60
    #for index in range(1, num_of_images + 1):
    #    plt.subplot(6, 10, index)
    #    plt.axis('off')
    #    plt.imshow(images[index].numpy().squeeze(), cmap='gray_r')
    #plt.show()

if __name__ == '__main__':
    main()