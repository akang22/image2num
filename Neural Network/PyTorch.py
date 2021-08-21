#Disclaimer, if you find yourself asking "what does this mean" or "what does this do" ask literally anyone else please

import numpy as np
import torch
import torchvision
import matplotlib.pyplot as plt
from time import time
from torchvision import datasets, transforms
from torch import nn, optim

#transformation being performed on the dataset
transform = transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.5,),(0.5))])

#download datasets and load them to DataLoader
trainset = datasets.MNIST('./data', download=False, train=True, transform=transform)
valset = datasets.MNIST('./data', download=False,train=False,transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)
valloader = torch.utils.data.DataLoader(valset, batch_size=64, shuffle=True)

dataiter = iter(trainloader)
images, labels = dataiter.next()

print(images.shape)
print(labels.shape)

#Neural network :DDDDD
input_size = 784
hidden_sizes = [128,64]
output_size = 10

model = nn.Sequential(nn.Linear(input_size, hidden_sizes[0]), nn.ReLU(),nn.Linear(hidden_sizes[0],hidden_sizes[1]), nn.ReLU(), nn.Linear(hidden_sizes[1],output_size),nn.LogSoftmax(dim=1))
print(model)

#negative log-likelihood loss (don't ask me what this means)

criterion = nn.NLLLoss()
images, labels = next(iter(trainloader))
images = images.view(images.shape[0], -1)

logps = model(images) #log probabilities
loss = criterion(logps, labels) #calculate the NLL loss

print('Before backward pass: \n', model[0].weight.grad)
loss.backward()
print('After backward pass: \n', model[0].weight.grad)

#core training process

optimizer = optim.SGD(model.parameters(), lr=0.003, momentum=0.9)
time0=time()
epochs = 30
for e in range (epochs):
    running_loss = 0
    for images, labels in trainloader:
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
        print("Epoch {} - Training loss: {}".format(e, running_loss/len(trainloader)))
print("\n Training Time (minutes) is", (time()-time0)/60)        

correct_count, all_count = 0, 0
for images,labels in valloader:
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
print("\nModel Accuracy: " (correct_count/all_count))   



#figure = plt.figure()
#num_of_images = 60
#for index in range(1, num_of_images + 1):
#    plt.subplot(6, 10, index)
#    plt.axis('off')
#    plt.imshow(images[index].numpy().squeeze(), cmap='gray_r')
#plt.show()