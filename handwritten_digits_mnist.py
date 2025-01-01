import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
from random import randint
from tqdm import tqdm #tqdm is not necessary.If you remove it change this --> for image, label in tqdm(train_loader):
                      #                                           to this --> for image,label in train_loader

#normalzing the data 
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.5,), std=(0.5,))
])

mnist_trainset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
mnist_testset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

train_size = int(0.8 * len(mnist_trainset))
val_size = len(mnist_trainset) - train_size
train_subset , val_subset = random_split(mnist_trainset, [train_size, val_size])


batch_size = 64
train_loader = DataLoader(train_subset, batch_size= batch_size , shuffle=True)
val_loader = DataLoader(val_subset, batch_size=batch_size , shuffle=True)
test_loader = DataLoader(mnist_testset, batch_size=batch_size , shuffle=True)



class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1) #output (32,26,26)
        self.conv2 = nn.Conv2d(32, 64, 3, 1) #output(64,24,24)
        self.Flatten = nn.Flatten() #convert the 4d Tensor to 1d Tensor
        self.pool = nn.MaxPool2d(2) #output(64,12,12)
        self.fc1 = nn.Linear(12*12*64, 128)
        self.fc2 = nn.Linear(128, 10)
        self.dropout = nn.Dropout(0.5)


    def forward(self,x):
      x = F.relu(self.conv1(x))
      x = F.relu(self.conv2(x))
      x = self.pool(x)
      x = self.Flatten(x)
      x = F.relu(self.fc1(x))
      x = self.dropout(x)
      x = self.fc2(x)
      return x

model = Net()
#hyparammeters 
epochs = 12 #epochs range should be between(10,20)
learning_rate = 1e-3 #1e-3 = 0.001
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters() , lr=learning_rate)

def model_training():
    print("Start training >>> ")
    for epoch in range(epochs):
        for image, label in tqdm(train_loader):
            optimizer.zero_grad()
            output = model(image)
            loss = criterion(output,label)
            loss.backward()
            optimizer.step()

        with torch.no_grad():
            val_loss = 0 
            for image,label in val_loader:
                output = model(image)
                val_loss += criterion(output,label).item() #The val_loss it the sum of all the losses 
        print(f"Epoch : {epoch+1}  | loss : {loss:.4f} | val_loss = {val_loss:.4f}")


def save_model(full_model = False):
    if full_model:
        torch.save(model, "full_model.pth")
        print("Entire model saved successfully!") #when you load it you will need to evaluate the model (model.eval())
    else:
        torch.save(model.state_dict(), "model.pth") #To use it you will need ot have the class Net in your code 
        print("Model saved successfully!")

def testing(x:int , model):
    prep = mnist_testset[x][0].unsqueeze(0)
    output = model(prep)
    _ , prediction = torch.max(output ,1)
    print(f"Prediction = {prediction.item()} | Label = {mnist_testset.targets[x]}")
    plt.imshow(mnist_testset[x][0].permute(1,2,0).squeeze().numpy(), cmap='gray')
    plt.show()


