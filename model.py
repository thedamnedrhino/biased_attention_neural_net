import dataset 

import torch
import torch.nn as nn
from torchvision.datasets import CIFAR10
from torchvision.transforms import transforms
from torch.utils.data import DataLoader
from torch.optim import Adam
from torch.autograd import Variable
import numpy as np

import optparse

print("CCCCCCCC")

def exit():
	import sys
	sys.exit()

KERNEL_SIZE=5	
HIDDEN_CHANNELS=12

class Unit(nn.Module):
    def __init__(self,in_channels,out_channels):
        super(Unit,self).__init__()
        self.conv = nn.Conv2d(in_channels=in_channels,kernel_size=KERNEL_SIZE,out_channels=out_channels,stride=1,padding=KERNEL_SIZE//2)
        self.bn = nn.BatchNorm2d(num_features=out_channels)
        self.relu = nn.ReLU()

    def forward(self,input):
        output = self.conv(input)
        output = self.bn(output)
        output = self.relu(output)

        return output

class SimpleNet(nn.Module):
    def __init__(self,num_classes=3, in_channels=3, hidden_channels=8, height=32, width=32):
        super(SimpleNet,self).__init__()

        self.hidden_channels = hidden_channels
        self.height = height
        self.width = width

        channels = hidden_channels

        self.unit1 = Unit(in_channels=in_channels,out_channels=channels)
        self.unit2 = Unit(in_channels=channels, out_channels=channels)
        self.pool1 = nn.MaxPool2d(kernel_size=2)

        self.unit3 = Unit(in_channels=channels, out_channels=channels)
        self.unit4 = Unit(in_channels=channels, out_channels=channels)
        # self.pool2 = nn.MaxPool2d(kernel_size=2)
        self.avgpool = nn.AvgPool2d(kernel_size=2)

        # self.pools = [2, 2, 2]
        self.pools = [2, 2]

        # self.pool2 = nn.MaxPool2d(kernel_size=2)

        # self.unit4 = Unit(in_channels=32, out_channels=64)
        # self.unit5 = Unit(in_channels=64, out_channels=64)
        # self.unit6 = Unit(in_channels=64, out_channels=64)
        # self.unit7 = Unit(in_channels=64, out_channels=64)

        # self.pool2 = nn.MaxPool2d(kernel_size=2)

        # self.unit8 = Unit(in_channels=64, out_channels=128)
        # self.unit9 = Unit(in_channels=128, out_channels=128)
        # self.unit10 = Unit(in_channels=128, out_channels=128)
        # self.unit11 = Unit(in_channels=128, out_channels=128)

        # self.pool3 = nn.MaxPool2d(kernel_size=2)

        # self.unit12 = Unit(in_channels=128, out_channels=128)
        # self.unit13 = Unit(in_channels=128, out_channels=128)
        # self.unit14 = Unit(in_channels=128, out_channels=128)

        # self.avgpool = nn.AvgPool2d(kernel_size=4)
        
        #Add all the units into the Sequential layer in exact order
        # self.net = nn.Sequential(self.unit1, self.unit2, self.unit3, self.pool1, self.unit4, self.unit5, self.unit6
                                 # ,self.unit7, self.pool2, self.unit8, self.unit9, self.unit10, self.unit11, self.pool3,
                                 # self.unit12, self.unit13, self.unit14, self.avgpool)
        self.net = nn.Sequential(self.unit1, self.unit2, self.pool1, self.unit3, self.unit4, self.avgpool)
        self.fc = nn.Linear(in_features=self.num_features(),out_features=num_classes)

    def num_features(self):
    	denom = 1

    	for p in self.pools:
    		denom *= p

    	denom = denom**2

    	return self.hidden_channels*self.height*self.width//denom

    def forward(self, input):
        output = self.net(input)
        output = output.view(-1, self.num_features())
        # output = output.view(-1,128)
        output = self.fc(output)
        return output


#Create a learning rate adjustment function that divides the learning rate by 10 every 30 epochs
def adjust_learning_rate(epoch):

    lr = 0.001

    if epoch > 180:
        lr = lr / 1000000
    elif epoch > 150:
        lr = lr / 100000
    elif epoch > 120:
        lr = lr / 10000
    elif epoch > 90:
        lr = lr / 1000
    elif epoch > 60:
        lr = lr / 100
    elif epoch > 30:
        lr = lr / 10

    for param_group in optimizer.param_groups:
        param_group["lr"] = lr




def save_models(epoch):
    torch.save(model.state_dict(), "convnet.model".format(epoch))
    print("Checkpoint saved")

def test():
    model.eval()
    test_acc = 0.0
    for i, (images, labels) in enumerate(test_loader):
      
        if cuda_avail:
                images = Variable(images.cuda())
                labels = Variable(labels.cuda())

        #Predict classes using images from the test set
        outputs = model(images)
        _,prediction = torch.max(outputs.data, 1)
        # prediction = prediction.cpu().numpy()
        test_acc += torch.sum(prediction == labels.data).float()
        


    #Compute the average acc and loss over all 10000 test images
    test_acc = test_acc / len(test_loader.dataset)

    return test_acc

def train(num_epochs):
    best_acc = 0.0

    for epoch in range(num_epochs):
        model.train()
        train_acc = 0.0
        train_loss = 0.0
        for i, (images, labels) in enumerate(train_loader):
            #Move images and labels to gpu if available
            if cuda_avail:
                images = Variable(images.cuda())
                labels = Variable(labels.cuda())

            #Clear all accumulated gradients
            optimizer.zero_grad()
            #Predict classes using images from the test set
            outputs = model(images)
            #Compute the loss based on the predictions and actual labels
            loss = loss_fn(outputs,labels)
            #Backpropagate the loss
            loss.backward()

            #Adjust parameters according to the computed gradients
            optimizer.step()

            train_loss += loss.cpu().item() * images.size(0)
            _, prediction = torch.max(outputs.data, 1)
            train_acc += torch.sum(prediction == labels.data).float()

        #Call the learning rate adjustment function
        adjust_learning_rate(epoch)

        #Compute the average acc and loss over all 50000 training images
        train_acc = train_acc / float(len(train_loader.dataset))
        train_loss = train_loss / len(train_loader.dataset)

        #Evaluate on the test set
        test_acc = test()

        # Save the model if the test acc is greater than our current best
        if test_acc > best_acc:
            save_models(epoch)
            best_acc = test_acc


        # Print the metrics
        print("Epoch {}, Train Accuracy: {} , TrainLoss: {} , Test Accuracy: {}".format(epoch, train_acc, train_loss,test_acc))


if __name__ == "__main__":
	optparser = optparse.OptionParser()
	optparser.add_option("-e", "--num-epochs", dest="epochs", default=10, help="number of epochs to train on")
	optparser.add_option("-k", "--kernel-size", dest="kernelsize", default=KERNEL_SIZE, help="the kernel size for the convulational filters")
	optparser.add_option("-c", "--channels", dest="hiddenchannels", default=HIDDEN_CHANNELS, help="number of channels(filters) in convulational filters")
	optparser.add_option("-a", "--augment", dest="augment", action="store_true", default=False, help="whether to augment the data")
	optparser.add_option("-d", "--data-directory", dest="datadir", default="./datasets", help="the dataset directory")
	#todo implement -n option
	(opts, _) = optparser.parse_args()
	epochs = int(opts.epochs)
	KERNEL_SIZE = int(opts.kernelsize)
	HIDDEN_CHANNELS = int(opts.hiddenchannels)
	datadir = opts.datadir

	#Define transformations for the training set, flip the images randomly, crop out and apply mean and std normalization
	train_transformations = transforms.Compose([
	    transforms.RandomHorizontalFlip(),
	    transforms.RandomCrop(32,padding=4),
	    transforms.ToTensor(),
	    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
	])


	#Define transformations for the test set
	test_transformations = transforms.Compose([
	   transforms.ToTensor(),
	    transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))

	])

	batch_size = 32
	train_loader = dataset.create_dataloader(datadir, 'train', batch_size)
	test_loader = dataset.create_dataloader(datadir, 'valid', batch_size)

	#Check if gpu support is available
	cuda_avail = torch.cuda.is_available()

	#Create model, optimizer and loss function
	model = SimpleNet(hidden_channels=HIDDEN_CHANNELS)

	if cuda_avail:
	    model.cuda()

	optimizer = Adam(model.parameters(), lr=0.001,weight_decay=0.0001)
	loss_fn = nn.CrossEntropyLoss()


	train(epochs)