import torch
import torch.nn as nn
from torchvision.datasets import CIFAR10
from torchvision.transforms import transforms
from torch.utils.data import DataLoader
from torch.optim import Adam
from torch.autograd import Variable
import numpy as np

import optparse
import pickle

import dataset

"""
This is a convulational network with 4 convulational layers and two pool layers after every two conv layers.
Each convulational layer has 12 filters. It is inspired by https://heartbeat.fritz.ai/basics-of-image-classification-with-pytorch-2f8973c51864
For the DATA IMBALANCE PROBLEM, I augmented the data to include transformed versions of the the images from the third category, such that there would
be an equal number of instances of each class.
Solving the data imbalance got the accuracy on the validation set from 0.75 to 0.80. The output file can be found as 'output.txt'.
The full runnable python code along with a readme can be found at  https://github.com/thedamnedrhino/first_image_classifier.

ADDITIONAL NOTES AND OBSERVATIONS:
- Increasing the epochs to train on did not have an effect, after reaching a certain value. When I added random transformations to the images
this changed and the accuracy kept increasing for longer. This was to be expected.

- The network was trained only on the train data for generating the test labels. I did also merge the validation set into the training set to see what difference
it would make on the test labels. Interestingly even though training on the merged trained-validation dataset gets the accuracy on the
training set and the validation set from ~0.80 BOTH, to 0.66 and 0.88 on the sets respectively, the labels generated for the test data
have minimal differences: less than 10 cases. I checked those cases manually and saw that the number that the network trained only on the train data
got right, was equal to the number that the network trained with the merged train and validation data got right, and they should have similar performances
on the test set.
"""

KERNEL_SIZE=5
HIDDEN_CHANNELS=12
MODEL_NAME='convnet.model'
MERGE_VALIDATION=False

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

class ExtendedNet(nn.Module):
	def __init__(self, nested_model, num_classes=3, in_channels=3, hidden_channels=8, height=32, width=32, nonlinear='sigmoid'):
		super(ExtendedNet, self).__init__()
		self.num_classes = num_classes
		self.hidden_channels = hidden_channels
		self.height = height
		self.width = width
		self.nested_model = nested_model
		self.fc1 = nn.Linear(in_features=self.num_features(), out_features=num_classes**2)
		self.fc2 = nn.Linear(in_features=num_classes**2 + num_classes ,out_features=num_classes)
		self.sigmoid = nn.Sigmoid()
		self.relu = nn.ReLU()
		nonlinearmap = {'sigmoid': self.sigmoid, 'relu': self.relu, 'none': lambda x: x}
		assert nonlinear in nonlinearmap
		self.nonlinear = nonlinearmap[nonlinear]

	def num_features(self):
		return self.nested_model.num_features() + self.num_classes

	def forward(self, input):
		nested_output = self.nested_model(input)
		nested_features = self.nested_model.features1d
		assert len(nested_features.size()) == 2
		assert len(nested_output.size()) == 2
		nested_probs = self.sigmoid(nested_output)
		extended_features = torch.cat((nested_output, nested_features), 1)
		output = self.fc1(extended_features)
		output = self.nonlinear(output)
		output = self.fc2(torch.cat((nested_output, output), 1))
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
		self.features = output
		output = output.view(-1, self.num_features())
		self.features1d = output
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




def save_models(epoch, model_name=MODEL_NAME):
	torch.save(model.state_dict(), model_name.format(epoch))
	print("Checkpoint saved")

def validate():
	if not MERGE_VALIDATION:
		model.eval()
	else:
		model.train()
	validate_acc = 0.0
	validate_loss = 0.0
	for i, (images, labels) in enumerate(validate_loader):

		if cuda_avail:
			images = Variable(images.cuda())
			labels = Variable(labels.cuda())

		if MERGE_VALIDATION:
			#Clear all accumulated gradients
			optimizer.zero_grad()
			#Predict classes using images from the validate set
			outputs = model(images)
			#Compute the loss based on the predictions and actual labels
			loss = loss_fn(outputs,labels)
			#Backpropagate the loss
			loss.backward()

			#Adjust parameters according to the computed gradients
			optimizer.step()

			validate_loss += loss.cpu().item() * images.size(0)
			_, prediction = torch.max(outputs.data, 1)
			validate_acc += torch.sum(prediction == labels.data).float()
		else:
			#Predict classes using images from the validate set
			outputs = model(images)
			_,prediction = torch.max(outputs.data, 1)
			# prediction = prediction.cpu().numpy()
			validate_acc += torch.sum(prediction == labels.data).float()



	#Compute the average acc and loss over all 10000 validate images
	validate_acc = validate_acc / len(validate_loader.dataset)

	print("validation accuracy: {}".format(validate_acc))
	return validate_acc

def train(num_epochs, model_name=MODEL_NAME):
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
			#Predict classes using images from the validate set
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

		#Evaluate on the validate set
		validate_acc = validate()

		# Save the model if the validate acc is greater than our current best
		if validate_acc > best_acc:
			save_models(epoch, model_name)
			best_acc = validate_acc


		# Print the metrics
		print("Epoch {}, Train Accuracy: {} , TrainLoss: {} , validate Accuracy: {}".format(epoch, train_acc, train_loss,validate_acc))

def load_checkpoint(model, checkpoint_name):
	if not torch.cuda.is_available():
		model.load_state_dict(torch.load(checkpoint_name, map_location='cpu'))
	else:
		model.load_state_dict(torch.load(checkpoint_name))

def test(model, test_loader):
	ls = []
	model.eval()
	import pickle
	for i, (images, labels) in enumerate(test_loader):
		outputs = model(images)
		_,prediction = torch.max(outputs.data, 1)
		print(prediction)

		with open(datadir+'/testlabel.pickle', 'rb') as f:
			labels = pickle.load(f)

		for i in range(len(labels)):
			labels[i] = prediction.data[i].item()

		with open(datadir+'/testlabel.pickle', 'wb') as f:
			pickle.dump(labels, f)

		with open(datadir+'/testlabel.pickle', 'rb') as f:
			print(pickle.load(f))


def create_model(extended=False, load_saved=False, checkpoint_name=None, extended_checkpoint=None, unfreeze_basefc=False, nonlinear='sigmoid'):
	model = SimpleNet(hidden_channels=HIDDEN_CHANNELS)
	if load_saved and not extended_checkpoint:
		load_checkpoint(model, checkpoint_name)
	if extended:
		for p in model.parameters():
			p.requires_grad = False
		if unfreeze_basefc:
			for p in model.fc.parameters():
				p.requires_grad = True

		model = ExtendedNet(model, nonlinear=nonlinear)
		if extended_checkpoint:
			load_checkpoint(model, checkpoint_name)
	return model

if __name__ == "__main__":
	optparser = optparse.OptionParser()
	optparser.add_option("-e", "--num-epochs", dest="epochs", default=10, help="number of epochs to train on")
	optparser.add_option("-k", "--kernel-size", dest="kernelsize", default=KERNEL_SIZE, help="the kernel size for the convulational filters")
	optparser.add_option("-c", "--channels", dest="hiddenchannels", default=HIDDEN_CHANNELS, help="number of channels(filters) in convulational filters")
	optparser.add_option("-a", "--augment", dest="augment", action="store_true", default=False, help="whether to augment the data")
	optparser.add_option("-v", "--validate_only", dest="validateonly", action="store_true", default=False, help="whether to only validate")
	optparser.add_option("-d", "--data-directory", dest="datadir", default="./datasets", help="the dataset directory")
	optparser.add_option("-m", "--model-name", dest="modelname", default=MODEL_NAME, help="the name to save the best model under")
	optparser.add_option("-t", "--transformers", dest="transformers", default=None, help="the transformers to use from {" + ', '.join(dataset.TRANSFORMERS.keys()) + "}")
	optparser.add_option("-l", "--load-checkpoint", dest="checkpointname", default=None, help="input the checkpoint for the model if you want to use one as base")
	optparser.add_option("--test", dest="test", action="store_true", default=False, help="whether to augment the data")
	optparser.add_option("-r", "--merge-validation", dest="mergevalidation", action="store_true", default=False, help="whether to augment the data")
	optparser.add_option("-x", "--extended", dest="extended", action="store_true", default=False, help="whether to use the extended model")
	optparser.add_option("--extended-checkpoint", dest="extendedcheckpoint", action="store_true", default=False, help="whether to use the supplied checkpoint is for the extended model and not the nested original")
	optparser.add_option("-u", "--unfreeze-fc", dest="unfreezefc", action="store_true", default=False, help="Unfreeze the fc of the base model. Only in effect with -x")
	optparser.add_option("--non-linear", dest="nonlinear", default="sigmoid", help="The non-linear function after the first fc of the extended net. Choose between 'relu', 'sigmoid', 'none'")

	#todo implement -n option
	(opts, _) = optparser.parse_args()
	epochs = int(opts.epochs)
	KERNEL_SIZE = int(opts.kernelsize)
	HIDDEN_CHANNELS = int(opts.hiddenchannels)
	datadir = opts.datadir
	transformers = opts.transformers
	checkpoint_name = opts.checkpointname
	load_saved = bool(checkpoint_name)
	validate_only = opts.validateonly
	test_only = opts.test
	MERGE_VALIDATION = opts.mergevalidation
	extended = opts.extended
	extended_checkpoint = opts.extendedcheckpoint
	unfreeze_basefc = opts.unfreezefc
	nonlinear = opts.nonlinear

	if transformers == '-':
		transformers = []
	else:
		transformers = transformers.split(',') if transformers is not None else None

	#Define transformations for the training set, flip the images randomly, crop out and apply mean and std normalization
	train_transformations = transforms.Compose([
		transforms.RandomHorizontalFlip(),
		transforms.RandomCrop(32,padding=4),
		transforms.ToTensor(),
		transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
	])


	#Define transformations for the validate set
	validate_transformations = transforms.Compose([
		transforms.ToTensor(),
		transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))

	])

	batch_size = 32
	train_loader = dataset.create_dataloader(datadir, 'train', batch_size, transformers=transformers)
	validate_loader = dataset.create_dataloader(datadir, 'valid', batch_size, transformers=[])


	#Check if gpu support is available
	cuda_avail = torch.cuda.is_available()

	#Create model, optimizer and loss function
	model = create_model(extended, load_saved,
			checkpoint_name=checkpoint_name, extended_checkpoint=extended_checkpoint, unfreeze_basefc=unfreeze_basefc,
			nonlinear=nonlinear)

	if cuda_avail:
		model.cuda()

	optimizer = Adam(model.parameters(), lr=0.001,weight_decay=0.0001)
	loss_fn = nn.CrossEntropyLoss()

	if not validate_only and not test_only:
		train(epochs, opts.modelname)
	if validate_only:
		validate()
	if test_only:
		test_loader = dataset.create_testloader(datadir)
		test(model, test_loader)

