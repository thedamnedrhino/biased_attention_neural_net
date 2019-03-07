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


"""
This is a convulational network with 4 convulational layers and two pool layers. 
Each convulational layer has 12 filters. It is inspired by https://heartbeat.fritz.ai/basics-of-image-classification-with-pytorch-2f8973c51864
For the data imbalance problem, I augmented the data to include transformed versions of the the images from the third category, such that there would 
be an equal number of instances of each class.
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



class dataset:

	class Dataset(torch.utils.data.Dataset):
		'Characterizes a dataset for PyTorch'
		def __init__(self, images, set='train', transformers=None):
			'Initialization'
			self.images = images
			self.transformer = self.create_transformer(set, transformers=transformers)

		def create_transformer(self, set, transformers=None):
			if transformers is None:
				transformers = [      
					transforms.RandomHorizontalFlip(p=0.5),
					transforms.RandomRotation(20)
				]

			ts = [transforms.ToPILImage()] + transformers + [transforms.ToTensor()]

			return transforms.Compose(ts)

		def __len__(self):
			return len(self.images)

		def __getitem__(self, index):
			image = self.images[index]
			X = self.transformer(np.array(image.shaped()))
			y = image.label

			return X, y

	def augment(d, augment_label):
		print ("augmenting++++++++++++++")
		s = [i.shaped() for i in data if i.label == augment_label]
		ts = [
		transforms.ColorJitter(brightness=2),
		transforms.ColorJitter(contrast=2),
		transforms.ColorJitter(saturation=2),
		]
		transformer = transformer.Compose([
		  transformers.ToPILImage(),
		  transformers.RandomChoice(ts)
		])
		for idx, i in enumerate(s):
		  s[idx] = transformer(i)

		return d + [data.Image(i, augment_label) for i in s]

	TRANSFORMERS = {
	'hor': transforms.RandomHorizontalFlip(p=0.5),
	'rot': transforms.RandomRotation(15),
	'gray': transforms.RandomGrayscale(p=0.1),
	'affine': transforms.RandomAffine(15),
	'rrcrop': transforms.RandomResizedCrop((32, 32))
	}

	def create_dataloader(datadir='./datasets', set='train', batch_size=10, augment=False, augment_label=2, transformers=None):
		d = data.get_data(datadir, set)
		if augment:
		  augmented = augment(d, augment_label)
		transformers = [TRANSFORMERS[i] for i in transformers] if transformers is not None else None
		dset = dataset.Dataset(d, set, transformers=transformers)
		dloader = torch.utils.data.DataLoader(dataset=dset, batch_size=batch_size, shuffle=True)
		return dloader
	def create_testloader(datadir='./datasets'):
		d = data.get_testdata(datadir)
		dset = dataset.Dataset(d, 'test', transformers=[])
		dloader = torch.utils.data.DataLoader(dataset=dset, batch_size=len(dset), shuffle=False)
		return dloader


class data:

	IMAGES = None
	LABELS = None

	class Image:
		def __init__(self, pixels, label, channels=3, rows=32, cols=32):
			self.pixels = pixels
			self.label = label
			self.shaped_img = None
			self.channels = channels
			self.rows = rows
			self.cols = cols
		
		def show(self, p=True):
			plt.imshow(self.shaped())
			plt.show()
			if p:
				print("LABEL: ++ {} ++\n".format(self.label))
			return plt
			
		def shape(self, merged=True):
			img = self.pixels
			assert len(img) == 1024 * 3
			channels = [img[i*1024:(i+1)*1024] for i in range(3)]
			channels = [ [channel[row*32:(row+1)*32] for row in range(32)] for channel in channels ]
			assert len(channels) == 3
			assert len(channels[0]) == 32
			assert len(channels[2][2]) == 32
			if merged:
				channels = [[ [channels[0][i][j], channels[1][i][j], channels[2][i][j]] for j in range(32)] for i in range(32)]
			return channels
					
		   
		def shaped(self):
			if self.shaped_img is None:
				self.shaped_img = self.shape()
			return self.shaped_img
		
	def get_data(datadir='./datasets', dataset='train'):
		"""
		param dataset: 'train', 'valid'
		"""
		filename = datadir + '/' + dataset + 'set.pickle'
		with open(filename, 'rb') as fo:
			dict = pickle.load(fo, encoding='bytes')
			d = dict['data']
			labels = dict['label']
		images = [data.Image(d[i], labels[i]) for i in range(len(d))]
		data.IMAGES = images if data.IMAGES is None else data.IMAGES
		return images

	def get_testdata(datadir='./datasets'):
		filename = datadir + '/' + 'test' + 'set.pickle'
		with open(filename, 'rb') as fo:
			dict = pickle.load(fo, encoding='bytes')
			d = dict['data']
		images = [data.Image(d[i], 10) for i in range(len(d))]
		return images

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
	model = SimpleNet(hidden_channels=HIDDEN_CHANNELS)

	if cuda_avail:
		model.cuda()

	optimizer = Adam(model.parameters(), lr=0.001,weight_decay=0.0001)
	loss_fn = nn.CrossEntropyLoss()
	if load_saved:
		load_checkpoint(model, checkpoint_name)

	if not validate_only and not test_only:
		train(epochs, opts.modelname)
	if validate_only:
		validate()
	if test_only:
		test_loader = dataset.create_testloader(datadir)
		test(model, test_loader)

