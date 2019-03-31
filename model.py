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
import nets

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
BATCH_SIZE=32
MODEL_FILE_NAME='convnet.model'
AUGMENT=False
MERGE_VALIDATION=False
LEARNING_RATE=0.001
STATIC_LEARNING_RATE=False

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
	def __init__(self, num_classes=3, in_channels=3, hidden_channels=HIDDEN_CHANNELS, height=32, width=32):
		"""
		in_channels: number of channels for the input image, e.g 3 for rgb
		hidden_channels: number of convolutional channels (filters) in each conv layer
		"""
		super(SimpleNet,self).__init__()

		self.hidden_channels = hidden_channels
		self.height = height
		self.width = width
		self.num_classes = num_classes

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

class NetworkManager:
	def __init__(self, batch_size=BATCH_SIZE, kernel_size=KERNEL_SIZE, hidden_channels=HIDDEN_CHANNELS, learning_rate=LEARNING_RATE, static_learning_rate=STATIC_LEARNING_RATE, datadir='datasets/', augment=AUGMENT, train_transformers=None, checkpoint_file_name=None, extended_net=False, extended_checkpoint=False, unfreeze_basefc=False, unfreeze_all=False, nonlinear=None, extended_net_args={}, train_on_validation=False, super_verbose=False):
		"""
		train_transformers: list from ['hor', 'rot', 'gray', 'affine', 'rrcrop'], uses default set if None is provided
		validation_labels_file: file name to save the validation labels under - only if validate_only=True
		checkpoint_file_name: name of the file to load a checkpoint from, falsey for no checkpoint
		extended_net: name of the extended network, choose from ['reg', 'fcN', 'featNRO_(1)', featNPO_(1)'] where (1) is one of ['R', 'S', 'Th']. Pass falsey for non-extended net.
		extended_checkpoint: checkpoint will be loaded into the extended network instead of the simple network - only in effect with extended_net != False
		super_verbose: False or float. if float, super verbosity will be toggled on when validation acc is above that threshold. If False, super verbosity will be off.(see self.model.super_verbose)
		"""

		self.batch_size = batch_size
		self.train_transformers = train_transformers
		self.datadir = datadir
		self.train_on_validation = train_on_validation
		self.augment = augment
		self.super_verbose = super_verbose
		self.toggle_super_verbosity(0)

		#Create model, optimizer and loss function
		self.model = self.create_model(hidden_channels, bool(extended_net), bool(checkpoint_file_name), extended_net, extended_net_args=extended_net_args,
				checkpoint_file_name=checkpoint_file_name, extended_checkpoint=extended_checkpoint, unfreeze_basefc=unfreeze_basefc, unfreeze_all=unfreeze_all, nonlinear=None)

		#Check if gpu support is available
		self.cuda_avail = torch.cuda.is_available()

		if self.cuda_avail:
			self.model.cuda()


		self.learning_rate = learning_rate
		self.static_learning_rate = static_learning_rate
		self.optimizer = Adam(self.model.parameters(), lr=self.learning_rate, weight_decay=0.0001)
		self.loss_fn = nn.CrossEntropyLoss()

	def create_model(self, hidden_channels, extended, load_saved, extended_net_name='', extended_net_args={}, checkpoint_file_name=None, extended_checkpoint=False, unfreeze_basefc=False, unfreeze_all=False, nonlinear=None):
		model = SimpleNet(hidden_channels=hidden_channels)
		if load_saved and not extended_checkpoint:
			self.load_checkpoint(model, checkpoint_file_name)

		if extended:
			if not unfreeze_all:
				for p in model.parameters():
					p.requires_grad = False
				if unfreeze_basefc:
					for p in model.fc.parameters():
						p.requires_grad = True

			# model = nets.ExtendedNetFactory().create_net(extended_net_name, model, nonlinear=nonlinear)
			if nonlinear is not None:
				assert 'nonlinear' not in extended_net_args
				extended_net_args['nonlinear'] = nonlinear
			model = nets.ExtendedNetFactory().create_net(extended_net_name, model, extended_net_args)

			if extended_checkpoint:
				self.load_checkpoint(model, checkpoint_file_name)

		return model

	def __init(self, set):
		if set == 'train':
			self.train_loader = dataset.create_dataloader(self.datadir, 'train', self.batch_size, self.augment, transformers=self.train_transformers)
		elif set == 'validate':
			if not self.train_on_validation:
				self.validate_loader = dataset.create_dataloader(self.datadir, 'valid', self.batch_size, False, shuffle=False, transformers=[])
			else:
				self.validate_loader = dataset.create_dataloader(self.datadir, 'valid', self.batch_size, self.augment, shuffle=True, transformers=train_transformers)
		elif set == 'test':
			self.test_loader = dataset.create_testloader(self.datadir)


	def train(self, num_epochs, save_model_file_name):
		self.__init('train')

		model = self.model
		train_loader = self.train_loader
		optimizer = self.optimizer
		loss_fn = self.loss_fn

		best_acc = 0.0
		for epoch in range(num_epochs):
			self.toggle_super_verbosity(best_acc)
			model.super_verbose = self.is_super_verbose
			model.train()
			train_acc = 0.0
			train_loss = 0.0
			for i, (images, labels) in enumerate(train_loader):
				#Move images and labels to gpu if available
				if self.cuda_avail:
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

			if self.is_super_verbose:
				model.print_outputs()
				if model.metrics is not None:
					model.metrics.aggregate()
					print("Diff_avg: {}, Relative_diff_avg: {}, Contradiction_avg: {}".format(model.metrics.diff_avg, model.metrics.relative_diff_avg, model.metrics.contradiction_avg))
					model.reset_metrics()

			#Call the learning rate adjustment function
			self.adjust_learning_rate(epoch)

			#Compute the average acc and loss over all 50000 training images
			train_acc = train_acc / float(len(train_loader.dataset))
			train_loss = train_loss / len(train_loader.dataset)

			#Evaluate on the validate set
			validate_acc, validate_labels = self.validate()

			# Save the model if the validate acc is greater than our current best
			if validate_acc > best_acc:
				self.save_models(epoch, save_model_file_name, accuracy={'validation_acc': validate_acc.item(), 'train_acc': train_acc.item(), 'train_loss': train_loss})
				best_acc = validate_acc


			# Print the metrics
			print("Epoch {}, Train Accuracy: {} , TrainLoss: {} , validate Accuracy: {}".format(epoch, train_acc, train_loss,validate_acc))

	def validate(self):
		self.__init('validate')

		model = self.model
		validate_loader = self.validate_loader
		optimizer = self.optimizer
		loss_fn = self.loss_fn
		train = self.train_on_validation

		if train:
			model.train()
		else:
			model.eval()
		validate_acc = 0.0
		validate_loss = 0.0
		validate_labels = []
		for i, (images, labels) in enumerate(validate_loader):

			if self.cuda_avail:
				images = Variable(images.cuda())
				labels = Variable(labels.cuda())

			if train:
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
				_, prediction = torch.max(outputs.data, 1)
				# prediction = prediction.cpu().numpy()
				validate_acc += torch.sum(prediction == labels.data).float()

			for i in range(len(prediction.data)):
				validate_labels.append(prediction.data[i].item())



		#Compute the average acc and loss over all 10000 validate images
		validate_acc = validate_acc / len(validate_loader.dataset)

		print("validation accuracy: {}".format(validate_acc))
		return validate_acc, validate_labels

	def test(self, label_file_name='testlabel.pickle'):
		self.__init('test')

		model = self.model
		test_loader = self.test_loader
		ls = []
		model.eval()
		import pickle
		for i, (images, labels) in enumerate(test_loader):
			outputs = model(images)
			_,prediction = torch.max(outputs.data, 1)
			print(prediction)

			with open(datadir+label_file_name, 'rb') as f:
				labels = pickle.load(f)

			for i in range(len(labels)):
				labels[i] = prediction.data[i].item()

			with open(datadir+label_file_name, 'wb') as f:
				pickle.dump(labels, f)

			with open(datadir+label_file_name, 'rb') as f:
				print(pickle.load(f))

	def load_checkpoint(self, model, checkpoint_name):
		if not torch.cuda.is_available():
			model.load_state_dict(torch.load(checkpoint_name, map_location='cpu'))
		else:
			model.load_state_dict(torch.load(checkpoint_name))


	def save_models(self, epoch, model_file_name, accuracy=None):
		torch.save(self.model.state_dict(), model_file_name.format(epoch))
		if accuracy is not None:
			file_name = model_file_name + '.accuracy'
			with open(file_name, 'w') as f:
				f.write("epoch: {}, accuracy: {}\n".format(epoch, accuracy))
		print("checkpoint saved")

	def save_labels(self, labels, file_name):
		with open(file_name, 'wb') as f:
			pickle.dump(labels, f)

#create a learning rate adjustment function that divides the learning rate by 10 every 30 epochs
	def adjust_learning_rate(self, epoch):

		if self.static_learning_rate:
			return self.learning_rate

		lr = self.learning_rate

		if epoch > 180:
			lr = lr / 10
		elif epoch > 150:
			lr = lr / 10
		elif epoch > 120:
			lr = lr / 10
		elif epoch > 90:
			lr = lr / 10
		elif epoch > 60:
			lr = lr / 10
		elif epoch > 30:
			lr = lr / 10

		for param_group in self.optimizer.param_groups:
			param_group["lr"] = lr

		self.learning_rate = lr

	def toggle_super_verbosity(self, best_validation_acc):
		if self.super_verbose is False:
			self.is_super_verbose = False
		elif self.super_verbose is True or best_validation_acc >= self.super_verbose:
			self.is_super_verbose = True



if __name__ == "__main__":
	import argparse
	optparser = argparse.ArgumentParser()
	optparser.add_argument("-e", "--num-epochs", dest="epochs", default=10, help="number of epochs to train on")
	optparser.add_argument("-b", "--batch-size", type=int, dest="batchsize", default=BATCH_SIZE, help="training batch size")
	optparser.add_argument("-k", "--kernel-size", dest="kernelsize", default=KERNEL_SIZE, help="the kernel size for the convulational filters")
	optparser.add_argument("-c", "--channels", dest="hiddenchannels", default=HIDDEN_CHANNELS, help="number of channels(filters) in convulational filters")
	optparser.add_argument("-a", "--augment", dest="augment", action="store_true", default=AUGMENT, help="enable augmenting the data")
	optparser.add_argument("-v", "--validate_only", dest="validateonly", nargs='?', const='validation_labels.pickle', default=False, help="whether to only validate and store the validation labels in the path provided as value to this option - defaults to validation_labels.pickle")
	optparser.add_argument("-d", "--data-directory", dest="datadir", default="./datasets", help="the dataset directory")
	optparser.add_argument("-m", "--model-file-name", dest="modelfilename", default=MODEL_FILE_NAME, help="the name to save the best model under")
	optparser.add_argument("-t", "--transformers", dest="transformers", default=None, help="the transformers to use from {" + ', '.join(dataset.TRANSFORMERS.keys()) + "}")
	optparser.add_argument("-l", "--load-checkpoint", dest="checkpointname", default=None, help="input the checkpoint for the model if you want to use one as base")
	optparser.add_argument("--test", dest="test", action="store_true", default=False, help="whether to augment the data")
	optparser.add_argument("--merge-validation", dest="mergevalidation", action="store_true", default=False, help="whether to augment the data")
	optparser.add_argument("-x", "--extended", dest="extended", action="store_true", default=False, help="whether to use the extended model")
	optparser.add_argument("--extended-checkpoint", dest="extendedcheckpoint", action="store_true", default=False, help="whether to use the supplied checkpoint is for the extended model and not the nested original")
	optparser.add_argument("-u", "--unfreeze-fc", dest="unfreezefc", action="store_true", default=False, help="Unfreeze the fc of the base model. Only in effect with -x")
	optparser.add_argument("--non-linear", dest="nonlinear", default="sigmoid", help="The non-linear function after the first fc of the extended net. Choose between 'relu', 'sigmoid', 'none'")
	optparser.add_argument("-n", "--network", dest="network", default="fcN", help="the extended network to use (only with -x). Choose from \n{}".format(" **|** ".join(["{}: {}".format(k, v) for k, v in nets.ExtendedNetFactory.NETS.items()])))
	optparser.add_argument("--net-args", dest="netargs", nargs="+", default=[], help="the arguments passed to the extended network. Check the documentation for options of each network. only in effect with -x")
	optparser.add_argument("-r", "--learning-rate", type=float, dest="learningrate", default=False, help="the static learning rate. defaults to a dynamic one starting at 0.001 and divided by 10 every 30 epochs")
	optparser.add_argument("--unfreeze-all", default=False, action="store_true", dest="unfreezeall", help="whether to unfreeze all layers in the base network (only with -x)")
	optparser.add_argument("--super-verbose", dest="superverbose", nargs="?", default=False, const=0.0,  help="whether to be super verbose. Optionally supply a value to start super verbosity when validation accuracy is above that value.")
	#todo implement -n option
	opts = optparser.parse_args()
	epochs = int(opts.epochs)
	batch_size = opts.batchsize
	kernel_size = int(opts.kernelsize)
	hidden_channels = int(opts.hiddenchannels)
	datadir = opts.datadir
	augment = opts.augment
	transformers = opts.transformers
	checkpoint_name = opts.checkpointname
	load_saved = bool(checkpoint_name)
	validate_only = opts.validateonly
	test_only = opts.test
	merge_validation = opts.mergevalidation
	extended = opts.extended
	extended_checkpoint = opts.extendedcheckpoint
	unfreeze_basefc = opts.unfreezefc
	nonlinear = opts.nonlinear
	network = opts.network
	super_verbose = opts.superverbose if opts.superverbose is False else float(opts.superverbose)
	learning_rate = LEARNING_RATE
	static_learning_rate = STATIC_LEARNING_RATE
	if opts.learningrate is not False:
		learning_rate = float(opts.learningrate)
		static_learning_rate = True

	extended_net_args = {k: v for k, v in [arg.split('=') for arg in opts.netargs]}
	def normalize_input_args(args):
		for k, v in args.items():
			if v in ['true', 'True']:
				v = True
			elif v in ['false', 'False']:
				v = False
			elif v in ['None', 'null', 'Null']:
				v = None
			elif v in ['[]']:
				v = []
			args[k] = v
		return args
	extended_net_args = normalize_input_args(extended_net_args)


	if transformers == '-':
		transformers = []
	else:
		transformers = transformers.split(',') if transformers is not None else None

	net_man = NetworkManager(batch_size, kernel_size, hidden_channels, learning_rate, static_learning_rate, datadir, augment, train_transformers=transformers, checkpoint_file_name=checkpoint_name, extended_net=network if extended else False, extended_checkpoint=extended_checkpoint, unfreeze_basefc=unfreeze_basefc, unfreeze_all=opts.unfreezeall, nonlinear=nonlinear, extended_net_args=extended_net_args,
			train_on_validation=merge_validation, super_verbose=super_verbose)

	if not validate_only and not test_only:
		net_man.train(epochs, opts.modelfilename)
	if validate_only:
		accuracy, labels = net_man.validate()
		# validate_only is also the file name!
		print(accuracy.item())
		print(len(labels))
		net_man.save_labels(labels, validate_only)
	if test_only:
		net_man.test()
"""
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
"""
