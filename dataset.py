import data

import torch
from torchvision import transforms
import numpy as np


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
	ts = [
	transforms.ColorJitter(brightness=2),
	transforms.ColorJitter(contrast=2),
	transforms.ColorJitter(saturation=2),
	]
	t = transforms.Compose([
	  transforms.ToPILImage(),
	  transforms.RandomChoice(ts)
	])

	s = [np.array(i.shaped()) for i in d if i.label == augment_label]
	augment_list = []
	for e in s:
		augment_list.append(t(e))

	return d + [data.Image(i, augment_label) for i in augment_list]

TRANSFORMERS = {
'hor': transforms.RandomHorizontalFlip(p=0.5),
'rot': transforms.RandomRotation(15),
'gray': transforms.RandomGrayscale(p=0.1),
'affine': transforms.RandomAffine(15),
'rrcrop': transforms.RandomResizedCrop((32, 32))
}

def create_dataloader(datadir='./datasets', set='train', batch_size=10, augment_enabled=False, augment_label=2, transformers=None):
	d = data.get_data(datadir, set)
	if augment_enabled:
		augmented = augment(d, augment_label)
	transformers = [TRANSFORMERS[i] for i in transformers] if transformers is not None else None
	dset = Dataset(d, set, transformers=transformers)
	dloader = torch.utils.data.DataLoader(dataset=dset, batch_size=batch_size, shuffle=True)
	return dloader
def create_testloader(datadir='./datasets'):
	d = data.get_testdata(datadir)
	dset = Dataset(d, 'test', transformers=[])
	dloader = torch.utils.data.DataLoader(dataset=dset, batch_size=len(dset), shuffle=False)
	return dloader


if __name__ == '__main__':
	l = create_dataloader()
	images, labels = next(iter(l))
	print(labels)
	print(images[0].size())
	i = images[0]
	i = transforms.ToPILImage()(i)
	i.show()

