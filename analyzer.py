from matplotlib import pyplot as plt
import torch
import torchvision
import torchvision.transforms as transforms
import numpy as np
import pickle

import data
import dataset

import collections

DIR = 'datasets/'

class Analyzer:
	def __init__(self, label_files, data_dir='./datasets', base_dir='.', dataset='valid'):
		self.label_files = label_files
		self.dataset = dataset
		self.data_dir = data_dir
		self.base_dir = base_dir
		self.initialized = False
		self.reference_label_set = None

	def initialize(self):
		if self.initialized:
			return

		self.images = data.get_data(self.data_dir, self.dataset)
		self.reference_label_set = [image.label for image in self.images]

		self.initialized = True

	def get_image_tensors(self, indices=None):
		tensors = []
		transform = transforms.Compose([transforms.ToPILImage(), transforms.ToTensor()])
		for i, image in enumerate(self.images):
			if indices is not None and i not in indices:
				continue
			tensors.append(transform(np.array(image.shaped())))
		tensors = torch.stack(tensors)
		return tensors

	def diff_grid(self):
		self.initialize()
		label_sets = []
		for file in self.label_files:
			with open("{}/{}".format(self.base_dir, file), 'rb') as f:
				label_sets.append(pickle.load(f))
		reference_set = self.reference_label_set
		diff_indices = []
		diff_sets = collections.defaultdict(list) # maps indices with diff to the sets with different labels from the reference
		for i in range(0, len(reference_set)):
			for j, set in enumerate(label_sets):
				if reference_set[i] != set[i]:
					diff_indices.append(i)
					diff_sets[i].append(j)
		print(len(diff_indices))
		print(len(reference_set))
		tensors = self.get_image_tensors(diff_indices)
		grid = torchvision.utils.make_grid(tensors, nrow=25)
		plt.imshow(grid.permute(1, 2, 0))
		plt.show()

if __name__ == '__main__':
	import argparse
	optparser = argparse.ArgumentParser()
	optparser.add_argument("-s", "--set", dest="set", default="valid", help="train, valid, or test")
	optparser.add_argument("--dataset-dir", dest="datasetdir", default="./datasets", help="the directory where the reference datasets are found")
	optparser.add_argument("--labelfile-basedir", dest="labelfilebasedir", default=".", help="the base directory where the label files are addressed from")
	optparser.add_argument("-f", "--label-files", dest="labelfiles", default=[], required=True, nargs='+', help="the pickle files for each label set")
	args = optparser.parse_args()

	analyzer = Analyzer(args.labelfiles, dataset=args.set, base_dir=args.labelfilebasedir, data_dir=args.datasetdir)
	analyzer.diff_grid()


