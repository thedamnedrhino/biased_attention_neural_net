from matplotlib import pyplot as plt
import torch
import torchvision
import data
import dataset

DIR = 'datasets/'

def compare(file1, file2, limit=10):
	import pickle
	d1, d2 = None, None
	with open(DIR + '/' + file1, 'rb') as f:
		d1 = pickle.load(f)
	with open(DIR + '/' + file2, 'rb') as f:
		d2 = pickle.load(f)

	diff = []
	for i in range(len(d1)):
		if d1[i] != d2[i]:
			diff.append((i, d1[i], d2[i]))
			if len(diff) >= limit:
				break

	print(diff)
	dl = dataset.create_testloader()
	for images, _ in dl:
		l = [images[t[0]] for t in diff]

	tensor = torch.stack(l)
	print(tensor.size())
	grid_img = torchvision.utils.make_grid(tensor)
	plt.imshow(grid_img.permute(1, 2, 0))
	plt.show()

if __name__ == '__main__':
	compare('testlabel.pickle', 'testlabel1.pickle')