import data

import torch
from torchvision import transforms
import numpy as np


class Dataset(torch.utils.data.Dataset):
  'Characterizes a dataset for PyTorch'
  def __init__(self, images, set='train', transformer=None):
        'Initialization'
        self.images = images
        self.transformer = None if transformer is False else self.create_transformer(set)

  def create_transformer(self, set):
    ts = [
      transforms.ToPILImage(),
      # transforms.RandomHorizontalFlip(),
      transforms.ToTensor(),
    ]
    return transforms.Compose(ts)

  def __len__(self):
        return len(self.images)

  def __getitem__(self, index):
        image = self.images[index]
        X = self.transformer(np.array(image.shaped()))
        y = image.label

        return X, y

def create_dataloader(set='train', batch_size=10):
  d = data.get_data(set)
  dset = Dataset(d, set)
  dloader = torch.utils.data.DataLoader(dataset=dset, batch_size=batch_size, shuffle=True)
  return dloader


if __name__ == '__main__':
  l = create_dataloader()
  images, labels = next(iter(l))
  print(labels)
  print(images[0].size())
  i = images[0]
  i = transforms.ToPILImage()(i)
  i.show()

