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
      # transforms.Normalize(mean=[0.485, 0.456, 0.406],
      #                            std=[0.229, 0.224, 0.225])
    ]
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
  transforms.RandomHorizontalFlip(p=1),
  transforms.RandomRotation(20)
  ]
  transformer = transformer.Compose([
    transformers.ToPILImage(),
    transformers.RandomChoice(ts)
  ])
  for idx, i in enumerate(s):
    s[idx] = transformer(i)

  return d + [data.Image(i, augment_label) for i in s]

def create_dataloader(set='train', batch_size=10, augment=False, augment_label=2):
  d = data.get_data(set)
  if augment:
    augmented = augment(d, augment_label)
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

