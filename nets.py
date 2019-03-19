import torch
import torch.nn as nn

class AbstractExtendedNet(nn.Module):
	def __init__(self, nested_model, num_classes=3, in_channels=3, hidden_channels=8, height=32, width=32, nonlinear='sigmoid', fc_include_class_prob=True, enable_fc_class_correlate=True):
		super(ExtendedNet, self).__init__()
		self.num_classes = num_classes
		self.hidden_channels = hidden_channels
		self.height = height
		self.width = width
		self.nested_model = nested_model
		self.sigmoid = nn.Sigmoid()
		self.relu = nn.ReLU()
		nonlinearmap = {'sigmoid': self.sigmoid, 'relu': self.relu, 'none': lambda x: x}
		assert nonlinear in nonlinearmap
		self.nonlinear = nonlinearmap[nonlinear]
		self.fc_include_class_prob = fc_include_class_prob
		self.fc_class_correlate = nn.Linear(in_features=num_classes, out_features=num_classes)

		self._init_layers()

	def num_features(self):
		return self.nested_model.num_features() + self.num_classes

	def forward(self, input):
		nested_output = self.nested_model(input)
		nested_features = self.nested_model.features1d

		assert len(nested_features.size()) == 2
		assert len(nested_output.size()) == 2

		nested_probs = self.sigmoid(nested_output)
		normalized_features = nested_features.unsqueeze(1).transpose(-1, -2).matmul(nested_probs.unsqueeze(1))

		assert normalized_features.size(1) == nested_features.size(1), "{} != {}".format(normalized_features.size(), nested_features.size())
		assert normalized_features.size(2) == nested_output.size(1), "{} != {}".format(normalized_features.size(), nested_features.size())

		output = self._process(nested_output, nested_probs, nested_features, normalized_features)
		if self.enable_fc_class_correlate:
			output = self.fc_class_correlate(self.sigmoid(output))

		return output

	def __linearize_features(self, features):
		return features.view(-1, features.size(-2) * features.size(-1))

	def _init_layers(self):
		raise Exception('implement this')
	def _process(self, nested_output, nested_probs, nested_features, normalized_features):
		raise Exception('implement this')

class RegularExtendedNet(AbstractExtendedNet):

	def _init_layers(self):
		self.fc1 = nn.Linear(in_features=self.num_features() + num_classes, out_features=num_classes**2)
		self.fc2 = nn.Linear(in_features=num_classes**2 + num_classes, out_features=num_classes)

	def process(self, nested_output, nested_probs, nested_features, normalized_features):
		nested_features = self.__linearize_features(nested_features)
		extended_features = torch.cat((nested_output, nested_features), 1)
		output = self.fc1(torch.cat((nested_features, nested_output), 1))
		output = self.nonlinear(output)
		output = self.fc2(torch.cat((output, nested_output), 1))
		return output

class FCNormalizedNet(AbstractExtendedNet):

	def _init_layers(self):
		# all the features plus the class probability for each class go in
		self.fc1 = nn.Linear(in_features=(self.num_features() + 1)*num_classes, out_features=num_classes)
		self.fc2 = nn.Linear(in_features=num_classes + num_classes, out_features=num_classes) # inputs are from previous layer and last layer in the base net

	def process(self, nested_output, nested_probs, nested_features, normalized_features):

		normalized_output = nested_output * nested_probs

		normalized_features = self.__linearize_features(normalized_features)

		if self.fc_include_class_prob:
			normalized_features = torch.cat((normalized_features, normalized_output), 1)

		output = self.fc1(normalized_features)
		output, nested_output = self.nonlinear(output), self.nonlinear(nested_output)
		output = self.fc2(torch.cat((output, nested_output), 1))

		return output

