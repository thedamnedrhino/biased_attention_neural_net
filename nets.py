import torch
import torch.nn as nn

class AbstractExtendedNet(nn.Module):
	def __init__(self, nested_model, nonlinear='sigmoid', fc_include_class_prob=True, enable_fc_class_correlate=True):
		super(AbstractExtendedNet, self).__init__()
		self.num_classes = nested_model.num_classes
		self.hidden_channels = nested_model.hidden_channels
		self.height = nested_model.height
		self.width = nested_model.width
		self.nested_model = nested_model
		self.sigmoid = nn.Sigmoid()
		self.relu = nn.ReLU()
		nonlinearmap = {'sigmoid': self.sigmoid, 'relu': self.relu, 'none': lambda x: x}
		assert nonlinear in nonlinearmap
		self.nonlinear = nonlinearmap[nonlinear]
		self.fc_include_class_prob = fc_include_class_prob
		self.fc_class_correlate = nn.Linear(in_features=self.num_classes, out_features=self.num_classes)
		self.enable_fc_class_correlate = enable_fc_class_correlate

		self._init_layers()

	def num_features(self):
		return self.nested_model.num_features()

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

	def _linearize_features_(self, features):
		return features.view(-1, features.size(-2) * features.size(-1))

	def _init_layers(self):
		raise Exception('implement this')
	def _process(self, nested_output, nested_probs, nested_features, normalized_features):
		raise Exception('implement this')

class RegularExtendedNet(AbstractExtendedNet):

	def _init_layers(self):
		self.fc1 = nn.Linear(in_features=self.num_features() + self.num_classes, out_features=self.num_classes**2)
		self.fc2 = nn.Linear(in_features=num_classes**2 + self.num_classes, out_features=self.num_classes)

	def _process(self, nested_output, nested_probs, nested_features, normalized_features):
		nested_features = self._linearize_features_(nested_features)
		extended_features = torch.cat((nested_output, nested_features), 1)
		output = self.fc1(torch.cat((nested_features, nested_output), 1))
		output = self.nonlinear(output)
		output = self.fc2(torch.cat((output, nested_output), 1))
		return output

class FCNormalizedNet(AbstractExtendedNet):

	def _init_layers(self):
		# all the features plus the class probability for each class go in
		self.fc1 = nn.Linear(in_features=self.num_features()*self.num_classes + self.num_classes, out_features=self.num_classes)
		self.fc2 = nn.Linear(in_features=self.num_classes + self.num_classes, out_features=self.num_classes) # inputs are from previous layer and last layer in the base net

	def _process(self, nested_output, nested_probs, nested_features, normalized_features):

		normalized_output = nested_output * nested_probs

		normalized_features = self._linearize_features_(normalized_features)

		normalized_features = torch.cat((normalized_features, normalized_output), 1)

		output = self.fc1(normalized_features)

		output, nested_output = self.nonlinear(output), self.nonlinear(nested_output)

		output = self.fc2(torch.cat((output, nested_output), 1))

		return output

class FeatureNormalizedWithRawOutputNet(AbstractExtendedNet):

	def _init_layers(self):
		self.fc2 = self._final_fc_(self.num_features())

	def _process(self, nested_output, nested_probs, nested_features, normalized_features):

		# discard the previous normalized features, we want to normalize the features independently in this net
		normalized_features = self.normalize_features(nested_output, nested_probs, nested_features)

		output = self.nonlinear(normalized_features)

		output = self.fc2(output)

		return output


class FeatureNormalizedWithProbability_Raw_OutputNet(FeatureNormalizedWithRawOutputNet):

	def get_normalization_coefficients(self, nested_output, nested_probs, nested_features):
		return self.fc1(nested_output)

class FeatureNormalizedWithProbability_Sigmoid_OutputNet(FeatureNormalizedWithRawOutputNet):

	def get_normalization_coefficients(self, nested_output, nested_probs, nested_features):
		return self.sigmoid(self.fc1(nested_output))

class AbstractFeatureNormalizationLayer(nn.Module):

	def __init__(self, num_features, num_classes):
		super(FeatureNormalizationLayer, self).__init__()
		self.num_features = num_features
		self.num_classes = num_classes
		self.normalizer_fc = nn.Linear(in_features=self.num_classes, out_features=num_features)

	def forward(self, nested_output, nested_probs, nested_features, normalized_features):
		return self.normalize_features(nested_output, nested_probs, nested_features)

	def normalize_features(self, nested_output, nested_probs, nested_features):
		feature_normalizers = self.feature_normalizers(nested_output, nested_probs, nested_features)
		normalized_features = nested_features * feature_normalizers
		return normalized_features

	def feature_normalizers(self, nested_output, nested_probs, nested_features):
		raise Exception('implement this')

class FeatureNormalizationLayer_RawOutput(AbstractFeatureNormalizationLayer):
	def feature_normalizers(self, nested_output, nested_probs, nested_features):
	 	return = self.sigmoid(self.fc1(nested_output))

class FeatureNormalizationLayer_ProbabilityOutput_Raw(AbstractFeatureNormalizationLayer):
	def feature_normalizers(self, nested_output, nested_probs, nested_features):
		return
