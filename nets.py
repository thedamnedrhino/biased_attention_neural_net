import torch
import torch.nn as nn



class ExtendedNetFactory:
	NETS = {'reg': 'regular', 'fcN': 'Fully Connected Layer Normalized', 'featNRO_R': 'Features normalized with normalizers generated from raw output with no non-linear after' ,'featNRO_S': 'Features normalized with normalizers generated from raw output with a tanh after', 'featNRO_Th': 'Features normalized with normalizers generated from raw output with a tanh after' , 'featNPO_R': 'Features normalized with normalizers generated from probability output with no non-linear after', 'featNPO_S': 'Features normalized with normalizers generated from probability output with a sigmoid after', 'featNPO_Th': 'Features normalized with normalizers generated from probability output with a tanh after'}

	def create_net(self, net_name, nested_net, net_args):
		assert net_name in ExtendedNetFactory.NETS, 'net_name argument must be ExtendedNetFactory.NETS: [{}]. "{}" provided.'.format(','.join(ExtendedNetFactory.NETS.keys()), net_name)
		net_factory = FeatureNormalizedNetFactory()
		constructors = {
				'reg': RegularExtendedNet,
				'fcN': FCNormalizedNet,
				'featNRO_R': net_factory.raw_output_raw,
				'featNRO_S': net_factory.raw_output_sigmoid,
				'featNRO_Th': net_factory.raw_output_tanh,
				'featNPO_R': net_factory.probability_output_raw,
				'featNPO_S': net_factory.probability_output_sigmoid,
				'featNPO_Th': net_factory.probability_output_tanh
				}

		return constructors[net_name](nested_net, **net_args)


class AbstractExtendedNet(nn.Module):
	def __init__(self, nested_model, nonlinear='sigmoid', fc_include_class_prob=True, enable_fc_class_correlate=True, include_original=True, **kwargs):
		super(AbstractExtendedNet, self).__init__()
		self.num_classes = nested_model.num_classes
		self.hidden_channels = nested_model.hidden_channels
		self.height = nested_model.height
		self.width = nested_model.width
		self.nested_model = nested_model
		self.sigmoid = nn.Sigmoid()
		self.tanh = nn.Tanh()
		self.relu = nn.ReLU()
		self.softmax = nn.Softmax()
		nonlinearmap = {'sigmoid': self.sigmoid, 'tanh': self.tanh, 'relu': self.relu, 'softmax': self.softmax, 'none': lambda x: x}
		assert nonlinear in nonlinearmap
		self.nonlinear = nonlinearmap[nonlinear]
		self.fc_include_class_prob = fc_include_class_prob
		self.fc_class_correlate = nn.Linear(in_features=self.num_classes, out_features=self.num_classes)
		self.enable_fc_class_correlate = enable_fc_class_correlate
		self.include_original = include_original
		for k, v in kwargs:
			self.__set_attr__(k, v)

		self.outputs = None
		self.add_outputs = []
		self.super_verbose = False
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
	def print_outputs(self):
		if self.outputs is None:
			return
		print(self.outputs)

	def save_output_values(self, new_output, old_output):
		outputs = {
				'new_output_sum': torch.sum(new_output).item(),
				'new_output^2_sum': torch.sum(new_output**2).item(),
				'new_output^2_max': torch.max(new_output**2).item(),
				'new_output^2_mean': torch.mean(new_output**2).item(),
				'old_output_sum': torch.sum(old_output).item(),
				'old_output^2_sum': torch.sum(old_output**2).item(),
				'old_output^2_max': torch.max(old_output**2).item(),
				'old_output^2_mean': torch.mean(old_output**2).item()
				}
		self.outputs = "new - size: {}, sum: {}, ^2sum: {}, ^2max: {}, ^2_mean: {}\n".format(new_output.size(), outputs['new_output_sum'], outputs['new_output^2_sum'], outputs['new_output^2_max'], outputs['new_output^2_mean'])
		self.outputs += "old - size: {}, sum: {}, ^2sum: {}, ^2max: {}, ^2_mean: {}\n".format(old_output.size(), outputs['old_output_sum'], outputs['old_output^2_sum'], outputs['old_output^2_max'], outputs['old_output^2_mean'])
		self.outputs += ("\n").join(["{}".format(e) for e in self.add_outputs])

class RegularExtendedNet(AbstractExtendedNet):

	def _init_layers(self):
		if self.include_original is False:
			raise Exception('the "regular" extended net does not work without the originals')
		self.fc1 = nn.Linear(in_features=self.num_features() + self.num_classes, out_features=self.num_classes**2)
		self.fc2 = nn.Linear(in_features=self.num_classes**2 + self.num_classes, out_features=self.num_classes)

	def _process(self, nested_output, nested_probs, nested_features, normalized_features):
		# nested_features = self._linearize_features_(nested_features)
		nested_features, nested_output = nested_features/self.num_features(), nested_output/self.num_classes
		extended_features = torch.cat((nested_output, nested_features), 1)
		self.save_output_values(nested_features, nested_output)
		output = self.fc1(extended_features)
		output = self.nonlinear(output)
		nested_output = self.nonlinear(nested_output)
		output = self.fc2(torch.cat((output, nested_output), 1))
		return output

class FCNormalizedNet(AbstractExtendedNet):

	def _init_layers(self):
		# all the features plus the class probability for each class go in
		# self.fc1 = nn.Linear(in_features=self.num_features()*self.num_classes + int(self.include_original)*self.num_classes**2, out_features=self.num_classes)
		# I decided against the above, because combining feature maps, with class values (i.e: result of passing maps through an FC) is hard see notes in self._process()
		self.fc1 = nn.Linear(in_features=self.num_features()*self.num_classes, out_features=self.num_classes)
		self.fc2 = nn.Linear(in_features=self.num_classes + int(self.include_original) * self.num_classes, out_features=self.num_classes) # inputs are from previous layer and last layer in the base net

	def _process(self, nested_output, nested_probs, nested_features, normalized_features):

		def save_stats(self):
			"""
			call this to print some additional stats about the inputs
			"""
			shaped = self.nested_model.features
			s = shaped**2
			s = s.view(s.size(0), self.hidden_channels, -1)
			atts = {
					'max': torch.max(s, 2),
					'mean': torch.mean(s, 2),
					}
			self.add_outputs.append("original nested feature stats: {}".format(atts))

		# we don't need the normalized output anymore, as we pass it through a softmax function. Since the new outputs will also go through the softmax, this will achieve
		# the "normalization" we require to combine the new outputs (i.e: class "values" - converted to probs through softmax) with the old outputs
		# normalized_output = nested_output.unsqueeze(1).transpose(-2, -1).matmul(nested_probs.unsqueeze(1)).view(-1, self.num_classes**2)

		# if self.include_original:
		if False:
			"""
			This if was supposed to combine the outputs from the original network with the new features in this network in the first fc layer
			I decided against this in favor of just combining the OUTPUT of the first layer with the outputs of the original network
			This approach would have required a lot of technical provisions to make it work properly, as the number of features vastly differs from the number of outputs (i.e: number of classes)
			"""
			normalized_features = normalized_features / self.num_features()
			normalized_output = normalized_output / self.num_classes
			self.save_output_values(normalized_features, normalized_output)
			normalized_features = torch.cat((normalized_features, normalized_output), 1)


		output = self.fc1(self._linearize_features_(normalized_features))

		output, nested_output = self.softmax(output), self.softmax(nested_output)

		if self.super_verbose:
			print("{}\n************\n{}\n--------------", output[0:9], nested_output[0:9])
		if self.include_original:
			output = torch.cat((output, nested_output), 1)

		output = self.fc2(output)

		# self.normalizeds, self.normals = output, nested_output

		return output

class FeatureNormalizedNet(AbstractExtendedNet):

	def __init__(self, nested_model, normalization_layer, nonlinear='sigmoid', fc_include_class_prob=True, enable_fc_class_correlate=True):
		super(FeatureNormalizedNet, self).__init__(nested_model, nonlinear, fc_include_class_prob, enable_fc_class_correlate)
		self.normalization_layer = normalization_layer
		self.final_fc = nn.Linear(in_features=self.num_features(), out_features=self.num_classes)

	def _init_layers(self):
		pass # already done in __init__()

	def _process(self, nested_output, nested_probs, nested_features, normalized_features):

		# discard the previous normalized features, we want to normalize the features independently in this net
		# normalized_features = self.normalize_features(nested_output, nested_probs, nested_features)
		normalized_features = self.normalization_layer(nested_output, nested_probs, nested_features, normalized_features)

		output = self.nonlinear(normalized_features)

		self.normalizeds, self.normals = output, []
		output = self.final_fc(output)

		return output

class FeatureNormalizedNetFactory:
	def __net(self, nested_model, layer, **net_kwargs):
		return FeatureNormalizedNet(nested_model, layer, **net_kwargs)

	def raw_output_raw(self, nested_model, **kwargs):
		normalization_layer = FeatureNormalizationLayer_RawOutput_Raw(nested_model.num_classes, nested_model.num_features())
		return self.__net(nested_model, normalization_layer, **kwargs)
	def raw_output_sigmoid(self, nested_model, **kwargs):
		normalization_layer = FeatureNormalizationLayer_RawOutput_Sigmoid(nested_model.num_classes, nested_model.num_features())
		return self.__net(nested_model, normalization_layer, **kwargs)
	def raw_output_tanh(self, nested_model, **kwargs):
		normalization_layer = FeatureNormalizationLayer_RawOutput_Tanh(nested_model.num_classes, nested_model.num_features())
		return self.__net(nested_model, normalization_layer, **kwargs)
	def probability_output_raw(self, nested_model, **kwargs):
		normalization_layer = FeatureNormalizationLayer_ProbabilityOutput_Raw(nested_model.num_classes, nested_model.num_features())
		return self.__net(nested_model, normalization_layer, **kwargs)
	def probability_output_sigmoid(self, nested_model, **kwargs):
		normalization_layer = FeatureNormalizationLayer_ProbabilityOutput_Sigmoid(nested_model.num_classes, nested_model.num_features())
		return self.__net(nested_model, normalization_layer, **kwargs)
	def probability_output_tanh(self, nested_model, **kwargs):
		normalization_layer = FeatureNormalizationLayer_ProbabilityOutput_Tanh(nested_model.num_classes, nested_model.num_features())
		return self.__net(nested_model, normalization_layer, **kwargs)

class AbstractFeatureNormalizationLayer(nn.Module):

	def __init__(self, num_classes, num_features):
		super(AbstractFeatureNormalizationLayer, self).__init__()
		self.num_classes = num_classes
		self.num_features = num_features
		self.normalizer_fc = nn.Linear(in_features=self.num_classes, out_features=num_features)
		self.sigmoid = nn.Sigmoid()
		self.tanh = nn.Tanh()

	def forward(self, nested_output, nested_probs, nested_features, normalized_features):
		return self.normalize_features(nested_output, nested_probs, nested_features)

	def normalize_features(self, nested_output, nested_probs, nested_features):
		feature_normalizers = self.feature_normalizers(nested_output, nested_probs, nested_features)
		normalized_features = nested_features * feature_normalizers
		return normalized_features

	def feature_normalizers(self, nested_output, nested_probs, nested_features):
		raise Exception('implement this')


class FeatureNormalizationLayer_RawOutput_Raw(AbstractFeatureNormalizationLayer):
	def feature_normalizers(self, nested_output, nested_probs, nested_features):
	 	return self.normalizer_fc(nested_output)
class FeatureNormalizationLayer_RawOutput_Sigmoid(AbstractFeatureNormalizationLayer):
	def feature_normalizers(self, nested_output, nested_probs, nested_features):
	 	return self.sigmoid(self.normalizer_fc(nested_output))
class FeatureNormalizationLayer_RawOutput_Tanh(AbstractFeatureNormalizationLayer):
	def feature_normalizers(self, nested_output, nested_probs, nested_features):
	 	return self.tanh(self.normalizer_fc(nested_output))

class FeatureNormalizationLayer_ProbabilityOutput_Raw(AbstractFeatureNormalizationLayer):
	def feature_normalizers(self, nested_output, nested_probs, nested_features):
		return self.normalizer_fc(nested_probs)
class FeatureNormalizationLayer_ProbabilityOutput_Sigmoid(AbstractFeatureNormalizationLayer):
	def feature_normalizers(self, nested_output, nested_probs, nested_features):
		return self.sigmoid(self.normalizer_fc(nested_probs))
class FeatureNormalizationLayer_ProbabilityOutput_Tanh(AbstractFeatureNormalizationLayer):
	def feature_normalizers(self, nested_output, nested_probs, nested_features):
		return self.tanh(self.normalizer_fc(nested_probs))

