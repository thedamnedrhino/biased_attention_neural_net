import random

import torch
import torch.nn as nn


class Regularizer:
	TYPES = ('l1', 'l2', 'l1_2')
	def l1(self, p):
		return p.abs().sum()
	def l2(self, p):
		return (p**2).sum()
	def l1over2(self, p):
		return torch.sqrt(p.abs()).sum()

	def __init__(self, reg_type, reg_rate):
		assert reg_type in type(self).TYPES
		self.reg_type = reg_type
		self.reg_rate = reg_rate
		parameter_wise_callback_map = {
				'l1': self.parameter_wise_reg_callback(self.l1),
				'l2': self.parameter_wise_reg_callback(self.l2),
				'l1_2': self.parameter_wise_reg_callback(self.l1over2),
				}
		if self.reg_type in parameter_wise_callback_map:
			self.reg = parameter_wise_callback_map[self.reg_type]
		else:
			raise Exception('don\'t have this regularization type: "%s"' % self.reg_type)

	def regularize(self, layer, loss):
		loss += self.reg(layer)
		return loss

	def parameter_wise_reg_callback(self, parameter_fn):
		def callback(layer):
			count = 0
			total = 0
			for p in layer.parameters():
				total += parameter_fn(p)
				count += 1
			return total/count * self.reg_rate
		return callback


class ExtendedNetFactory:
	NETS = {
			'reg': 'regular',

			'fcN': 'Fully Connected Layer Normalized',

			'featNRO_R': 'Features normalized with normalizers generated from raw output with no non-linear after' ,
			'featNRO_S': 'Features normalized with normalizers generated from raw output with a tanh after',
			'featNRO_Th': 'Features normalized with normalizers generated from raw output with a tanh after' ,
			'featNRO_Smax_g': 'Features normalized with normalizers generated from raw output with a global after',
			'featNRO_Smax_ch': 'same as before, but the softmax is channelwise, i.e: at each locations the channel feature values sum up to 1',

			'featNPO_R': 'Features normalized with normalizers generated from probability output with no non-linear after',
			'featNPO_S':'Features normalized with normalizers generated from probability output with a sigmoid after',
			'featNPO_Th': 'Features normalized with normalizers generated from probability output with a tanh after',
			'featNPO_Smax_g': 'Features normalized with normalizers generated from probability output with a global after',
			'featNPO_Smax_ch': 'same as before, but the softmax is channelwise, i.e: at each locations the channel feature values sum up to 1'}

	def create_net(self, net_name, nested_net, net_args):
		assert net_name in ExtendedNetFactory.NETS, 'net_name argument must be ExtendedNetFactory.NETS: [{}]. "{}" provided.'.format(','.join(ExtendedNetFactory.NETS.keys()), net_name)
		net_factory = FeatureNormalizedNetFactory()
		constructors = {
				'reg': RegularExtendedNet,

				'fcN': FCNormalizedNet,

				'featNRO_R': net_factory.raw_output_raw,
				'featNRO_S': net_factory.raw_output_sigmoid,
				'featNRO_Th': net_factory.raw_output_tanh,
				'featNRO_Smax_g': net_factory.raw_output_softmax_global,
				'featNRO_Smax_ch': net_factory.raw_output_softmax_channelwise,

				'featNPO_R': net_factory.probability_output_raw,
				'featNPO_S': net_factory.probability_output_sigmoid,
				'featNPO_Th': net_factory.probability_output_tanh,
				'featNPO_Smax_g': net_factory.probability_output_softmax_global,
				'featNPO_Smax_ch': net_factory.probability_output_softmax_channelwise
				}

		return constructors[net_name](nested_net, **net_args)


class AbstractExtendedNet(nn.Module):
	def __init__(self, nested_model, nonlinear='sigmoid', fc_include_class_prob=True, enable_fc_class_correlate=True, include_original=True, regularization_rate=0.0, regularization_type='l2', **kwargs):
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
		regularization_rate = float(regularization_rate) if regularization_rate is not None else 0
		self.regularize = bool(regularization_rate)
		if self.regularize:
			self.regularizer = Regularizer(regularization_type, regularization_rate)
		for k, v in kwargs:
			raise Exception('what are these argument %s, this is is\'nt supposed to happen!' % (kwargs.items()))


		self.outputs = None
		self.add_outputs = []
		# super verbosity is toggled from outside based on the validation accuracy. But we should fix this line regardless.
		self.super_verbose = False
		# each network will have their own metric
		self.metrics = None
		self.reset_reg_diffs()
		self._init_layers()

	def num_features(self):
		return self.nested_model.num_features()

	def forward(self, input):
		nested_output = self.nested_model(input)
		nested_features = self.nested_model.features1d
		shaped_nested_features = self.nested_model.features
		self.nested_output = nested_output
		self.nested_features = nested_features
		self.shaped_nested_features = shaped_nested_features

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


	def _reguralize_layer(self, layer, loss):
		if not self.regularize:
			return loss
		# loss is mutable so we can only keep it this way
		old_loss = loss.item()
		# regularize parameters in fc1 to selectively choose features
		loss = self._add_regularization(layer, loss)
		new_loss = loss.item()
		diff = new_loss - old_loss
		if self.super_verbose:
			# print("{} - {} = {} = {}% * {}".format(new_loss, old_loss, diff, diff/old_loss * 100, old_loss))
		# else:
			# save output every 10 batches - we don't want to keep too much data: (5 can be any other number in [0, 10])
			self.reg_diffs[0].append(diff)
			self.reg_diffs[1].append(diff/old_loss)
		return loss

	def _add_regularization(self, layer, loss):
		return self.regularizer.regularize(layer, loss)
		l2 = 0
		count = 0
		for p in layer.parameters():
			l2 += (p**2).sum()
			count += 1
		l2 /= count
		loss += l2 * self.regularization_rate
		return loss

	def loss_hook(self, loss):
		return loss

	def reset_reg_diffs(self):
		"""
		To log regularization... very badly put down, I know.
		"""
		self.reg_diffs = [[], []]

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
		if self.super_verbose:
			self.save_output_values(nested_features, nested_output)
		output = self.fc1(extended_features)
		output = self.nonlinear(output)
		nested_output = self.nonlinear(nested_output)
		output = self.fc2(torch.cat((output, nested_output), 1))
		return output

	def loss_hook(self, loss):
		return self._reguralize_layer(self.fc1, loss)

class FCNormalizedNet(AbstractExtendedNet):

	def _init_layers(self):
		self.metrics = FCNormalizedNet.Metrics(self)
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
			if self.super_verbose:
				self.save_output_values(normalized_features, normalized_output)
			normalized_features = torch.cat((normalized_features, normalized_output), 1)


		output = self.fc1(self._linearize_features_(normalized_features))

		output, nested_output = self.softmax(output), self.softmax(nested_output)

		self.metrics.batch_run(output, nested_output)
		if self.include_original:
			output = torch.cat((output, nested_output), 1)

		output = self.fc2(output)

		# self.normalizeds, self.normals = output, nested_output

		return output

	def loss_hook(self, loss):
		return self._reguralize_layer(self.fc1, loss)

	def reset_metrics(self):
		self.metrics.reset()

	class Metrics:
		def __init__(self, fcnet):
			self.fcnet = fcnet
			self.reset()
		def reset(self):
			self.batch_data = []
			self.diff_avg = None
			self.relative_diff_avg = None
			self.contradiction_avg = None

		def __add_batch_data(self, diff_avg, relative_diff_avg, contradiction_avg):
			self.batch_data.append((diff_avg, relative_diff_avg, contradiction_avg))

		def batch_run(self, output, nested_output):
			sms = (self.fcnet.softmax(output), self.fcnet.softmax(nested_output))
			diff = torch.abs(sms[0] - sms[1])
			relative_diff = diff/sms[0]
			diff, relative_diff = torch.sum(diff).item(), torch.sum(relative_diff).item()
			ls = (torch.argmax(output, 1), torch.argmax(nested_output, 1))
			contradictions = torch.sum(ls[0] != ls[1]).item()
			batch_size = output.size(0)
			diff_avg, relative_diff_avg, contradiction_avg = diff/batch_size, relative_diff/batch_size, contradictions/batch_size
			self.__add_batch_data(diff_avg, relative_diff_avg, contradiction_avg)

		def aggregate(self):
			aggregated = [0.0, 0.0, 0.0]
			for d in self.batch_data:
				for i, v in enumerate(d):
					aggregated[i] += v

			for i in range(len(aggregated)):
				aggregated[i] /= len(self.batch_data)

			self.diff_avg = aggregated[0]
			self.relative_diff_avg = aggregated[1]
			self.contradiction_avg = aggregated[2]


class FeatureNormalizedNet(AbstractExtendedNet):

	def set_normalization_layer(self, layer):
		self.normalization_layer = layer

	def _init_layers(self):
		self.fc1 = nn.Linear(in_features=self.num_features(), out_features=self.num_classes**2)
		self.fc2 = nn.Linear(in_features=self.num_classes**2, out_features=self.num_classes)
		self.final_fc = nn.Sequential(self.fc1, self.fc2)

	def _process(self, nested_output, nested_probs, nested_features, normalized_features):

		assert not self.normalization_layer is None, 'The normalization layer must first be set before using this network. use FeatureNormalizedNetFactory to instantiate feature normalized networks'
		# discard the previous normalized features, we want to normalize the features independently in this net
		# normalized_features = self.normalize_features(nested_output, nested_probs, nested_features)
		normalized_features = self.normalization_layer(nested_output, nested_probs, nested_features, self.shaped_nested_features, normalized_features)

		output = self.nonlinear(normalized_features)

		self.normalizeds, self.normals = output, []
		output = self.final_fc(output)

		return output

	def loss_hook(self, loss):
		return self._reguralize_layer(self.fc1, loss)

class FeatureNormalizedNetFactory:
	def __net(self, nested_model, layer, **net_kwargs):
		net = FeatureNormalizedNet(nested_model, **net_kwargs)
		net.set_normalization_layer(layer)
		return net

	def raw_output_raw(self, nested_model, **kwargs):
		normalization_layer = FeatureNormalizationLayer_RawOutput_Raw(nested_model.num_classes, nested_model.num_features())
		return self.__net(nested_model, normalization_layer, **kwargs)
	def raw_output_sigmoid(self, nested_model, **kwargs):
		normalization_layer = FeatureNormalizationLayer_RawOutput_Sigmoid(nested_model.num_classes, nested_model.num_features())
		return self.__net(nested_model, normalization_layer, **kwargs)
	def raw_output_tanh(self, nested_model, **kwargs):
		normalization_layer = FeatureNormalizationLayer_RawOutput_Tanh(nested_model.num_classes, nested_model.num_features())
		return self.__net(nested_model, normalization_layer, **kwargs)
	def raw_output_softmax_global(self, nested_model, **kwargs):
		normalization_layer = FeatureNormalizationLayer_RawOutput_Softmax_Global(nested_model.num_classes, nested_model.num_features())
		return self.__net(nested_model, normalization_layer, **kwargs)
	def raw_output_softmax_channelwise(self, nested_model, **kwargs):
		normalization_layer = FeatureNormalizationLayer_RawOutput_Softmax_Channelwise(nested_model.num_classes, nested_model.num_features())
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
	def probability_output_softmax_global(self, nested_model, **kwargs):
		normalization_layer = FeatureNormalizationLayer_ProbabilityOutput_Softmax_Global(nested_model.num_classes, nested_model.num_features())
		return self.__net(nested_model, normalization_layer, **kwargs)
	def probability_output_softmax_channelwise(self, nested_model, **kwargs):
		normalization_layer = FeatureNormalizationLayer_ProbabilityOutput_Softmax_Channelwise(nested_model.num_classes, nested_model.num_features())
		return self.__net(nested_model, normalization_layer, **kwargs)

class AbstractFeatureNormalizationLayer(nn.Module):

	def __init__(self, num_classes, num_features):
		super(AbstractFeatureNormalizationLayer, self).__init__()
		self.num_classes = num_classes
		self.num_features = num_features
		self.normalizer_fc = nn.Linear(in_features=self.num_classes, out_features=num_features)
		self.sigmoid = nn.Sigmoid()
		self.tanh = nn.Tanh()
		self.softmax = nn.Softmax()
		self.softmax2d = nn.Softmax2d()

	def forward(self, nested_output, nested_probs, nested_features, shaped_nested_features, normalized_features):
		return self.normalize_features(nested_output, nested_probs, nested_features, shaped_nested_features)

	def normalize_features(self, nested_output, nested_probs, nested_features, shaped_nested_features):
		feature_normalizers = self.feature_normalizers(nested_output, nested_probs, nested_features, shaped_nested_features)
		normalized_features = nested_features * feature_normalizers
		return normalized_features

	def feature_normalizers(self, nested_output, nested_probs, nested_features):
		raise Exception('implement this')


class FeatureNormalizationLayer_RawOutput_Raw(AbstractFeatureNormalizationLayer):
	def feature_normalizers(self, nested_output, nested_probs, nested_features, shaped_nested_features):
	 	return self.normalizer_fc(nested_output)
class FeatureNormalizationLayer_RawOutput_Sigmoid(AbstractFeatureNormalizationLayer):
	def feature_normalizers(self, nested_output, nested_probs, nested_features, shaped_nested_features):
	 	return self.sigmoid(self.normalizer_fc(nested_output))
class FeatureNormalizationLayer_RawOutput_Tanh(AbstractFeatureNormalizationLayer):
	def feature_normalizers(self, nested_output, nested_probs, nested_features, shaped_nested_features):
	 	return self.tanh(self.normalizer_fc(nested_output))
class FeatureNormalizationLayer_RawOutput_Softmax_Global(AbstractFeatureNormalizationLayer):
	def feature_normalizers(self, nested_output, nested_probs, nested_features, shaped_nested_features):
	 	return self.softmax(self.normalizer_fc(nested_output))
class FeatureNormalizationLayer_RawOutput_Softmax_Channelwise(AbstractFeatureNormalizationLayer):
	def feature_normalizers(self, nested_output, nested_probs, nested_features, shaped_nested_features):
	 	return self.softmax2d(self.normalizer_fc(nested_output).view(shaped_nested_features.size())).view(nested_features.size())

class FeatureNormalizationLayer_ProbabilityOutput_Raw(AbstractFeatureNormalizationLayer):
	def feature_normalizers(self, nested_output, nested_probs, nested_features, shaped_nested_features):
		return self.normalizer_fc(nested_probs)
class FeatureNormalizationLayer_ProbabilityOutput_Sigmoid(AbstractFeatureNormalizationLayer):
	def feature_normalizers(self, nested_output, nested_probs, nested_features, shaped_nested_features):
		return self.sigmoid(self.normalizer_fc(nested_probs))
class FeatureNormalizationLayer_ProbabilityOutput_Tanh(AbstractFeatureNormalizationLayer):
	def feature_normalizers(self, nested_output, nested_probs, nested_features, shaped_nested_features):
		return self.tanh(self.normalizer_fc(nested_probs))
class FeatureNormalizationLayer_ProbabilityOutput_Softmax_Global(AbstractFeatureNormalizationLayer):
	def feature_normalizers(self, nested_output, nested_probs, nested_features, shaped_nested_features):
		return self.softmax(self.normalizer_fc(nested_probs))
class FeatureNormalizationLayer_ProbabilityOutput_Softmax_Channelwise(AbstractFeatureNormalizationLayer):
	def feature_normalizers(self, nested_output, nested_probs, nested_features, shaped_nested_features):
		return self.softmax2d(self.normalizer_fc(nested_probs).view(shaped_nested_features.size())).view(nested_features.size())

