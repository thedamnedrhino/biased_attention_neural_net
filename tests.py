import torch

import math

import nets


def regularization_mask_test():
	r = nets.Regularizer('cos', 1)
	t = torch.randn(1000)*10
	for v in (torch.randn(20)*10).data:
		gm, lm = [f(t, v) for f in (r.greater_than_mask, r.less_than_mask)]
		for e, eg, el in zip(t.data, gm.data, lm.data):
			if e > v:
				assert eg == 1
			else:
				assert eg == 0
			if e < v:
				assert el == 1
			else:
				assert el == 0

		for v in (torch.randn(20)*10).data:
			if v < 0:
				v = -v # we are only testing this for absolute values
			norm_mask, inverse_mask = r.norm_masks(t, v)
			for e, en, ei in zip(t.data, norm_mask.data, inverse_mask.data):
				if e >= v or e <= -v:
					assert en == 0
					assert ei == 1
				else:
					assert en == 1, '%s, %s, %s, %s' % (e, v, en, ei)
					assert ei == 0

def cos_tensor_test():
	nets.Regularizer.cos_threshold = lambda self, p: 2.000001
	r = nets.Regularizer('cos', 1)
	d = torch.tensor([-10, -2, -1, 0, 0.5, 1, 1.5, 2, 3])
	ground_truth = torch.tensor([0, 2, 1, 0, 1 - 0.7, 1, 1.7, 2, 0])
	c = r.cos_tensor(d)
	diff = c - ground_truth
	assert ((c - ground_truth).abs() < 0.1).all()

def full_cos_reg_test():
	nets.Regularizer.cos_threshold = lambda self, p: 2.000001
	r = nets.Regularizer('cos', 1)
	d = torch.tensor([-10, -2, -1, 0, 0.5, 1, 1.5, 2, 3])
	cos_ground_truth = torch.tensor([0, 2, 1, 0, 1 - 0.7, 1, 1.7, 2, 0])
	ground_truth_sum = cos_ground_truth.sum() + 10 + 3
	assert abs(ground_truth_sum - r.cos(d, out_of_norm_reg_type='l1')) < 0.1

if __name__== '__main__':
	import sys
	if len(sys.argv) > 1 and sys.argv[1] == 'all':
		regularization_mask_test()
		cos_tensor_test()
	full_cos_reg_test()
