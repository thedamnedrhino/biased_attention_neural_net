import torch
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

regularization_mask_test()
