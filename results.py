import matplotlib.pyplot as plt
import numpy as np

import math

import outputparse

class Results:
	OUTPUT_FOLDERS=[
			('experiment_outputs/feat_norm_outputs/2/', 'acc'),
			('outputs/full/vanilla/', 'acc'),
			('full_diff_base_nets_outputs/', 'acc'),
	]

	def __init__(self, sort='valid_acc'):
		self.p = outputparse.Parser(sort_column=sort)

	def top_10(self, table=False):
		return self.top_k(10, table=table)

	def top_k(self, k, table=False):
		results = self.p.parse_folders([folder for folder, mode in type(self).OUTPUT_FOLDERS], sort=True)
		top_k = results[len(results) - k:]
		return self.p.tabulate(top_k) if table else top_k

class Plots:
	def cos_reg(self):
		k = 5
		x = np.arange(-3, 3, 0.01)
		f = (lambda x: math.cos(math.pi*x - math.pi) + 1 if abs(x) < 1 else k*abs(x) - k + 2)
		y = [f(xi) for xi in x]
		fig, (a0, a1) = plt.subplots(nrows=2)
		a0.set_title('lcos(w)', y=1.05)
		a0.grid(True)
		a0.margins(x=0)
		a0.plot(x, y, 'k')
		#plt.subplot(211)
		#plt.grid(True)
		#plt.title('p(w)')
		#plt.plot(x, y, 'k')

		fp = (lambda x: -math.sin(math.pi*x - math.pi) * math.pi if abs(x) < 1 else k*abs(x)/x)
		yp = [fp(xi) for xi in x]
		a1.set_title('lcos\'(w)', y=1.05)
		a1.grid(True)
		a1.plot(x, yp, 'r', x, [k for i in x], 'y--', x, [-k for i in x], 'y--')
		a1.annotate('lcos\'=K', xy=(0.0, k), xytext=(-1, k-1.1))
		a1.annotate('lcos\'=-K', xy=(0.0, -k), xytext=(1, -k+0.5))
		#plt.subplot(212)
		#plt.title('p\'(w)')
		#plt.plot(x, yp, 'r--')
		#plt.grid(True)
		#plt.subplots_adjust(top=1.1)
		plt.subplots_adjust(hspace=0.4)
		a1.margins(x=0.0)
		plt.show()

	def top_5_acc(self):
		N = 5
		extended = (81.8, 81.6, 81.5, 81.4, 81.4)
		base = [80.0] * 5
		print(base)
		# menStd = (2, 3, 4, 1, 2)
		# womenStd = (3, 5, 2, 3, 3)
		ind = np.arange(N)    # the x locations for the groups
		width = 0.35       # the width of the bars: can also be len(x) sequence

		p2 = plt.bar(ind, extended, width)
		p1 = plt.bar(ind, base, width)

		bottom=79.0
		top=82
		plt.ylabel('Scores (%)')
		plt.title('Extended Network Scores')
		plt.xticks(ind, ('FNN', 'AN(1)', 'AN(2)', 'AN(3)', 'EN'))
		plt.yticks(np.arange(bottom, top, 0.5))
		plt.ylim(bottom, top)
		plt.legend((p1[0], p2[0]), ('Base 80%', 'Extended Network'))
		def autolabel(rects, xpos='center'):
			"""
			Attach a text label above each bar in *rects*, displaying its height.

			*xpos* indicates which side to place the text w.r.t. the center of
			the bar. It can be one of the following {'center', 'right', 'left'}.
			"""

			xpos = xpos.lower()  # normalize the case of the parameter
			ha = {'center': 'center', 'right': 'left', 'left': 'right'}
			offset = {'center': 0.5, 'right': 0.57, 'left': 0.43}  # x_txt = x + w*off

			for rect in rects:
				height = rect.get_height()
				plt.text(rect.get_x() + rect.get_width()*offset[xpos], 1*height,
						 '{}'.format(height), ha=ha[xpos], va='bottom')

		autolabel(p2)
		plt.show()
P = Plots()

def run():
	r = Results()
	print(r.top_10(table=True))

def cos_reg():
	p = Plots()
	p.cos_reg()

def top_5_acc():
	P.top_5_acc()

if __name__ == '__main__':
	#run()
	#cos_reg()
	top_5_acc()
