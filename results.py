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


def run():
	r = Results()
	print(r.top_10(table=True))

if __name__ == '__main__':
	run()
