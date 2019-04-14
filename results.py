import outputparse

class Results:
	OUTPUT_FOLDERS=[
			('outputs/full/vanilla/', 'acc'),
			('full_diff_base_net_outputs', 'acc'),
	]

	def __init__(self, sort='valid_acc'):
		self.p = outputparse.Parser(sort_column=sort)

	def top_10(self):
		return self.top_k(10)

	def top_k(self, k):
		results = self.p.parse_folders([folder for folder, mode in type(self).OUTPUT_FOLDERS], sort=True)
		return results[len(results) - k:]


def run():
	r = Results()
	r.top_10()

if __name__ == '__main__':
	run()
