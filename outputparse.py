import tabulate

import os
import json

class Parser:
	COLUMNS = ['valid_acc', 'train_acc', 'train_loss']
	def __init__(self, base_folder='', sort_column='valid_acc'):
		self.base_folder = base_folder
		self.sort_column = sort_column

	def _parse_accuracy_line(self, line):
		parts = line.split()
		epoch = parts[1]
		# the 0th and 2nd items are the literals 'epoch:' and 'accuracy:'
		# The 3rd item and afterward constitute the accuracy json, stick them back together, and do some json specific correction:
		accuracy_json = ' '.join(parts[3:]).replace("'", "\"")

		epoch = int(epoch.strip(','))

		accuracy_dict = json.loads(accuracy_json)
		return {
			'valid_acc': accuracy_dict['validation_acc'],
			'train_acc': accuracy_dict['train_acc'],
			'train_loss': accuracy_dict['train_loss']
			}

	def _parse_file(self, file_name):
		with open(file_name, 'r') as f:
			return self._parse_accuracy_line(f.readline())
	def parse_file(self, file_name):
		return self._parse_file(self.resolve_path(file_name))

	def _parse_folder(self, folder_name):
		results = []
		for file_name in os.listdir(folder_name):
			if self._get_suffix(file_name, len('.accuracy')) == '.accuracy':
				data = {'label': self.labelize_file_name(file_name)}
				data.update(self._parse_file('%s/%s' % (folder_name, file_name)))
				results.append(data)
		return results

	def parse_folder(self, folder_name):
		return self._parse_folder(self.resolve_path(folder_name))

	def get_folder_table(self, folder_name):
		data = self.sort(self.parse_folder(folder_name))
		return tabulate.tabulate(data, headers='keys', showindex=True)

	def sort(self, data_list):
		return sorted(data_list, key=lambda x: x[self.sort_column])

	def labelize_file_name(self, file_name):
		return file_name.replace('.accuracy', '').replace('.model', '')

	def resolve_path(self, path):
		return path if not self.base_folder else "%s/%s" % (self.base_folder, path)

	def _get_suffix(self, string, length):
		return string[len(string) - length:len(string)]



if __name__ == '__main__':
	import argparse
	optparser = argparse.ArgumentParser()
	optparser.add_argument('path', help="the path to the accuracy file(s)")
	optparser.add_argument('-f', '--file', dest='file', default=False, action='store_true', help='whether <path> should be interpreted as a file instead of a folder')
	optparser.add_argument('-s', '--sort-column', dest='sortcolumn', default='valid_acc', choices=Parser.COLUMNS, help='what column to sort the data on')
	args = optparser.parse_args()

	parser = Parser(sort_column=args.sortcolumn)
	print(parser.get_folder_table(args.path))

