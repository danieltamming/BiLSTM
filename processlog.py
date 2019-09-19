import numpy as np
import matplotlib.pyplot as plt

num_folds = 10

def process_file(f):
	results_by_pct = {}
	for _ in range(11):
		pct, arr = process_pct(f)
		results_by_pct[pct] = arr
	return results_by_pct

def process_pct(f):
	line = f.readline()
	pct = line.split()[1]
	fold_accs = []
	line = f.readline().replace('INFO:BiLSTMAgent:','').rstrip('\n')
	for _ in range(num_folds):
		accs = process_fold(f)
		fold_accs.append(accs)
	return pct, fold_accs


def process_fold(f):
	'''
	Gets file with 'Fold number x' as the next line to be read
	'''
	accs = []
	f.readline()
	line = f.readline().replace('INFO:BiLSTMAgent:','').rstrip('\n')
	while line[0] == 'V' or line[0] == 'T':
		if line[0] == 'V':
			accs.append(float(line.split()[-1]))
		line = f.readline().replace('INFO:BiLSTMAgent:','').rstrip('\n')
	return accs

def get_avg_accs(results_by_pct):
	avg_accs_by_pct = {}
	for pct, fold_accs in results_by_pct.items():
		min_epochs = min([len(fold) for fold in fold_accs])
		print(max([len(fold) for fold in fold_accs]))
		for i, fold in enumerate(fold_accs):
			fold_accs[i] = fold[:min_epochs]
		accs = np.array(fold_accs)
		avg_accs = np.mean(accs, axis=0)
		avg_accs_by_pct[pct] = avg_accs
	return avg_accs_by_pct


results = {}

f = open('logs/2019:09:18-18:42:41.log')
results_by_pct = process_file(f)
f.close()

avg_accs_by_pct = get_avg_accs(results_by_pct)

for pct, avg_accs in avg_accs_by_pct.items():
	print(pct)
	plt.plot(avg_accs)
	plt.show()