import numpy as np
import matplotlib.pyplot as plt

from mpl_toolkits.mplot3d import Axes3D

num_folds = 10

def process_file(f):
	# num_pctgs = 14
	num_pctgs = 1
	# num_pctgs = 7
	results_by_pct = {}
	for _ in range(num_pctgs):
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
	accs = []
	f.readline()
	line = f.readline().replace('INFO:BiLSTMAgent:','').rstrip('\n')
	while line[0] == 'V' or line[0] == 'T':
		if line[0] == 'V':
			accs.append(float(line.split()[-1]))
		line = f.readline().replace('INFO:BiLSTMAgent:','').rstrip('\n')
	return accs

def avg_folds(fold_accs):
	min_epochs = min([len(fold) for fold in fold_accs])
	for i, fold in enumerate(fold_accs):
		fold_accs[i] = fold[:min_epochs]
	accs = np.array(fold_accs)
	avg_accs = np.mean(accs, axis=0)
	return avg_accs

def get_avg_accs(results_by_pct):
	avg_accs_by_pct = {}
	for pct, fold_accs in results_by_pct.items():
		min_epochs = min([len(fold) for fold in fold_accs])
		for i, fold in enumerate(fold_accs):
			fold_accs[i] = fold[:min_epochs]
		accs = np.array(fold_accs)
		avg_accs = np.mean(accs, axis=0)
		avg_accs_by_pct[pct] = avg_accs
	return avg_accs_by_pct

def process_crossval_log(filename):
	f = open(filename)
	results_by_pct = process_file(f)
	f.close()
	avg_accs_by_pct = get_avg_accs(results_by_pct)
	for pct, avg_accs in avg_accs_by_pct.items():
		print(pct)
		print(np.argmax(avg_accs))
		print(np.max(avg_accs))
		plt.plot(avg_accs)
		plt.title('Learning Curve With '+str(100*float(pct))+'% of Dataset')
		plt.ylabel('Cross Validation Accuracy (%)')
		plt.xlabel('Epoch')
		plt.show()

def read_test_log(filename):
	with open(filename) as f:
		pcts, accs = [], []
		line = f.readline()
		while line:
			line = line.replace('INFO:BiLSTMAgent:','').rstrip('\n')
			pcts.append(float(line.split()[1]))
			line = f.readline().replace('INFO:BiLSTMAgent:','').rstrip('\n')
			accs.append(float(line.split()[-1]))
			line = f.readline()
	pcts = 100*np.array(pcts)
	accs = 100*np.array(accs)
	return pcts, accs

def process_test_log(filename):
	pcts, accs = read_test_log(filename)
	plt.plot(pcts,accs, '-bo')
	plt.axis((0,100,70,100))
	plt.title('Learning Curve')
	plt.xlabel('Percent of Dataset (%)')
	plt.ylabel('Test Accuracy (%)')
	plt.show()

def compare_test_logs(def_filename, aug_filename):
	def_pcts, def_accs = read_test_log(def_filename)
	aug_pcts, aug_accs = read_test_log(aug_filename)
	plt.plot(aug_pcts, aug_accs, '-ro', label='Synonym Replacement')
	plt.plot(def_pcts,def_accs, '-bo', label='No Augmentation')
	plt.axis((0,100,70,100))
	plt.title('Learning Curve')
	plt.xlabel('Percent of Dataset (%)')
	plt.ylabel('Test Accuracy (%)')
	plt.legend()
	plt.show()

def get_pct(line):
	return line.split()[1]

def get_frac(line):
	return float(line.split(':')[-1].split('%', 1)[0])/100

def get_geo(line):
	return float(line.split()[-1].rstrip('.'))

def process_grid(f):
	pct = get_pct(f.readline())
	frac = get_frac(f.readline())
	geo = get_geo(f.readline())
	fold_accs = []
	line = f.readline().replace('INFO:BiLSTMAgent:','').rstrip('\n')
	for _ in range(num_folds):
		accs = process_fold(f)
		fold_accs.append(accs)
	return (frac, geo), fold_accs

def process_crossval_gridsearch_log(filename):
	num_fracs = 5
	num_geos = 5
	f = open(filename)
	results_grid = {}
	for _ in range(num_fracs*num_geos):
		(frac, geo), arr = process_grid(f)
		avgs = avg_folds(arr)
		results_grid[(frac, geo)] = avgs
	f.close()

	# for (frac, geo), avg_accs in results_grid.items():
	# 	print('Using '+str(frac)+' of the original dataset, geo of '+str(geo)+':')
	# 	print(np.argmax(avg_accs))
	# 	print(np.max(avg_accs))
	# 	plt.plot(avg_accs)
	# 	plt.title('Learning Curve With '+str(frac)+' of the original dataset, geo of '+str(geo))
	# 	plt.ylabel('Cross Validation Accuracy (%)')
	# 	plt.xlabel('Epoch')
	# 	plt.show()
	fracs = [key[0] for key in results_grid.keys()]
	geos = [key[1] for key in results_grid.keys()]
	# accs = [arr[-1] for arr in results_grid.values()]
	accs = [np.max(arr) for arr in results_grid.values()]
	# accs = [np.argmax(arr) for arr in results_grid.values()]

	fig = plt.figure()
	# ax = fig.add_subplot(111, projection='3d')
	# for (frac, geo), avg_accs in results_grid.items():
	# 	ax.scatter(geo,frac,np.max(avg_accs), cmap='gray')
	# ax.set_xlabel('Frac')
	# ax.set_ylabel('Geo')
	# ax.set_zlabel('Acurracy')
	# plt.show()

	plt.scatter(fracs, geos, c=accs)
	plt.colorbar()
	plt.xlabel('Frac')
	plt.ylabel('Geo')
	plt.show()

filename = 'logs/sr_crossval500.log'
sr_filename = 'logs/sr_test500.log'
filename = 'logs/crossval500.log'
# process_crossval_log(filename)
# process_test_log(filename)
filename = 'logs/test500.log'
compare_test_logs(filename, sr_filename)

# filename = 'logs/sr_gridsearch_crossval.log'
# process_crossval_gridsearch_log(filename)