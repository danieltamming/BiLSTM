import numpy as np
import matplotlib.pyplot as plt

from mpl_toolkits.mplot3d import Axes3D

def is_training(line):
	return line.split(' ', 1)[0] == 'INFO:BiLSTMAgent:Training'

def is_validating(line):
	return line.split(' ', 1)[0] == 'INFO:BiLSTMAgent:Training'

def get_acc(line):
	return float(line.split()[-1])

def get_pct(line):
	return float(line.split()[1])

def get_frac(line):
	return float(line.split(':')[-1].split('%', 1)[0])/100

def get_geo(line):
	return float(line.split()[-1].rstrip('.'))

def read_val(filename, early_stopping, aug):
	'''
	Process the val log
	early_stopping => ignore extra line
	aug => frac, geo not None
	'''
	records = {}
	with open(filename) as f:
		line = f.readline()
		while line:
			train_accs = []
			val_accs = []
			pct_usage = get_pct(line)
			if aug:
				frac = get_frac(f.readline())
				geo = get_geo(f.readline())
			else:
				frac = None
				geo = None
			line = f.readline()
			while is_training(line):
				train_accs.append(get_acc(line))
				line = f.readline()
				val_accs.append(get_acc(line))
				line = f.readline()
			train_accs = 100*np.array(train_accs)
			val_accs = 100*np.array(val_accs)
			records[(pct_usage, frac, geo)] = (train_accs, val_accs)
			if early_stopping:
				# Ignore previously read line taht explains early stop
				line = f.readline()
	return records

def plot_val_gridsearch(filename, desired_pct_usage, early_stopping, aug):
	'''
	Plots heatmap graph with frac and geo on axes
	TODO test this
	'''
	records = read_val(filename, early_stopping, aug)
	fig = plt.figure()
	fracs = []
	geos = []
	accs = []
	for (pct_usage, frac, geo), (train_accs, val_accs) in records.items():
		if pct_usage != desired_pct_usage: 
			continue
		fracs.append(frac)
		geos.append(geo)
		accs.append(np.max(val_accs))
	plt.scatter(fracs, geos, c=accs)
	plt.colorbar()
	plt.title('{}% of the dataset used'.format(100*desired_pct_usage))
	plt.xlabel('Frac')
	plt.ylabel('Geo')
	plt.show()

def plot_val_vary_pct(filename, desired_pct_usage, early_stopping, desired_frac, desired_geo):
	aug = (desired_frac, desired_geo) != (None, None)
	records = read_val(filename, early_stopping, aug)
	xs = []
	ys = []
	for (pct_usage, frac, geo), (train_accs, val_accs) in records.items():
		if (frac, geo) != (desired_frac, desired_geo):
			continue
		xs.append(pct_usage)
		ys.append(np.max(val_accs))
	plt.plot(xs, ys, '-o')
	axes = plt.gca()
	axes.set_ylim(70, 100)
	plt.xlabel('pct_usage')
	plt.ylabel('Validation Accuracy')
	plt.show()

def process_gridsearch_log(filename, num_fracs, num_geos):
	f = open(filename)
	results_grid = {}
	for _ in range(num_fracs*num_geos):
		(frac, geo), arr = process_grid(f)
		results_grid[(frac, geo)] = arr
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

if __name__ == "__main__":
	# filename = 'logs/val_no_aug.log'
	# plot_val_vary_pct(filename, 1, True, None, None)

	# filename = 'logs/val_sr_100p.log'
	# plot_val_gridsearch(filename, 1, True, True)

	filename = 'logs/val_sr_allpcts.log'
	percentages = ([0.02, 0.04, 0.06, 0.08] 
				  + [round(0.1*i,2) for i in range(1,11)])
	for desired_pct in percentages:
		plot_val_gridsearch(filename, desired_pct, True, True)