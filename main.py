from agents.bilstm import BiLSTMAgent
from utils.logger import initialize_logger
from utils.parsing import get_config

# percentages = [0.02, 0.04, 0.06, 0.08] + [round(0.1*i,2) for i in range(1,11)]
# fracs = [float(1)/(1+k) for k in [1, 2, 4, 8, 16]]
# geos = [0.5, 0.6, 0.7, 0.8, 0.9]

# for pct_usage in percentages:
# 	agent = BiLSTMAgent(config, pct_usage)
# 	agent.run()

# best crossval 500 params are geo = 0.9 and frac = 0.5, 391 epochs

def grid_search(pct_usage):
	# fracs = [float(1)/(1+k) for k in [1, 2, 4, 8, 16]]
	fracs = [float(1)/(1+k) for k in [6, 12, 18, 24, 30]]
	# geos = [0.3, 0.5, 0.6, 0.7, 0.8, 0.9]
	geos = [0.5, 0.6, 0.7, 0.8]
	for frac in fracs:
		for geo in geos:
			agent = BiLSTMAgent(config, pct_usage, frac, geo)
			agent.run()

def estimate_optimal_epochs(pct_usage):
	'''
	Estimate optimal number of epochs for a pct_usage, 
	using midrange frac and geo parameters
	'''
	frac = 1/4
	geo = 0.5
	agent = BiLSTMAgent(config, pct_usage, frac, geo)
	agent.run()

def grid_search_pcts():
	percentages = ([0.02, 0.04, 0.06, 0.08] 
				   + [round(0.1*i,2) for i in range(1,11)])
	for pct_usage in percentages:
		grid_search(pct_usage)

if __name__ == "__main__":
	config = get_config()
	initialize_logger()

	grid_search(1)

	# agent = BiLSTMAgent(config, pct_usage)
	# agent.run()