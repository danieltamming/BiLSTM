from agents.bilstm import BiLSTMAgent
from utils.logger import initialize_logger
from utils.parsing import get_config

config = get_config()
initialize_logger()

percentages = [0.02, 0.04, 0.06, 0.08] + [round(0.1*i,2) for i in range(1,11)]
# fracs = [0.1, 0.3, 0.5, 0.7, 0.9]
fracs = [float(1)/(1+k) for k in [1, 2, 4, 8, 16]]
geos = [0.5, 0.6, 0.7, 0.8, 0.9]
# pct_usage = 1
# percentages = [round(0.1*i,2) for i in range(8,11)]
for pct_usage in percentages:
	agent = BiLSTMAgent(config, pct_usage)
	agent.run()

# agent = BiLSTMAgent(config, pct_usage)
# agent.run()

# for frac in fracs:
# 	for geo in geos:
# 		agent = BiLSTMAgent(config, pct_usage, frac, geo)
# 		agent.run()

# best crossval 500 params are geo = 0.9 and frac = 0.5, 391 epochs