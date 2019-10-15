from agents.bilstm import BiLSTMAgent
from utils.logger import initialize_logger
from utils.parsing import get_config

config = get_config()
initialize_logger()

percentages = [0.02, 0.04, 0.06, 0.08] + [round(0.1*i,2) for i in range(1,11)]
# percentages = [round(0.1*i,2) for i in range(8,11)]
for pct_usage in percentages:
	agent = BiLSTMAgent(config, pct_usage)
	agent.run()

# agent = BiLSTMAgent(config, 1)
# agent.run()