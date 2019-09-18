from agents.bilstm import BiLSTMAgent
from utils.logger import initialize_logger
from utils.parsing import get_config

config = get_config()
initialize_logger()

percentages = [0.05] + [round(0.1*i,2) for i in range(1,11)]
for pct_usage in percentages:
	agent = BiLSTMAgent(config, pct_usage)
	agent.run()