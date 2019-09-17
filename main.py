import json
from easydict import EasyDict

from agents.bilstm import BiLSTMAgent
from utils.logger import initialize_logger

with open('configs/bilstm.json') as f:
	config = EasyDict(json.load(f))

initialize_logger()
agent = BiLSTMAgent(config)
agent.run()