import json
from easydict import EasyDict

from agents.bilstm import BiLSTMAgent

with open('configs/bilstm.json') as f:
	config = EasyDict(json.load(f))
agent = BiLSTMAgent(config)
agent.run()