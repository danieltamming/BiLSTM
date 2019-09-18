import torch

class AverageMeter:
	def __init__(self):
		self.sum = 0
		self.count = 0
		self.avg = 0

	def reset(self):
		self.sum = 0
		self.count = 0
		self.avg = 0

	def update(self, val, n=1):
		self.sum += n*val
		self.count += n
		self.avg = self.sum / self.count

	@property
	def val(self):
		return self.avg
	

def get_accuracy(pred, target):
	_, pred = torch.max(pred, 1)
	correct = (pred==target).sum().item()
	return float(correct)/target.shape[0]

class EarlyStopper:
	def __init__(self, patience):
		self.patience = patience
		self.best = 0
		self.since_improved = 0

	def update_and_check(self, acc):
		if self.best < acc:
			self.best = acc
			self.since_improved = 0
		else:
			self.since_improved += 1
		return self.since_improved > self.patience