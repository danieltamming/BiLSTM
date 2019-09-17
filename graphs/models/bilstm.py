import torch
import torch.nn as nn

class BiLSTM(nn.Module):
	def __init__(self, embed_dim, num_classes):
		super(BiLSTM, self).__init__()
		self.lstm1 = nn.LSTM(input_size=embed_dim, hidden_size=64, batch_first=True, num_layers=1, bidirectional=True)
		self.drop1 = nn.Dropout(p=0.5)
		self.lstm2 = nn.LSTM(input_size=2*64, hidden_size=32, batch_first=True, num_layers=1, bidirectional=True)
		self.drop2 = nn.Dropout(p=0.5)
		self.dense1 = nn.Linear(2*32, 20)
		self.relu = nn.ReLU()
		self.dense2 = nn.Linear(20, num_classes)
		self.sm = nn.Softmax(dim=1)

	def forward(self, x):
		output1, _ = self.lstm1(x)
		_, (h2, _) = self.lstm2(output1)
		h2 = h2.permute(1,0,2)
		h2 = h2.contiguous().view(h2.shape[0],-1)
		d1 = self.dense1(h2)
		r1 = self.relu(d1)
		d2 = self.dense2(r1)
		output = self.sm(d2)
		return output