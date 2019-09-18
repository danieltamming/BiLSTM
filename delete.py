import numpy as np
import random

from torch.utils.data import Dataset

num_classes = 2
input_length = 25

class ProConDataset(Dataset):
	def __init__(self, data_path, embedding_file, pct_usage=1):
		self.labels, self.reviews, embed_list = [], [], []
		for label in range(num_classes):
			data_file = data_path + str(label) + '.txt'
			with open(data_file, 'r') as f:
				num_reviews = int(f.readline())
				num_to_read = int(pct_usage*num_reviews)
				f.readline()
				for _ in range(num_to_read):
					line = f.readline()
					self.labels.append(label)
					review = [int(num) for num in line.split()]
					if len(review) < input_length: review += (input_length-len(review))*[0]
					self.reviews.append(review[:input_length])
		# self.embeddings = get_embeddings(embedding_file)

def get_sequences(data_path, pct_usage=1):
	seqs, labels = [], []
	for label in range(num_classes):
		data_file = data_path + str(label) + '.txt'
		with open(data_file) as f:
			num_seqs = f.readline()
			num_to_read = int(pct_usage*num_seqs)
			f.readline()
			for _ in range(num_to_read):
				labels.append(label)
				line = f.readline()
				seq = [int(tok) for tok in line.split()]
				if len(seq) < input_length: seq += (input_length-len(seq))*[0]
				seqs.append(seq[:input_length])
	return seqs, labels

dataset = ProConDataset('data/procon/test/', 'data/procon/embeddings.txt')
reviews1 = dataset.reviews

reviews2, _ = get_sequences('data/procon/test/')

for i, (rev1, rev2) in enumerate(zip(reviews1, reviews2)):
	print(i)
	if rev1 != rev2: print('uh oh')

# THIS WORKS SO PORT TO DATA AND PROCON