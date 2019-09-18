import os
import string
import re
from collections import Counter
from tqdm import tqdm

word2vec_orig = './../DownloadedData/enwiki_20180420_300d.txt'
word2vec_target = 'data/procon/embeddings.txt'
# embeddings = KeyedVectors.load_word2vec_format(word2vec_orig, binary=False, limit=10)

reviews_path = './../DownloadedData/pros-cons/'

def get_reviews(review_path):
	reviews = []
	with open(review_path, 'r') as f:
		reviews_raw = f.read()
	reviews_raw = [tup[1] for tup in re.findall(r'(<Cons>|<Pros>)(.*?)(</Cons>|</Pros>)', reviews_raw)]
	for i, review in enumerate(reviews_raw):
		review = review.lower()
		review = review.replace('&amp', ' ')
		to_remove = string.punctuation + string.digits
		trans = str.maketrans(to_remove, len(to_remove)*' ')
		review = review.translate(trans)
		review = review.split()
		if len(review) > 0: reviews.append(review)
	return reviews

def get_embed_vocab(embed_path, review_vocab):
	data_path = 'data/'
	if not os.path.exists(data_path): os.mkdir(data_path)
	data_path = 'data/procon/'
	if not os.path.exists(data_path): os.mkdir(data_path)
	embed_vocab = set()
	with open(word2vec_orig, 'r') as f1, open(word2vec_target, 'w') as f2:
		num_lines, embed_dims = [int(num) for num in f1.readline().split()]
		for _ in tqdm(range(num_lines)):
			line = f1.readline()
			word = line.split(' ',1)[0]
			if word in review_vocab:
				embed_vocab.add(word)
				f2.write(line)
	return embed_vocab

def get_tokenizer(review_word_list, embed_vocab):
	word_counter = Counter([word for word in review_word_list if word in embed_vocab])
	word_counter_list = word_counter.most_common(len(word_counter))
	word_to_num = {word:i+1 for i, (word, _) in enumerate(word_counter_list)}
	return word_to_num

def tknz(reviews_list, word_to_num):
	for i, reviews in enumerate(reviews_list):
		for j, review in enumerate(reviews):
			tknzd_review = []
			for word in review:
				if word in word_to_num: tknzd_review.append(word_to_num[word])
			reviews[j] = tknzd_review
		reviews_list[i] = reviews
	return reviews_list

def write_tknzd_splits(tknzd_reviews_list, test_pct=0.10):
	data_path = 'data/procon/'
	if not os.path.exists(data_path): os.mkdir(data_path)
	test_path = data_path + 'test/'
	if not os.path.exists(test_path): os.mkdir(test_path)
	train_path = data_path + 'train/'
	if not os.path.exists(train_path): os.mkdir(train_path)
	first_train_idx = int(test_pct*(sum([len(reviews) for reviews in tknzd_reviews_list])/len(tknzd_reviews_list)))
	test_list = [review[:first_train_idx] for review in tknzd_reviews_list]
	train_list = [review[first_train_idx:] for review in tknzd_reviews_list]

	for i, tknzd_reviews in enumerate(test_list):
		with open(test_path+str(i)+'.txt', 'w') as f:
			f.write(str(len(tknzd_reviews))+'\n\n')
			for review in tknzd_reviews:
				f.write(' '.join([str(num) for num in review])+'\n')

	for i, tknzd_reviews in enumerate(train_list):
		with open(train_path+str(i)+'.txt', 'w') as f:
			f.write(str(len(tknzd_reviews))+'\n\n')
			for review in tknzd_reviews:
				f.write(' '.join([str(num) for num in review])+'\n')


reviews_pro = get_reviews(reviews_path+'IntegratedPros.txt')
reviews_con = get_reviews(reviews_path+'IntegratedCons.txt')
review_word_list = [word for review in reviews_pro+reviews_con for word in review]
review_vocab = set(review_word_list)
embed_vocab = get_embed_vocab(word2vec_orig, review_vocab)
word_to_num = get_tokenizer(review_word_list, embed_vocab)
tknzd_reviews_list = tknz([reviews_con, reviews_pro], word_to_num)
write_tknzd_splits(tknzd_reviews_list)