import string
import re

reviews_path = 'pros-cons/'

def get_reviews(review_path, type):
	reviews = []
	with open(review_path, 'r') as f:
		reviews_raw = f.read()
	reviews_raw = [tup[1] for tup in re.findall(r'(<Cons>|<Pros>)(.*?)(</Cons>|</Pros>)', reviews_raw)]
	for i, review in enumerate(reviews_raw):
		review = review.lower()
		review = review.replace('&amp', ' ')
		to_remove = string.punctuation + string.digits
		if type == 1: trans = str.maketrans('','',to_remove)
		else: trans = str.maketrans(to_remove, len(to_remove)*' ')
		review = review.translate(trans)
		review = review.split()
		if len(review) > 0: reviews.append(review)
	return reviews


reviews_pro = get_reviews(reviews_path+'IntegratedPros.txt', 1)
reviews_con = get_reviews(reviews_path+'IntegratedCons.txt', 1)
review_word_list = [word for review in reviews_pro+reviews_con for word in review]
vocab_1 = set(review_word_list)

reviews_pro = get_reviews(reviews_path+'IntegratedPros.txt', 2)
reviews_con = get_reviews(reviews_path+'IntegratedCons.txt', 2)
review_word_list = [word for review in reviews_pro+reviews_con for word in review]
vocab_2 = set(review_word_list)


print(len(vocab_1))
print(len(vocab_2))

with open('temp1', 'w') as f:
	f.write('Words in no space, not in space \n')
	for word in sorted(list(vocab_1.difference(vocab_2))):
		f.write(word+' ')

with open('temp2', 'w') as f:
	for word in sorted(list(vocab_2)):
		f.write(word+' ')