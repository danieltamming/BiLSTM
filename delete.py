from bs4 import BeautifulSoup 
from googletrans import Translator

reviews = [
['horrible', 'customer', 'service', 'delivery', 'problems'],
['doesn', 't', 'come', 'with', 'enough', 'accessories'],
['none'],
['like', 'all', 'digital', 'cameras', 'dynamic', 'range', 'of', 'contrast', 'is', 'poor', 'compared', 'to', 'film'],
['none'],
['need', 'more', 'zoom', 'and', 'ibm', 'mb', 'microdrive'],
['mediocre', 'low', 'light', 'high', 'speed', 'performance'],
['can', 'be', 'costly', 'to', 'get', 'the', 'most', 'out', 'of', 'it'],
['need', 'a', 'lot', 'of', 'light'],
['no', 'rechargeable', 'battery', 'came', 'with', 'it'],
['short', 'battery', 'life', 'weak', 'flash', 'and', 'urrrrr', 'needs', 'more', 'megapixels'],
['still', 'comes', 'with', 'only', 'a', 'mb', 'cf', 'card'],
['wimpy', 'mb', 'compact', 'flash', 'card'],
['mb', 'flash', 'card', 'included', 'semi', 'weak', 'flash', 'short', 'battery', 'life', 'no', 'zoom', 'with', 'movies'],
['battery', 'life', 'lack', 'of', 'memory'],
['soft', 'photos', 'proprietary', 'battery', 'with', 'mediocre', 'life'],
['canon', 'specific', 'a', 'c', 'adapter', 'not', 'included', 'and', 'the', 'measly', 'mb', 'flash', 'card'],
['bad', 'picture', 'quality', 'very', 'inadequate', 'flash'],
['software', 'is', 'bulky']
]

translator = Translator()

for review in reviews:
	pivot = 'german'
	pivoted = translator.translate(' '.join(review), dest=pivot, src='english').text
	back = translator.translate(pivoted, dest='english', src=pivot).text
	print(back)