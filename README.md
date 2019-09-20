# BiLSTM
A stacked bidirectional LSTM used as a baseline for training curve analysis in academic research.

The model is trained on various fractions of a training set and tested on a test set of fixed size. 
This allows the learning curves **INSERT HERE** to be created.

### Current Use Case
Performs sentiment classification on the pro-con dataset introduced by Ganapathibhotla.

Uses pretrained wikipedia2vec embeddings, a species of word2vec that is trained on a corpus of Wikipedia entries. 

### Current Configurations
The code can be run with two configurations, crossval and test.

Cross validation is used to determine the optimal number of training epochs for each percentage of the training set. To execute, run the following command:

    python main.py --config crossval.json

Test is used to determine the accuracy (as measured on the test set) of a model that has been trained on each percentage of the training set. To execute, run the following command:

    python main.py --config test.json

### Logs
Each run creates a timestamped log file containing training accuracy and other valuable information about the experiment.

### Organization
The organization of the code is loosely based on the PyTorch project template outlined in Hager Radi's excellent [post](https://www.linkedin.com/pulse/pytorch-project-template-do-smart-way-hager-rady/).

### Package Installation
Create a virtual environment  and run the following code from inside the repository:

    pip install -r requirements.txt

### Downloading and Processing Raw Data and Embeddings
Note that the preprocessed data and embeddings necessary to run the code are already contained in this repository. To download and process the code oneself, follow the steps below.

Download the dataset from [here](https://www.cs.uic.edu/~liub/FBS/sentiment-analysis.html), at the hyperlink with the anchor text "Pros and cons dataset".

Download the embeddings from [here](https://wikipedia2vec.github.io/wikipedia2vec/pretrained/) , at the hyperlink with the anchor text "300d (txt)" at the bullet point "enwiki_20180420".

Decompress both files and place both in a folder outside this repository. Run the following command:

    python utils/preprocess.py -datapath <path to folder>/<folder name>/


## References
Pro-con dataset:

    @inproceedings{GanapathibhotlaCITE,
    author = {Ganapathibhotla, Murthy and Liu, Bing},
    title = {Mining Opinions in Comparative Sentences},
    booktitle = {Proceedings of the 22Nd International Conference on Computational Linguistics - Volume 1},
    series = {COLING '08},
    year = {2008},
    isbn = {978-1-905593-44-6},
    location = {Manchester, United Kingdom},
    pages = {241--248},
    numpages = {8},
    url = {http://dl.acm.org/citation.cfm?id=1599081.1599112},
    acmid = {1599112},
    publisher = {Association for Computational Linguistics},
    address = {Stroudsburg, PA, USA}
    }

Wikipedia2vec embeddings:

    @article{yamada2018wikipedia2vec,
    title={Wikipedia2Vec: An Optimized Tool for Learning Embeddings of Words and Entities from Wikipedia},
    author={Yamada, Ikuya and Asai, Akari and Shindo, Hiroyuki and Takeda, Hideaki and Takefuji, Yoshiyasu},
    journal={arXiv preprint 1812.06280},
    year={2018}
    }
