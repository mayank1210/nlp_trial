# NLP Trial
Natural Language Processing Trial for NUVI

# The Data

You will be given an unlabeled data set `/data/posts.csv` with ~11,000 `posts`. A `post` is a social media post from some network (Twitter, Facebook, etc.). Each post is written in English. It is described as an object/row with a set of attributes such as `_id, author_username, hashtags, location, network, raw_body_text, etc`. The first line of the csv has the column names.  The content of the post is in the `raw_body_text` column, but you may use other columns to accomplish the task. 

# The Task

You will need to do at least one of the following:

- [ ] Find & count the named entities in the posts. NER [https://www.wikiwand.com/en/Named-entity_recognition]
- [ ] Use a clustering algorithm to group the entities/themes [https://noisy-text.github.io/2015/pdf/WNUT21.pdf]
- [ ] Write a classification algorithm that puts each post into 3-10 classifications of your choosing (ex: is_feedback, is_product_review, etc.).

#### Extra Credit

- [ ] Visualize the data set
- [ ] Compare the output of 2 or more NLP libraries and discuss the strengths/weaknesses
- [ ] Add benchmarking to show the speed of making predictions

#### Things not to do

Don't do sentiment analysis

# Submission
You will need to make a pull request to this repository with your name as the branch. Ex. `git checkout -b firstname_lastname`.
In the main.py file you will need to include instructions about how to run your program.

# Required Libraries for running main.py
--Pandas (for python data structure)
pip install pandas  OR  conda install pandas

--Numpy (for mathematical imputation)
pip install numpy  OR  conda install numpy

--Plotly (for graphs)
pip insatll plotly or conda install plotly

-- Cufflins
pip install cufflinks  OR  conda install cufflinks

--Gensim (for vector model)
pip install gensim  OR  conda install gensim

--Spacy (for NER)
pip install spacy  OR  conda install spacy

--ntlk (for NLP)
pip install nltk  OR  conda install nltk

--sklearn (tools for data mining and data analysis)
pip install sklearn  OR  conda install sklearn

--matplotlib (for graph)
pip install matplotlib  OR  conda install matplotlib

--seaborn (for graph)
pip install seaborn  OR  conda install seaborn

--multiprocessing (for spawning processes)
pip install multiprocessing OR conda install multiprocessing

Also run "python -m spacy download en" for englsih vocab

#Main File
Python version 2.7 was used.
All the code is in the jupyter notebook. It is also present in py and pdf format.


