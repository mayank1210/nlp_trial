# nlp_trial
Natural Language Processing Trial for NUVI

# The Data

You will be given an unlabeled data set `/data/posts.csv` with ~11,000 `posts`. A `post` is a social media post from some network (Twitter, Facebook, etc.). It is described as an object/row with a set of attributes such as `_id, author_username, hashtags, location, network, raw_body_text, etc`. The first line of the csv has the column names. Each post is written in English. The content of the post is in the `raw_body_text` column, but you may use other columns to accomplish the task. 

# The Task

You will need to do at least one of the following:

- [ ] Find & count the named entities in the post. NER [https://www.wikiwand.com/en/Named-entity_recognition]
- [ ] Use a clustering algorithm to group the entities/themes [https://noisy-text.github.io/2015/pdf/WNUT21.pdf]
- [ ] Write a classification algorithm that puts each post into at least 3 classes.
- [ ] 

####Extra Credit

- [ ] Visualize the data set
- [ ] Compare two libraries and discuss the strengths/weaknesses
- [ ] Add benchmarking to show the speed of making predictions

####Things not to do

Don't worry about doing sentiment analysis

# Submission
You will need to make a pull request to this repository with your name as the branch. Ex. `git checkout -b firstname_lastname`.
In the main.py file you will need to include instructions about how to run your program.

