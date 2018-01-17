import os
import pdb
import csv
import numpy as np
import pandas as pd

class NuviNer:

  IDX = {
   "text": 18,
   "author_real_name": 7,
   "hashes": 9,
   "urls": 14,
   "sentiment": 19
  }

  def __init__(self, file):
    with open(file) as f:
      csv_data = pd.read_csv(file)
      self.mentions = np.array(csv_data)

  def filter_mentions(self, idx, val, operator='='):
    if operator == '=':
      filter = self.mentions[:,  idx] == val
    elif operator == '!=':
      filter = self.mentions[:,  idx] != val
    elif operator == '>':
      filter = self.mentions[:,  idx] > val
    elif operator == '<':
      filter = self.mentions[:,  idx] < val

    return self.mentions[filter, :]

  def write(self, file, data):
    f = open(file, 'w')
    f.seek(0) & f.truncate()
    f.close()

    pd_file = pd.DataFrame(data)
    pd_file.to_csv(file, index=False)


if __name__ == "__main__":
  mentions_sample = './data/mentions.csv'
  ner = NuviNer(mentions_sample)
  print("Shape of mention data: ", ner.mentions.shape)
  ## Dictionary of the authorName and post counts
  # unique, counts = np.unique(ner.mentions[:,NuviNer.IDX["author_real_name"]], return_counts=True)
  # print("Author count:", dict(zip(unique, counts)))

  output_file = './data/mentions_text.csv'
  mentions = ner.filter_mentions(NuviNer.IDX["author_real_name"], 'American Express', '!=')
  ner.write(output_file, mentions[:,NuviNer.IDX["text"]])

  # ner2 = NuviNer(output_file)
  # print("Shape of output: ", ner2.mentions.shape)

