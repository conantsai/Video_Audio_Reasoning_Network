import os
import csv
from random import random
import pandas as pd


train_dir = 'ucfInfo/ced_training.csv'
valid_dir = 'ucfInfo/ced_valid.csv'
new_train_dir = 'ucfInfo/ced_new_train.csv'


info = pd.read_csv(train_dir)

if os.path.exists(new_train_dir):
    csv_writer = csv.writer(open(new_train_dir, 'a'))
else:
    csv_writer = csv.writer(open(new_train_dir, 'w'))
    csv_writer.writerow(['name', 'label', 'score'])

if os.path.exists(valid_dir):
    csv_writer2 = csv.writer(open(valid_dir, 'a'))
else:
    csv_writer2 = csv.writer(open(valid_dir, 'w'))
    csv_writer2.writerow(['name', 'label', 'score'])

for idx, row in info.iterrows():
    print(row)
    print('[{}/{} - {}]'.format(idx, len(info), row['name']))

    if random() < 0.9:
        csv_writer.writerow([row['name'], row.label, row.score])
    else:
        csv_writer2.writerow([row['name'], row.label, row.score])
