import os
import csv
from random import random

def get_infolist(file):
    with open(file, 'r') as f:
        content = f.read()
        content = content.strip('\r\n').split('\n')
    f.close()

    return content


def main():
    root_path = 'ucfTrainTestlist'
    dest_path = 'ucfInfo'
    class_path = 'ucfTrainTestlist/classInd.txt'

    content = get_infolist(class_path)
    class_to_idx = {line.split(' ')[1]: line.split(' ')[0] for line in content}

    for i in range(1, 4):
        train_file = 'trainlist{:02d}.txt'.format(i) 
        train_csv = 'trainlist{:02d}.csv'.format(i)
        valid_csv = 'validlist{:02d}.csv'.format(i)

        train_path = os.path.join(root_path, train_file)
        train_csv = os.path.join(dest_path, train_csv)
        valid_csv = os.path.join(dest_path, valid_csv)

        train_w = csv.writer(open(train_csv, 'w'))
        valid_w = csv.writer(open(valid_csv, 'w'))

        train_w.writerow(['video_name', 'target'])
        valid_w.writerow(['video_name', 'target'])

        content = get_infolist(train_path)           
        
        for line in content:
            target = line.split('/')[0]
            file_name = line.split('/')[1]
            file_name = file_name.split('.')[0]

            if random() < 0.9:
                train_w.writerow([file_name, class_to_idx.get(target)])
            else:
                valid_w.writerow([file_name, class_to_idx.get(target)])
        

if __name__ == '__main__':
    main()