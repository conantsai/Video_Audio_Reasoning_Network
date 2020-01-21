import os
import re
import csv


def main(root_dir):
    dic = {}
    if os.path.isdir(root_dir):
        all_files = os.listdir(root_dir)
        all_files.sort()
        for file in all_files:
            frames = os.listdir(os.path.join(root_dir, file))
            frames.sort()
            last_frame = frames[-1]
            count = re.sub('([a-zA-Z.]*)', '', last_frame)
            dic[file] = int(count)
    else:
        print('It is not dir.')

    return dic


if __name__ == '__main__':
    count_dic = main('ucf_jpegs_256')
    print(len(count_dic))
    csv_writer = csv.writer(open('ucf101_count.csv', 'w'))
    csv_writer.writerow(['video_name', 'frame_count'])

    for key, val in count_dic.items():
        csv_writer.writerow([key, val])
