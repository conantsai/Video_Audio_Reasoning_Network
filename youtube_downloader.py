import pandas as pd
import numpy as np
from random import random
from pytube import YouTube
import csv
import os


def yt_download(row, csv_w, idx):
    video_name = 'v_fight_' + '{:05d}'.format(idx)
    res = ['480p', '720p', '1080p']

    try:
        yt = YouTube(row.video_link).streams.filter(
            progressive=True, subtype='mp4')
    except Exception as e:
        print(idx, row.video_link, e)
        return

    try:
        for r in res:
            if yt.filter(res=r).count() != 0:
                yt = yt.filter(res=r).first()
                break
    except Exception as e:
        print(idx, e)
        return

    try:
        if random() < 0.8:
            yt.download('/home/uscc/train', video_name)
            csv_w.writerow([idx, video_name, yt.filesize, 'train'])
            print([video_name, yt.filesize, idx, 'train'])
        else:
            yt.download('/home/uscc/test', video_name)
            csv_w.writerow([idx, video_name, yt.filesize, 'test'])
            print([video_name, yt.filesize, idx, 'test'])
    except AttributeError as e:
        print(idx, e)
        return


def main():
    input_file = 'youtube_conflict/youtube_filter_v2.csv'
    output_file = 'youtube_conflict/youtube_download_info_v2.csv'
    file = pd.read_csv(input_file)

    if os.path.exists(output_file):
        csv_w = csv.writer(open(output_file, 'a'))
    else:
        csv_w = csv.writer(open(output_file, 'w'))
        csv_w.writerow(['idx', 'video_name', 'file_size', 'mode'])

    m = file.shape[0]

    for i in range(578, m):
        yt_download(file.iloc[i, :], csv_w, i)


if __name__ == '__main__':
    main()
