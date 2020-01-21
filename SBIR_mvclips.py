import os
import shutil
import pandas as pd
import cv2


def main():
    df = pd.read_csv('B-1/Jinag_thesis/backup_thesis/fight_videos/test_train2.csv')

    for _, row in df.iterrows():
        name = row['name'].split('/')[-1]
        # print((name, row['name']))
        capture_frame(name, "B-1/Jinag_thesis/backup_thesis/" + row['name'])



def capture_frame(video_name, dst):
    print(video_name)
    temp = dst + '/' + video_name + '_c.mp4'
    if os.path.exists(temp):
        cap = cv2.VideoCapture(temp)
    else:
        cap = cv2.VideoCapture(dst + '/' + video_name + '.mp4')

    frame_count = 1
    success = True

    while(success):
        success, frame = cap.read()
        file_name = dst + '/' + video_name + '_%d.jpg' % frame_count
        if os.path.exists(file_name):
            break
        if success:
            cv2.imwrite(file_name, frame)
            frame_count += 1


if __name__ == '__main__':
    main()
    # test_frame()
