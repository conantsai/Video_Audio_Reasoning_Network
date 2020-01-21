import os
import shutil
import pandas as pd
import cv2


def main():
    # df = pd.read_csv('ucfInfo/ced_training.csv')

    # for _, row in df.iterrows():
    #     name = row['name'].split('/')[-1]
    #     capture_frame(name, row['name'])

        # verify_dir(row['name'])
        #     temp = row['name']
        #     name = temp.split('/')[-1]

        #     print(idx, temp)

        #     if not os.path.isdir(temp):
        #         os.mkdir(temp)

        #     move_file(temp + '.mp4', temp + '/' + name + '.mp4')
        #     move_file(temp + '.mp3', temp + '/' + name + '.mp3')
        #     move_file(temp + '.npy', temp + '/' + name + '.npy')

    df = pd.read_csv('ucfInfo/ced_testing.csv')

    for _, row in df.iterrows():
        name = row['name'].split('/')[-1]
        capture_frame(name, row['name'])
        # verify_dir(row['name'])
    #     temp = row['name']
    #     name = temp.split('/')[-1]

    #     print(idx, temp)

    #     if ~os.path.isdir(temp):
    #         os.mkdir(temp)

    #     move_file(temp + '.mp4', temp + '/' + name + '.mp4')
    #     move_file(temp + '.mp3', temp + '/' + name + '.mp3')
    #     move_file(temp + '.npy', temp + '/' + name + '.npy')


def move_file(src, dst):
    if os.path.exists(src):
        shutil.move(src, dst)


def verify_dir(src):
    dirs = os.listdir(src)
    if len(dirs) != 3:
        for file in dirs:
            print(src, file)


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


def test_frame():
    cap = cv2.VideoCapture(
        'fight_videos/ced/v0164_0_000/v0164_0_000_c.mp4')
    frame_count = 1
    success = True

    while(success):
        success, frame = cap.read()
        file_name = 'fight_videos/ced/v0164_0_000/v0164_0_000' + \
            each_video_name + '_{:d}.jpg'.format(frame_count)
        if os.path.exists(file_name):
            break
        if success:
            cv2.imwrite(file_name, frame)
            frame_count += 1


if __name__ == '__main__':
    main()
    # test_frame()
