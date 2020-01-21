import os
import sys
import subprocess


def v2f_process(src_dir, dst_dir, label):
    get_path = os.path.join
    src_path = get_path(src_dir, label)

    if not os.path.isdir(src_path):
        print('This is not directory.')
        return

    dst_path = get_path(dst_dir, label)
    if not os.path.exists(dst_path):
        os.mkdir(dst_path)

    for file_name in os.listdir(src_path):
        if '.avi' not in file_name:
            continue

        name, ext = os.path.splitext(file_name)

        save_path = get_path(dst_path, name)
        video_path = get_path(src_path, file_name)
        print(save_path)
        print(video_path)
        '''try:
            if os.path.exists(save_path):
                if not os.path.exists(get_path(save_path, 'image_{%5d}.jpg'.format(1)))
                    subprocess.call(
                        'rm -r \"{}\"'.format(save_path), shell=True)
                    os.mkdir(save_path)
                else:
                    continue
            else:
                os.mkdir(save_path)
        except:
            print(save_path)
            continue

        cmd = 'ffmpeg -i \"{}\" -vf scale=-1:240 \"{}/image_%5d.jpg\"'.format(
            video_path, save_path)
        print(cmd)
        subprocess.call(cmd, shell=True)
        print('\n')'''


if __name__ == '__main__':
    src_dir = 'UCF-101'
    dst_dir = 'UCF-101-IMAGE'
    c = 0

    for label in os.listdir(src_dir):
        v2f_process(src_dir, dst_dir, label)
        c += 1
        if c == 20:
            break
