import os
import csv
import pandas as pd
from random import random
from moviepy.editor import VideoFileClip, AudioFileClip


def main():
    file_name = '/home/uscc/USAI_Outsourcing/SBIR_fight_videos/ced_info_SBIR.csv'
    output_file1 = 'ucfInfo/ced_training.csv'
    output_file2 = 'ucfInfo/ced_testing.csv'
    v_ext = '.mp4'
    a_ext = '.mp3'

    subclips_info = pd.read_csv(file_name)

    # if os.path.exists(output_file1):
    #     csv_writer = csv.writer(open(output_file1, 'a'))
    # else:
    #     csv_writer = csv.writer(open(output_file1, 'w'))
    #     csv_writer.writerow(['name', 'label', 'score'])

    # if os.path.exists(output_file2):
    #     csv_writer2 = csv.writer(open(output_file2, 'a'))
    # else:
    #     csv_writer2 = csv.writer(open(output_file2, 'w'))
    #     csv_writer2.writerow(['name', 'label', 'score'])

    for index, row in subclips_info.iterrows():
        # if index < 316:
        #     continue
        print('[{}/{}] {}'.format(index, len(subclips_info), row.Name))
        # print('Name: {row.Name}\t'
        #       'Label: {row.label}\t'
        #       'start: {row.start}\t'
        #       'end: {row.end}\t'
        #       'score: {row.score}'.format(row=row))
        video_name = row.Name + v_ext
        audio_name = row.Name + a_ext
        output_name = row.Name[:-11] + 'v{:04d}_{}_{:03d}_n'.format(index, row.label, row.score)


        # if random() < 0.9:
        #     csv_writer.writerow([output_name, row.label, row.score])
        # else:
        #     csv_writer2.writerow([output_name, row.label, row.score])

        ## cut vedio and audio
        vedio_clip = VideoFileClip(video_name).subclip(row.start, row.end)
        audio_clip = AudioFileClip(audio_name).subclip(row.start, row.end)

        vedio_clip.write_videofile(output_name + v_ext)
        audio_clip.write_audiofile(output_name + a_ext)

        ## replace origional file
        os.remove(row.Name + v_ext)
        os.remove(row.Name + a_ext)
        os.rename(row.Name[:-11] + 'v{:04d}_{}_{:03d}_n'.format(index, row.label, row.score) +  v_ext, 
            row.Name[:-11] + row.Name[50:-12] +  v_ext)
        os.rename(row.Name[:-11] + 'v{:04d}_{}_{:03d}_n'.format(index, row.label, row.score) +  a_ext, 
            row.Name[:-11] + row.Name[50:-12] +  a_ext)
        
        # ffmpeg_extract_subclip(
        #     video_name, , , '')


def time_to_sec(time):
    ts = time.split(':')
    if len(ts) == 1:
        return int(time)
    else:
        _sum = int(ts.pop())
        i = 1
        while(len(ts) != 0):
            _sum += (60**i) * int(ts.pop())
            i += 1

        return _sum


if __name__ == '__main__':
    main()
