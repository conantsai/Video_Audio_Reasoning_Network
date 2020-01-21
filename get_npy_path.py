import os

dirs = os.listdir('./fight_videos/ced')
text_file = open("audio_list.txt", "w")

for file in dirs:
    if file.find('mp3') != -1:
        text_file.write(os.path.join('../fight_videos/ced', file) + '\n')

text_file.close()
