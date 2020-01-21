import pandas as pd
import numpy as np

input_file = 'youtube_conflict/youtube_fight_v2.csv'
output_file = 'youtube_conflict/youtube_filter_v2.csv'
df = pd.read_csv(input_file)


# print(file1.shape)
# print(file2.shape)
# print(file3.shape)

# file = pd.concat([file1, file2, file3])
print(df.shape)

df = df[~df['video_title'].str.contains('部落冲突', case=False)]
print(df.shape)

df = df[~df['video_title'].str.contains('皇室战争', case=False)]
print(df.shape)

df = df[~df['video_title'].str.contains('歌', case=False)]
print(df.shape)

df = df[~df['video_title'].str.contains('綜藝', case=False)]
print(df.shape)

df = df[~df['video_title'].str.contains('游戏', case=False)]
print(df.shape)

df = df[~df['video_title'].str.contains('廣播', case=False)]
print(df.shape)

# df = df[df['video_time'] >= '0:30']
# print(df.shape)

out = df.to_csv(output_file)
