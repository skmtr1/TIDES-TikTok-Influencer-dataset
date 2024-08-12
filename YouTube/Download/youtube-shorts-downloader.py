# -*- coding: utf-8 -*-
# very ezzz code
import os
from pytube import YouTube
from pytube.helpers import install_proxy
import time
import pandas as pd

download_directory = './videos'

df_proxy = pd.read_csv('Free_Proxy_List.csv')


df_proxy['ip-port'] = df_proxy.apply(lambda x: "http://"+str(x['ip']) + ":" + str(x['port']), axis = 1)

proxy_list = list(df_proxy['ip-port'].values)
print(proxy_list)

proxy_counter = 0


def download_video_max_res(url, videoId, proxy_counter):
    try:
        install_proxy({'http': proxy_list[proxy_counter]})  


        yt = YouTube(url)
        stream = yt.streams.get_highest_resolution()

        if stream:
            video_title = stream.title
            print(f"Downloading ... [{url}] - [{video_title.strip()}]")
            ret = stream.download(output_path='./videos',
                            filename=videoId+'.mp4')
            
            print(ret)
            return f"Video downloaded successfully."
        else:
            return "Unable to find a suitable video stream."

    except Exception as e:
        error = str(e)
        with open('download_failed.txt', 'a') as f:
            f.write(f'{url}: {error}\n')

        return "An error occurred: {}".format(error)


if not os.path.exists(download_directory):
    os.makedirs(download_directory)


df_videos = pd.read_csv("./youtube_df_videos.csv")


videos = df_videos['url'].values.tolist()
# video_url = input("Type video URL here: ")

print(videos)


for url in videos:
    print(url)
    if "shorts" in url: #("https://youtube.com/shorts/"): #only shorts, bro...
        videoId = url.split('/')[-1]

        if os.path.exists(os.path.join(download_directory, videoId+'.mp4')): # skip if file already downloaded
            print(url, 'is already exists')
            continue

        response = download_video_max_res(url=url, videoId=videoId, proxy_counter=proxy_counter)
        print(response)
    else:
        print("This is not a youtube shorts link!")
    # print("sleeping...")
    time.sleep(1)
    proxy_counter += 1
    if proxy_counter == len(proxy_list):
        proxy_counter = 0 
