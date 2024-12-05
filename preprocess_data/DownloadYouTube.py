# quick script to download a youtube video
from yt_dlp import YoutubeDL

if __name__ == '__main__':
    url = 'https://youtu.be/BV7vi3VJgKI'

    ydl_opts = {
        'format': 'best',
        'outtmpl': f'./videos/input.%(ext)s'
    }
    with YoutubeDL(ydl_opts) as ydl:
        ydl.download([url])