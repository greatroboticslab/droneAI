#saves each frame in a video to a folder
import os
import cv2
from yt_dlp import YoutubeDL

output_crash_frames_path = './labeled_frames/Flight'

def video_classification(video_path):
    # start video process
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    count = 0

    while True:
        ret, og_frame = cap.read() # read current frame
        if not ret:
            break

        print(f"\rProcessing frame {count + 1}/{total_frames}", end='', flush=True)
        frame = cv2.resize(og_frame, (640, 640))
        cv2.imwrite(os.path.join(output_crash_frames_path,f'flight_frame{count + 1}.png'), frame)
        count += 1 # frame is over

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    url = 'https://www.youtube.com/watch?v=2SMgYeEiv64'
    ydl_opts = {
        'format': 'best',
        'outtmpl': f'./videos/frame_me.%(ext)s'
    }
    with YoutubeDL(ydl_opts) as ydl:
        ydl.download([url])
    
    video_classification('./videos/frame_me.mp4')
