# download a youtube video and add the timestamp in the top left
from yt_dlp import YoutubeDL
import os
import cv2

def make_timestamp(video_path, output_path):
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    count = 0
    
    # go frame by frame in the video
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        print(f"\rProcessing frame {count + 1}/{total_frames}", end='', flush=True)
        
        # get the frame's timestamp
        current_time_ms = cap.get(cv2.CAP_PROP_POS_MSEC)
        current_time_sec = current_time_ms / 1000.0
        
        # add the timestamp to the top left
        timestamp_text = f"{int(current_time_sec // 60)}:{current_time_sec % 60:05.2f}"
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.6
        font_color = (255, 255, 255)
        thickness = 2
        position = (10, 30)
        text_size = cv2.getTextSize(timestamp_text, font, font_scale, thickness)[0]
        text_x, text_y = position
        cv2.rectangle(frame, (text_x - 5, text_y - text_size[1] - 5), 
                      (text_x + text_size[0] + 5, text_y + 5), 
                      (0, 0, 0), -1)
        cv2.putText(frame, timestamp_text, position, font, font_scale, font_color, thickness)
        
        # output the frame in the new video
        out.write(frame)
        count += 1
    
    cap.release()
    out.release()
    cv2.destroyAllWindows()
    print("\nVideo processing complete.")

if __name__ == '__main__':
    # download the youtube video
    url = 'https://youtu.be/BV7vi3VJgKI?si=nNZljBEjzgyO4Nnz'
    output = './videos/train_video.mp4'
    vid_input = './videos/input.mp4'
    ydl_opts = {
        'format': 'best',
        'outtmpl': f'./videos/input.%(ext)s'
    }
    with YoutubeDL(ydl_opts) as ydl:
        ydl.download([url])
    
    # timestamp it
    make_timestamp(vid_input, output)
    os.remove(vid_input)

    # run this command before upload video to roboflow: ffmpeg -i ./videos/train_video.mp4 -vf scale=640:640 -c:v libx264 ./videos/train-h264.mp4
    