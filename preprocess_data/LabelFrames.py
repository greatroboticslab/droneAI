# load a video and output frames to label folders based on timestamps
import cv2
import os

# timestamp in seconds
time_ranges = [
    ("No signal", 0, 1),
    ("Started", 1, 13),
    ("Flight", 13, 24),
    ("No signal", 24, 32),
    ("Crash", 32, 37),
    ("Flight", 37, 48),
    ("Crash", 49, 54),
    ("No signal", 55, 66),
    ("Started", 66, 68),
    ("Flight", 68, 227),
    ("Crash", 227, 229),
    ("No signal", 229, 242),
    ("Flight", 242, 279),
    ("Landing", 279, 289),
    ("Flight", 289, 361),
    ("Landing", 361, 383),
    ("Started", 383, 390),
    ("Flight", 390, 404),
    ("Landing", 404, 416),
    ("Started", 416, 418),
    ("Flight", 418, 513),
    ("Landing", 513, 531),
    ("Started", 531, 534),
    ("Flight", 534, 606),
    ("Landing", 606, 617)
]

def save_frames_by_label(video_path, output_dir):
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # create label folders if it doesnt exist
    label_dirs = {}
    for label, _, _ in time_ranges:
        label_dir = os.path.join(output_dir, label)
        if not os.path.exists(label_dir):
            os.makedirs(label_dir)
        label_dirs[label] = label_dir

    # go frame by frame in the video
    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.resize(frame, (640, 640))

        # get the frame's time in the video
        current_time_sec = frame_idx / fps

        # get label based on current time
        label = None
        for lbl, start, end in time_ranges:
            if start <= current_time_sec < end:
                label = lbl
                break

        # output frame to its label folder
        if label:
            output_path = os.path.join(label_dirs[label], f"frame_{frame_idx:06d}.png")
            cv2.imwrite(output_path, frame)

        frame_idx += 1
        print(f"\rProcessed frame {frame_idx}, Time: {current_time_sec:.2f} sec", end='', flush=True)

    cap.release()
    cv2.destroyAllWindows()
    print("\nFrames saved by label.")

if __name__ == '__main__':
    video_path = "./videos/input.mp4"
    output_dir = "./labeled_frames"
    save_frames_by_label(video_path, output_dir)