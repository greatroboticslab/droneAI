
#
# needs updating to match new structure in preprocess_data/MultipleModelVideoClassifier.py
#

import os
import cv2
from ultralytics import YOLO

output_crash_frames_path = './labeled_frames/Crashes'

def get_model():
    return YOLO("./results/landing_class_v3/weights/best.pt")

def format_time(seconds):
    # seconds to minutes:seconds
    return f"{int(seconds // 60)}:{seconds % 60:05.2f}"

def merge_crash_events(crash_events):
    # many crash events have the same start time
    if not crash_events:
        return []

    merged_events = [crash_events[0]] # start with the first crash event

    for current_start, current_end in crash_events[1:]:
        last_start, last_end = merged_events[-1] # get the last start and end
        if current_start - last_end <= 5.0: # check if less than merge time threshold
            merged_events[-1] = (last_start, max(last_end, current_end)) # update end time to the latest time
        else: 
            merged_events.append((current_start, current_end)) # no merging needed
    return merged_events

def video_classification(video_path,label_vid_output,crash_vid_output, model, min_crash_duration=2.0): # min_crash_duration weeds out false flags
    # start video process
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    count = 0

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    labeled_out = cv2.VideoWriter(label_vid_output, fourcc, fps, (width, height))
    crash_out = cv2.VideoWriter(crash_vid_output, fourcc, fps, (width, height))

    # label dictionary
    label_counts = {
        "Crash": 0,
        "Flight": 0,
        "NoSignal": 0,
        "Started": 0,
        "Landing": 0,
        "Unknown": 0
    }

    is_crash = False # is current frame a crash
    crash_events = [] # store crash event tuples (start,end)
    crash_start_time = None # keep track of start time

    non_crash_frame_threshold = int(fps * 1.0) # number of frames to consider crash over
    non_crash_frame_count = 0 # count current non crash frames
    print('\n')
    while True:
        ret, og_frame = cap.read() # read current frame
        if not ret:
            break

        print(f"\rProcessing frame {count + 1}/{total_frames}", end='', flush=True)
        
        # preprocess frame for prediction
        frame = cv2.resize(og_frame, (640, 640))
        #frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # get timestamp of the frame
        current_time_ms = cap.get(cv2.CAP_PROP_POS_MSEC)
        current_time_sec = current_time_ms / 1000.0

        # make prediction
        results = model.predict(source=frame, imgsz=640, verbose=False)
        current_label = "Unknown"

        # update frame's label
        if results and hasattr(results[0], 'probs') and results[0].probs is not None:
            top1_index = results[0].probs.top1

            if top1_index == 0:
                label_counts["Crash"] += 1
                current_label = "Crash"
            elif top1_index == 1:
                label_counts["Flight"] += 1
                current_label = "Flight"
            elif top1_index == 2:
                label_counts["NoSignal"] += 1
                current_label = "NoSignal"
            elif top1_index == 3:
                label_counts["Started"] += 1
                current_label = "Started"
            elif top1_index == 4:
                label_counts["Landing"] += 1
                current_label = "Landing"
            else:
                label_counts["Unknown"] += 1

        if current_label == "Crash":
            crash_out.write(og_frame)
            #cv2.imwrite(os.path.join(output_crash_frames_path,f'crash_frame{count + 1}.png'), frame)
            if not is_crash:
                # start new crash event
                crash_start_time = current_time_sec
                is_crash = True
            non_crash_frame_count = 0
        else:
            if is_crash:
                # currently in crash event
                non_crash_frame_count += 1
                if non_crash_frame_count >= non_crash_frame_threshold:
                    # end crash event if non_crash is more than threshold
                    crash_end_time = current_time_sec
                    crash_duration = crash_end_time - crash_start_time
                    if crash_duration >= min_crash_duration:
                        crash_events.append((crash_start_time, crash_end_time))
                    is_crash = False
                    crash_start_time = None
                    non_crash_frame_count = 0
            else:
                # not in a crash event
                non_crash_frame_count = 0

        # write label to frame
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.6
        font_color = (255, 255, 255)
        thickness = 2
        position = (10, 30)
        text_size = cv2.getTextSize(current_label, font, font_scale, thickness)[0]
        text_x, text_y = position
        cv2.rectangle(og_frame, (text_x - 5, text_y - text_size[1] - 5), (text_x + text_size[0] + 5, text_y + 5), (0, 0, 0), -1)
        cv2.putText(og_frame, current_label, position, font, font_scale, font_color, thickness)
        labeled_out.write(og_frame)

        count += 1 # frame is over
    
    # video is over
    if is_crash:
        # ended on a crash frame
        crash_end_time = total_frames / fps
        crash_duration = crash_end_time - crash_start_time
        if crash_duration >= min_crash_duration:
            crash_events.append((crash_start_time, crash_end_time))

    cap.release()
    labeled_out.release()
    crash_out.release()
    cv2.destroyAllWindows()

    # merge events to only have unique crashes
    merged_crash_events = merge_crash_events(crash_events)

    return label_counts, merged_crash_events

if __name__ == '__main__':
    model = get_model()

    label_counts, crash_events = video_classification('./videos/test.mkv', model, min_crash_duration=2.0)
    for label, count in label_counts.items():
        print(f"\n{label}: {count}")

    print(f"\nNumber of unique crashes: {len(crash_events)}")
    for i, (start, end) in enumerate(crash_events):
        start_formatted = format_time(start)
        end_formatted = format_time(end)
        duration = end - start
        print(f"Crash {i+1}: Start time = {start_formatted}, End time = {end_formatted}, Duration = {duration:.2f}s")
