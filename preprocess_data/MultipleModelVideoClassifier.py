# purpose of this file is to run multiple trained models, see which model performs better
import os
import cv2
from ultralytics import YOLO

def get_model(version):
    if version == 1:
        return YOLO("./results/landing_class_v1/weights/best.pt")
    elif version == 2:
        return YOLO("./results/landing_class_v2/weights/best.pt")
    else: return YOLO("./results/landing_class_v3/weights/best.pt")

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

def video_classification(video_path, model, label_dict, is_v1, min_crash_duration):
    # start video process
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    duration = format_time(total_frames / fps)
    count = 0

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
            if is_v1 == False:
                if top1_index == 0:
                    label_dict["Crash"] += 1
                    current_label = "Crash"
                elif top1_index == 1:
                    label_dict["Flight"] += 1
                    current_label = "Flight"
                elif top1_index == 2:
                    label_dict["Landing"] += 1
                    current_label = "Landing"
                elif top1_index == 3:
                    label_dict["No signal"] += 1
                    current_label = "No signal"
                elif top1_index == 4:
                    label_dict["Started"] += 1
                    current_label = "Started"
                else:
                    label_dict["Unknown"] += 1
            else:
                if top1_index == 0:
                    label_dict["Crash"] += 1
                    current_label = "Crash"
                elif top1_index == 1:
                    label_dict["Flight"] += 1
                    current_label = "Flight"
                elif top1_index == 2:
                    label_dict["No drone"] += 1
                    current_label = "No drone"
                elif top1_index == 3:
                    label_dict["No signal"] += 1
                    current_label = "No signal"
                elif top1_index == 4:
                    label_dict["No started"] += 1
                    current_label = "No started"
                elif top1_index == 5:
                    label_dict["Started"] += 1
                    current_label = "Started"
                elif top1_index == 6:
                    label_dict["Unstable"] += 1
                    current_label = "Unstable"
                elif top1_index == 7:
                    label_dict["Landing"] += 1
                    current_label = "Landing"
                else:
                    label_dict["Unknown"] += 1

        if current_label == "Crash":
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


        count += 1 # frame is over
    
    # video is over
    if is_crash:
        # ended on a crash frame
        crash_end_time = total_frames / fps
        crash_duration = crash_end_time - crash_start_time
        if crash_duration >= min_crash_duration:
            crash_events.append((crash_start_time, crash_end_time))

    cap.release()
    cv2.destroyAllWindows()

    # merge events to only have unique crashes
    merged_crash_events = merge_crash_events(crash_events)

    return label_dict, len(merged_crash_events), duration, total_frames
