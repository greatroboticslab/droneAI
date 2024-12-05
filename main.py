# get youtube links, download the videos, process the videos, make classification prediction, and output results

#
# needs updating to match new structure in preprocess_data/TestModels.py
#
from yt_dlp import YoutubeDL
import video_classifier as classifier
from dataclasses import dataclass
import os
import csv

@dataclass
class VideoResult:
    video_path: str
    label_counts: dict
    crash_events: list
    video_link: str
    is_simulation: int

def get_input():
    print("============================================")
    print("Getting Student Name...")
    print("============================================")
    student = input("Student's Name for Save File: ").strip()
    # get how many videos
    sim_count = 0
    real_count = 0
    print("============================================")
    print("Getting Video Count...")
    print("============================================")
    while sim_count <= 0:
        while True:
            try:
                sim_count = int(input("How many simulation videos?: ").strip())
                break
            except ValueError:
                print("Invalid input. Please enter an integer.")
        if sim_count <= 0: print("Need at least 1 simulation video.")
    print("--------------------------------------------")
    while real_count <= 0:
        while True:
            try:
                real_count = int(input("How many real flying videos?: ").strip())
                break
            except ValueError:
                print("Invalid input. Please enter an integer.")
        if real_count <= 0: print("Need at least 1 real flying video.")
    # get youtube links
    print("============================================")
    print("Getting YouTube Links...")
    print("Provide YouTube Links. Examples: https://www.youtube.com/watch?v=8HkfQjC_2WU OR https://youtu.be/Gd12g9XqSo4")
    simulation_links = []
    real_links = []
    print("============================================")
    for i in range(sim_count):
        simulation_links.append(input(f"Provide Simulation YouTube Video #{i+1} > ").strip())
    print("--------------------------------------------")
    for i in range(real_count):
        real_links.append(input(f"Provide Real Flying YouTube Video #{i+1} > ").strip())

    return student, simulation_links, real_links

def download_videos(simulation_links, real_links,video_folder):
    # download each youtube video
    print("============================================")
    print("Downloading YouTube Videos...")
    print("============================================")
    simulation_paths = []
    real_path = []
    for i, url in enumerate(simulation_links, start=1):
        print(f"\nDownloading Simulation Video {url} to {video_folder}")
        ydl_opts = {
            'format': 'best',
            'outtmpl': f'{video_folder}/sim_video{i}.%(ext)s'
        }
        with YoutubeDL(ydl_opts) as ydl:
            ydl.download([url])
        simulation_paths.append(f'{video_folder}/sim_video{i}.mp4')
    print("--------------------------------------------")
    for i, url in enumerate(real_links, start=1):
        print(f"\nDownloading Real Flying Video {url} to {video_folder}")
        ydl_opts = {
            'format': 'best',
            'outtmpl': f'{video_folder}/real_video{i}.%(ext)s'
        }
        with YoutubeDL(ydl_opts) as ydl:
            ydl.download([url])
        real_path.append(f'{video_folder}/real_video{i}.mp4')
    
    return simulation_paths, real_path

def detect_crashes(simulation_links, real_links, simulation_paths, real_path):
    print("============================================")
    print("Detecting Crashes...")
    model = classifier.get_model()
    simulation_results = []
    real_results = []
    print("============================================")
    for i, path in enumerate(simulation_paths, start=1):
        print(f"Detecting Simulation Crashes in {path}...")

        crash_video_filename = f'Simulation_Video_{i}_Crashes.mp4'
        crash_video_path = os.path.join(output_folder,crash_video_filename)
        labeled_video_filename = f'Simulation_Video_{i}_Labeled.mp4'
        labeled_video_path = os.path.join(output_folder,labeled_video_filename)

        label_counts, crash_events = classifier.video_classification(path,labeled_video_path,crash_video_path, model, min_crash_duration=2.0)
        simulation_results.append(VideoResult(video_path=path, label_counts=label_counts, crash_events=crash_events,video_link=simulation_links[i-1],is_simulation=1))
    print("\n--------------------------------------------")
    for i, path in enumerate(real_path, start=1):
        print(f"Detecting Real Flight Crashes in {path}...")

        crash_video_filename = f'Real_Flying_Video_{i}_Crashes.mp4'
        crash_video_path = os.path.join(output_folder,crash_video_filename)
        labeled_video_filename = f'Real_Flying_Video_{i}_Labeled.mp4'
        labeled_video_path = os.path.join(output_folder,labeled_video_filename)

        label_counts, crash_events = classifier.video_classification(path,labeled_video_path,crash_video_path, model, min_crash_duration=2.0)
        real_results.append(VideoResult(video_path=path, label_counts=label_counts, crash_events=crash_events,video_link=real_links[i-1],is_simulation=0))
    
    return simulation_results, real_results

def print_results(simulation_results, real_results):
    print("\n============================================")
    print("Results...")
    print("============================================")
    print("Simulation Video Results:")
    for i, result in enumerate(simulation_results, start=1):
        print(f"\nResults for Simulation Video {i} ({result.video_path}):")
        for label, count in result.label_counts.items():
            print(f"{label}: {count}")
        print(f"\nNumber of unique crashes: {len(result.crash_events)}")
        for j, (start, end) in enumerate(result.crash_events, start=1):
            start_formatted = classifier.format_time(start)
            end_formatted = classifier.format_time(end)
            duration = end - start
            print(f"Crash {j}: Start time = {start_formatted}, End time = {end_formatted}, Duration = {duration:.2f}s")
        print("+++++++++++++++++++++++++++++++++++")
    print("--------------------------------------------")
    print("Real Flight Video Results:")
    for i, result in enumerate(real_results, start=1):
        print(f"\nResults for Real Flight Video {i} ({result.video_path}):")
        for label, count in result.label_counts.items():
            print(f"{label}: {count}")
        print(f"\nNumber of unique crashes: {len(result.crash_events)}")
        for j, (start, end) in enumerate(result.crash_events, start=1):
            start_formatted = classifier.format_time(start)
            end_formatted = classifier.format_time(end)
            duration = end - start
            print(f"Crash {j}: Start time = {start_formatted}, End time = {end_formatted}, Duration = {duration:.2f}s")
        print("+++++++++++++++++++++++++++++++++")

def result_csv(student,simulation_results, real_results,output_folder):
    print("============================================")
    print("Saving Results as CSV File...")
    print("============================================")
    results = simulation_results + real_results
    file_name = 'crash_results.csv'
    csv_file = os.path.join(output_folder,file_name)
    csv_columns = [
        'Student','YouTubeLink', 'IsSimulation', 'UniqueCrashes', 'CrashFrames','FlightFrames', 'NoSignalFrames','StartedFrames','LandingFrames','CrashTimes','CrashDurations'
    ]
    # start writing file
    with open(csv_file, 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=csv_columns)
        writer.writeheader()
        
        for result in results:
            unique_crashes = len(result.crash_events)

            # write results
            if unique_crashes > 0:

                # build string for durations and time stamps
                crash_times = ''
                crash_durations = ''
                for j, (start, end) in enumerate(result.crash_events, start=1):
                    start_formatted = classifier.format_time(start)
                    end_formatted = classifier.format_time(end)
                    duration = end - start

                    crash_times = crash_times + f"[{start_formatted},{end_formatted}]"
                    crash_durations = crash_durations + f"{duration:.2f}"
                    if unique_crashes - j != 0:
                        crash_times = crash_times + ","
                        crash_durations = crash_durations + ","
                
                writer.writerow({
                    'Student': student,
                    'YouTubeLink': result.video_link,
                    'IsSimulation': result.is_simulation,
                    'UniqueCrashes': unique_crashes,
                    'CrashFrames': result.label_counts["Crash"],
                    'FlightFrames': result.label_counts["Flight"],
                    'NoSignalFrames': result.label_counts["NoSignal"],
                    'StartedFrames': result.label_counts["Started"],
                    'LandingFrames': result.label_counts["Landing"],
                    'CrashTimes': crash_times,
                    'CrashDurations': crash_durations
                })  
            else:
                # no crashes
                writer.writerow({
                    'Student': student,
                    'YouTubeLink': result.video_link,
                    'IsSimulation': result.is_simulation,
                    'UniqueCrashes': unique_crashes,
                    'CrashFrames': result.label_counts["Crash"],
                    'FlightFrames': result.label_counts["Flight"],
                    'NoSignalFrames': result.label_counts["NoSignal"],
                    'StartedFrames': result.label_counts["Started"],
                    'LandingFrames': result.label_counts["Landing"],
                    'CrashTimes': None,
                    'CrashDurations': None 
                })
    print(f"Results have been saved to {csv_file}")

def delete_youtube_videos():
    print("============================================")
    print("Finishing Script...")
    print("============================================")
    while True:
        delete_videos = input("Do you want to delete the downloaded videos? (y/n): ").strip().lower()
        if delete_videos in ['y', 'yes']:
            print("Deleting downloaded videos...")
            # Combine all video paths
            all_video_paths = simulation_paths + real_path
            for video_file in all_video_paths:
                os.remove(video_file)
            print("All downloaded videos have been deleted.")
            break
        elif delete_videos in ['n', 'no']:
            print("Videos have not been deleted.")
            break
        else: print("Incorrect input...")

if __name__ == '__main__':
    video_folder = './videos'
    output_folder = './results'
    
    student, simulation_links, real_links = get_input()
    simulation_paths, real_path = download_videos(simulation_links, real_links,video_folder)
    simulation_results, real_results = detect_crashes(simulation_links, real_links, simulation_paths, real_path)
    print_results(simulation_results, real_results)
    result_csv(student,simulation_results, real_results,output_folder)
    delete_youtube_videos()
    
    
    
   