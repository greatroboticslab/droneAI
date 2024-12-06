# get youtube links, download the videos, process the videos, make classification prediction, and output results

from yt_dlp import YoutubeDL
import video_classifier as classifier
from dataclasses import dataclass
import os
import csv
import pandas as pd

@dataclass
class VideoResult:
    video_path: str
    label_counts: dict
    unique_crashes: int
    video_link: str
    is_simulation: int
    duration: str
    total_frames: int
    video_title: str


def parse_inputs(simulation_input, real_input):
    simulation_links = [url.strip() for url in simulation_input.split(',') if url.strip()]
    real_links = [url.strip() for url in real_input.split(',') if url.strip()]
    return simulation_links, real_links

def get_input():
    print("============================================")
    print("Getting Student Name...")
    print("============================================")
    student = input("Student's Name for Save File: ").strip()
    # get youtube links
    print("============================================")
    print("Getting YouTube Links...")
    print("Provide YouTube Links. Examples: \'https://youtu.be/Gd12g9XqSo4,https://youtu.be/Gd12g9XqSo4,https://youtu.be/Gd12g9XqSo4\'")
    print("============================================")
    simulation_input = input("Provide Simulation YouTube Videos > ")
    print("--------------------------------------------")
    real_input = input("Provide Real Flying YouTube Videos > ")
    # parse links
    simulation_links, real_links = parse_inputs(simulation_input, real_input)

    return student, simulation_links, real_links

def download_videos(simulation_links, real_links,video_folder):
    # download each youtube video
    print("============================================")
    print("Downloading YouTube Videos...")
    print("============================================")
    simulation_paths = []
    real_path = []
    for i, url in enumerate(simulation_links, start=1):
        ydl_opts = {
            'format': 'best',
            'outtmpl': f'{video_folder}/sim_video{i}.%(ext)s'
        }
        with YoutubeDL(ydl_opts) as ydl:
            ydl.download([url])
        simulation_paths.append(f'{video_folder}/sim_video{i}.mp4')
    print("--------------------------------------------")
    for i, url in enumerate(real_links, start=1):
        ydl_opts = {
            'format': 'best',
            'outtmpl': f'{video_folder}/real_video{i}.%(ext)s'
        }
        with YoutubeDL(ydl_opts) as ydl:
            ydl.download([url])
        real_path.append(f'{video_folder}/real_video{i}.mp4')
    
    return simulation_paths, real_path

def run_each_model(simulation_links, real_links, simulation_paths, real_path,student,output_folder):
    print("============================================")
    print("Processing Each Video...")
    print("============================================")
    model = classifier.get_model()
    sim_results, real_results = detect_crashes(model, simulation_links, real_links, simulation_paths, real_path)

    print("============================================")
    print("Saving Results...")
    print("============================================")

    df_results = result_dataframe(student,sim_results, real_results)

    df_results.to_csv(os.path.join(output_folder,'results.csv'), index=False)

def detect_crashes(model, simulation_links, real_links, simulation_paths, real_path):
    simulation_results = []
    real_results = []
    for i, video_path in enumerate(simulation_paths, start=1):
        crash_video_filename = f'Simulation_Video_{i}_Crashes.mp4'
        crash_video_path = os.path.join(output_folder,crash_video_filename)
        labeled_video_filename = f'Simulation_Video_{i}_Labeled.mp4'
        labeled_video_path = os.path.join(output_folder,labeled_video_filename)

        label_dict = {"Crash": 0,"Flight": 0,"No drone": 0,"No signal": 0,"Started": 0,"Landing": 0,"Unknown": 0}
                                                       
        label_counts, unique_crashes, duration, total_frames = classifier.video_classification(label_dict,video_path,labeled_video_path,crash_video_path, model,2.0)
        simulation_results.append(
            VideoResult(
                video_path=video_path, # current video's file path
                label_counts=label_counts, # how many times each label got predicted
                unique_crashes=unique_crashes, # how many unique crashes were calculated
                video_link=simulation_links[i-1], # the video's youtube link
                is_simulation=1, # 1 if it's a simulation video
                duration = duration, # duration of the video
                total_frames = total_frames, # total amount of frames for the video
                video_title = f"Simulation Flying #{i}" # title for the video in the results file
            )
        )
    for i, video_path in enumerate(real_path, start=1):
        crash_video_filename = f'Real_Flying_Video_{i}_Crashes.mp4'
        crash_video_path = os.path.join(output_folder,crash_video_filename)
        labeled_video_filename = f'Real_Flying_Video_{i}_Labeled.mp4'
        labeled_video_path = os.path.join(output_folder,labeled_video_filename)

        label_dict = {"Crash": 0,"Flight": 0,"No drone": 0,"No signal": 0,"No started": 0,"Started": 0,"Unstable": 0,"Landing": 0,"Unknown": 0}
        label_counts, unique_crashes, duration, total_frames = classifier.video_classification(label_dict,video_path,labeled_video_path,crash_video_path, model,2.0)
        real_results.append(
            VideoResult(
                video_path=video_path,
                label_counts=label_counts,
                unique_crashes=unique_crashes,
                video_link=real_links[i-1],
                is_simulation=0,
                duration = duration,
                total_frames = total_frames,
                video_title = f"Real Flying #{i}"
            )
        )
    return simulation_results, real_results


def result_dataframe(student,simulation_results, real_results):
    all_results = simulation_results + real_results
    data = []
    for result in all_results:
        total_labels = sum(result.label_counts.values())
        row = {
            "Student": student,
            "Video Instance": result.video_title,
            "Video Path": result.video_path,
            "Predicted Crashes": result.unique_crashes,
            "Video Link": result.video_link,
            "Is Simulation": result.is_simulation,
            "Video Duration": result.duration,
            "Total Frames": result.total_frames,
            "Crash Frames %": (result.label_counts["Crash"] / result.total_frames * 100),
            "Flight Frames %": (result.label_counts["Flight"] / total_labels * 100),
            "NoSignal Frames %": (result.label_counts["No signal"] / total_labels * 100),
            "Started Frames %": (result.label_counts["Started"] / total_labels * 100),
            "Landing Frames %": (result.label_counts["Landing"] / total_labels * 100)
        }
        data.append(row)
    return pd.DataFrame(data)

def delete_youtube_videos(simulation_paths,real_path):
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
    run_each_model(simulation_links, real_links, simulation_paths, real_path,student,output_folder)
    delete_youtube_videos(simulation_paths,real_path)
    
    
    
   