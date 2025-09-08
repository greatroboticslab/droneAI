# Drone Classification AI
The purpose of this repository is to research how well student's drone flying improve. The model is built off yolov8 pretrained classification model. Images used for training can be found here: https://app.roboflow.com/soil-moisture/drone-firstperson-classification/1
## Docker

Docker is not necessary.

Here are the steps to run this codebase in Docker:
- docker build -f Dockerfile -t yolov8:1.0 .
- docker run --gpus all --shm-size=8g -it -d -e DISPLAY=$DISPLAY -v ${PWD}:/app yolov8:1.0

## Validation GUI
Here are the steps to run the validation GUI. The purpose is for getting the ground truth for the model's test videos.
- follow the docker steps above
- run the script ValidationGUI/app.py
- open http://localhost:5000/ in your local browser

## Requirements:
- Python 3.11.10
- pip install yt-dlp (for downloading YouTube links)
- pip install flask (for validation GUI)

## Update:
Successfully labeled first video, and marked 2 potential crashes. After the video finished processing, the program
saved the 2 crash clips along with a crash log marking the time of crash in a folder.
## Crash Log
<img width="689" height="158" alt="image" src="https://github.com/user-attachments/assets/8eb10fe3-6e28-4687-97f3-8c62a53b894f" />

## Saved Folder
<img width="508" height="181" alt="image" src="https://github.com/user-attachments/assets/938245ed-981c-478c-bf67-f7278f528f3e" />


