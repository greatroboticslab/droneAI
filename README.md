# Drone Classification AI
The purpose of this repository is to research how well student's drone flying improve. The model is built off yolov8 pretrained classification model. Images used for training can be found here: https://app.roboflow.com/soil-moisture/drone-firstperson-classification/1
## Docker
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