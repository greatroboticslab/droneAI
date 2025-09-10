# Drone Classification AI
The purpose of this repository is to research how well student's drone flying improve. The model is built off yolov8 pretrained classification model. Images used for training can be found here: https://app.roboflow.com/soil-moisture/drone-firstperson-classification/1
## Docker

Docker is not necessary.

Here are the steps to run this codebase in Docker:
- docker build -f Dockerfile -t yolov8:1.0 .
- docker run --gpus all --shm-size=8g -it -d -e DISPLAY=$DISPLAY -v ${PWD}:/app yolov8:1.0

## Requirements:
- Python 3.11.10
- pip install yt-dlp (for downloading YouTube links)
- pip install flask (for validation GUI)

## Steps To Run:
1. Clone the directory
   git clone https://github.com/greatroboticslab/droneAI.git

2. Get in the folder
   cd droneAI

3. Activate a virtual enviornment
   python -m venv venv
   source venv/Scripts/activate

4. Install Dependencies
   pip install flask yt-dlp ultralytics

5. Run the file
   python LabelGUI/app.py

6. Open the website:
   http://localhost:5000/

You should then see the GUI:
<img width="1833" height="969" alt="image" src="https://github.com/user-attachments/assets/b2920f4f-b0b6-4a22-a4c0-18c93d9349ca" />

7. Click on Label Validation GUI (Blue Option)

8. Enter the youtube video link and the corresponding student's name:
<img width="1043" height="447" alt="image" src="https://github.com/user-attachments/assets/39dda6c4-3f39-46f4-930d-6385ce96a25d" />

Check on "Delete original downloaded video after processing"
Press "Start Processing"

10. Click on "Mark Observation" whenever you see a crash

11. Once the video finishes, the folder should be automatically saved in your computer.
    Inside the folder you will see:
    - A text file that will show you the log of all crashes you marked
    - Small video clips of all the crashes

## Crash Log
<img width="689" height="158" alt="image" src="https://github.com/user-attachments/assets/8eb10fe3-6e28-4687-97f3-8c62a53b894f" />

## Saved Folder
<img width="508" height="181" alt="image" src="https://github.com/user-attachments/assets/938245ed-981c-478c-bf67-f7278f528f3e" />


