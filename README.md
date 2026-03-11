DroneAI is a web-based labeling platform designed to help teams analyze drone flight videos, create machine-learning training data, and verify crash detection results. The system allows multiple people to work on the same dataset across different computers while keeping labeling progress synchronized.

The platform combines three major components:

• Validation labeling for identifying flight events such as takeoff, landing, and crashes
• Training dataset generation for machine learning
• Crash analysis and verification for comparing predicted crashes against human review

To support team collaboration, DroneAI includes:

• A shared SQLite database for storing labeling progress
• Import/export tools so teammates can continue work on different computers
• MQTT real-time communication to show labeling activity across the team
• Video locking to prevent duplicate labeling of the same video

The entire system runs through a Flask web interface, making it easy for labelers to use without needing programming knowledge.

System Overview

DroneAI works as a labeling and collaboration tool for drone flight datasets.

The system has four main functional areas:

Validation Labeling

Users watch drone videos and mark events such as:

Takeoff

Landing

Minor crashes

Severe crashes

Each labeled event is stored and used to generate labeled video clips and logs.

Training Data Collection

Users label video frames or segments to build datasets that can be used for machine learning models.

Crash Analysis / Verification

The system compares predicted crash counts from automated detection with human verification results.

Database Collaboration

All labeling data is stored in a SQLite database. Users can export and import this database to continue labeling work across multiple computers.

Collaboration Features

DroneAI supports team collaboration through two mechanisms.

Shared Database Workflow

All labeling sessions are stored in a SQLite database.

Users can:

Export the current database

Share it with teammates

Import the latest version before continuing labeling

This allows multiple people to work on the same dataset across different computers.

Recommended workflow:

Import the latest shared database.

Perform validation or training labeling.

Export the updated database.

Share it with the next teammate.

This ensures everyone works from the latest dataset.

MQTT Real-Time Communication

DroneAI also supports real-time communication using MQTT.

MQTT is a lightweight messaging protocol commonly used in IoT systems.

When connected to an MQTT broker, the system can broadcast live labeling activity such as:

A user starting validation on a video

A user starting training labeling

A labeling session finishing

These events allow teammates to see activity from other computers in real time.

Example event messages:

Haider started validation on video X
Sarah started training labeling on video Y
Alex finished validation on video Z
Video Locking System

To prevent duplicate labeling, DroneAI includes a locking system.

When a user starts labeling a video:

The system publishes a lock message through MQTT.

Other users attempting to label the same video will be warned or blocked.

When the labeling session finishes:

The lock is released automatically.

Other teammates are free to label that video.

This prevents two people from labeling the same video simultaneously.

Login System

DroneAI uses a simple team login system.

Each user logs in using:

• their own username
• a shared team password

Example:

Username: haider
Password: droneai2025

This approach allows:

easy team access

identification of which user labeled each video

MQTT activity tracking per user

The username is stored in the session and used when publishing MQTT events.

Project Structure
DroneAI/
│
├── LabelGUI/
│   ├── app.py
│   ├── mqtt_client.py
│   ├── training_backend.py
│   ├── validation_backend.py
│   ├── crash_verify_backend.py
│   ├── db_store.py
│   ├── video_utils.py
│
│   ├── templates/
│   │   ├── home.html
│   │   ├── login.html
│   │   ├── training_index.html
│   │   ├── training_preview.html
│   │   ├── training_results.html
│   │   ├── validation_index.html
│   │   ├── validation_results.html
│   │   ├── crash_analysis.html
│   │   └── mqtt.html
│
│   └── static/
│       └── style.css
│
├── analysis/
│   ├── data/
│   └── results/
│
└── db/
    └── droneai.sqlite
Installation
1. Clone the repository
git clone https://github.com/your-repo/droneai.git
cd DroneAI
2. Create a virtual environment
python -m venv venv

Activate it:

Windows:

venv\Scripts\activate

Mac/Linux:

source venv/bin/activate
3. Install dependencies
pip install flask
pip install pandas
pip install opencv-python
pip install paho-mqtt

You can also install from a requirements file if provided:

pip install -r requirements.txt
Running the Application

Start the Flask server:

python LabelGUI/app.py

Then open a browser and go to:

http://localhost:5000

You will see the DroneAI dashboard.

Using the Application
1. Validation Labeling

Click Start Validation

Enter a YouTube link or choose an entry from Excel

Watch the video

Mark events using the buttons

When the video finishes:

labeled clips are saved

event logs are generated

MQTT lock is released

2. Training Labeling

Click Start Training

Enter a video link

Choose labeling settings

Label frames for training data generation

Training results are saved for later machine learning use.

3. Crash Analysis / Verification

This module compares predicted crash events with human verification.

Users can review results and export analysis reports.

4. Database Tools

The DB Tools page allows users to:

• Download the current database
• Upload a previous database
• Export database tables to Excel

This is used for team collaboration across computers.

MQTT Setup

MQTT enables real-time collaboration across different machines.

To configure MQTT:

Open the MQTT page in the GUI.

Enter broker settings.

Example for testing:

Host: broker.hivemq.com
Port: 1883
Topic Prefix: droneai
Username: (leave blank)
Password: (leave blank)

Click Connect.

Once connected, the system will broadcast labeling events.

Example Team Workflow

Recommended workflow for a team:

Import the latest shared database.

Connect to MQTT.

Start validation or training labeling.

Export the updated database when finished.

Share the database with teammates.

This ensures all labeling work stays synchronized.

Technologies Used

DroneAI is built using:

Python

Flask

OpenCV

SQLite

MQTT

HTML / CSS / JavaScript
