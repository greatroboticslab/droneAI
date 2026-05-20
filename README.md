# DroneAI

DroneAI is a local web-based GUI for labeling drone video datasets. The current version is focused on local dataset upload, video validation labeling, training data collection, and crash review.

This project runs as a Flask application on your computer.

---

## Requirements

Before running the app, make sure you have:

- Python 3.10 or newer
- Git
- pip
- A virtual environment
- FFmpeg installed and added to PATH
- A supported Excel dataset file

The Excel file should include columns for:

- `Persona Name`
- `Youtube Link`


---

## Setup

Clone the repository:

```bash
git clone https://github.com/YOUR_USERNAME/DroneAI.git
cd DroneAI
