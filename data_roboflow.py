from roboflow import Roboflow

API_KEY = ""

# This is to download the datset from Roboflow to custom train the ball detection model. The dataset is already trained and the best.pt file is used in the shot_type.py file.
project = Roboflow(api_key=API_KEY).workspace("viren-dhanwani").project("tennis-ball-detection")
version = project.version(6)
dataset = version.download("yolov8")