# Fruit_detection_yolov5
A simple object detection project using **YOLOv5** to detect **9 types of fruits** in images. 
The pipeline covers dataset preparation, training, and real-time inference.

# Classes Detected
`apple`, `avocado`, `banana`, `guava`, `kiwi`, `mango`, `orange`, `peach`, `pineapple`

# Dataset
- **Source:** https://www.kaggle.com/datasets/itsmeaman03/fruit-detection-yolo
- **Format:** YOLO format (`.txt` labels)
- **Splits:**
  - Training images: `X` images
  - Validation images: `Y` images
 
# Setup
- **Clone YOLOv5**
  - git clone https://github.com/ultralytics/yolov5
  - cd yolov5

- **Create environment**
  - conda create -n yolov5 python=3.9 -y
  - conda activate yolov5
  - pip install -r requirements.txt
  # OR
- **python environment**
  - python -m venv yolov5
  - yolov5/Scripts/actiavte

- **yaml file**
  - Create yaml file
  - Includes number of classes, name of classes and address to the dataset train and val folders

- **yolov5 weights**
  - yolov5s.pt : small speed + moderate accuracy
  - yolov5l.pt : large speed + high accuracy
  - yolov5m.pt : medium speed + better accuracy
  - yolov5x.pt : slow speed + high accuracy
  - yolov5n.pt : fast speed + low accuracy

# Training
  - cd to env and yolov5 clone folder
  - python train.py --img (img_size) --batch (batch_size) --epochs (no.of epochs) --data (add to yaml file) --weights (preferred weight) --device (cpu/gpu)

# Saved model
  - Default model saves in clone yolov5/runs/train/exp
  - During training, two main model checkpoint files are generated
    - **best.pt** : Model with best performance on validation set
    - **last.pt** : Model at end of final epoch

# Inference
  - python detect.py --weights (path to the weights in run/exp) --img (img size as of training) --source (path to img to test)
  - **Saves results at** : yolov5clone folder/runs/detect/exp 

# Tools Used
  - Python 3.9
  - PyTorch
  - YOLOv5
  - Jupyter (for testing)

