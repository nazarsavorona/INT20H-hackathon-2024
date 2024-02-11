# RSNA Pneumonia Detection Challenge
## About
This is experimental approach for the Pneumonia detection challenge available [here](https://www.kaggle.com/competitions/rsna-pneumonia-detection-challenge/).

### Running Inference Script

To run the inference script, follow these steps:

1. **Clone the Repository:**
   ```
   git clone https://github.com/nazarsavorona/INT20H-hackathon-2024
   cd INT20H-hackathon-2024
   ```
   
2. **Install Dependencies:**
   ```
   pip install -r requirements.txt
   ```
   
3. **Run the Script:**
   ```
   python inference.py --binary-weights /path/to/binary_weights.pt --detector-weights /path/to/detector_weights.pt --source /path/to/image/directory --batch-size 128
   ```
   
   Replace the following arguments with your desired values:

   * **--binary-weights**: Path to the binary weights file.
   * **--detector-weights**: Path to the detector weights file.
   * **--source**: Source directory containing images for inference.
   * **--batch-size**: Batch size for inference (default is 128).


4. **Output:**

    After running the script, the results will be saved to submission.csv in the current directory.
   
## Implementation
### Approach/model selection
After some consideration, the final choice fell on using the YOLO model because of its high speed and good detection 
of multiple objects in the picture. The main idea was to break the task into two subtasks - pneumonia binary 
classification and object detection on previously classified data.
### Data preprocessing
As required by YOLO models, we use 640*640 image rescaling saved in ```.png``` format to proceed with further training.
Labels are stored as ```.txt``` file for each picture containing elements in format ```class x-center y-center width height```.
### Model building
For the classification part we use *YOLOv8M-cls*, which is a specific mid-sized variant of YOLOv8 for classification 
purposes and offers a pretty good balance between accuracy and speed.

Main selected parameters for *YOLOv8M-cls*:
* *dropout = 0.25*
* *optimizer = AdamW*
* *lr0 = 1e-4*
* *lrf = 1e-3*

For the object detection part we use same variant *YOLOv8m*, which is used for this type of tasks specifically.

Main selected parameters for *YOLOv8m*:
* *dropout = 0.1*
* *optimizer = AdamW*
* *lr0 = 1e-3*
* *lrf = 1e-2*

### Prediction
In this challenge, we are expected to orginize our predictions by the following format: 

```confidence x-min y-min width height```

Also we should provide only one prediction per image, althought there might be several bounding boxes, which have to be 
written one by one without separation.