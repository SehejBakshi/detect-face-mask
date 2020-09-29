# COVID-19: Face Mask Detector with OpenCV, and Deep Learning
This program detects whether a person is wearing a face mask or not both in images, and in real-time video stream.
Right now, this works only for one person wearing a mask in a frame. So there's room for improvement!

## Dataset
The dataset here was made by @prajnasb (<a href="https://github.com/prajnasb/observations">Prajna's Github Repo</a>)
This dataset consists of 1,376 images belonging to two classes:
1. with_mask: 690 images
2. without_mask: 686 images

## Program Structure
There are 3 python scripts in this repo:
1. train_mask_detector.py: Accepts the input dataset and fine-tunes MobileNetV2 upon it to create the mask_detector.model. 
A training history plot.png containing accuracy/loss curves is also produced.

![Accuracy/Loss curves](images/plot.png?raw=true "Accuracy/Loss curves")

2. detect_mask_image.py: Performs face mask detection in static images.
3. detect_mask_video.py: Using the webcam, this script applies face mask detection to every frame in the stream.

## Running the scripts
1. Although the train_mask_detector.py script has been already run and its weights saved, you can still run it by opening the terminal/cmd in
the script directory and typing this command: <code>python train_mask_detector.py --dataset dataset</code>

2. To detect masks on a static image, run detect_mask_image.py in the terminal/cmd by typing: <code>python detect_mask_image.py --image images/example_01.png</code>

3. To detect masks in real time, run detect_mask_video.py in the terminal/cmd by typing: <code>python detect_mask_video.py</code>
