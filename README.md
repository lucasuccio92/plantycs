## <div align="center">Yolov5 Custom Implementation for Crop Detection</div>

<div align="center">

<br>

</div>

<p>
YOLOv5 is a family of object detection architectures and models pretrained on the COCO dataset, and represents <a href="https://ultralytics.com">Ultralytics</a>
 open-source research into future vision AI methods, incorporating lessons learned and best practices evolved over thousands of hours of research and development.<br>
   This is a customised version for corn plant identification as part of an ongoing research project.
</p>


### Training Checkpoints

<p>

   
   
cornmodem864i12002<br>
Colab trained model with 1,200 images (First50 dataset) containing 2,727 class instances<br>
```--img 864 --batch 8 --epochs 150 --cfg models/yolov5l.yaml```<br>
Peak performance in 23rd epoch<br>
mAP@0.5:
  
kornmodel960m<br>
Locally trained model with 684 images<br>
```--img 960 batch 4 --epochs 100 --data korn.yaml --cfg models/yolov5m.yaml```
<br>
Computing time: ~31 hours<br>
mAP@0.5: unknown
<br><br>
kornmodel960l_1200<br>
Colab trained model with 1200 images<br>
```--img 960 --batch 8 --epochs 150 --data korn.yaml --cfg models/yolov5l.yaml```
<br>
Computing time: ?<br>
mAP@0.5: unknown
<br>
</p>


<br>

### The Data

<p>Labelled with <a href="https://github.com/tzutalin/labelImg">labelImg</a></p>

<br>


### Workflow

<li>Feed image into Yolov5 object detector trained on corn plant centres</li>
<li>Find the orientation of rows (well-differentiated lines in point cloud)</li>
<li>Find all rows with >= 2 plants</li>
<li>Measure variability in planting distance</li>
<li>Output Coefficient of Variability per row and per image</li>

<br>

### To use

1. Detect corn plants in input images and generate label files:
```python detect.py --source [test-data] --weights [weights-file] --conf [confidence] --img [image-size] --save-txt --name [name]```
<br>e.g.:
```python detect.py --source test-data/ --weights runs/train/cornmodem864i12002/weights/best.pt --conf 0.2 --img 3072 --save-txt --name testrun1```

2. Run single samples through find_kornrows.py:
```python find_kornrows.py --image_name [image]```
<br>Run multiple samples through main.py:
```python main.py --imgdir [folder-name]```

### Utilities

<ul>slice_images.py for resizing HR input images</ul>
<ul>split_train_val.py for dividing data and structuring into Yolo-readable format</ul>
<ul>count_instances.py to count class instances in labels</ul>
<ul>find_kornrows.py to find the orientation of cornrows in the field</ul>





