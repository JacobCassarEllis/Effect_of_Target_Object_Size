To obtain the models presented in this Final Year Project, please use the following links:

Detectron2 - https://github.com/facebookresearch/detectron2
Yolov3 - https://github.com/eriklindernoren/PyTorch-YOLOv3
EfficientDet - https://github.com/zylo117/Yet-Another-EfficientDet-Pytorch

Each of these repositories contain a ReadMe file that was used to set up the environments for these models. It was noted that Detectron2 requires a CUDA environment which can be downloaded from:
https://developer.nvidia.com/cuda-downloads?target_os=Windows&target_arch=x86_64

Some modifications were made to the original code to allow each model to be tested on a custom dataset:

Detectron2 - "run.py" is a custom python file used to make predictions of a folder of images which can be found in the folder detectroncode. To use, the user must simply change the input and output 	
filepaths. To apply the custom dataset, each object instance was tested seperately. Thus for each model, the user must run the model four times to evaluate all four object instances, changing the
filepaths each time.

Yolov3 - "detect.py" can be found in the original github repository however some slight modifications such as the threshold limit and filepaths were altered to maintain consistency throughout this
experiment. A new "detect.py" can be found in the folder yolocode. The user must follow a similar process to detectron2 to process the custom dataset.

EfficientDet - "testimage.py" can be found in the folder efficientdetcode. This file is a modified version of "efficientdet_test.py" which was used to apply the model to a webcam. Again, a similar
process to the two previous models was applied to this model. To use, the user must simply alter the filepaths and run the model.

The Results obtained from this experiment can be located within the folder Results. Within this folder are the resultant output images for each model. The manually extracted results can be located in
the subfolder Manually Extracted Results and contains the results for all 12 tests along with a final result sheet labelled Final Calculations which was used to visualize the results.

The original dataset used can be found within the folder Custom Dataset. This contains five subfolders for the baselines and the four object instances. For each object instances, twelve images may be found
labelled according to the distance from the object.

Additionally, Additional Resources contains the full code of each model that is found in the github repository links. It is still recommended to clone the repository directly first.