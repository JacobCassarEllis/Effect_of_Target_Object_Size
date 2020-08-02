from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog
import cv2
import os

def load_images_from_folder(folder):

    cfg = get_cfg()
    cfg.merge_from_file("detectron2_repo/configs/COCO-Detection/faster_rcnn_R_101_FPN_3x.yaml") #load model config
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.1  # set threshold for this model
    cfg.MODEL.WEIGHTS = "detectron2://COCO-Detection/faster_rcnn_R_101_FPN_3x/137851257/model_final_f6e8b1.pkl" #load weights

    # Create predictor
    predictor = DefaultPredictor(cfg)

    images = []
    #iterates through a folder of images, one object instance at a time
    for filename in os.listdir(folder):
        im = cv2.imread(os.path.join(folder,filename))
        if im is not None:
            outputs = predictor(im)

            v = Visualizer(im[:, :, ::-1], MetadataCatalog.get(cfg.DATASETS.TRAIN[0]), scale=1.2)
            v = v.draw_instance_predictions(outputs["instances"].to("cpu"))

            test = outputs["instances"].to("cpu")
            path_text = 'experiment/chair/results' #store results
            result_filename = "result_" + filename + ".txt"
            f = open(os.path.join(path_text, result_filename), "a")
            f.write(str(test))
            f.close()

            output_file = "output_" + filename
            path_image = 'experiment/chair/output' #store results
            cv2.imwrite(os.path.join(path_image, output_file), v.get_image()[:, :, ::-1])


load_images_from_folder("experiment/chair/images")
# get image
