import numpy as np
import model.inferencer as inf
import cv2
import os
from matplotlib import pyplot as plt

model = inf.load_model("model/densenet121nobn.ckpt")
model_semyon = inf.load_model_semyon("model/densenet_fold2_epoch29.pth")


def process(input_image, saved_cam=None):
    if saved_cam is not None:
        result, visualization = inf.predict_visual(model, input_image)
        cv2.imwrite(saved_cam, visualization)
    else:
        result = inf.predict(model, input_image)

    # 1 argument: probability, 2 argument: diseases.
    return result.detach().cpu().numpy(), \
           np.array(["Atelectasis",
                     "Cardiomegaly",
                     "Consolidation",
                     "Edema",
                     "Effusion",
                     "Emphysema",
                     "Fibrosis",
                     "Hernia",
                     "Infiltration",
                     "Mass",
                     "Nodule",
                     "Pleural_Thickening",
                     "Pneumonia",
                     "Pneumothorax"])


def process_semyon(input_image, saved_cam = None):


    if saved_cam is not None:
        result, visualization = inf.predict_visual_semyon(model_semyon, input_image)
        plt.imsave(saved_cam, visualization)

    else:
        result = inf.predict_semyon(model_semyon, input_image)

    # 1 argument: probability, 2 argument: diseases.
    return result.detach().cpu().numpy(), \
        np.array(["Atelectasis",
                  "Cardiomegaly",
                  "Consolidation",
                  "Edema",
                  "Effusion",
                  "Emphysema",
                  "Fibrosis",
                  "Hernia",
                  "Infiltration",
                  "Mass",
                  "Nodule",
                  "Pleural_Thickening",
                  "Pneumonia",
                  "Pneumothorax"])
