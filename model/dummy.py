import numpy as np
import model.inferencer as inf
import cv2
import os

model = inf.load_model("model/densenet121nobn.ckpt")
model_semyon = inf.load_model_semyon("model/densenet_fold2_epoch29.pth")


def process(input_image, saved_cam=None):
    filename = os.path.splitext(input_image)[0].split("/")[-1]

    if saved_cam is not None:
        result, visualization = inf.predict_visual(model, input_image)
        cv2.imwrite("static/{0}_cam.png".format(filename), visualization)
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

    filename = os.path.splitext(input_image)[0].split("/")[-1]

    if saved_cam is not None:
        result, visualization = inf.predict_visual_semyon(model_semyon, input_image)
        cv2.imwrite("static/{0}_cam.png".format(filename), visualization)
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
