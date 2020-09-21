import re

import torch.nn as nn

import albumentations as A

import torchvision

import numpy as np
import torch
import cv2
import lungs_finder as lf  # pip install git+https://github.com/dirtmaxim/lungs-finder
import matplotlib.pyplot as plt

from skimage.color import rgb2gray, rgba2rgb, gray2rgb

from albumentations.core.composition import *
from albumentations.pytorch import ToTensor, ToTensorV2
from albumentations.augmentations.transforms import *

threshold = 0.15
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def normalizer(img, **params):
    img = img.astype(np.float32)
    if img.max() > 255:
        img /= 65535.
    elif img.max() > 1:
        img /= 255.
    return (img-0.5)*2


preprocess = Compose(
    [Lambda(image=normalizer), Resize(224, 224), ToTensorV2()])

# ---------- HELPERS FOR VISUALIZATION
"""
gradcam wrapperg
"""


class GradCam:
    def __init__(self, model, layers):
        self.model = model
        self.layers = layers
        self.hooks = []
        self.fmap_pool = dict()
        self.grad_pool = dict()

        def forward_hook(module, input, output):
            self.fmap_pool[module] = output.detach().cpu()

        def backward_hook(module, grad_in, grad_out):
            self.grad_pool[module] = grad_out[0].detach().cpu()

        for layer in layers:
            self.hooks.append(layer.register_forward_hook(forward_hook))
            self.hooks.append(layer.register_backward_hook(backward_hook))

    def close(self):
        for hook in self.hooks:
            hook.remove()

    def __enter__(self):
        return self

    def __exit__(self, type, value, traceback):
        self.close()

    def __call__(self, *args, **kwargs):
        self.model.zero_grad()
        return self.model(*args, **kwargs)

    def get(self, layer):
        assert layer in self.layers, f'{layer} not in {self.layers}'
        fmap_b = self.fmap_pool[layer]  # [N, C, fmpH, fmpW]
        grad_b = self.grad_pool[layer]  # [N, C, fmpH, fmpW]

        grad_b = torch.nn.functional.adaptive_avg_pool2d(
            grad_b, (1, 1))  # [N, C, 1, 1]
        gcam_b = (fmap_b * grad_b).sum(dim=1,
                                       keepdim=True)  # [N, 1, fmpH, fmpW]
        gcam_b = torch.nn.functional.relu(gcam_b)

        return gcam_b


"""
guidedbackprop wrapper
"""


class GuidedBackPropogation:
    def __init__(self, model):
        self.model = model
        self.hooks = []

        def backward_hook(module, grad_in, grad_out):
            if isinstance(module, torch.nn.ReLU):
                return tuple(grad.clamp(min=0.0) for grad in grad_in)

        for name, module in self.model.named_modules():
            self.hooks.append(module.register_backward_hook(backward_hook))

    def close(self):
        for hook in self.hooks:
            hook.remove()

    def __enter__(self):
        return self

    def __exit__(self, type, value, traceback):
        self.close()

    def __call__(self, *args, **kwargs):
        self.model.zero_grad()
        return self.model(*args, **kwargs)

    def get(self, layer):
        return layer.grad.cpu()


"""
gradcam predictor (also returns model prediction)
"""


def get_grad(model, img, dsize=[224, 224]):

    torch.set_grad_enabled(True)

    gcam = GradCam(model, [model.features])

    out = gcam(img)

    mask = (out >= threshold)[0]
    out[0:1, out[0].argmax()].sum().backward()

    grad = gcam.get(model.features)[0:1]
    grad = torch.nn.functional.interpolate(
        grad, dsize, mode='bilinear', align_corners=False)

    # we return everything only for 1 image
    return out[0], grad[0, 0, :, :]


"""
guided backprop predictor
"""


def get_gbprop(model, img):

    torch.set_grad_enabled(True)

    gdbp = GuidedBackPropogation(model)
    inp_b = img.requires_grad_()  # Enable recording inp_b's gradient
    out_b = gdbp(inp_b)
    mask = (out_b >= threshold)[0]
    out_b[0:1, out_b[0].argmax()].sum().backward()

    grad_b = gdbp.get(img)[0:1]  # [N, 3, inpH, inpW]
    grad_b = grad_b.mean(dim=1, keepdim=True).abs()  # [N, 1, inpH, inpW]

    return grad_b.squeeze()


def normalize(img):
    out = img-img.min()
    out /= (out.max()+1e-7)
    return out


# --------- END OF HELPERS FOR VISUALIZATION--------


# --------- model functions

def load_model(ckpt_path):
    """
    function us used to load our model from checkpoint and return it
    """

    from .densenet import densenet121

    model = densenet121(num_classes=14)

    class Fixer(torch.nn.Module):
        def __init__(self, model):
            super(Fixer, self).__init__()
            self.model = model

    model = Fixer(model)
    model.load_state_dict(torch.load(
        ckpt_path, map_location=torch.device('cpu'))['state_dict'])
    model = model.model

    model.train()

    return model


def convert_prediction_to_pathology(y_pred, threshold=threshold):
    """
    this function is used to convert vector of anwers to vector of string representations
    e.g. [1,1,0,0,0,0,...] to ['Atelec...','Cadioomeg...']
    """
    pathologies = np.asarray(['Atelectasis',
                              'Cardiomegaly',
                              'Consolidation',
                              'Edema',
                              'Effusion',
                              'Emphysema',
                              'Fibrosis',
                              'Hernia',
                              'Infiltration',
                              'Mass',
                              'Nodule',
                              'Pleural_Thickening',
                              'Pneumonia',
                              'Pneumothorax'])

    y_pred = y_pred.clone().detach().cpu().numpy()
    mask = (y_pred >= threshold).astype(bool)

    return pathologies[mask].tolist()


def prepare_image(path_to_image):
    """
    image preprocessor
    
    Args:
        path_to_image: string with image location
    """
    image = cv2.imread(path_to_image).astype(float)

    if image.ndim == 3:
        if image.shape[2] == 4:
            image = rgb2gray(rgba2rgb(image))
        else:
            image = rgb2gray(image)
    image = np.expand_dims(image, -1)

    image = preprocess(image=image)['image'].unsqueeze(0)
    return image


def predict_visual(model, path_to_image, isCuda=False):
    """
    function is used to predict labels and visualise image
    
    Args:
        path_to_image: str with image location (0-255)
        model: our loaded model
        isCuda: bool flag, set to Treu to run on gpu
    """
    # sadly, gbrop can not be extract at the same time as gradcam
    # so we have to do two predictions for the same image
    image = prepare_image(path_to_image)

    image = image.to(device)
    model.to(device)

    image.require_grad = True
    pred, gcam = get_grad(model, image)
    gbprop = get_gbprop(model, image)

    # plt.imshow((np.repeat(normalize(img.squeeze().detach().cpu().numpy())[:,:,np.newaxis],3,2)+plt.cm.hot(normalize(gbprop*grad).detach().cpu().numpy())[:,:,:3])/2)
    visualization = normalize((gbprop*gcam).detach().cpu().numpy())
    visualization = normalize(visualization)

    orig = (normalize(image.detach().cpu().numpy()
                      [0, 0, :, :])*255).astype(np.uint8)

    visualization = plt.get_cmap('hot')(visualization)[:, :, :3]

    final = np.zeros(visualization.shape)
    right_lung_haar_rectangle = lf.find_right_lung_haar(orig)
    left_lung_haar_rectangle = lf.find_left_lung_haar(orig)

    if (right_lung_haar_rectangle is not None) and (left_lung_haar_rectangle is not None):
        x, y, width, height = right_lung_haar_rectangle
        final[y:y + height, x:x + width] = visualization[y:y + height, x:x + width]
        x, y, width, height = left_lung_haar_rectangle
        final[y:y + height, x:x + width] = visualization[y:y + height, x:x + width]
    else:
        final = visualization

    final = normalize(final +
                      cv2.cvtColor(orig, cv2.COLOR_GRAY2RGB).astype(
                          np.float32)/255)

    return (pred>threshold).astype(int), final


def predict(model, path_to_image, isCuda=False):
    """
    used to predit labels only (to speed up the process)
    isCuda: bool flag, set to Treu to run on gpu
    """
    image = prepare_image(path_to_image)

    image = image.to(device)
    model.to(device)

    return model(image)[0]


# ---------- SEMYON


preprocess_semyon = A.Compose([A.Resize(256, 256),
                               A.Normalize(
    mean=[0.485, 0.456, 0.406],
    std=[0.229, 0.224, 0.225],
),
    ToTensor()
])


def new_densenet121(imagenet=True, path_to_weights=None):
    net = torchvision.models.densenet121()
    if imagenet:
        state_dict = torch.load(
            '../weights/misc/densenet121_pretrained.pth', map_location=torch.device('cpu'))
        # '.'s are no longer allowed in module names, but pervious _DenseLayer
        # has keys 'norm.1', 'relu.1', 'conv.1', 'norm.2', 'relu.2', 'conv.2'.
        # They are also in the checkpoints in model_urls. This pattern is used
        # to find such keys.
        pattern = re.compile(
            r'^(.*denselayer\d+\.(?:norm|relu|conv))\.((?:[12])\.(?:weight|bias|running_mean|running_var))$')
        for key in list(state_dict.keys()):
            res = pattern.match(key)
            if res:
                new_key = res.group(1) + res.group(2)
                state_dict[new_key] = state_dict[key]
                del state_dict[key]
        net.load_state_dict(state_dict)
        num_ftrs = net.classifier.in_features
        net.classifier = nn.Linear(num_ftrs, 14)
    else:
        if path_to_weights == None:
            num_ftrs = net.classifier.in_features
            net.classifier = nn.Linear(num_ftrs, 14)
        else:
            state_dict = torch.load(
                path_to_weights, map_location=torch.device('cpu'))
            num_ftrs = net.classifier.in_features
            net.classifier = nn.Linear(num_ftrs, 14)
            net.load_state_dict(state_dict)
    return net.to(device)


def new_inceptionV3(imagenet=True, path_to_weights=None):
    net = torchvision.models.inception_v3()
    if imagenet:
        state_dict = torch.load(
            '../weights/misc/inception_v3_pretrained_imagenet.pth', map_location=torch.device('cpu'))
        # '.'s are no longer allowed in module names, but pervious _DenseLayer
        # has keys 'norm.1', 'relu.1', 'conv.1', 'norm.2', 'relu.2', 'conv.2'.
        # They are also in the checkpoints in model_urls. This pattern is used
        # to find such keys.
        pattern = re.compile(
            r'^(.*denselayer\d+\.(?:norm|relu|conv))\.((?:[12])\.(?:weight|bias|running_mean|running_var))$')
        for key in list(state_dict.keys()):
            res = pattern.match(key)
            if res:
                new_key = res.group(1) + res.group(2)
                state_dict[new_key] = state_dict[key]
                del state_dict[key]
        net.load_state_dict(state_dict)
        num_ftrs = net.fc.in_features
        net.fc = nn.Linear(num_ftrs, 14)
    else:
        if path_to_weights == None:
            num_ftrs = net.fc.in_features
            net.fc = nn.Linear(num_ftrs, 14)
        else:
            state_dict = torch.load(
                path_to_weights, map_location=torch.device('cpu'))
            num_ftrs = net.fc.in_features
            net.fc = nn.Linear(num_ftrs, 14)
            net.load_state_dict(state_dict)
    net.aux_logits = False
    return net.to(device)


def prepare_image_semyon(path_to_image):
    """
    image preprocessor
    
    Args:
        path_to_image: string with image location
    """
    image = cv2.imread(path_to_image)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = preprocess_semyon(image=image)['image'].unsqueeze(0)
    return image


# './lungs_disease_classification/weights/DenseNet121_FocalLoss_40epochs/densenet121_FocalLoss_fold4_epoch39.pth'
def load_model_semyon(path):
    model = new_densenet121(False, path_to_weights=path)
    model.classifier = torch.nn.Sequential(
        model.classifier, torch.nn.Sigmoid())
    return model


def predict_semyon(model, path_to_image, isCuda=False):
    """
    used to predit labels only (to speed up the process)
    isCuda: bool flag, set to Treu to run on gpu
    """
    image = prepare_image_semyon(path_to_image)

    image = image.to(device)
    model.to(device)

    model.eval()

    return model(image)[0]


def predict_visual_semyon(model, path_to_image, isCuda=False):
    """
    function is used to predict labels and visualise image
    
    Args:
        path_to_image: str with image location (0-255)
        model: our loaded model
        isCuda: bool flag, set to Treu to run on gpu
    """
    # sadly, gbrop can not be extract at the same time as gradcam
    # so we have to do two predictions for the same image
    image = prepare_image_semyon(path_to_image)

    image = image.to(device)
    model.to(device)
    model.eval()

    image.require_grad = True
    pred, gcam = get_grad(model, image, dsize=[256, 256])

    # plt.imshow((np.repeat(normalize(img.squeeze().detach().cpu().numpy())[:,:,np.newaxis],3,2)+plt.cm.hot(normalize(gbprop*grad).detach().cpu().numpy())[:,:,:3])/2)
    visualization = normalize((gcam).detach().cpu().numpy())
    visualization = normalize(visualization)

    orig = (normalize(image.detach().cpu().numpy()
                      [0, 0, :, :])*255).astype(np.uint8)

    visualization = plt.get_cmap('hot')(visualization)[:, :, :3]

    final = np.zeros(visualization.shape)
    right_lung_haar_rectangle = lf.find_right_lung_haar(orig)
    left_lung_haar_rectangle = lf.find_left_lung_haar(orig)

    if (right_lung_haar_rectangle is not None) and (left_lung_haar_rectangle is not None):
        x, y, width, height = right_lung_haar_rectangle
        final[y:y + height, x:x + width] = visualization[y:y + height, x:x + width]
        x, y, width, height = left_lung_haar_rectangle
        final[y:y + height, x:x + width] = visualization[y:y + height, x:x + width]
    else:
        final = visualization

    final = normalize(final +
                      cv2.cvtColor(orig, cv2.COLOR_GRAY2RGB).astype(
                          np.float32)/255)
    final = normalize(final)

    return (pred>=threshold).astype(int), final


"""
Code to use:

# must have
model = load_model("/home/alexander/work/hackathon/models/temp_models/_ckpt_epoch_4.ckpt")

# --- prediction

# visualization
pred, vis = predict_visual(model,"/home/alexander/work/hackathon/chest-14/images/00014022_084.png", isCuda = True) # cardiomegaly

# pure prediction
pred = predict(model,"/home/alexander/work/hackathon/chest-14/images/00014022_084.png", isCuda = True)

"""
