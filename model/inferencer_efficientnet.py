import numpy as np
import torch
import cv2
import lungs_finder as lf  # pip install git+https://github.com/dirtmaxim/lungs-finder
import matplotlib.pyplot as plt
import os

from albumentations.core.composition import *
from albumentations.pytorch import ToTensor, ToTensorV2
from albumentations.augmentations.transforms import *

threshold = 0.5

fake_images = ["./fake_images/"+img for img in os.listdir('./fake_images/')]

preprocess = Compose([Normalize(mean=[0.485, 0.456, 0.406], std=[
                     0.229, 0.224, 0.225]), Resize(224, 224), ToTensorV2()])

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


def get_grad(model, img):

    torch.set_grad_enabled(True)

    gcam = GradCam(model, [model.conv_head])

    out = gcam(img)

    mask = (out >= threshold)[0]
    out[0:1, mask].sum().backward()

    grad = gcam.get(model.conv_head)[0:1]
    grad = torch.nn.functional.interpolate(
        grad, [224, 224], mode='bilinear', align_corners=False)

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
    out_b[0:1, mask].sum().backward()

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
    torch.hub.list('rwightman/gen-efficientnet-pytorch', force_reload=False)
    model = torch.hub.load(
        'rwightman/gen-efficientnet-pytorch', 'efficientnet_b3', pretrained=True)

    model.classifier = torch.nn.Sequential(
        torch.nn.Linear(1536, 14), torch.nn.Sigmoid())

    class Fixer(torch.nn.Module):
        def __init__(self, model):
            super(Fixer, self).__init__()
            self.model = model

    model = Fixer(model)
    model.load_state_dict(torch.load(ckpt_path)['state_dict'])
    model = model.model

    model.train()

    return model


def convert_prediction_to_pathology(y_pred):
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

    if image.ndim == 2 or (image.ndim == 3 and image.shape[2] == 1):
        image = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    elif image.ndim == 3 and image.shape[2] == 4:
        image = image[:, :, :3]

    image = preprocess(image=image)['image'].unsqueeze(0)
    return image


def prepare_fake():
    return torch.stack([prepare_image(img) for img in fake_images]).squeeze()


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
    image = torch.cat([image, prepare_fake()])

    if isCuda:
        image.to(torch.device('cuda'))
        model.to(torch.device('cuda'))

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

    return pred, final


def predict(model, path_to_image, isCuda=False):
    """
    used to predit labels only (to speed up the process)
    isCuda: bool flag, set to Treu to run on gpu
    """
    image = prepare_image(path_to_image)
    image = torch.cat([image, prepare_fake()])

    if isCuda:
        image.to(torch.device('cuda'))
        model.to(torch.device('cuda'))

    return model(image)[0]
    
"""
Code to use:

# must have
model = load_model("/home/alexander/work/hackathon/models/temp_models/_ckpt_epoch_4.ckpt")

# --- prediction

# visualization
pred, vis = predict_visual(model,"/home/alexander/work/hackathon/chest-14/images/00014022_084.png") # cardiomegaly

# pure prediction
pred = predict(model,"/home/alexander/work/hackathon/chest-14/images/00014022_084.png")

"""
