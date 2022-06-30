import cv2
import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F

# https://github.com/kamata1729/visualize-pytorch

class GradCAM():
    def __init__(self, model, target_layer, use_cuda):
        self.model = model.eval()
        self.target_layer = target_layer
        self.use_cuda = use_cuda
        self.feature_map = 0
        self.grad = 0
        
        if self.use_cuda:
            self.model = self.model.cuda()
        
        for module in self.model.named_modules():
            if module[0] == target_layer:
                module[1].register_forward_hook(self.save_feature_map)
                module[1].register_backward_hook(self.save_grad)
    
    def save_feature_map(self, module, input, output):
        self.feature_map =  output.detach()
        
    def save_grad(self, module, grad_in, grad_out):
        self.grad = grad_out[0].detach()
        
    def __call__(self, x, index=None):
        if self.use_cuda:
            x = x.cuda()
        with torch.autograd.detect_anomaly():
            output = self.model(x)
            if index==None:
                index = output.argmax(dim=1)
                
            self.model.zero_grad()
            (F.one_hot(index, 2) * output).sum().backward()

            cam = F.relu((self.feature_map[0] * (self.grad.mean(dim=(2,3))[0, :])[:, None, None]).sum(dim=0))

            print(f"Max: {cam.max()}")
            # Normalized between 0 and 1
            cam = cv2.resize((cam/cam.max()).cpu().numpy(), (x.shape[-2], x.shape[-1]))

            # Without normalization
            #cam = cv2.resize((cam).cpu().numpy(), (x.shape[-2], x.shape[-1]))
            return cam, index

def show_cam_on_image(img, mask):
    heatmap = cv2.applyColorMap(np.uint8(255*mask), cv2.COLORMAP_JET)
    heatmap = np.float32(heatmap) / 255
    cam = heatmap + np.float32(img)/255
    cam = cam / np.max(cam)
    return np.uint8(255 * cam)