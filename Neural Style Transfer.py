from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

import torch
from torchvision import models
import torchvision.transforms.v2 as tfs_v2
import torch.nn as nn
import torch.optim as optim

# Определение устройства (GPU или CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_image(path):
    image = Image.open(path).convert('RGB')
    transform = tfs_v2.Compose([
        tfs_v2.ToImage(),
        tfs_v2.ToDtype(torch.float32, scale=True)
    ])
    return transform(image).unsqueeze(0)

class ImageStyle(nn.Module):
    def __init__(self):
        super().__init__()
        model_ = models.vgg19(weights=models.VGG19_Weights.DEFAULT)
        self.mf = model_.features
        self.mf.requires_grad_(False)
        self.mf.eval()
        self.ind_out = (0, 5, 10, 19, 28, 34)
        self.num_style_layers = len(self.ind_out) - 1

    def forward(self, x):
        outputs = []
        for indx, layer in enumerate(self.mf):
            x = layer(x)
            if indx in self.ind_out:
                outputs.append(x.squeeze(0))
        return outputs

def get_content_loss(content, target):
    return torch.mean(torch.square(content - target))

def gram_matrix(x):
    channels = x.size(dim=0)
    g = x.view(channels, -1)
    gram = torch.mm(g, g.mT) / g.size(dim=1)
    return gram

def get_style_loss(base_style, gram_target):
    style_weights = [1.0, 0.8, 0.5, 0.3, 0.1]
    _loss = 0
    for i, (style, target) in enumerate(zip(base_style, gram_target)):
        gram_style = gram_matrix(style)
        _loss += style_weights[i] * torch.mean(torch.square(gram_style - target))
    return _loss

def save_and_show(tensor_img, path='result.jpg'):
    x = tensor_img.detach().cpu().squeeze()
    low, hi = torch.amin(x), torch.amax(x)
    x = (x - low) / (hi - low) * 255.0
    x = x.permute(1, 2, 0).cpu().numpy()
    x = np.clip(x, 0, 255).astype('uint8')
    image = Image.fromarray(x, 'RGB')
    image.save(path)
    plt.imshow(x)
    plt.axis('off')
    plt.show()

if __name__ == '__main__':
    img_path = 'img.jpg'
    style_path = 'img_style.jpg'
    epochs = 1000
    lr = 0.01
    content_weight = 1
    style_weight = 1700

    img = load_image(img_path).to(device)
    img_style = load_image(style_path).to(device)
    img_create = img.clone().to(device)
    img_create.requires_grad_(True)

    model = ImageStyle().to(device)
    outputs_img = model(img)
    outputs_img_style = model(img_style)
    gram_matrix_style = [gram_matrix(x) for x in outputs_img_style[:model.num_style_layers]]

    best_loss = float('inf')
    best_img = img_create.clone()
    optimizer = optim.Adam(params=[img_create], lr=lr)

    for epoch in range(epochs):
        outputs_img_create = model(img_create)

        loss_content = get_content_loss(outputs_img_create[-1], outputs_img[-1])
        loss_style = get_style_loss(outputs_img_create, gram_matrix_style)
        loss = content_weight * loss_content + style_weight * loss_style

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        with torch.no_grad():
            img_create.clamp_(0, 1)

        if loss.item() < best_loss:
            best_loss = loss.item()
            best_img = img_create.clone()

        print(f'Epoch {epoch+1}/{epochs}, Loss: {loss.item():.4f}')

    save_and_show(best_img)
