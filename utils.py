import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image

from loss_functions import ContentLoss, Normalization, StyleLoss


def load_image(image_path, device, max_size=512, shape=None):
    """Converts input image into tensors of appropriate size

    Args:
        image_path (str): absolute path to image file
        device (bool): check if cuda available
        max_size (int, optional): . Defaults to 512
        shape (int, optional): _description_. Defaults to None

    Returns:
        tensor: tensor of dimension B x C x W x H
    """
    image = Image.open(image_path)
    if max(image.size) > max_size:
        size = max_size
    else:
        size = max(image.size)
    if shape is not None:
        size = shape

    transform = transforms.Compose([transforms.Resize(size), transforms.ToTensor()])
    image = transform(image).unsqueeze(0)
    if device:
        image = image.cuda()
    return image


def save_image(image_tensor, device):
    """Transforms tensors to it's native format PNG

    Args:
        image_tensor (tensor): input image tensor 
        device (bool): returns true on cuda enabled device
    """
    transform = transforms.ToPILImage()
    if device:
        image = image_tensor.cpu().clone().detach().squeeze()
    else:
        image = image_tensor.clone().detach().squeeze()
    image = transform(image_tensor)
    image.save("output_nst.png")


def compute_loss(
    content_image, style_image, normalization_mean, normalization_sd, device
):
    """Modifies the VGG19 network to extract loss (style and content) from defined layers as in Gatys "https://arxiv.org/pdf/1508.06576.pdf" 


    Args:
        content_image (tensor): content image
        style_image (tensor): style image
        normalization_mean (tensor): 0.485, 0.456, 0.406 (default)
        normalization_sd (tensor): 0.229, 0.224, 0.225 (default)  
        device (bool): check if cuda is available

    Raises:
        RuntimeError: Raises error for unrecogonized layers in VGG19

    Returns:
        model(sequential module list), style and content loss
    
    """

    if device:
        vgg19 = models.vgg19(pretrained=True).features.cuda().eval()
        normalization = Normalization(normalization_mean, normalization_sd).cuda()
    else:
        vgg19 = models.vgg19(pretrained=True).features.eval()
    normalization = Normalization(normalization_mean, normalization_sd)

    content_layers = ["conv_10"]
    style_layers = ["conv_1", "conv_3", "conv_5", "conv_9", "conv_13"]

    content_losses = []
    style_losses = []

    model = nn.Sequential(normalization)

    i = 0
    for layer in vgg19.children():
        if isinstance(layer, nn.Conv2d):
            i += 1
            name = f"conv_{i}"
        elif isinstance(layer, nn.ReLU):
            name = f"relu_{i}"
            layer = nn.ReLU(inplace=False)
        elif isinstance(layer, nn.MaxPool2d):
            name = f"max_pool_{i}"
        else:
            raise RuntimeError(f"Unrecognized layer: {layer.__class__.__name__}")

        model.add_module(name, layer)

        if name in content_layers:
            target = model(content_image).detach()
            content_loss = ContentLoss(target)
            model.add_module(f"content_loss_{i}", content_loss)
            content_losses.append(content_loss)

        if name in style_layers:
            target = model(style_image).detach()
            style_loss = StyleLoss(target)
            model.add_module(f"style_loss_{i}", style_loss)
            style_losses.append(style_loss)

    return model, content_losses, style_losses

