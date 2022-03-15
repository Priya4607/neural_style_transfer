import argparse
import os

import torch
import torch.optim as optim

from utils import compute_loss, load_image, save_image


def stylize(
    input_image,
    content_image,
    style_image,
    content_weight,
    style_weight,
    num_iter,
    normalization_mean,
    normalization_sd,
    device,
):
    """
        Function to implement artistic style effects on the content image

    Args:
        input_image (tensor): typically a clone of content image
        content_image (tensor): content image
        style_image (tensor): style image
        content_weight (float): 1 (default)
        style_weight (float): 1e6 (default)
        num_iter (int): number of iterations 
        normalization_mean (tensor): 0.485, 0.456, 0.406 (default)
        normalization_sd (tensor): 0.229, 0.224, 0.225 (default)
        device (bool): check if cuda is available

    Returns:
        tensor: target image with stylized effects 
    """

    model, content_losses, style_losses = compute_loss(
        content_image, style_image, normalization_mean, normalization_sd, device
    )
    input_image.requires_grad_(True)
    model.requires_grad_(False)

    optimizer = optim.LBFGS([input_image])

    step = [0]
    while step[0] < int(num_iter):

        def closure():
            with torch.no_grad():
                input_image.clamp_(0, 1)

            optimizer.zero_grad()

            model(input_image)

            layer_content_loss = 0
            layer_style_loss = 0

            for cl in content_losses:
                layer_content_loss += cl.loss
            for sl in style_losses:
                layer_style_loss += sl.loss

            layer_content_loss *= content_weight
            layer_style_loss *= style_weight

            total_loss = layer_content_loss + layer_style_loss

            total_loss.backward()

            step[0] += 1

            if step[0] % 100 == 0:
                print(f"Epochs {step[0]}/{num_iter}:")
                print(
                    "Content Loss: {:4f} Style Loss: {:4f}".format(
                        layer_content_loss.item(), layer_style_loss.item(),
                    )
                )

            return total_loss

        optimizer.step(closure)

    with torch.no_grad():
        input_image.clamp_(0, 1)

    return input_image


def _main():
    """ Function to parse arguments and produce stylized effects on the input image
    """

    arg_parser = argparse.ArgumentParser(
        description="Parser for neural style transfer for images"
    )
    arg_parser.add_argument(
        "--style_image",
        type=str,
        dest="style_image",
        help="Absolute path to style image",
    )
    arg_parser.add_argument(
        "--content_image",
        type=str,
        dest="content_image",
        help="Absolute path to content image",
    )
    arg_parser.add_argument(
        "--style_weight",
        dest="style_weight",
        default=1e6,
        help="Set the weight of style image",
    )
    arg_parser.add_argument(
        "--content_weight",
        dest="content_weight",
        default=1,
        help="Set the weight of content image",
    )
    arg_parser.add_argument(
        "--num_iter", dest="num_iter", default=300, help="Set the number of iterations"
    )

    args = vars(arg_parser.parse_args())

    # Check if cuda is enabled in the device
    device = torch.cuda.is_available()

    if not device:
        print("CUDA is not available. Using CPU ...")
    else:
        print("CUDA is available. Using GPU ...")

    content_image = load_image(os.path.expanduser(args["content_image"]), device)
    style_image = load_image(
        os.path.expanduser(args["style_image"]), device, shape=content_image.shape[-2:],
    )

    input_image = content_image.clone()
    if device:
        normalization_mean = torch.tensor([0.485, 0.456, 0.406]).cuda()
        normalization_sd = torch.tensor([0.229, 0.224, 0.225]).cuda()
    else:
        normalization_mean = torch.tensor([0.485, 0.456, 0.406])
        normalization_sd = torch.tensor([0.229, 0.224, 0.225])

    assert (
        style_image.size() == content_image.size()
    ), f"Need style image and content image of same size. Style_image: {style_image.size()} Content_image: {content_image.size()}"

    image = stylize(
        input_image,
        content_image,
        style_image,
        args["content_weight"],
        args["style_weight"],
        args["num_iter"],
        normalization_mean,
        normalization_sd,
        device,
    )

    save_image(image, device)


if __name__ == "__main__":
    _main()
