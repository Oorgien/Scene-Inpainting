import os

import torch
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms

from models.RaGAN.model.model import InpaintingDiscriminator, InpaintingGenerator


def eval_image(args, image, mask, save_dir):
    # switch to evaluate mode
    args.model_G.eval()

    with torch.no_grad():
        mean = [0.5, 0.5, 0.5]
        std = [0.5, 0.5, 0.5]
        normalize = transforms.Normalize(mean=mean,
                                         std=std)
        transform = transforms.Compose([
            transforms.ToTensor(),
            normalize])
        image = transform(image)
        while len(mask.shape) != 4:
            mask = mask.unsqueeze(0)
        while len(image.shape) != 4:
            image = image.unsqueeze(0)

        # Input images and masks
        image = image.to(args.device)
        mask = (torch.ones(*mask.shape) - mask).to(args.device)

        mask_3x = torch.cat((mask, mask, mask), dim=1)
        masked_image = torch.mul(image, mask_3x)

        input = torch.cat((masked_image, mask), dim=1)

        # Prediction
        predicted = args.model_G(input)
        predicted = masked_image + torch.mul(predicted, (torch.ones(mask_3x.shape).to(args.device) - mask_3x))

        if not os.path.isdir(f"{save_dir}"):
            os.makedirs(f"{save_dir}")

        t = transforms.ToPILImage()
        mean = torch.Tensor([0.5, 0.5, 0.5]).view(1, 3, 1, 1)
        std = torch.Tensor([0.5, 0.5, 0.5]).view(1, 3, 1, 1)
        img = (predicted.cpu() * std + mean)
        img = t(img.squeeze(0))
        img.save(os.path.join(save_dir, "result.tiff"))

    return img


def test(args, image, mask, save_dir):
    if not args.parallel:
        args.device = torch.device(f"cuda:{args.gpu_id}" if torch.cuda.is_available() else "cpu")
    elif args.parallel:
        print(f"Multiple GPU devices found: {torch.cuda.device_count()}")
        args.device = torch.device(f"cuda" if torch.cuda.is_available() else "cpu")
    args.model_G = InpaintingGenerator(in_nc=4, out_nc=3, nf=64, n_blocks=8).to(args.device)

    # Loading states
    try:
        print("=> loading checkpoint '{}'".format(args.resume))
        checkpoint = torch.load(args.resume)
        args.model_G.load_state_dict(checkpoint['state_dict_G'])
        print("=> loaded checkpoint '{}' (epoch {})"
              .format(args.resume, checkpoint['epoch']))
    except FileNotFoundError:
        print("=> no checkpoint found at '{}'".format(args.resume))

    img = eval_image(args, image, mask, save_dir)
    return img

