import torch
from os import path

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def save_network_state(model, width, height, ndf, norm, upsample, num_classes, decoder_type, anatomy_out_channels, z_length, num_mask_channels, optimizer, epoch , name , save_path):
    if not path.exists(save_path):
        raise ValueError("{} not a valid path to save model state".format(save_path))
    torch.save(
        {
            'epoch' : epoch,
            'width': width,
            'height': height,
            'ndf' : ndf,
            'norm' : norm,
            'upsample' : upsample,
            'num_classes': num_classes,
            'decoder_type' : decoder_type,
            'anatomy_out_channels' : anatomy_out_channels,
            'z_length' : z_length,
            'num_mask_channels' : num_mask_channels,
            'model_state_dict' : model.state_dict(),
            'optimizer_state_dict' : optimizer.state_dict()
        }, path.join(save_path, name))