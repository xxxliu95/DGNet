import torch
from tqdm import tqdm
import torch.nn.functional as F
from metrics.dice_loss import dice_coeff

def eval_dgnet(net, loader, device, mode):
    """Evaluation without the densecrf with the dice coefficient"""
    net.eval()
    mask_type = torch.float32
    n_val = len(loader)  # the number of batch
    tot = 0
    tot_lv = 0
    tot_myo = 0
    tot_rv = 0
    with tqdm(total=n_val, desc='Validation round', unit='batch', leave=False) as pbar:
        if mode=='val':
            for imgs, true_masks, _ in loader:
                imgs = imgs.to(device=device, dtype=torch.float32)
                true_masks = true_masks.to(device=device, dtype=mask_type)

                with torch.no_grad():
                    reco, z_out, mu_tilde, a_out, mask_pred, mu, logvar, _, _ = net(imgs, true_masks, 'test')

                mask_pred = a_out[:, :4, :, :]
                pred = F.softmax(mask_pred, dim=1)
                pred = (pred > 0.5).float()
                tot += dice_coeff(pred[:, 0:3, :, :], true_masks[:, 0:3, :, :], device).item()
                tot_lv += dice_coeff(pred[:, 0, :, :], true_masks[:, 0, :, :], device).item()
                tot_myo += dice_coeff(pred[:, 1, :, :], true_masks[:, 1, :, :], device).item()
                tot_rv += dice_coeff(pred[:, 2, :, :], true_masks[:, 2, :, :], device).item()
                pbar.update()
        else:
            for imgs, true_masks in loader:
                imgs = imgs.to(device=device, dtype=torch.float32)
                true_masks = true_masks.to(device=device, dtype=mask_type)

                with torch.no_grad():
                    reco, z_out, mu_tilde, a_out, mask_pred, mu, logvar, _, _ = net(imgs, true_masks, 'test')

                mask_pred = a_out[:, :4, :, :]
                pred = F.softmax(mask_pred, dim=1)
                pred = (pred > 0.5).float()
                tot += dice_coeff(pred[:, 0:3, :, :], true_masks[:, 0:3, :, :], device).item()
                tot_lv += dice_coeff(pred[:, 0, :, :], true_masks[:, 0, :, :], device).item()
                tot_myo += dice_coeff(pred[:, 1, :, :], true_masks[:, 1, :, :], device).item()
                tot_rv += dice_coeff(pred[:, 2, :, :], true_masks[:, 2, :, :], device).item()
                pbar.update()

    net.train()
    return tot / n_val, tot_lv / n_val, tot_myo / n_val, tot_rv / n_val