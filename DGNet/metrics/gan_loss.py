""" Full assembly of the parts to form the complete network """

import torch


def ls_discriminator_loss(scores_real, scores_fake):
    """
    Compute the Least-Squares GAN metrics for the discriminator.

    Inputs:
    - scores_real: PyTorch Variable of shape (N,) giving scores for the real data.
    - scores_fake: PyTorch Variable of shape (N,) giving scores for the fake data.

    Outputs:
    - metrics: A PyTorch Variable containing the metrics.
    """
    loss = (torch.mean((scores_real - 1) ** 2) + torch.mean(scores_fake ** 2)) / 2
    return loss


def ls_generator_loss(scores_fake):
    """
    Computes the Least-Squares GAN metrics for the generator.

    Inputs:
    - scores_fake: PyTorch Variable of shape (N,) giving scores for the fake data.

    Outputs:
    - metrics: A PyTorch Variable containing the metrics.
    """
    loss = torch.mean((scores_fake - 1) ** 2) / 2
    return loss


# %%

def bce_loss(input, target):
    """
    Numerically stable version of the binary cross-entropy metrics function.

    As per https://github.com/pytorch/pytorch/issues/751
    See the TensorFlow docs for a derivation of this formula:
    https://www.tensorflow.org/api_docs/python/tf/nn/sigmoid_cross_entropy_with_logits

    Inputs:
    - input: PyTorch Variable of shape (N, ) giving scores.
    - target: PyTorch Variable of shape (N,) containing 0 and 1 giving targets.

    Returns:
    - A PyTorch Variable containing the mean BCE metrics over the minibatch of input data.
    """
    neg_abs = - input.abs()
    loss = input.clamp(min=0) - input * target + (1 + neg_abs.exp()).log()
    return loss.mean()


# %%

def discriminator_loss(logits_real, logits_fake, device):
    """
    Computes the discriminator metrics described above.

    Inputs:
    - logits_real: PyTorch Variable of shape (N,) giving scores for the real data.
    - logits_fake: PyTorch Variable of shape (N,) giving scores for the fake data.

    Returns:
    - metrics: PyTorch Variable containing (scalar) the metrics for the discriminator.
    """
    true_labels = torch.ones(logits_real.size()).to(device=device, dtype=torch.float32)
    loss = bce_loss(logits_real, true_labels) + bce_loss(logits_fake, true_labels - 1)
    return loss


def generator_loss(logits_fake, device):
    """
    Computes the generator metrics described above.

    Inputs:
    - logits_fake: PyTorch Variable of shape (N,) giving scores for the fake data.

    Returns:
    - metrics: PyTorch Variable containing the (scalar) metrics for the generator.
    """
    true_labels = torch.ones(logits_fake.size()).to(device=device, dtype=torch.float32)
    loss = bce_loss(logits_fake, true_labels)
    return loss