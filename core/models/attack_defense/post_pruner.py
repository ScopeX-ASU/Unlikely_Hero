'''
Date: 2024-10-01 13:49:57
LastEditors: Jiaqi Gu && jiaqigu@asu.edu
LastEditTime: 2024-10-01 14:09:29
FilePath: /ONN_Reliable/core/models/attack_defense/post_pruner.py
'''
import torch
from torch import nn

from core.models.layers.gemm_conv2d import GemmConv2d
from core.models.layers.gemm_linear import GemmLinear


def model_reset_weight(model: nn.Sequential):
    """
    Description: Restore model weights to clean weights
    """
    for layer in model.modules():
        if isinstance(layer, (GemmConv2d, GemmLinear)):
            layer.weight_quantizer.w_q_com.data.copy_(layer._clean_weight)
            layer.weight_quantizer.from_two_com()


class post_pruner(object):
    def __init__(self, dirty_model: nn.Sequential, device) -> None:
        self.dirty_model = dirty_model
        self.device = device

        for layer in self.dirty_model.modules():
            if isinstance(layer, (GemmConv2d, GemmLinear)):
                self.N_bits = layer.weight_quantizer.N_bits

    def perform_correction(self, G_size: int):
        """
        Description:
        """
        for _, layer in self.dirty_model.named_modules():
            if isinstance(layer, (GemmConv2d, GemmLinear)):
                self.layerwise_pruning(layer=layer, G_size=G_size)

        return self.dirty_model

    def layerwise_pruning(self, layer, G_size: int):
        layer.num_group = int(layer.weight.data.numel() / G_size)
        # Calculate new signature
        if G_size == 1:
            layer._MSB_new = torch.tensor(
                [
                    int(bin(x)[2:].zfill(self.N_bits)[0])
                    for x in layer.weight_quantizer.w_q_com.data.view(-1).int()
                ]
            )
            layer._signature = layer._MSB_new
        else:
            signature = []
            for i in range(layer.num_group):
                weight_chunk = layer.weight_quantizer.w_q_com.data.view(-1)[
                    i :: layer.num_group
                ]
                SA = bin((weight_chunk.sum() / (2**self.N_bits)).floor().int())[-1]
                SB = bin((weight_chunk.sum() / (2 ** (self.N_bits - 1))).floor().int())[
                    -1
                ]
                signature.append(int(SA + SB))
            layer._signature = torch.tensor(signature)

        dff_indice = (layer._golden_signature != layer._signature).nonzero(
            as_tuple=True
        )[0]

        # Replace the groups with centers (0 in pruning)
        if G_size == 1:
            for i in range(dff_indice.shape[0]):
                layer.weight_quantizer.w_q_com.data.view(-1)[dff_indice[i]] = 0
        else:
            for i in range(dff_indice.shape[0]):
                layer.weight_quantizer.w_q_com.data.view(-1)[
                    dff_indice[i] :: layer.num_group
                ] = 0

        layer.weight_quantizer.from_two_com()
