'''
Date: 2024-10-01 13:49:57
LastEditors: Jiaqi Gu && jiaqigu@asu.edu
LastEditTime: 2024-10-01 14:09:38
FilePath: /ONN_Reliable/core/models/attack_defense/post_recovery.py
'''
import torch
from torch import Tensor, nn

from core.models.layers.gemm_conv2d import GemmConv2d
from core.models.layers.gemm_linear import GemmLinear


class post_corrector(object):
    def __init__(self, dirty_model: nn.Sequential, device) -> None:
        self.dirty_model = dirty_model
        self.device = device

        for layer in self.dirty_model.modules():
            if isinstance(layer, (GemmConv2d, GemmLinear)):
                self.N_bits = layer.weight_quantizer.N_bits

    def perform_correction(
        self, L_K: dict[Tensor], W_K: dict[Tensor], G_size: dict[int]
    ):
        """
        Description: Perrform attack under Unary + Locking protection
        """
        for name, layer in self.dirty_model.named_modules():
            if isinstance(layer, (GemmConv2d, GemmLinear)):
                self.layerwise_correction(
                    layer=layer, L_K=L_K[name], W_K=W_K[name], G_size=G_size[name]
                )

        return self.dirty_model

    def layerwise_correction(self, layer, L_K: Tensor, W_K: Tensor, G_size: int):
        """
        Description: Calculate new signature and find those groups attacked. Replace weights in those
        groups with pre-calculated centers.
        """
        layer.num_group = (layer.weight.data.numel() / G_size).__ceil__()
        # Calculate new signature
        if G_size == 1:
            layer._MSB_new = torch.tensor(
                [
                    int(bin(x)[2:].zfill(self.N_bits)[0])
                    for x in layer.weight_quantizer.w_q_com.data.view(-1).int()
                ]
            )
            layer._signature = layer._MSB_new
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

        # Compare the signature with golden ones and identify the group index that be attacked
        dff_indice = (layer._golden_signature != layer._signature).nonzero(
            as_tuple=True
        )[0]

        # Replace the groups with pre-calculated centers
        if G_size == 1:
            for i in range(dff_indice.shape[0]):
                layer.weight_quantizer.w_q_com.data.view(-1)[dff_indice[i]] = W_K[
                    L_K[dff_indice[i]]
                ]
        else:
            for i in range(dff_indice.shape[0]):
                layer.weight_quantizer.w_q_com.data.view(-1)[
                    dff_indice[i] :: layer.num_group
                ] = W_K[L_K[dff_indice[i]]]

        layer.weight_quantizer.from_two_com()
