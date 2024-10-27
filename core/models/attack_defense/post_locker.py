import math

import numpy as np
import torch
from pyutils.general import logger as lg
from pyutils.torch_train import set_torch_deterministic
from sklearn.cluster import KMeans  # Original K-mean clustering problem
from torch import nn
from torch.utils.data import DataLoader

from core.models.layers.gemm_conv2d import GemmConv2d
from core.models.layers.gemm_linear import GemmLinear
from core.models.layers.utils import CustomDistanceKMeans, calculate_grad_hessian
from train_pretrain import validate


def model_reset_weight(model: nn.Sequential):
    """
    Description: Restore model weights to clean weights
    """
    for layer in model.modules():
        if isinstance(layer, (GemmConv2d, GemmLinear)):
            layer.weight_quantizer.w_q_com.data.copy_(layer._clean_weight)
            layer.weight_quantizer.from_two_com()


def calculate_taylor_expansion(model: nn.Sequential, max_value):
    """
    Description: Calculate Taylor Expansion terms caused by possible Bit-flips
    """
    for layer in model.modules():
        if isinstance(layer, (GemmConv2d, GemmLinear)):
            layer.weight._taylor_expansion = (
                layer.weight._first_grad.data
                * (layer.weight_quantizer.w_q_com.data - max_value).sign()
                + layer.weight._second_grad.data * max_value / 2
            )


def calculate_KMeans_obj(
    XA: np.ndarray, XB: np.ndarray, first_grad: np.ndarray, sec_grad: np.ndarray
) -> np.ndarray:
    XA = XA.reshape(-1, 1)
    XB = XB.reshape(1, -1)
    first_grad = first_grad.reshape(-1, 1)
    sec_grad = sec_grad.reshape(-1, 1)
    #
    distance = first_grad * (XA - XB) + 0.5 * sec_grad * (XA - XB) ** 2
    return distance


class smart_locker(object):
    def __init__(
        self,
        model: nn.Sequential,
        criterion: nn.CrossEntropyLoss,
        device: torch.device,
        cluster_method: str = "normal",
        HD_con: int = 100,
        temperature: float = 1.0,
    ) -> None:
        self.model = model
        # Defination of w_percent here: Every layer has w_percent (i.e., 1%) weights
        # can be locked, layers share a same w_percent
        self.criterion = criterion
        self.HD_con = HD_con
        self.device = device
        self.cluster_method = cluster_method
        assert self.cluster_method in ["normal", "custom"]

        self.num_layer = 0
        self.num_weight = 0
        # temperature to control softmax
        self.T = temperature

        self.train_mode = self.model.training
        self.model.eval()

        self.max_lock_bit = 0
        self.total_bit = 0
        self.num_weight_layer_acc = []

        for name, layer in self.model.named_modules():
            if isinstance(layer, (GemmConv2d, GemmLinear)):
                lg.info(f"Layer {name}, with weight shape {layer.weight.shape}")
                layer.weight_quantizer.to_two_com()
                # Store the clean weight for each layer
                layer._clean_weight = (
                    layer.weight_quantizer.w_q_com.data.detach().clone()
                )
                self.num_layer += 1
                self.num_weight += layer.weight_quantizer.w_q_com.data.numel()
                self.N_bits = layer.weight_quantizer.N_bits
                self.num_weight_layer_acc.append(self.num_weight)
                # Calculate total bits
                self.total_bit += layer.weight_quantizer.w_q_com.numel() * self.N_bits

        lg.info(
            f"Total number of weights is {self.num_weight}, layer number is {self.num_layer}"
        )
        self.max_value = 2 ** (self.N_bits - 1)

    def layerwise_locking(self, layer, eta: float, val_loader: DataLoader):
        """
        Description: In each layer, determine the group size and cluster centers that fulfill the
        requirement of acceptable accuracy degradation.
        Return: Group size (G) in Tensor, Locked weight index () in Tensor, int
        """
        # Initialization of group size and number of cluster
        G, K = 256, 1
        self.acc_0 = validate(
            self.model, val_loader, -3, self.criterion, [], [], self.device
        )

        while G >= 1:
            # Stop criteria: after G = 1
            K = 1
            while K <= 2**self.N_bits / 8:
                # Perform K-means using current G and K
                if G == 1:
                    K_mean_solver = CustomDistanceKMeans(
                        first_grad=layer.weight._first_grad.data.view(-1)
                        .detach()
                        .cpu()
                        .numpy(),
                        sec_grad=layer.weight._second_grad.data.view(-1)
                        .detach()
                        .cpu()
                        .numpy(),
                        n_clusters=K,
                        max_iter=500,
                        tol=0.00001,
                        calculate_obj=calculate_KMeans_obj,
                    )

                    W_np = (
                        layer.weight_quantizer.w_q_com.data.view(-1)
                        .detach()
                        .cpu()
                        .numpy()
                        .reshape(-1, 1)
                    )
                    K_mean_solver.fit(X=W_np)

                    labels, centroids = (
                        K_mean_solver.labels_,
                        K_mean_solver.cluster_centers_,
                    )
                    # lg.info(f"Centers are {centroids}")

                else:
                    num_group = (layer.weight_quantizer.w_q_com.numel() / G).__ceil__()
                    centroids_group = []
                    for i in range(num_group):
                        solver = CustomDistanceKMeans(
                            first_grad=layer.weight._first_grad.data.view(-1)[
                                i::num_group
                            ]
                            .detach()
                            .cpu()
                            .numpy()
                            .reshape(-1, 1),
                            sec_grad=layer.weight._second_grad.data.view(-1)[
                                i::num_group
                            ]
                            .detach()
                            .cpu()
                            .numpy()
                            .reshape(-1, 1),
                            n_clusters=1,
                            max_iter=500,
                            tol=0.00001,
                            calculate_obj=calculate_KMeans_obj,
                        )
                        weight_cluster = (
                            layer.weight_quantizer.w_q_com.data.view(-1)[i::num_group]
                            .detach()
                            .cpu()
                            .numpy()
                            .reshape(-1, 1)
                        )

                        solver.fit(X=weight_cluster)
                        _, centroids = solver.labels_, solver.cluster_centers_

                        centroids_group.append(centroids.item())

                    centroids_group = np.array(centroids_group).reshape(-1, 1)

                    solver_group = KMeans(
                        n_clusters=K if K <= num_group else num_group,
                        init="k-means++",
                        max_iter=500,
                        tol=0.00001,
                        n_init="auto",
                    )
                    solver_group.fit(X=centroids_group)

                    labels, centroids = (
                        solver_group.labels_,
                        solver_group.cluster_centers_,
                    )

                L_K = torch.tensor(labels)
                W_K = torch.tensor(centroids.round())
                # lg.info(f"Centers are {W_K}")

                acc_lock_list = []
                for i in range(5):
                    set_torch_deterministic(42 + i * 1000)

                    layer.perform_lock(L_K=L_K, W_K=W_K, G_size=G, HD_con=self.HD_con)
                    acc_lock = validate(
                        self.model, val_loader, -3, self.criterion, [], [], self.device
                    )
                    acc_lock_list.append(acc_lock)
                    model_reset_weight(self.model)
                acc_drop = self.acc_0 - np.mean(acc_lock_list).item()
                lg.info(
                    f"For G={G} and K={K}, Accuracy drop is {acc_drop}, Number of cluster is {K}"
                )

                model_reset_weight(self.model)

                if acc_drop < eta:
                    return L_K, W_K, G
                else:
                    # Continue on next iteration
                    K *= 2
            G /= 2

        # Cannot lock, turn to normal locking
        lg.info("Cannot lock this layer using customized locking")
        G, K = 256, 1
        while G >= 1:
            K = 1
            while K <= 2**self.N_bits / 8:
                # Perform K-means using current G and K
                if G == 1:
                    K_mean_solver = KMeans(
                        n_clusters=K,
                        init="k-means++",
                        max_iter=500,
                        tol=0.0001,
                        n_init="auto",
                    )

                    W_np = (
                        layer.weight_quantizer.w_q_com.data.view(-1)
                        .detach()
                        .cpu()
                        .numpy()
                        .reshape(-1, 1)
                    )

                    K_mean_solver.fit(X=W_np)

                    labels, centroids = (
                        K_mean_solver.labels_,
                        K_mean_solver.cluster_centers_,
                    )
                    # lg.info(f"Centers are {centroids}")

                else:
                    num_group = (layer.weight_quantizer.w_q_com.numel() / G).__ceil__()

                    centroids_group = []
                    for i in range(num_group):
                        solver = KMeans(
                            n_clusters=1,
                            init="k-means++",
                            max_iter=500,
                            tol=0.00001,
                            n_init="auto",
                        )

                        weight_cluster = (
                            layer.weight_quantizer.w_q_com.data.view(-1)[i::num_group]
                            .detach()
                            .cpu()
                            .numpy()
                            .reshape(-1, 1)
                        )

                        solver.fit(X=weight_cluster)
                        _, centroids = solver.labels_, solver.cluster_centers_

                        centroids_group.append(centroids.item())

                    centroids_group = np.array(centroids_group).reshape(-1, 1)

                    solver_group = KMeans(
                        n_clusters=K if K <= num_group else num_group,
                        init="k-means++",
                        max_iter=500,
                        tol=0.00001,
                        n_init="auto",
                    )
                    solver_group.fit(X=centroids_group)

                    labels, centroids = (
                        solver_group.labels_,
                        solver_group.cluster_centers_,
                    )

                L_K = torch.tensor(labels)
                W_K = torch.tensor(centroids.round())
                # lg.info(f"Centers are {W_K}")

                acc_lock_list = []
                for i in range(5):
                    set_torch_deterministic(42 + i * 1000)

                    layer.perform_lock(L_K=L_K, W_K=W_K, G_size=G, HD_con=self.HD_con)
                    acc_lock = validate(
                        self.model, val_loader, -3, self.criterion, [], [], self.device
                    )
                    acc_lock_list.append(acc_lock)
                    model_reset_weight(self.model)

                acc_drop = self.acc_0 - np.mean(acc_lock_list).item()
                lg.info(
                    f"For G={G} and K={K}, Accuracy drop is {acc_drop}, Number of cluster is {K}"
                )

                model_reset_weight(self.model)

                if acc_drop < eta:
                    return L_K, W_K, G
                else:
                    # Continue on next iteration
                    K *= 2
            G /= 2

        # Still cannot lock
        lg.info(
            "Cannot lock by Normal or Customized Locking"
        )  # Cannot perform Weight Locking on this layer
        return torch.empty(0), torch.empty(0), 0

    def smart_locking(self, eta: float, val_loader: DataLoader):
        """
        Description: Return G, LK, WK of every layer
        """
        calculate_grad_hessian(
            model=self.model,
            train_loader=val_loader,
            criterion=self.criterion,
            mode="defender",
            num_samples=1,
            device=self.device,
        )

        L_K_res, W_K_res, G_res = {}, {}, {}

        for name, layer in self.model.named_modules():
            if isinstance(layer, (GemmConv2d, GemmLinear)):
                # lg.info(f"Performing Locking in Layer: {name}")
                L_K, W_K, G = self.layerwise_locking(
                    layer=layer, eta=eta, val_loader=val_loader
                )  # TENSOR, TENSOR, INT

                L_K_res[name] = L_K_res.get(name, 0) + L_K
                W_K_res[name] = W_K_res.get(name, 0) + W_K
                G_res[name] = G_res.get(name, 0) + G

        self.L_K = L_K_res
        self.W_K = W_K_res
        self.G = G_res

        return L_K_res, W_K_res, G_res

    def calculate_mem_ov(self):
        """
        Description: Calculate memory overhead required by Weight Locking
        """
        self.bit_consumption = {}
        total_consumption = 0

        for name, layer in self.model.named_modules():
            if isinstance(layer, (GemmConv2d, GemmLinear)):
                num_group = (
                    layer.weight_quantizer.w_q_com.numel() / self.G[name]
                ).__ceil__()

                group_bit_consumption = (
                    math.log2(self.W_K[name].numel()).__ceil__() + 1
                    if self.G[name] == 1
                    else math.log2(self.W_K[name].numel()).__ceil__() + 2
                )  # log_2 N

                total_consumption += num_group * group_bit_consumption

        lg.info(f"Memory overhead is {total_consumption / self.total_bit}")

        return total_consumption / self.total_bit
