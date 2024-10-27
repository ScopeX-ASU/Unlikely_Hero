import operator

import numpy as np
import torch
from pyutils.general import logger as lg
from torch import Tensor, nn
from torch.types import Tuple
from torch.utils.data import DataLoader

from core.models.layers.gemm_conv2d import GemmConv2d
from core.models.layers.gemm_linear import GemmLinear


class grad_attacker(object):
    def __init__(
        self,
        model: nn.Sequential,
        criterion: nn.CrossEntropyLoss,
        N_sample: int,
        inf_ov: int,
        HD_con: int,
        protected_index: dict = {},
        random_int: int = 1,
        device: torch.device = torch.device("cuda"),
    ) -> None:
        # Initialization
        self.model = model
        self.criterion = criterion
        self.N_sample = N_sample
        # Inference overhead and Hamming Distance
        self.inf_ov = inf_ov
        self.HD_con = HD_con

        # Protected index and locked index, two dict with layer name and tensor
        self.protected_index = protected_index
        self.device = device

        self.training_mode = self.model.training
        self.loss_dict = {}

        self.n_bits2flip = 0
        self.bit_counter = 0
        self.num_layer = 0

        # Record the number and shape of weight in each layer
        self.num_weight = 0
        self.num_weight_layer_acc = []
        self.num_weight_layer = []
        self.weight_shape = []
        self.grad_layer = []
        self.grad_global = []
        # =============== For global grad sorting ==============

        self.grad_mask = []
        self.grad_sign = []
        # ========== For filter out the unneccessary flip ======

        self.model.eval()

        for module in self.model.modules():
            if isinstance(module, (GemmConv2d, GemmLinear)):
                self.num_layer += 1
                self.num_weight += module.weight.data.numel()

                self.num_weight_layer.append(module.weight.data.numel())
                self.num_weight_layer_acc.append(self.num_weight)
                self.weight_shape.append(module.weight.data.shape)

                module.weight_quantizer.to_two_com()
                self.N_bits = module.weight_quantizer.N_bits

        self.iteration = 0
        # hvls and LSB
        self.max_value = 1 << (self.N_bits - 1)
        self.LSB_value = 1 << 0
        self.random_int = random_int

    def data_preparation(
        self,
        dataloader: DataLoader,
    ) -> Tuple[Tensor, Tensor]:
        """
        Description: Generate a minibatch dataset according to self.random_int
        """
        # random.seed(self.random_seed)

        for i, (data, target) in enumerate(dataloader):
            if i == self.random_int:
                dataset = data.to(self.device)
                labels = target.to(self.device)
                break
        return dataset, labels  # two tensors

    def get_gradient(self, dataset: Tensor, labels: Tensor):
        """
        Description:
        Perform forward and backward to calculate first-order gradient.
        """
        grad_weight_stat, loss_list = {}, []
        loss = 0
        for _ in range(self.N_sample):
            for i in range(len(dataset)):
                output = self.model(dataset)
                loss = self.criterion(output, labels)

                loss.backward()
                loss_list.append(loss.detach().item())

            for name, module in self.model.named_modules():
                if isinstance(module, (GemmConv2d, GemmLinear)):
                    grad_weight = module.weight.grad.data.clone()
                    grad_weight_stat[name] = grad_weight_stat.get(name, 0) + grad_weight

        for name, module in self.model.named_modules():
            if isinstance(module, (GemmConv2d, GemmLinear)):
                grad_weight_stat[name] = grad_weight_stat[name] / self.N_sample

        return grad_weight_stat, np.mean(loss_list)  # dict and float

    def flip_bits_bfa(
        self, module: tuple[GemmConv2d, GemmConv2d], grad_module: Tensor, prot_idx: list
    ):
        """
        Description: Flip the MSB of weight with highest sensitivity in a layer leading to increase of Loss\\
        Use first-order gradient to select sensitivity
        """
        # Choose the weights with largest gradients
        self.topk = self.HD_con * 4
        w_grad_topk, w_idx_topk = (
            grad_module.data.detach().abs().view(-1).topk(self.topk)
        )

        w_grad_topk = grad_module.data.detach().view(-1)[w_idx_topk]
        w_grad_topk_sign = (w_grad_topk.sign() + 1) * 0.5

        w_bin_topk = module.weight_quantizer.w_q_com.data.view(-1)[w_idx_topk]
        # Filter out those bits cannot be fliped

        attack_weight = w_bin_topk.clone()
        n_bits_fliped = 0
        w_idx_topk = w_idx_topk.detach().cpu().numpy().tolist()
        # Change to set, acclerate searching
        prot_idx = set(prot_idx)

        for i in range(self.topk):
            # Case.1: Weights being attacked is protected, thus LSB will be flipped
            if w_idx_topk[i] in prot_idx:  # ! set
                if w_grad_topk_sign[i] == 1 and attack_weight[i] <= self.max_value:
                    # lg.info(f"Weight before attack is {attack_weight[i]}")
                    attack_weight[i] = attack_weight[i].round().int() ^ self.LSB_value
                    # lg.info(f"Weight after attack is {attack_weight[i]}")
                    # lg.info("...Protection Hits!...")
                    n_bits_fliped += 1
                elif w_grad_topk_sign[i] == 0 and attack_weight[i] >= self.max_value:
                    # lg.info(f"Weight before attack is {attack_weight[i]}")
                    attack_weight[i] = attack_weight[i].round().int() ^ self.LSB_value
                    # lg.info(f"Weight after attack is {attack_weight[i]}")
                    # lg.info("...Protection Hits!...")
                    n_bits_fliped += 1

            # Case.2: Weights being attacked is non-protected, thus MSB will be flipped
            else:
                if w_grad_topk_sign[i] == 1 and attack_weight[i] <= self.max_value:
                    attack_weight[i] = attack_weight[i].round().int() ^ self.max_value
                    n_bits_fliped += 1
                elif w_grad_topk_sign[i] == 0 and attack_weight[i] >= self.max_value:
                    attack_weight[i] = attack_weight[i].round().int() ^ self.max_value
                    n_bits_fliped += 1

            if n_bits_fliped == self.n_bits2flip:
                break

        # lg.info(f"Check original weights {module.weight_quantizer.w_q_com.data.view(-1)[w_idx_topk]}")
        module.weight_quantizer.w_q_com.data.view(-1)[w_idx_topk] = (
            attack_weight.float()
        )
        # lg.info(f"Check whether attacked weights are changed {module.weight_quantizer.w_q_com.data.view(-1)[w_idx_topk]}")

        return module.weight_quantizer.w_q_com.data

    def progressive_bit_search_bfa(self, dataloader: DataLoader) -> None:
        """
        Description: Progressive bit seearch, reference to https://github.com/elliothe/BFA.git
        """
        # lg.info("Attacking mode == Gradient-based attacker......")
        self.iteration += 1

        # Get the grad data by forward a dataset and backward
        dataset, target = self.data_preparation(dataloader)

        self.model.zero_grad()
        self.grad_module_stat, self.loss = self.get_gradient(
            dataset=dataset, labels=target
        )
        self.loss_max = self.loss

        with torch.no_grad():
            while self.loss_max <= self.loss:
                self.n_bits2flip += 1
                for name, module in self.model.named_modules():
                    if isinstance(module, (GemmConv2d, GemmLinear)):
                        # lg.info(f"Attacking module:{name}")
                        clean_weight = module.weight_quantizer.w_q_com.data.clone()
                        self.flip_bits_bfa(
                            module=module,
                            grad_module=self.grad_module_stat[name],
                            prot_idx=[]
                            if self.protected_index == {}
                            or self.protected_index[name] == []
                            else self.protected_index[name].tolist(),
                        )
                        # change the weight to attacked weight and calculate loss
                        module.weight_quantizer.from_two_com()

                        output = self.model(dataset)
                        self.loss_dict[name] = (
                            self.loss_dict.get(name, 0)
                            + self.criterion(output, target).item()
                        )

                        # change back to clean weights
                        module.weight_quantizer.w_q_com.data = clean_weight
                        module.weight_quantizer.from_two_com()
                # lg.info(self.loss_dict)

                # Get the layer with largest loss
                max_loss_module = max(
                    self.loss_dict.items(), key=operator.itemgetter(1)
                )[0]
                self.loss_max = self.loss_dict[max_loss_module]

            # if the loss_max does lead to the performance degradation compared to the self.loss,
            # then change that layer's weight without putting back the clean weight
            for _, (name, module) in enumerate(self.model.named_modules()):
                if name == max_loss_module and isinstance(
                    module, (GemmConv2d, GemmLinear)
                ):
                    attack_weight = self.flip_bits_bfa(
                        module=module,
                        grad_module=self.grad_module_stat[name],
                        # Change back to self.protected_index[name].tolist()
                        prot_idx=[]
                        if self.protected_index == {}
                        or self.protected_index[name] == []
                        else self.protected_index[name].tolist(),
                    )

                    lg.info(f"attack {name} in this iteration")
                    module.weight_quantizer.w_q_com.data = attack_weight
                    module.weight_quantizer.from_two_com()

            self.bit_counter += self.n_bits2flip

            # lg.info(f"Actual {self.n_bits2flip} to be attacked in {max_loss_module} in iteration {self.iteration}")
            # Calculate remaining inference overhead to conduct next attack
            self.inf_ov -= (
                3 + self.num_layer * ((1 + self.n_bits2flip) * self.n_bits2flip / 2)
            ) * self.N_sample
            self.n_bits2flip = 0

    def progressive_bit_search_select(self, grad_module_stat: dict):
        """
        Description: After the inference budget exhausted, select a group of sensitive weights in one-time
        """
        with torch.no_grad():
            self.grad_layer_masked = []
            self.weight_layer = []
            for name, layer in self.model.named_modules():
                if isinstance(layer, (GemmConv2d, GemmLinear)):
                    # apply a mask to filter out those unneccessary bit-flip
                    grad_sign = (
                        (grad_module_stat[name].data.view(-1).sign() + 1) / 2
                    ).int()
                    grad_mask = (
                        (
                            (
                                layer.weight_quantizer.w_q_com.data.view(-1)
                                - 2 ** (self.N_bits - 1)
                            ).sign()
                            + 1
                        )
                        / 2
                    ).int()

                    grad_mask = grad_mask ^ (~grad_sign)

                    self.grad_layer_masked.append(
                        grad_module_stat[name].data.view(-1) * grad_mask.float()
                    )
                    self.weight_layer.append(
                        layer.weight_quantizer.w_q_com.data.view(-1)
                    )

            self.grad_global = torch.cat(
                [grad_layer for grad_layer in self.grad_layer_masked]
            )
            self.weight_global = torch.cat(
                [weight_layer for weight_layer in self.weight_layer]
            )

            if self.update_grad == True:
                _, grad_topk_global = self.grad_global.topk(
                    self.HD_con - self.bit_counter
                )
            else:
                _, grad_topk_global = self.grad_global.topk(
                    self.HD_con - self.bit_counter + 1
                )
                grad_topk_global = grad_topk_global[1:]

            self.protected_index_global = []
            self.locked_index_global = []
            self.layer_pointer = 0

            for name, layer in self.model.named_modules():
                if isinstance(layer, (GemmConv2d, GemmLinear)):
                    if self.layer_pointer == 0:
                        idx_ptct_global = (
                            []
                            if self.protected_index == {}
                            or self.protected_index[name] == []
                            else self.protected_index[name].tolist()
                        )
                    else:
                        idx_ptct_global = (
                            []
                            if self.protected_index == {}
                            or self.protected_index[name] == []
                            else [
                                idx + self.num_weight_layer_acc[self.layer_pointer - 1]
                                for idx in self.protected_index[name].tolist()
                            ]
                        )

                    self.protected_index_global.extend(idx_ptct_global)
                    idx_ptct_global.clear()
                    self.layer_pointer += 1

            self.protected_index_global = set(self.protected_index_global)
            # lg.info(self.protected_index_global)

            grad_topk_global_list = grad_topk_global.detach().cpu().numpy().tolist()
            # lg.info(grad_topk_global_list)
            for i in range(len(grad_topk_global)):
                if grad_topk_global_list[i] in self.protected_index_global:
                    # If the weight being attacked is being protected, then only LSB will be flipped
                    self.weight_global[grad_topk_global[i]] = (
                        self.weight_global[grad_topk_global[i]].round().int()
                        ^ self.LSB_value
                    ).float()
                else:
                    # Attack MSB, ordinary attack
                    self.weight_global[grad_topk_global[i]] = (
                        self.weight_global[grad_topk_global[i]].round().int()
                        ^ self.max_value
                    ).float()

            # Split the global weights to layers and give them back to original layer weights
            self.layer_pointer = 0
            for _, layer in self.model.named_modules():
                # Split the weights into original layers, shapes
                if isinstance(layer, (GemmConv2d, GemmLinear)):
                    layer.weight_quantizer.w_q_com.data = (
                        self.weight_global[
                            0 : self.num_weight_layer_acc[self.layer_pointer]
                        ].reshape(shape=self.weight_shape[self.layer_pointer])
                        if self.layer_pointer == 0
                        else self.weight_global[
                            self.num_weight_layer_acc[
                                self.layer_pointer - 1
                            ] : self.num_weight_layer_acc[self.layer_pointer]
                        ].reshape(shape=self.weight_shape[self.layer_pointer])
                    )

                    self.layer_pointer += 1
                    layer.weight_quantizer.from_two_com()

    def pbs_top(self, attacker_loader: DataLoader):
        while (
            self.bit_counter < self.HD_con
            and self.inf_ov >= (3 + self.num_layer) * self.N_sample
        ):
            self.progressive_bit_search_bfa(dataloader=attacker_loader)

        # lg.info(f" Remaining budget: HD= {self.HD_con - self.bit_counter}")
        # lg.info(f" Remaining budget: Inf= {self.inf_ov}")
        # If there is HD left, use the gradient information to perform global sorting and select
        # those most sensitive weights to perform bit-flip attack
        self.update_grad = True

        if self.inf_ov >= 3 * self.N_sample:
            # If inference overhead is still greater than 3, we can calculate new gradient and
            dataset, target = self.data_preparation(dataloader=attacker_loader)
            self.model.zero_grad()
            self.grad_module_stat, self.loss = self.get_gradient(
                dataset=dataset, labels=target
            )

        else:
            # Use the old gradient to select weights (index), but BFA was conducted before, so we flip the weights from [1:]
            self.update_grad = False

        self.progressive_bit_search_select(grad_module_stat=self.grad_module_stat)
        # lg.info(f"Actually attacked {self.bit_counter} bits")
        # lg.info(f"Actually left {self.inf_ov} inference cycles not used")
        # lg.info(f"Actually experienced {self.iteration} iterations")


class grad_attacker_LSB(object):
    def __init__(
        self,
        model: nn.Sequential,
        criterion: nn.CrossEntropyLoss,
        N_sample: int,
        inf_ov: int,
        HD_con: int,
        protected_index: dict = {},
        random_int: int = 1,
        device: torch.device = torch.device("cuda"),
    ) -> None:
        # Initialization
        self.model = model
        self.criterion = criterion
        self.N_sample = N_sample
        # Inference overhead and Hamming Distance
        self.inf_ov = inf_ov
        self.HD_con = HD_con

        # Protected index and locked index, two dict with layer name and tensor
        self.protected_index = protected_index

        self.device = device

        self.training_mode = self.model.training
        self.loss_dict = {}

        self.n_bits2flip = 0
        self.bit_counter = 0
        self.num_layer = 0

        # Record the number and shape of weight in each layer
        self.num_weight = 0
        self.num_weight_layer_acc = []
        self.num_weight_layer = []
        self.weight_shape = []
        self.grad_layer = []
        self.grad_global = []

        self.grad_mask = []
        self.grad_sign = []

        self.model.eval()

        for module in self.model.modules():
            if isinstance(module, (GemmConv2d, GemmLinear)):
                self.num_layer += 1
                self.num_weight += module.weight.data.numel()

                self.num_weight_layer.append(module.weight.data.numel())
                self.num_weight_layer_acc.append(self.num_weight)
                self.weight_shape.append(module.weight.data.shape)

                module.weight_quantizer.to_two_com()
                self.N_bits = module.weight_quantizer.N_bits

        self.iteration = 0
        # hvls and LSB
        self.max_value = 1 << (self.N_bits - 1)
        self.LSB_value = 1 << 0
        self.random_int = random_int

    def data_preparation(
        self,
        dataloader: DataLoader,
    ) -> Tuple[Tensor, Tensor]:
        """
        Description: Generate a minibatch dataset according to self.random_int
        """
        for i, (data, target) in enumerate(dataloader):
            if i == self.random_int:
                dataset = data.to(self.device)
                labels = target.to(self.device)
                break
        return dataset, labels  # two tensors

    def get_gradient(self, dataset: Tensor, labels: Tensor):
        """
        Description: Perform forward and backward to calculate first-order gradient.
        """
        grad_weight_stat, loss_list = {}, []
        loss = 0
        for _ in range(self.N_sample):
            for i in range(len(dataset)):
                output = self.model(dataset)
                loss = self.criterion(output, labels)

                loss.backward()
                loss_list.append(loss.detach().item())

            for name, module in self.model.named_modules():
                if isinstance(module, (GemmConv2d, GemmLinear)):
                    grad_weight = module.weight.grad.data.clone()
                    grad_weight_stat[name] = grad_weight_stat.get(name, 0) + grad_weight

        for name, module in self.model.named_modules():
            if isinstance(module, (GemmConv2d, GemmLinear)):
                grad_weight_stat[name] = grad_weight_stat[name] / self.N_sample

        return grad_weight_stat, np.mean(loss_list)  # dict and float

    def flip_bits_bfa(
        self,
        module: tuple[GemmConv2d, GemmConv2d],
        grad_module: Tensor,
        prot_idx: list,
    ):
        """
        Description: Flip the MSB of weight with highest sensitivity in a layer\\
        Use first-order gradient to select sensitivity
        """
        # Choose the weights with largest gradients
        self.topk = self.HD_con * 4
        w_grad_topk, w_idx_topk = (
            grad_module.data.detach().abs().view(-1).topk(self.topk)
        )
        # lg.info(f"Bits to flip in this iteration ({self.iteration}) is {self.n_bits2flip}")

        w_grad_topk = grad_module.data.detach().view(-1)[w_idx_topk]
        w_grad_topk_sign = (w_grad_topk.sign() + 1) * 0.5

        w_bin_topk = module.weight_quantizer.w_q_com.data.view(-1)[w_idx_topk]
        # Filter out those bits cannot be fliped

        attack_weight = w_bin_topk.clone()
        n_bits_fliped = 0
        w_idx_topk = w_idx_topk.detach().cpu().numpy().tolist()
        # Change to set, acclerate searching
        prot_idx = set(prot_idx)

        for i in range(self.topk):
            if w_idx_topk[i] in prot_idx:  # change to set
                if w_grad_topk_sign[i] == 1 and attack_weight[i] <= self.max_value:
                    # lg.info(f"Weight before attack is {attack_weight[i]}")
                    attack_weight[i] = attack_weight[i].round().int() ^ self.LSB_value
                    # lg.info(f"Weight after attack is {attack_weight[i]}")
                    # lg.info("...Protection Hits!...")
                    n_bits_fliped += 1
                elif w_grad_topk_sign[i] == 0 and attack_weight[i] >= self.max_value:
                    # lg.info(f"Weight before attack is {attack_weight[i]}")
                    attack_weight[i] = attack_weight[i].round().int() ^ self.LSB_value
                    # lg.info(f"Weight after attack is {attack_weight[i]}")
                    # lg.info("...Protection Hits!...")
                    n_bits_fliped += 1

            # Weights being attacked is non-protected, thus MSB will be flipped
            else:
                if w_grad_topk_sign[i] == 1 and attack_weight[i] <= self.max_value:
                    attack_weight[i] = attack_weight[i].round().int() ^ self.LSB_value
                    n_bits_fliped += 1
                elif w_grad_topk_sign[i] == 0 and attack_weight[i] >= self.max_value:
                    attack_weight[i] = attack_weight[i].round().int() ^ self.LSB_value
                    n_bits_fliped += 1

            if n_bits_fliped == self.n_bits2flip:
                break

        # lg.info(f"Check original weights {module.weight_quantizer.w_q_com.data.view(-1)[w_idx_topk]}")
        module.weight_quantizer.w_q_com.data.view(-1)[w_idx_topk] = (
            attack_weight.float()
        )
        # lg.info(f"Check whether attacked weights are changed {module.weight_quantizer.w_q_com.data.view(-1)[w_idx_topk]}")

        return module.weight_quantizer.w_q_com.data

    def progressive_bit_search_bfa(self, dataloader: DataLoader) -> None:
        """
        Description: PBFA, reference to https://github.com/elliothe/BFA.git
        """
        # lg.info("Attacking mode == Gradient-based attacker......")
        self.iteration += 1

        # Get the grad data by forward a dataset and backward
        dataset, target = self.data_preparation(dataloader)

        self.model.zero_grad()
        self.grad_module_stat, self.loss = self.get_gradient(
            dataset=dataset, labels=target
        )
        self.loss_max = self.loss

        with torch.no_grad():
            while self.loss_max <= self.loss:
                self.n_bits2flip += 1
                for name, module in self.model.named_modules():
                    if isinstance(module, (GemmConv2d, GemmLinear)):
                        # lg.info(f"Attacking module:{name}")
                        clean_weight = module.weight_quantizer.w_q_com.data.clone()
                        self.flip_bits_bfa(
                            module=module,
                            grad_module=self.grad_module_stat[name],
                            # Change back to self.protected_index[name].tolist()
                            prot_idx=[]
                            if self.protected_index == {}
                            or self.protected_index[name] == []
                            else self.protected_index[name].tolist(),
                        )
                        # change the weight to attacked weight and calculate loss
                        module.weight_quantizer.from_two_com()

                        output = self.model(dataset)
                        self.loss_dict[name] = (
                            self.loss_dict.get(name, 0)
                            + self.criterion(output, target).item()
                        )

                        # change back to clean weights
                        module.weight_quantizer.w_q_com.data = clean_weight
                        module.weight_quantizer.from_two_com()

                # Get the layer with largest loss
                max_loss_module = max(
                    self.loss_dict.items(), key=operator.itemgetter(1)
                )[0]
                self.loss_max = self.loss_dict[max_loss_module]

            # if the loss_max does lead to the degradation compared to the self.loss,
            # then change that layer's weight without putting back the clean weight
            for _, (name, module) in enumerate(self.model.named_modules()):
                if name == max_loss_module and isinstance(
                    module, (GemmConv2d, GemmLinear)
                ):
                    attack_weight = self.flip_bits_bfa(
                        module=module,
                        grad_module=self.grad_module_stat[name],
                        # Change back to self.protected_index[name].tolist()
                        prot_idx=[]
                        if self.protected_index == {}
                        or self.protected_index[name] == []
                        else self.protected_index[name].tolist(),
                    )

                    lg.info(f"attack {name} in this iteration")
                    module.weight_quantizer.w_q_com.data = attack_weight
                    module.weight_quantizer.from_two_com()

            self.bit_counter += self.n_bits2flip

            # lg.info(f"Actual {self.n_bits2flip} to be attacked in {max_loss_module} in iteration {self.iteration}")
            # Calculate remaining inference overhead to conduct next attack
            self.inf_ov -= (
                3 + self.num_layer * ((1 + self.n_bits2flip) * self.n_bits2flip / 2)
            ) * self.N_sample
            self.n_bits2flip = 0

    def pbs_top(self, attacker_loader: DataLoader):
        while (
            self.bit_counter < self.HD_con
            and self.inf_ov >= (3 + self.num_layer) * self.N_sample
        ):
            self.progressive_bit_search_bfa(dataloader=attacker_loader)

        # lg.info(f" Remaining budget: HD= {self.HD_con - self.bit_counter}")
        # lg.info(f" Remaining budget: Inf= {self.inf_ov}")
