import torch
from pyutils.general import logger as lg
from torch import Tensor, nn
from torch.utils.data import DataLoader, WeightedRandomSampler

from core.models.layers.gemm_conv2d import GemmConv2d
from core.models.layers.gemm_linear import GemmLinear
from core.models.layers.utils import calculate_grad_hessian
from train_pretrain import validate


class unary_defender(object):
    def __init__(
        self,
        model: nn.Sequential,
        mem_ov: float,
        w_percent: float,
        rt_ov: int,
        HD_con: int,
        criterion: nn.CrossEntropyLoss,
        device: torch.device,
        temperature: float = 1.0,
    ) -> None:
        self.model = model
        self.criterion = criterion
        # Mem overhead
        self.mem_ov = mem_ov
        self.rt_ov = rt_ov
        self.HD_con = HD_con
        # percentage of weights being protected
        self.w_percent = w_percent

        # For softmax in choosing index to defend
        self.T = temperature

        self.num_layer = 0
        self.num_weight = 0
        self.device = device

        self.layer_num_weight = {}
        self.defend_ws_layer = {}
        # define a dict to record number of weights being protected, by percantage of weights
        self.defend_w_per_layer = {}

        self.train_mode = self.model.training
        self.model.eval()

        for name, module in self.model.named_modules():
            if isinstance(module, (GemmConv2d, GemmLinear)):
                self.num_layer += 1
                self.num_weight += module.weight.numel()
                self.N_bits = module.weight_quantizer.N_bits
                lg.info(f"Layer: {name}, with weights {module.weight.shape}")

        lg.info(f"Total weights: {self.num_weight}")

        # The budget is scheduled across layers
        self.total_bits = self.num_weight * self.N_bits

        self.mem_budget = (self.mem_ov * self.total_bits).__floor__()

        for name, module in self.model.named_modules():
            if isinstance(module, (GemmConv2d, GemmLinear)):
                self.defend_ws_layer[name] = (
                    self.defend_ws_layer.get(name, 0)
                    + (self.mem_budget / (2**self.N_bits) / self.num_layer).__floor__()
                )
                self.layer_num_weight[name] = (
                    self.layer_num_weight.get(name, 0) + module.weight.data.numel()
                )

        self.redundant_bits = (
            self.mem_budget
            - self.num_layer
            * (2**self.N_bits)
            * (self.mem_budget / (2**self.N_bits) / self.num_layer).__floor__()
        )
        # lg.info(f'remaining bits = {self.redundant_bits}')
        for name, module in self.model.named_modules():
            if isinstance(module, (GemmConv2d, GemmLinear)):
                if self.redundant_bits >= (2**self.N_bits):
                    self.defend_ws_layer[name] += 1
                    self.redundant_bits -= 2**self.N_bits

        self.Wper_budget = (self.num_weight * self.w_percent).__floor__()
        # define self.defend_w_per_layer accroding to w_percent
        self.num_module = self.num_layer

        self.max_value = 2 ** (self.N_bits - 1)

    def distribute_budget(self, method: str = "even", salience: str = "second-order"):
        """
        Description: Calculate the number of weights that should be protected for each layer
        """
        assert method in ["even", "importance"], f"{method} not supported"

        if method == "even":
            numel_list, name_list = [], []
            num_rank = []
            self.layer_num_weight = {}
            for name, layer in self.model.named_modules():
                if isinstance(layer, (GemmConv2d, GemmLinear)):
                    numel_list.append(layer.weight.data.numel())
                    name_list.append(name)
                    self.layer_num_weight[name] = (
                        self.layer_num_weight.get(name, 0) + layer.weight.data.numel()
                    )

            num_idx = sorted(
                range(len(numel_list)), key=lambda x: numel_list[x], reverse=False
            )

            for i in range(len(num_idx)):
                num_rank.append(name_list[num_idx[i]])

            lg.info(f"Ranked layers are {num_rank}")
            for layer_name in num_rank:
                w_intend = (self.Wper_budget / self.num_module).__floor__()
                if w_intend <= self.layer_num_weight[layer_name]:
                    self.defend_w_per_layer[layer_name] = self.defend_w_per_layer.get(
                        layer_name, 0
                    ) + (w_intend)
                else:
                    self.defend_w_per_layer[layer_name] = (
                        self.defend_w_per_layer.get(layer_name, 0)
                        + (self.layer_num_weight[layer_name])
                    )
                self.Wper_budget -= self.defend_w_per_layer[layer_name]
                self.num_module -= 1

        elif method == "importance":
            sen_rank, name_list = [], []
            sensitivity_mean = []
            for name, layer in self.model.named_modules():
                if isinstance(layer, (GemmConv2d, GemmLinear)):
                    if salience == "second-order":
                        layer_grad_abs = layer.weight._second_grad.data.abs()
                    elif salience == "first-order":
                        layer_grad_abs = layer.weight._first_grad.data.abs()
                    elif salience == "taylor-series":
                        # If we found that taylor series return a negative value, no need to protect?
                        layer_grad_abs = layer.weight._taylor_series.data
                    else:
                        not NotImplementedError

                    quartile_10 = torch.quantile(layer_grad_abs, 0.1)
                    quartile_5 = torch.quantile(layer_grad_abs, 0.05)
                    quartile_2 = torch.quantile(layer_grad_abs, 0.02)
                    quartile_1 = torch.quantile(layer_grad_abs, 0.01)

                    sensitivity_mean.append(
                        quartile_10 + quartile_5 + quartile_2 + quartile_1
                    )
                    name_list.append(name)

            # sort the sensitivity according to list
            sensitivity_idx = sorted(
                range(len(sensitivity_mean)),
                key=lambda x: sensitivity_mean[x],
                reverse=True,
            )
            # lg.info(f"sorted sensitivity index is {sensitivity_idx}")

            for i in range(len(sensitivity_idx)):
                sen_rank.append(name_list[sensitivity_idx[i]])
            lg.info(f"Layer rank is {sen_rank}")

            for layer_name in sen_rank:
                if self.Wper_budget >= self.layer_num_weight[layer_name]:
                    self.defend_w_per_layer[layer_name] = self.layer_num_weight[
                        layer_name
                    ]
                    self.Wper_budget -= self.layer_num_weight[layer_name]
                else:
                    self.defend_w_per_layer[layer_name] = self.Wper_budget
                    self.Wper_budget = 0
        else:
            raise NotImplementedError()

    def in_layer_search(
        self,
        module: tuple[GemmConv2d, GemmLinear],
        candidate_sensitivity: Tensor,
        candidate_idx: Tensor,
    ):
        probe_sensitivity = torch.nn.functional.softmax(
            candidate_sensitivity / 0.5, dim=0
        )
        probe_sampler = WeightedRandomSampler(
            weights=probe_sensitivity, num_samples=self.HD_con, replacement=False
        )
        probe_idx = candidate_idx[torch.tensor([idx for idx in probe_sampler])]
        module.weight_quantizer.w_q_com.data.view(-1)[probe_idx] = self.perturb(
            module.weight_quantizer.w_q_com.data.view(-1)[probe_idx], self.N_bits
        )
        # return the HD_group that can cause largest accuracy drop to represent the protection effect of this group

    def perturb(self, x: Tensor, N: int):
        return (x.int() ^ (1 << (N - 1))).float()

    def layerwise_protection(
        self, val_loader: DataLoader, small_loader: DataLoader, salience: str
    ):
        assert salience in ["first-order", "second-order", "taylor-series"]
        ptct_idx = {}
        acc_list, acc_list_group = [], []
        dfd_idx_list = []
        for name, module in self.model.named_modules():
            if isinstance(module, (GemmConv2d, GemmLinear)):
                lg.info(f"Defending module: {name}......")

                if salience == "first-order":
                    sensitivity = module.weight._first_grad.data.abs().view(-1)
                    norm_sensitivity = torch.nn.functional.softmax(
                        input=sensitivity / self.T, dim=0
                    )
                elif salience == "second-order":
                    sensitivity = module.weight._second_grad.data.abs().view(-1)
                    norm_sensitivity = torch.nn.functional.softmax(
                        input=sensitivity / self.T, dim=0
                    )
                elif salience == "taylor-series":
                    sensitivity = module.weight._taylor_series.data.view(-1)
                    norm_sensitivity = torch.zeros_like(sensitivity)
                    sensitivity_masked = sensitivity[sensitivity > 0]
                    norm_sensitivity_masked = torch.nn.functional.softmax(
                        input=sensitivity_masked / self.T, dim=0
                    )
                    norm_sensitivity[sensitivity > 0] = norm_sensitivity_masked
                else:
                    raise NotImplementedError

                if (
                    module.weight.data.numel()
                    <= self.defend_w_per_layer[name] + self.HD_con
                    and self.defend_w_per_layer[name] != 0
                ):
                    sampler = WeightedRandomSampler(
                        weights=norm_sensitivity,
                        num_samples=self.defend_w_per_layer[name],
                        replacement=False,
                    )
                    grad_idx_topk = torch.tensor([idx for idx in sampler])
                    ptct_idx[name] = ptct_idx.get(name, 0) + grad_idx_topk

                elif self.defend_w_per_layer[name] == 0:
                    ptct_idx[name] = ptct_idx.get(name, []) + []

                else:
                    dfd_idx_list.clear()
                    acc_list.clear()
                    clean_weight = module.weight_quantizer.w_q_com.data.clone()
                    total_idx = torch.tensor(range(module.weight.data.numel()))
                    lg.info(
                        f"Check sensitivity {torch.sort(module.weight._taylor_series.data.view(-1), descending=True)[0]}"
                    )
                    check_sensitivity = torch.nn.functional.softmax(
                        torch.sort(
                            module.weight._taylor_series.data.view(-1), descending=True
                        )[0],
                        dim=0,
                    )
                    check_sensitivity_T = torch.nn.functional.softmax(
                        torch.sort(
                            module.weight._taylor_series.data.view(-1) / self.T,
                            descending=True,
                        )[0],
                        dim=0,
                    )

                    lg.info(f"Check softmaxed sensitivity {check_sensitivity}")
                    lg.info(
                        f"Check temperatured softmaxed sensitivity {check_sensitivity_T}"
                    )

                    for i in range(20):
                        # ? "100" here denotes the defending index sampling
                        sampler = WeightedRandomSampler(
                            weights=norm_sensitivity,
                            num_samples=self.defend_w_per_layer[name],
                            replacement=False,
                        )

                        sampled_idx = torch.tensor([idx for idx in sampler])

                        mask = torch.isin(total_idx, sampled_idx)
                        candidate_idx = (total_idx * (~mask))[total_idx * (~mask) != 0]
                        candidate_sensitivity = norm_sensitivity[candidate_idx]

                        acc_list_group.clear()
                        for _ in range(self.rt_ov):
                            self.in_layer_search(
                                module=module,
                                candidate_sensitivity=candidate_sensitivity,
                                candidate_idx=candidate_idx,
                            )
                            module.weight_quantizer.from_two_com()

                            accuracy = validate(
                                self.model,
                                small_loader,
                                -3,
                                self.criterion,
                                [],
                                [],
                                self.device,
                            )

                            acc_list_group.append(accuracy)
                            module.weight_quantizer.w_q_com.data.copy_(clean_weight)
                            module.weight_quantizer.from_two_com()

                        _, min_index = max(
                            (value, index) for index, value in enumerate(acc_list_group)
                        )
                        accuracy_repre = acc_list_group[min_index]
                        lg.info(f"Accuracy for {i}-th group is {accuracy_repre}")

                        dfd_idx_list.append(sampled_idx)
                        acc_list.append(accuracy_repre)

                    acc_candidate = torch.tensor(acc_list)
                    _, indices = torch.topk(acc_candidate, 5)

                    lg.info(f"Selected top-5 index are {indices}")
                    # Perform full validation
                    acc_list_topk = []
                    accuracy_repre_topk = []
                    for i in indices.tolist():
                        sampled_idx = dfd_idx_list[i]

                        mask = torch.isin(total_idx, sampled_idx)
                        candidate_idx = (total_idx * (~mask))[total_idx * (~mask) != 0]
                        candidate_sensitivity = norm_sensitivity[candidate_idx]

                        for _ in range(self.rt_ov):
                            self.in_layer_search(
                                module=module,
                                candidate_sensitivity=candidate_sensitivity,
                                candidate_idx=candidate_idx,
                            )
                            module.weight_quantizer.from_two_com()

                            accuracy = validate(
                                self.model,
                                val_loader,
                                -3,
                                self.criterion,
                                [],
                                [],
                                self.device,
                            )

                            acc_list_topk.append(accuracy)
                            module.weight_quantizer.w_q_com.data.copy_(clean_weight)
                            module.weight_quantizer.from_two_com()

                        _, min_index = max(
                            (value, index) for index, value in enumerate(acc_list_topk)
                        )

                        accuracy_repre = acc_list_topk[min_index]
                        accuracy_repre_topk.append(accuracy_repre)

                    _, max_index = max(
                        (value, index)
                        for index, value in enumerate(accuracy_repre_topk)
                    )
                    dfd_idx = dfd_idx_list[indices[max_index]]
                    lg.info(
                        f"Coverage: {len(set(module.weight._taylor_series.data.view(-1).topk(self.defend_w_per_layer[name])[1].tolist()).intersection(set(dfd_idx.tolist()))) / len(dfd_idx.tolist())}"
                    )

                    ptct_idx[name] = ptct_idx.get(name, 0) + dfd_idx

        # Return a dictionary which records the weights which are protected (dict contains lists)
        return ptct_idx

    def weight_protection(
        self,
        val_loader: DataLoader,
        small_loader: DataLoader,
        method: str = "importance",
        salience: str = "second-order",
    ):
        # Use full-batch of training data to calculate gradient
        calculate_grad_hessian(
            self.model,
            train_loader=val_loader,
            criterion=self.criterion,
            mode="defender",
            num_samples=1,
            device=self.device,
        )

        self.calculate_taylor_series()

        # determine memory budget for each layer
        self.distribute_budget(method=method)

        with torch.no_grad():
            self.ptct_idx = self.layerwise_protection(
                val_loader, small_loader=small_loader, salience=salience
            )

        self.model.train(self.train_mode)

        return self.ptct_idx

    def cal_mem_ov(self, ptct_idx: dict, mode: str = "truncated"):
        """
        Description: Calculate memory overhead required bu TCU or Unary protection.
        """
        assert mode in [
            "truncated",
            "original",
        ], f"{mode} not support"  # Truncated as TCU
        import math

        self.ptct_bits = 0
        self.ptct_idx = ptct_idx if ptct_idx != {} else self.ptct_idx

        for name, layer in self.model.named_modules():
            if isinstance(layer, (GemmConv2d, GemmLinear)):
                pointer_bit = math.log2(layer.weight.data.numel()).__ceil__()
                ptct_weight = layer.weight_quantizer.w_q_com.data.view(-1).clone()[
                    self.ptct_idx[name]
                ]

                if mode == "original":
                    self.ptct_bits += ptct_weight.data.numel() * (
                        self.max_value * 2 - 1 + pointer_bit
                    )
                else:  # TCU
                    # lg.info(ptct_weight)
                    mask = ptct_weight > self.max_value
                    ptct_weight[mask] = 2**self.N_bits - ptct_weight[mask]
                    # lg.info(ptct_weight)
                    aligned_ptct_weight = (
                        self.next_power_2_tensor(x=ptct_weight.int()) - 1
                    )
                    self.ptct_bits += (
                        aligned_ptct_weight.sum() + ptct_weight.numel() * pointer_bit
                    )

        self.mem_ov = self.ptct_bits / (self.N_bits * self.num_weight)

        return self.mem_ov

    def next_power_2_tensor(self, x: Tensor):
        assert (x >= 0).all()
        n = x.clone()
        n -= 1
        n |= n >> 1
        n |= n >> 2
        n |= n >> 4
        n |= n >> 8
        n |= n >> 16
        n += 1
        return n

    def cal_statistics(self, ptct_idx: dict):
        self.weight_statistics = torch.zeros(size=[self.N_bits], device=self.device)
        print(self.weight_statistics)
        for name, layer in self.model.named_modules():
            if isinstance(layer, (GemmConv2d, GemmLinear)):
                ptct_weight = layer.weight_quantizer.w_q_com.data.view(-1)[
                    torch.tensor(ptct_idx[name], dtype=int)
                ]

                mask = ptct_weight > self.max_value
                ptct_weight[mask] = 2**self.N_bits - ptct_weight[mask]

                aligned_ptct_weight = self.next_power_2_tensor(
                    ptct_weight.int()
                ).float()

                for i in range(self.N_bits):
                    self.weight_statistics[i] += (
                        (aligned_ptct_weight == 2**i).int().sum()
                    )
                self.weight_statistics[0] += (aligned_ptct_weight == 0).int().sum()

        lg.info(f" Statistics is {self.weight_statistics}")

    def calculate_taylor_series(self):
        for layer in self.model.modules():
            if isinstance(layer, (GemmConv2d, GemmLinear)):
                series_term = (
                    layer.weight._first_grad.data
                    * (layer.weight_quantizer.w_q_com.data - self.max_value).sign()
                    + layer.weight._second_grad.data * self.max_value / 2
                )
                layer.weight._taylor_series = series_term
