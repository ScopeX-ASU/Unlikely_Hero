"""
Description:
Author: Jiaqi Gu (jqgu@utexas.edu)
Date: 2021-10-24 16:35:39
LastEditors: ScopeX-ASU jiaqigu@asu.edu
LastEditTime: 2023-10-04 16:30:46
"""

import math
import os
import random
import sys
from functools import lru_cache
from typing import Callable, List, Optional, Tuple, Union

import einops
import matplotlib.pyplot as plt
import numpy as np
import torch
import tqdm
from matplotlib import cm
from pyutils.compute import (
    gen_boolean_mask,
    gen_gaussian_filter2d,
    merge_chunks,
    partition_chunks,
)
from pyutils.general import logger
from pyutils.torch_train import set_torch_deterministic
from scipy.spatial.distance import cdist
from sklearn.cluster import KMeans
from torch import Tensor
from torch.types import Device, _size
from torchonn.op.mzi_op import (
    checkerboard_to_vector,
    upper_triangle_to_vector,
    vector_to_checkerboard,
    vector_to_upper_triangle,
)

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../.."))


__all__ = [
    "CustomDistanceKMeans",
    "Noise_scheduler",
    "PhaseQuantizer",
    "LinearFeatureSampler",
    "Conv2dFeatureSampler",
    "FeedbackSampler",
    "SingularValueGradientSampler",
    "LearningProfiler",
    "DeterministicCtx",
    "PhaseVariationScheduler",
    "GlobalTemperatureScheduler",
    "CrosstalkScheduler",
    "calculate_grad_hessian",
]


class CustomDistanceKMeans:
    def __init__(
        self,
        first_grad: np.ndarray,
        sec_grad: np.ndarray,
        calculate_obj,
        n_clusters=8,
        max_iter=300,
        tol=1e-4,
    ):
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.tol = tol
        self.calculate_obj = calculate_obj
        self.first_grad = first_grad
        self.sec_grad = sec_grad

    def fit(self, X: np.ndarray):
        self.kmeans = KMeans(
            n_clusters=self.n_clusters, max_iter=self.max_iter, tol=self.tol
        )
        self.kmeans.fit(X)

        self.cluster_centers_ = self.kmeans.cluster_centers_

        for _ in range(self.max_iter):
            # 计算自定义距离
            if self.calculate_obj == cdist:
                distances = self.calculate_obj(X, self.cluster_centers_, "euclidean")
            else:
                distances = self.calculate_obj(
                    X, self.cluster_centers_, self.first_grad, self.sec_grad
                )
            # 找到最近的聚类中心
            self.labels_ = np.argmin(distances, axis=1)
            new_centers = np.array(
                [
                    X[self.labels_ == j].mean(axis=0)
                    if np.sum(self.labels_ == j) > 0
                    else self.cluster_centers_[j]
                    for j in range(self.n_clusters)
                ]
            )

            if np.allclose(self.cluster_centers_, new_centers, atol=self.tol):
                break

            self.cluster_centers_ = new_centers
            # logger.info(f"Cneters are {self.cluster_centers_}")


def apply_remap_weight(self, weight, col_ind, require_size=[4, 4, 8, 8]):
    ## FIXME canonly handle one to one remapping..
    weight = self.layer_weight_partition_chunk(
        weight, require_size=require_size
    )  # [b0, b1, R, C, K, K]
    # print(self.weight.shape, weight.shape, self.col_ind.shape)
    weight = weight.flatten(0, 1)[
        torch.arange(weight.shape[0] * weight.shape[1])[..., None],
        col_ind.flatten(0, 1),
    ].reshape(weight.shape)
    weight = self.layer_weight_merge_chunk(weight)[
        : self.grid_dim_y, : self.grid_dim_x
    ]  # [P,Q,K,K]
    return weight


def unapply_remap_weight(self, weight, col_ind, require_size=[4, 4, 8, 8]):
    ## FIXME canonly handle one to one remapping..
    weight = self.layer_weight_partition_chunk(
        weight, require_size=require_size
    )  # [b0, b1, R, C, K, K]
    weight.flatten(0, 1)[
        torch.arange(weight.shape[0] * weight.shape[1])[..., None],
        col_ind.flatten(0, 1),
    ] = weight.flatten(0, 1).clone()
    weight = self.layer_weight_merge_chunk(weight)[
        : self.grid_dim_y, : self.grid_dim_x
    ]  # [P,Q,K,K]
    return weight


def apply_remap_noise(noise_map, col_ind):
    ## noise_map: [b0, b1, R, C, K, K]
    ## col_ind: [b0, b1, R]
    ## FIXME canonly handle one to one remapping..

    # print(noise_map)
    ## [0, 1, 1, 2] means W0 -> T0, W1 -> T1, W2 -> T1, W3 -> T2
    noise_map = noise_map.flatten(0, 1)[
        torch.arange(noise_map.shape[0] * noise_map.shape[1])[..., None],
        col_ind.flatten(0, 1),
    ].reshape(noise_map.shape)
    # print(noise_map)
    # exit(0)

    return noise_map  # [b0, b1 ,R, C, K, K]


class Noise_scheduler(object):
    def __init__(
        self,
        noise_flag: bool,
        noise_level: float,
        out_noise_level: float,
        device: torch.device = torch.device("cuda"),
    ):
        super().__init__()
        self.noise_flag = noise_flag
        self.noise_level = noise_level
        self.out_noise_level = out_noise_level
        self.device = device

    def add_input_noise(self, x: Tensor) -> Tensor:
        if self.noise_flag:
            noise_map = torch.tensor(self.noise_level, device=self.device).repeat(
                x.shape
            )
            noise = torch.normal(mean=0.0, std=noise_map)
            return x + noise
        else:
            return x

    def add_weight_noise(self, x: Tensor) -> Tensor:
        if self.noise_flag:
            noise_map = torch.tensor(self.noise_level, device=self.device).repeat(
                x.shape
            )
            noise = torch.normal(mean=0.0, std=noise_map)
            return x + noise
        else:
            return x

    def add_output_noise(self, x: Tensor) -> Tensor:
        if self.noise_flag:
            noise_map = torch.tensor(self.out_noise_level, device=self.device).repeat(
                x.shape
            )
            noise = torch.normal(mean=0.0, std=noise_map)
            return x + noise
        else:
            return x


class PhaseVariationScheduler(object):
    def __init__(
        self,
        size: _size = [
            4,
            4,
            8,
            8,
        ],  # this one should be the architecture dimension, [R, C, K, K], not workload dimension [P, Q, K, K]
        T_max: int = 1000,  # total number of steps
        mean_schedule_fn: Callable = lambda: 0.02,  # a function that returns a mean value for a given step
        std_schedule_fn: Callable = lambda: 0.01,  # a function that returns a std value for a given step
        smoothing_kernel_size: int = 5,  # kernel size for the gaussian filter
        smoothing_factor: float = 0.05,  # how smooth is the distribution
        smoothing_mode: str = "core",  # smoothing mode, core: smooth core-by-core, arch: smooth over all cores
        min_std: float = 0.001,
        momentum: float = 0.9,  # how much is on previous noise std distribution, momenutm * old_map + (1-momentum) * new_map
        noise_scenario_src: str = "",  # corner, edge
        noise_scenario_tgt: str = "",
        random_state: int = 0,
        device="cuda:0",
    ) -> None:
        """
        Each device has a zero-mean random phase noise, the phase noise follows N(0, std_i^2) for the i-th device
        Then we need a tensor `noise_std_map` with the same shape as `phase` for each device.
        The noise intensity for each device will gradually drift to an unknown direction.
        To create a random std drift curve, e.g., std_i=0.01 -> std_i=0.008 -> std_i=0.012 -> std_i=0.011 -> std_i=0.009
        we construct a random process, std_i = momentum * std_i_old + (1 - momentum) * std_i_new
        , where std_i_new is randomly sampled from a Gaussian distribution N(std_mean_i, std_std_i),
        std_i_new are spatially smooth across all devices, therefore we apply gaussian filter to smooth `noise_std_map`.
        std_mean_i is controlled by mean_schedule_fn, std_std_i is controlled by std_schedule_fn.
        For example, if std_mean increases, it means the average noise intensity increases across all devices. Maybe the environment gets worse or background noises become larger.
        For example, if std_std increases, it means the noise intensity becomes more diverse across all devices. Maybe there is some local perturbation that makes devices behave diversely.

        """
        # std of the phase noise follows Gaussian distribution ~ N(noise_std_mean, noise_std_std^2)
        super().__init__()
        self.size = size
        self.T_max = T_max
        self.mean_schedule_fn = mean_schedule_fn
        self.std_schedule_fn = std_schedule_fn
        self.smoothing_kernel_size = smoothing_kernel_size
        assert (
            smoothing_kernel_size == 0 or smoothing_kernel_size % 2 == 1
        ), "Must have 0 or odd size of kernel"
        self.smoothing_factor = smoothing_factor
        self.smoothing_mode = smoothing_mode
        self.momentum = momentum
        self.min_std = min_std
        self.noise_scenario_src = noise_scenario_src
        self.noise_scenario_tgt = noise_scenario_tgt

        self.random_state = random_state
        self.device = device
        self.core_noise_mean_map = None

        if self.smoothing_factor > 0 and self.smoothing_kernel_size > 0:
            self.gaussian_filter = gen_gaussian_filter2d(
                self.smoothing_kernel_size,
                std=self.smoothing_factor,
                center_one=False,
                device=self.device,
            )[None, None, ...].to(device)
            # print(self.gaussian_filter)
            # exit(0)
            pad = self.smoothing_kernel_size // 2
            self.padder = torch.nn.ReflectionPad2d((pad, pad, pad, pad))
        else:
            self.gaussian_filter = None
        self.noises = None

        self.reset()

    def reset(self):
        self._step = 0
        self.noise_std_mean = self.mean_schedule_fn(
            0
        )  # the mean of the phase noise std
        self.noise_std_std = self.std_schedule_fn(0)  # the std of the phase noise std
        self.noise_std_map = None
        self.noises = None
        self.noise_scenario_transition()
        self.update_noise_std_map()

    def step(self):
        # one time step to change noise distribution
        # you can call this at any frequency you want, e.g., every step, or even more fine-grained (every layer)
        self._step += 1  # enable periodic scheduling
        self.noise_std_mean = self.mean_schedule_fn(
            (self._step % self.T_max) / self.T_max  # 0.002 # enable periodic scheduling
        )  # normalized value to query the mean schedule function
        self.noise_std_std = self.std_schedule_fn(
            (self._step % self.T_max) / self.T_max  # enable periodic scheduling
        )  # normalized value to query the std schedule function
        self.update_noise_std_map()
        self.noise_scenario_transition()
        # print(f'noise_std_mean={self.noise_std_mean}, noise_std_std={self.noise_std_std}')

    def noise_scenario_transition(self):
        if self.noise_scenario_tgt == "edge":
            target_core_noise_mean_map = torch.tensor(
                [
                    [0.008, 0.006, 0.004, 0.002],
                    [0.008, 0.006, 0.004, 0.002],
                    [0.008, 0.006, 0.004, 0.002],
                    [0.008, 0.006, 0.004, 0.002],
                ],
                device=self.device,
            )
        elif self.noise_scenario_tgt == "corner":
            target_core_noise_mean_map = torch.tensor(
                [
                    [0.008, 0.006, 0.006, 0.004],
                    [0.006, 0.006, 0.004, 0.004],
                    [0.006, 0.004, 0.004, 0.002],
                    [0.004, 0.004, 0.002, 0.002],
                ],
                device=self.device,
            )

        core_noise_mean_map = self._generate_core_noise_mean_map()
        if self.core_noise_mean_map is None:
            self.core_noise_mean_map = core_noise_mean_map
        else:
            self.core_noise_mean_map = (
                self.momentum * self.core_noise_mean_map
                + (1 - self.momentum) * target_core_noise_mean_map
            )

    def _generate_core_noise_mean_map(self) -> Tensor:
        core_noise_mean_map = torch.zeros(self.size[:-2])
        if self.noise_scenario_src == "corner":
            self.core_noise_mean_map = torch.tensor(
                [
                    [0.0008, 0.0006, 0.0006, 0.0004],
                    [0.0006, 0.0006, 0.0004, 0.0004],
                    [0.0006, 0.0004, 0.0004, 0.0002],
                    [0.0004, 0.0004, 0.0002, 0.0002],
                ]
            )
        elif self.noise_scenario_src == "edge":
            self.core_noise_mean_map = torch.tensor(
                [
                    [0.0008, 0.0006, 0.0004, 0.0002],
                    [0.0008, 0.0006, 0.0004, 0.0002],
                    [0.0008, 0.0006, 0.0004, 0.0002],
                    [0.0008, 0.0006, 0.0004, 0.0002],
                ]
            )
        else:
            raise NotImplementedError

        core_noise_mean_map = self.core_noise_mean_map / 2
        self.core_noise_mean_map = core_noise_mean_map.to(self.device)
        return core_noise_mean_map.to(self.device)

    def _generate_noise_std_map(self):
        # determinstic, do not affect external random state, different across steps
        # this is device-wise noise std map. Each MRR has a noise_std, representing its noise intensity
        # this can create difference within each core for intra-core remapping
        noise_std_map = torch.normal(
            self.noise_std_mean,
            self.noise_std_std,
            size=self.size,
            generator=torch.Generator(device=self.device).manual_seed(
                self.random_state + self._step
            ),
            device=self.device,
        ).clamp_min_(
            self.min_std
        )  # [c, r, k, k] # the std needs to be at least some small value, if std is zero, then it is not random at all.
        ## we assume each core has a background noise intensity(std) specific to this core.
        ## this core-wise std will be added to the noise_std_map. then different cores can have different noise intensity
        ## this core-wise noise intensity leads to unbalanced/uneven noise levels across c x r cores. Then enable inter-core remapping.
        # self._generate_core_noise_mean_map()

        core_noise_std_map = (
            torch.normal(
                mean=self.core_noise_mean_map,  # self.noise_std_mean, #core_noise_mean_map, # core-wise std level
                std=self.noise_std_std,  # core-wise std diversity
                # size=self.size[:-2], # std_mean for this core, approximated by the std_mean averaged across kxk rings
                generator=torch.Generator(device=self.device).manual_seed(
                    self.random_state + self._step
                ),
                # device=self.device,
            )
            .clamp_min_(self.min_std)
            .to(self.device)[..., None, None]
        )  # [c,r,1,1]  # the std needs to be at least some small value, if std is zero, then it is not random at all.

        # print(core_noise_std_map.shape)
        # print(noise_std_map.shape)
        # =========================================================  core-wise noise_mean_map  ========================================================#
        ## core-wise noise_mean_map, different core has different noise intensity, and we define 2 modes for noise_mean distribution
        ## 1: corner mode, noise intensity is most significant at left-up core, and smoothly distributed along x- and y- axis
        ## 2: edge mode, noise intensity is most significant at left column and distributed along x-axis

        noise_std_map = (core_noise_std_map + noise_std_map) / 2
        if self.gaussian_filter is not None:
            # we assume the noise intensity (i.e., std) distribution is smooth locally
            if self.smoothing_mode == "core":
                noise_std_map = torch.nn.functional.conv2d(
                    self.padder(noise_std_map).flatten(0, 1).unsqueeze(1),
                    self.gaussian_filter,
                    padding="valid",
                ).view_as(noise_std_map)
            elif self.smoothing_mode == "arch":
                noise_std_map = partition_chunks(
                    torch.nn.functional.conv2d(
                        self.padder(merge_chunks(noise_std_map)[None, None]),
                        self.gaussian_filter,
                        padding="valid",
                    )[0, 0],
                    bs=noise_std_map.shape[-1],
                )
        return noise_std_map

    def update_noise_std_map(self):
        noise_std_map = self._generate_noise_std_map()
        if self.noise_std_map is None:
            self.noise_std_map = noise_std_map
        else:
            # every time step, we gradually update the noise std map to another random map, the momentum controls how much we keep the old map
            self.noise_std_map = (
                self.momentum * self.noise_std_map + (1 - self.momentum) * noise_std_map
            )

    def sample_noise(self, size=None, enable_remap: bool = False, col_ind=None):
        ## size [P, Q, k, k]: the workload size you want to map to this [R, C, K, K] multi-core MRR accelerator
        ## If size is None, then the workload is assumed to be [R, C, K, K]
        ## need to return [P, Q, k, k] phase noises for this workload
        ## assume the archiecture is [R, C, k, k]

        # when size=self.size, i.e., batch = [1, 1], then P=R, Q=C, i.e., each block in the layer weight matrix is mapped to a photonic core.
        # when batch = [u, v], we assume u=\ceil{P/R}, v=\ceil{Q/C}, i.e., the matrix needs to be partition into multiple RkxCk blocks and mapped sequentially to the same accelerator.
        size = size or self.size
        batch = (
            int(np.ceil(size[0] / self.size[0])),
            int(np.ceil(size[1] / self.size[1])),
        )

        # we assume the phase noise has zero mean, only std is determined by the noise_std_map
        # the P, Q, K, K workload will be chunked into u-by-v chunks (with same padding), each chunk is R, C, K, K, and thus can be mapping to the arch.
        # The u-by-v chunks require u-by-v times inference. The u-by-v inferences will see the same noise distribution, but different noise samples.
        # noise_std_map = einops.repeat(self.noise_std_map, "r c k l-> (u r) (v c) k l", u=batch[0], v=batch[1])[:size[0], :size[1]]
        noise_std_map = einops.repeat(
            self.noise_std_map, "r c k l-> u v r c k l", u=batch[0], v=batch[1]
        )
        if enable_remap and col_ind is not None:
            ## we remap noise distribution
            noise_std_map = apply_remap_noise(noise_std_map, col_ind=col_ind)
        noise_std_map = (
            noise_std_map.permute(0, 2, 1, 3, 4, 5)
            .flatten(0, 1)
            .flatten(1, 2)[: size[0], : size[1]]
        )

        noises = torch.normal(
            mean=0.0, std=noise_std_map
        )  # n ~ N(0, noise_std_map^2) different device has different std
        # noises = torch.normal(
        #     mean=0.0, std=self.noise_std_map
        # )  # n ~ N(0, noise_std_map^2) different device has different std
        self.noises = noises  ## add this to record the noise sampled.
        return noises


class GlobalTemperatureScheduler(object):
    def __init__(
        self,
        size=[4, 4, 8, 8],
        T_max: int = 1000,  # total number of steps
        n_g: float = 4.3,  # Bogaerts et al. 2012
        n_eff: float = 1.89,  # Bogaerts et al. 2012, TM Mode
        dwl_dT: float = 0.102,  # Bogaerts et al. 2012, TM Mode, d wavelength / d T. unit nm / K
        schedule_fn: Callable = lambda: 300,  # a function that returns a temperature in K unit, bu default is room temp
        T0: float = 300,  # initial room temperature
        lambda_res: List | Tensor | np.ndarray = [],
        L_list: List | Tensor | np.ndarray = [],
        hotspot_mode: str = "uniform",
        device="cuda:0",
    ) -> None:
        """
        just gradually set global temperature based on schedule_fn
        """
        # std of the phase noise follows Gaussian distribution ~ N(noise_std_mean, noise_std_std^2)
        super().__init__()
        self.size = size
        self.T_max = T_max
        self.schedule_fn = schedule_fn
        self.n_g = n_g
        self.n_eff = n_eff
        self.dwl_dT = dwl_dT
        self.T0 = T0
        self._last_T = T0
        self.L_list = L_list
        assert hotspot_mode in {"uniform", "corner"}
        self.hotspot_mode = hotspot_mode
        self.lambda_res = lambda_res
        if isinstance(lambda_res, list):
            self.lambda_res = torch.tensor(lambda_res, device=device)
        elif isinstance(lambda_res, np.ndarray):
            self.lambda_res = torch.from_numpy(lambda_res).to(device)

        if self.lambda_res.device != device:
            self.lambda_res = self.lambda_res.to(device)

        if isinstance(L_list, list):
            self.L_list = torch.tensor(L_list, device=device)
        elif isinstance(L_list, np.ndarray):
            self.L_list = torch.from_numpy(L_list).to(device)

        if self.L_list.device != device:
            self.L_list = self.L_list.to(device)

        self.device = device
        self.reset()

    def reset(self) -> None:
        self._step = 0
        self.T = self.schedule_fn(0)

    def step(self) -> None:
        self._step += 1
        self.T = self.schedule_fn(self._step / self.T_max)

    def get_global_temp(self) -> float:
        return self.T

    def record_current_temp(self):
        self._last_T = self.T

    def get_hotspot_map(self) -> Tensor:
        if self.hotspot_mode == "uniform":
            hotspot_map = torch.ones(self.size[0:2], device=self.device)
        elif self.hotspot_mode == "corner":
            X, Y = torch.meshgrid(
                torch.arange(self.size[0], device=self.device),
                torch.arange(self.size[1], device=self.device),
            )
            hotspot_map = torch.exp(-1 * (X.square() + Y.square()).sqrt())
        else:
            raise NotImplementedError
        return hotspot_map

    def get_phase_drift(
        self, phase, T, enable_remap: bool = False, col_ind=None
    ) -> Tensor:
        """
        temperature drift will trigger lambda shift, i.e., delta_lambda, we assume lambda is linear to T, then
        delta_lambda = delta_T * d lambda / dT

        delta_lambda means there is a change on the neff, i.e., delta_neff
        delta_neff = delta_lambda * n_g / lambda_res

        the neff change leads to extra round-trip phase shift, e.g., delta_phi
        delta_phi = delta_neff * 2pi * R / lambda_res * 2pi
        The temperature change induced phase drift is only a function of T and wavelengths/Radius.
        For this [R,C,K,K] MRR weight bank architecture, only K different wavelengths/Radius, T is global.
        return delta_Phi [Tensor]: [K]-shaped tensor, each element is the phase drift for each wavelength/Radius.
        This can be naturally broadcast to [P,Q,K,K] workload (corresponding to the last dimension).
        """

        n_g = self.n_g  # Bogaerts et al. 2012
        n_eff = self.n_eff  # Bogaerts et al. 2012, TM Mode
        # delta_lambda = (T - self.T0) * self.dwl_dT
        hotspot_map = self.get_hotspot_map()[..., None, None]  # [R, C, 1, 1]
        delta_T = (T - self.T0) * hotspot_map
        delta_lambda = delta_T * self.dwl_dT  # [R, C, 1, 1]
        K = phase.shape[-1]
        lambda_res = self.lambda_res[
            self.lambda_res.shape[0] // 2 - K // 2 : self.lambda_res.shape[0] // 2
            - K // 2
            + K
        ]  # we only need k lambdas, so you need pass the central k wavelengths
        L_list = self.L_list[
            self.L_list.shape[0] // 2 - K // 2 : self.L_list.shape[0] // 2 - K // 2 + K
        ]
        delta_neff = delta_lambda * n_g / lambda_res  # [R, C, 1, k]
        delta_Phi = delta_neff * L_list * 1000 / lambda_res * 2 * np.pi  # [R, C, 1, k]

        size = phase.shape  # [P,Q,K,K]
        batch = (
            int(np.ceil(size[0] / self.size[0])),
            int(np.ceil(size[1] / self.size[1])),
        )

        # we assume the phase noise has zero mean, only std is determined by the noise_std_map
        # the P, Q, K, K workload will be chunked into u-by-v chunks (with same padding), each chunk is R, C, K, K, and thus can be mapping to the arch.
        # The u-by-v chunks require u-by-v times inference. The u-by-v inferences will see the same noise distribution, but different noise samples.
        delta_Phi = einops.repeat(
            delta_Phi, "r c k l-> u v r c k l", u=batch[0], v=batch[1]
        )
        if enable_remap and col_ind is not None:
            ## we remap noise distribution
            # print("here")
            # print(col_ind)
            # exit(0)
            delta_Phi = apply_remap_noise(delta_Phi, col_ind=col_ind)
        delta_Phi = (
            delta_Phi.permute(0, 2, 1, 3, 4, 5)
            .flatten(0, 1)
            .flatten(1, 2)[: size[0], : size[1]]
        )

        return delta_Phi  # [P, Q, 1, k]


class CrosstalkScheduler(object):
    def __init__(
        self,
        Size=[4, 4, 8, 8],
        crosstalk_coupling_factor: float = 4.8,
        interv_h: float = 60.0,
        interv_v: float = 200.0,
        cutoff_calue: float = 1e-3,
        device="cuda:0",
    ) -> None:
        super().__init__()
        self.crosstalk_coupling_factor = crosstalk_coupling_factor
        self.interv_h = interv_h
        self.interv_v = interv_v
        self.cutoff_calue = cutoff_calue
        self.vh_coeff = self.interv_v / self.interv_h
        self.device = device

        self.crosstalk_mask = None

    def get_crosstalk_matrix_old(self, phase: Tensor) -> Tensor:
        crosstalk_mask = torch.eye(
            phase.size(-1) * phase.size(-2), phase.size(-1) * phase.size(-2)
        )
        for count in range(1, phase.size(-2)):
            for i in (
                range(0, phase.size(-2) * phase.size(-2) - count)
                if phase.size(-2) > count
                else range(0, 0)
            ):
                crosstalk_mask[i, i + count] = (
                    math.e ** (-self.crosstalk_coupling_factor * self.interv_h * count)
                    if i % phase.size(-2) < phase.size(-2) - count
                    else crosstalk_mask[i + count, i]
                )
                crosstalk_mask[i + count, i] = (
                    math.e ** (-self.crosstalk_coupling_factor * self.interv_h * count)
                    if i % phase.size(-2) < phase.size(-2) - count
                    else crosstalk_mask[i + count, i]
                )

        # vertical distance equals to several interv_v
        for count in range(1, phase.size(-1)):
            for i in (
                range(0, (phase.size(-1) - count) * phase.size(-1))
                if phase.size(-1) > count
                else range(0, 0)
            ):
                crosstalk_mask[i, i + count * phase.size(-1)] = math.e ** (
                    -self.crosstalk_coupling_factor * count * self.interv_v
                )
                crosstalk_mask[i + count * phase.size(-1), i] = math.e ** (
                    -self.crosstalk_coupling_factor * count * self.interv_v
                )

        # Kings-graph (i, 1) directions
        for count in range(1, phase.size(-2)):
            for i in range(0, phase.size(-2) * (phase.size(-1) - 1) - count):
                crosstalk_mask[i, i + phase.size(-2) + count] = (
                    math.e
                    ** (
                        -self.crosstalk_coupling_factor
                        * math.sqrt(self.interv_v**2 + count**2)
                    )
                    if i % phase.size(-1) <= phase.size(-1) - 1 - count
                    else crosstalk_mask[i, i + phase.size(-1) + count]
                )
                crosstalk_mask[i + phase.size(-1) + count, i] = (
                    math.e
                    ** (
                        -self.crosstalk_coupling_factor
                        * math.sqrt(self.interv_v**2 + count**2)
                    )
                    if i % phase.size(-1) <= phase.size(-1) - 1 - count
                    else crosstalk_mask[i + phase.size(-1) + count, i]
                )
            for i in range(1, phase.size(-2) * (phase.size(-1) - 1)):
                crosstalk_mask[i, i + phase.size(-1) - count] = (
                    math.e
                    ** (
                        -self.crosstalk_coupling_factor
                        * math.sqrt(self.interv_v**2 + (count * self.interv_h) ** 2)
                    )
                    if i % phase.size(-1) > count - 1
                    else crosstalk_mask[i, i + phase.size(-1) - count]
                )
                crosstalk_mask[i + phase.size(-1) - count, i] = (
                    math.e
                    ** (
                        -self.crosstalk_coupling_factor
                        * math.sqrt(self.interv_v**2 + (count * self.interv_h) ** 2)
                    )
                    if i % phase.size(-1) > count - 1
                    else crosstalk_mask[i + phase.size(-1) - count, i]
                )

        # Kings-graph (i, 2) directions,
        for count in range(1, phase.size(-2)):
            for i in range(0, phase.size(-2) * (phase.size(-1) - 2) - count):
                crosstalk_mask[i, i + 2 * phase.size(-1) + count] = (
                    math.e
                    ** (
                        -self.crosstalk_coupling_factor
                        * math.sqrt(
                            (self.interv_v * 2) ** 2 + (count * self.interv_h) ** 2
                        )
                    )
                    if i % phase.size(-1) <= phase.size(-1) - 1 - count
                    else crosstalk_mask[i, i + 2 * phase.size(-1) + count]
                )
                crosstalk_mask[i + 2 * phase.size(-1) + count, i] = (
                    math.e
                    ** (
                        -self.crosstalk_coupling_factor
                        * math.sqrt(
                            (self.interv_v * 2) ** 2 + (count * self.interv_h) ** 2
                        )
                    )
                    if i % phase.size(-1) <= phase.size(-1) - 1 - count
                    else crosstalk_mask[i, i + 2 * phase.size(-1) + count]
                )
            for i in range(1, phase.size(-2) * (phase.size(-1) - 2)):
                crosstalk_mask[i, i + 2 * phase.size(-1) - count] = (
                    math.e
                    ** (
                        -self.crosstalk_coupling_factor
                        * math.sqrt(
                            (self.interv_v * 2) ** 2 + (count * self.interv_h) ** 2
                        )
                    )
                    if i % phase.size(-1) > count - 1
                    else crosstalk_mask[i, i + 2 * phase.size(-1) - count]
                )
                crosstalk_mask[i + 2 * phase.size(-1) - count, i] = (
                    math.e
                    ** (
                        -self.crosstalk_coupling_factor
                        * math.sqrt(
                            (self.interv_v * 2) ** 2 + (count * self.interv_h) ** 2
                        )
                    )
                    if i % phase.size(-1) > count - 1
                    else crosstalk_mask[i + 2 * phase.size(-1) - count, i]
                )
        # print('crosstalk factor is:',
        #       crosstalk_mask)
        # # Kings-graph (i, 3) directions,
        # for count in range(1, K):
        #     for i in range(0, K * (K-3) - count):
        #         crosstalk_mask[i, i + 3 * K + count] = math.e ** (-self.crosstalk_coupling_factor * math.sqrt((self.vh_coeff * 2) ** 3 + count ** 2)) if i % K <= K - 1 - count else crosstalk_mask[i, i + 3 * K + count]
        #         crosstalk_mask[i + 3 * K + count, i] = math.e ** (-self.crosstalk_coupling_factor * math.sqrt((self.vh_coeff * 2) ** 3 + count ** 2)) if i % K <= K - 1 - count else crosstalk_mask[i, i + 3 * K + count]
        #     for i in range(1, K * (K-3)):
        #         crosstalk_mask[i, i + 3 * K - count] = math.e ** (-self.crosstalk_coupling_factor * math.sqrt((self.vh_coeff * 2) ** 3 + count ** 2)) if i % K > count - 1 else crosstalk_mask[i, i + 3 * K - count]
        #         crosstalk_mask[i + 3 * K - count, i] = math.e ** (-self.crosstalk_coupling_factor * math.sqrt((self.vh_coeff * 2) ** 3 + count ** 2)) if i % K > count - 1 else crosstalk_mask[i + 3 * K - count, i]

        # # Kings-graph (i, 4) directions,
        # for count in range(1, K):
        #     for i in range(0, K * (K-4) - count):
        #         crosstalk_mask[i, i + 4 * K + count] = math.e ** (-self.crosstalk_coupling_factor * math.sqrt((self.vh_coeff * 2) ** 4 + count ** 2)) if i % K <= K - 1 - count else crosstalk_mask[i, i + 4 * K + count]
        #         crosstalk_mask[i + 4 * K + count, i] = math.e ** (-self.crosstalk_coupling_factor * math.sqrt((self.vh_coeff * 2) ** 4 + count ** 2)) if i % K <= K - 1 - count else crosstalk_mask[i, i + 4 * K + count]
        #     for i in range(1, K * (K-4)):
        #         crosstalk_mask[i, i + 4 * K - count] = math.e ** (-self.crosstalk_coupling_factor * math.sqrt((self.vh_coeff * 2) ** 4 + count ** 2)) if i % K > count - 1 else crosstalk_mask[i, i + 4 * K - count]
        #         crosstalk_mask[i + 4 * K - count, i] = math.e ** (-self.crosstalk_coupling_factor * math.sqrt((self.vh_coeff * 2) ** 4 + count ** 2)) if i % K > count - 1 else crosstalk_mask[i + 4 * K - count, i]
        self.crosstalk_mask = crosstalk_mask
        return crosstalk_mask

    @lru_cache(maxsize=8)
    def get_crosstalk_matrix(self, size) -> Tensor:
        k1, k2 = size[-2], size[-1]
        X, Y = torch.meshgrid(
            torch.arange(k1, device=self.device), torch.arange(k2, device=self.device)
        )
        X, Y = X.flatten().float(), Y.flatten().float()
        distance = (
            X.unsqueeze(1).sub(X.unsqueeze(0)).square() * self.interv_v**2
            + Y.unsqueeze(1).sub(Y.unsqueeze(0)).square() * self.interv_h**2
        ).sqrt()
        # print(X.shape, distance.shape)
        self.crosstalk_mask = torch.exp(-self.crosstalk_coupling_factor * distance)
        return self.crosstalk_mask


class DeterministicCtx:
    def __init__(self, random_state: Optional[int] = None) -> None:
        self.random_state = random_state

    def __enter__(self):
        self.random_state = random.getstate()
        self.numpy_random_state = np.random.get_state()
        self.torch_random_state = torch.random.get_rng_state()
        self.torch_cuda_random_state = torch.cuda.get_rng_state()
        set_torch_deterministic(self.random_state)
        return self

    def __exit__(self, *args):
        random.setstate(self.random_state)
        np.random.seed(self.numpy_random_state)
        np.random.set_state(self.numpy_random_state)
        torch.random.set_rng_state(self.torch_random_state)
        torch.cuda.set_rng_state(self.torch_cuda_random_state)


class PhaseQuantizer(torch.nn.Module):
    __mode_list__ = {"rectangle", "triangle", "diagonal"}

    def __init__(
        self,
        bit: int,
        v_pi: float = 4.36,
        v_max: float = 10.8,
        gamma_noise_std: float = 0.0,
        crosstalk_factor: float = 0.0,
        crosstalk_filter_size: int = 5,
        random_state: Optional[int] = None,
        mode: str = "rectangle",
        device: torch.device = torch.device("cuda"),
    ) -> None:
        """2021/04/01: Uniform phase-space quantization. Support gamma noise and thermal crosstalk simulation
        Args:
            bit (int): bitwidth
            phase_onise_std (float, optional): std dev of Gaussian phase noise. Defaults to 0.
            random_state (None or int, optional): random_state for noise injection. Defaults to None.
            device (torch.Device, optional): torch.Device. Defaults to torch.device("cuda").
        """
        super().__init__()
        self.bit = bit
        self.v_pi = v_pi
        self.v_max = v_max
        self.gamma = np.pi / v_pi**2
        self.gamma_noise_std = gamma_noise_std
        self.crosstalk_factor = crosstalk_factor
        self.crosstalk_filter_size = crosstalk_filter_size
        self.random_state = random_state
        self.mode = mode
        assert mode in self.__mode_list__, logger.error(
            f"Only support {self.__mode_list__}, but got {mode}."
        )
        self.device = device

        self.crosstal_simulator = ThermalCrosstalkSimulator(
            plotting=False,
            filter_size=crosstalk_filter_size,
            crosstalk_factor=crosstalk_factor,
            device=self.device,
        )
        self.register_buffer("noisy_gamma", None)  # can be saved in checkpoint

    def set_gamma_noise(
        self, noise_std: float, size: _size, random_state: Optional[int] = None
    ):
        self.gamma_noise_std = noise_std
        self.random_state = random_state
        if random_state is not None:
            set_torch_deterministic(random_state)
        self.noisy_gamma = (
            torch.nn.init.trunc_normal_(torch.zeros(size, device=self.device))
            .mul_(noise_std)
            .add_(self.gamma)
        )

    def set_crosstalk_factor(self, crosstalk_factor):
        self.crosstalk_factor = crosstalk_factor
        self.crosstal_simulator.set_crosstalk_factor(crosstalk_factor)

    def set_bitwidth(self, bit: int) -> None:
        self.bit = bit

    def forward(self, x):
        x = x % (2 * np.pi)
        if self.bit < 16:
            if self.mode in {"rectangle", "triangle"}:  # [0, 2pi] quantize
                ratio = 2 * np.pi / (2**self.bit - 1)
                x.div_(ratio).round_().mul_(ratio)
            elif self.mode in {"diagonal"}:  # [0, pi] quantize
                x = torch.where(x > np.pi, 2 * np.pi - x, x)
                ratio = np.pi / (2**self.bit - 1)
                x.div_(ratio).round_().mul_(ratio)
            else:
                raise NotImplementedError(self.mode)

        if self.noisy_gamma is not None:
            x.mul_(self.noisy_gamma.div(self.gamma))

        if self.crosstalk_factor > 1e-5:
            x = self.crosstal_simulator.simple_simulate(
                x, mixedtraining_mask=None, mode=self.mode
            )

        return x


class ThermalCrosstalkSimulator(object):
    __mode_list__ = {"rectangle", "triangle", "diagonal"}

    def __init__(
        self,
        # interval bet/ heat source (um)
        heat_source_interval: float = 8.0,
        # SetPad=0,
        grid_precision: float = 10.0,  # um
        power_density_multipier: float = 1e-3,
        # W/(um K) thermal conductivity
        thermal_conductivity: float = 1.4e-6,
        max_iter: int = 2000,  # max # of iterations
        # material options
        boundary_cond: bool = False,
        # plotting options
        plotting: bool = True,
        display_iter: int = 10,
        hold_time: float = 0.00001,
        filter_size: int = 3,
        crosstalk_factor: float = 0.01,
        device: Device = torch.device("cuda:0"),
    ):
        super().__init__()

        self.heat_source_interval = heat_source_interval
        self.grid_precision = grid_precision
        self.power_density_multiplier = power_density_multipier
        self.thermal_conductivity = thermal_conductivity
        self.max_iter = max_iter
        self.boundary_cond = boundary_cond
        self.plotting = plotting
        self.display_iter = display_iter
        self.hold_time = hold_time
        self.filter_size = filter_size
        self.crosstalk_factor = crosstalk_factor
        self.device = device
        self.power_density = None

        # self.init_phase_distribution(self.phases)
        self.init_filter(filter_size, crosstalk_factor)
        self.mixedtraining_mask = None

    def init_filter(self, filter_size: int, crosstalk_factor: float) -> None:
        c = crosstalk_factor
        if filter_size == 3:
            self.filter = torch.tensor(
                [[0, c, 0], [c, 1, c], [0, c, 0]], device=self.device
            )
        elif filter_size == 5:
            self.filter = torch.tensor(
                [[0, c, 0], [c, 0, c], [0, 1, 0], [c, 0, c], [0, c, 0]],
                device=self.device,
            )
        else:
            raise ValueError(
                f"Does not support filter sizes other than 3 or 5, but got {filter_size}"
            )
        self.filter.unsqueeze_(0).unsqueeze_(0)

        self.filter_zero_center = self.filter.clone()
        self.filter_zero_center[
            0, 0, self.filter.size(-2) // 2, self.filter.size(-1) // 2
        ] = 0

    def init_phase_distribution(self, phases: Tensor, dim: int) -> None:
        self.power_density = np.zeros(
            [self.heat_source_interval * dim, self.heat_source_interval * dim]
        )
        cnt = 0
        # for i in range(1, dim):
        #     for j in range(1, dim - i + 1):
        #         self.power_density[self.heat_source_interval*i, self.heat_source_interval*j] = phases[cnt]
        #         cnt = cnt + 1
        pointer = 0
        for i in range(1, dim):
            number_of_sources = dim - i
            interval = self.heat_source_interval
            self.power_density[
                interval * i, interval : number_of_sources * interval + 1 : interval
            ] = phases[pointer : pointer + number_of_sources]
            pointer += number_of_sources

    def simulate(self, phases: Tensor, dim: int) -> None:
        self.init_phase_distribution(phases, dim)
        # *SetSpace      # number of steps in x
        nx = self.power_density.shape[0]
        ny = self.power_density.shape[1]  # *SetSpace   # number of steps in y
        dx = self.grid_precision  # nx/(nx-1) # width of step
        dy = self.grid_precision  # ny/(ny-1) # width of step

        # Initial Conditions
        p = torch.zeros((1, 1, nx, ny)).float().to(self.device)
        power_density = (
            (
                torch.from_numpy(self.power_density.copy()).unsqueeze(0).unsqueeze(0)
                * dx
                * dx
                * dy
                * dy
                * self.thermal_conductivity
                / (2 * (dx * dx + dy * dy))
            )
            .float()
            .to(self.device)
        )
        kernel = torch.from_numpy(
            np.array(
                [[0, dy * dy, 0], [dx * dx, 0, dx * dx], [0, dy * dy, 0]],
                dtype=np.float32,
            )
        ) / (2 * (dx * dx + dy * dy))
        kernel = kernel.unsqueeze(0).unsqueeze(0).to(self.device)
        mask = torch.zeros(nx, ny, dtype=torch.float32, device=self.device)
        for row in range(1, nx - 2):
            mask[row, 1 : ny - row - 1] = 1

        conv_err = []
        if self.plotting is True:
            plt.ion()  # continuous SetPlotting
            fig = plt.figure()
            ax = fig.gca(projection="3d")
            x = np.linspace(dx / 2, nx - dx / 2, nx)
            y = np.linspace(dy / 2, ny - dy / 2, ny)  # orig no setspace
            X, Y = np.meshgrid(x, y)

        for it in range(self.max_iter + 1):
            # print(f"[I] iteration: {it}")
            out = torch.nn.functional.conv2d(p, kernel, padding=(1, 1))
            out.add_(power_density).mul_(mask)

            conv_err.append((it, (out - p).abs().max().data.item()))
            p = out

            if self.plotting is True and it % (self.display_iter) == 0:
                surf = ax.plot_surface(
                    X,
                    Y,
                    p.squeeze(0).squeeze(0).numpy(),
                    cmap=cm.rainbow,
                    linewidth=0,
                    antialiased=False,
                )
                # ax.set_zlim(0,80)
                # ax.set_xlim(0,0.1)
                # ax.set_ylim(0,0.1)
                plt.title("it#%d" % it, y=1)
                ax.set_xlabel("Distance (x%d um)" % (self.grid_precision))
                ax.set_ylabel("Distance (x%d um)" % (self.grid_precision))
                ax.set_zlabel("Temperature (C)")
                # for tick in ax.xaxis.get_major_ticks():
                #     tick.label.set_fontsize(80)
                # for tick in ax.yaxis.get_major_ticks():
                #     tick.label.set_fontsize(80)

                plt.show()
                plt.pause(self.hold_time)

        return p.cpu().numpy().astype(np.float64)

    def set_crosstalk_factor(self, crosstalk_factor: float) -> None:
        self.crosstalk_factor = crosstalk_factor
        self.init_filter(self.filter_size, crosstalk_factor)

    def simple_simulate_triangle(
        self, phases: Tensor, mixedtraining_mask: Optional[Tensor]
    ) -> Tensor:
        size = phases.size()
        phases = phases % (2 * np.pi)
        if mixedtraining_mask is None:
            # batchify phases [bs, k(k-1)/2]
            phases = phases.view(-1, 1, phases.size(-1))  # [bs, 1, k(k-1)/2]
            phases = vector_to_checkerboard(phases)
            filter = self.filter
            padding1, padding2 = self.filter.size(-2) // 2, self.filter.size(-1) // 2
            phases = torch.nn.functional.conv2d(
                phases, filter, padding=(padding1, padding2)
            )
            phases = checkerboard_to_vector(phases)
            phases = phases.view(size)
        else:
            # only active devices marked as 1/True in the mixed training mask will influcence others
            # passive devices will be influenced by active devices, but will not incluence others
            # batchify phases [bs, k(k-1)/2]
            phase_mat_active = vector_to_upper_triangle(
                phases.mul(mixedtraining_mask.float()).view(-1, 1, phases.size(-1))
            )
            phases = phases.view(-1, 1, phases.size(-1))  # [bs, 1, k(k-1)/2]
            filter = self.filter_zero_center
            padding1, padding2 = self.filter.size(-2) // 2, self.filter.size(-1) // 2
            # influence map
            phase_mat_active = torch.nn.functional.conv2d(
                phase_mat_active, filter, padding=(padding1, padding2)
            )
            # add influence map and original phases together
            phases = upper_triangle_to_vector(phase_mat_active) + phases
            phases = phases.view(size)

        return phases

    def simple_simulate_diagonal(
        self, phases: Tensor, mixedtraining_mask: Optional[Tensor]
    ) -> Tensor:
        return phases

    def simple_simulate_butterfly(
        self, phases: Tensor, mixedtraining_mask: Optional[Tensor]
    ) -> Tensor:
        phases = phases % (2 * np.pi)
        ## [n_level, k/2, 2]
        size = phases.size()

        if mixedtraining_mask is None:
            # [1, 1, n_level, k]
            phases = phases.view(
                [1, 1] + list(size)[:-2] + [phases.size(-1) * phases.size(-2)]
            )
            filter = self.filter
            padding = self.filter_size // 2
            phases = torch.nn.functional.conv2d(
                phases, filter, padding=(padding, padding)
            )
            phases = phases.view(size)

        else:
            # only active devices marked as 1/True in the mixed training mask will influcence others
            # poassive devices will be influenced by active devices, but will not incluence others

            phases_active = phases * mixedtraining_mask.float()
            filter = self.filter_zero_center
            padding = self.filter_size // 2
            # influence map
            phases_active = torch.nn.functional.conv2d(
                phases_active.view(
                    [1, 1] + list(size)[:-2] + [phases.size(-1) * phases.size(-2)]
                ),
                filter,
                padding=(padding, padding),
            )
            # add influence map and original phases together
            phases = phases_active.view_as(phases) + phases

        return phases

    def simple_simulate_rectangle(
        self, phases: Tensor, mixedtraining_mask: Optional[Tensor]
    ) -> Tensor:
        size = phases.size()
        phases = phases % (2 * np.pi)
        if mixedtraining_mask is None:
            # batchify phases [bs, k(k-1)/2]
            phases = phases.view(-1, 1, phases.size(-1))  # [bs, 1, k(k-1)/2]
            phases = vector_to_checkerboard(phases)
            filter = self.filter
            padding1, padding2 = self.filter.size(-2) // 2, self.filter.size(-1) // 2
            phases = torch.nn.functional.conv2d(
                phases, filter, padding=(padding1, padding2)
            )
            phases = checkerboard_to_vector(phases)
            phases = phases.view(size)
        else:
            # only active devices marked as 1/True in the mixed training mask will influcence others
            # passive devices will be influenced by active devices, but will not incluence others
            # batchify phases [bs, k(k-1)/2]
            phase_mat_active = vector_to_upper_triangle(
                phases.mul(mixedtraining_mask.float()).view(-1, 1, phases.size(-1))
            )
            phases = phases.view(-1, 1, phases.size(-1))  # [bs, 1, k(k-1)/2]
            filter = self.filter_zero_center
            padding1, padding2 = self.filter.size(-2) // 2, self.filter.size(-1) // 2
            # influence map
            phase_mat_active = torch.nn.functional.conv2d(
                phase_mat_active, filter, padding=(padding1, padding2)
            )
            # add influence map and original phases together
            phases = upper_triangle_to_vector(phase_mat_active) + phases
            phases = phases.view(size)

        return phases

    def simple_simulate(
        self,
        phases: Tensor,
        mixedtraining_mask: Optional[Tensor] = None,
        mode: str = "rectangle",
    ) -> Tensor:
        assert mode in self.__mode_list__, logger.error(
            f"Only support {self.__mode_list__}. But got {mode}"
        )
        if mode == "triangle":
            return self.simple_simulate_triangle(phases, mixedtraining_mask)
        elif mode == "rectangle":
            return self.simple_simulate_rectangle(phases, mixedtraining_mask)
        elif mode == "diagonal":
            return self.simple_simulate_diagonal(phases, mixedtraining_mask)
        elif mode == "butterfly":
            return self.simple_simulate_butterfly(phases, mixedtraining_mask)
        else:
            return phases


class LinearFeatureSampler(torch.nn.Module):
    def __init__(
        self,
        sparsity: float = 0,
        miniblock: int = 8,
        normalize: str = "none",
        random_state: Optional[int] = None,
    ) -> None:
        super().__init__()
        self.sparsity = sparsity
        self.miniblock = miniblock
        self.normalize = normalize
        self.random_state = random_state
        self.mask = None

    def set_sparsity(self, sparsity: float, random_state: Optional[int] = None) -> None:
        assert 0 <= sparsity <= 1, logger.error(
            f"Illegal sparsity, must within [0,1] but got {sparsity}."
        )
        self.sparsity = sparsity
        self.random_state = random_state

    def sample(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        """Block-level structured sampling of input tensor. return sampled blocks and a boolean mask
        Args:
            x (Tensor): 2D padded hidden features

        Raises:
            NotImplementedError: Not supported tensor shape

        Returns:
            Tuple[Tensor, Tensor]: sampled blocks and boolean mask
        """

        # padded 2D input for Linear layer [bs, inc] = [bs, p*k]
        # samples must be different for different examples
        if not self.training:  # DO NOT sampling during inference
            return x, None
        self.input_size = x.size()
        batch_size = x.size(0)
        self.n_block = x.size(-1) // self.miniblock
        self.mask = gen_boolean_mask(
            (batch_size, self.n_block), true_prob=1 - self.sparsity, device=x.device
        )  # [bs, p]
        x = x.view(batch_size, self.n_block, -1)[self.mask, :]  # [n_samples, k]
        if self.normalize == "exp":  # expectation maintained (unbiased)
            x = x.mul(1 / (self.mask.float().sum() / self.mask.numel() + 1e-12))
        elif self.normalize == "var":  # variance maintained
            x = x.mul(1 / (self.mask.float().sum() / self.mask.numel() + 1e-12).sqrt())
        return x, self.mask

    def reconstruct(self, x: Tensor, mask: Optional[Tensor] = None) -> Tensor:
        # x [n_samples, k]
        # mask [bs, n_block]
        if mask is None:
            mask = self.mask
        if mask is None:
            return x
        out = torch.zeros(self.input_size, device=x.device).view(
            -1, self.n_block, self.miniblock
        )
        out[mask, :] = x
        out = out.flatten(1)
        return out


class Conv2dFeatureSampler(torch.nn.Module):
    def __init__(
        self,
        spatial_sparsity: float = 0,
        column_sparsity: float = 0,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = 1,
        normalize: str = "none",
        random_state: Optional[int] = None,
    ) -> None:
        super().__init__()
        self.spatial_sparsity = spatial_sparsity
        self.column_sparsity = column_sparsity
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.normalize = normalize
        self.random_state = random_state
        self.spatial_mask = None
        self.column_mask = None
        self.set_sparsity(spatial_sparsity, column_sparsity, random_state)

    def set_sparsity(
        self,
        spatial_sparsity: float,
        column_sparsity: float,
        random_state: Optional[int] = None,
    ) -> None:
        assert 0 <= spatial_sparsity <= 1, logger.error(
            f"Illegal spatial_sparsity, must within [0,1] but got {spatial_sparsity}."
        )
        assert 0 <= column_sparsity <= 1, logger.error(
            f"Illegal column_sparsity, must within [0,1] but got {column_sparsity}."
        )
        self.spatial_sparsity = spatial_sparsity
        self.column_sparsity = column_sparsity
        self.random_state = random_state
        if (
            self.kernel_size == 1
        ):  # merge column sampling to spatial sampling to save memory
            self.spatial_sparsity = 1 - (1 - self.spatial_sparsity) * (
                1 - self.column_sparsity
            )
            self.column_sparsity = 0

    def sample(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        """Block-level structured sampling of input tensor. return sampled blocks and a boolean mask
        Args:
            x (Tensor): 4D feature maps

        Raises:
            NotImplementedError: Not supported tensor shape

        Returns:
            Tuple[Tensor, Tensor, Tensor]: sampled blocks, boolean spatial mask and boolean column mask
        """

        # 4D input for Conv2d layer [bs, inc, h, w]
        # share samples between different channels and examples are OK
        # unstructured spatial sampling to save memory and strcutured column sampling to save backward runtime
        if not self.training:  # DO NOT sampling during inference
            return x, None, None, None
        self.input_size = x.size()
        h_x, w_x = x.size(2), x.size(3)
        h_out = int((h_x - self.kernel_size + 2 * self.padding) / self.stride + 1)
        w_out = int((w_x - self.kernel_size + 2 * self.padding) / self.stride + 1)
        self.h_out, self.w_out = h_out, w_out
        if self.spatial_sparsity > 0:
            self.spatial_mask = gen_boolean_mask(
                (h_x, w_x),
                true_prob=1 - self.spatial_sparsity,
                random_state=self.random_state,
                device=x.device,
            )  # [h, w]
        else:
            self.spatial_mask = None
        if self.column_sparsity > 0:
            self.column_mask = gen_boolean_mask(
                [h_out * w_out],
                true_prob=1 - self.column_sparsity,
                random_state=self.random_state,
                device=x.device,
            )  # [h_out*w_out]
        else:
            self.column_mask = None
        if self.spatial_mask is not None:
            x = x[:, :, self.spatial_mask]
        if self.normalize in {"exp", "var"}:
            # Do not scale feature maps. Too costly. We also need to scale based on column_scaling_factor. Equivalently, we can scale grad_weight instead.
            real_spatial_sparsity = (
                (1 - self.spatial_mask.float().sum() / self.spatial_mask.numel())
                if self.spatial_mask is not None
                else 0
            )
            real_column_sparsity = (
                (1 - self.column_mask.float().sum() / self.column_mask.numel())
                if self.column_mask is not None
                else 0
            )
            self.scaling_factor = 1 / (
                (1 - real_spatial_sparsity) * (1 - real_column_sparsity) + 1e-12
            )
            if self.normalize == "var":
                self.scaling_factor = self.scaling_factor**0.5
        else:
            self.scaling_factor = None

        return x, self.spatial_mask, self.column_mask, self.scaling_factor

    def reconstruct(self, x: Tensor, spatial_mask: Optional[Tensor] = None) -> Tensor:
        # reconstruct using the spatial mask
        if spatial_mask is None:
            spatial_mask = self.spatial_mask
        if spatial_mask is not None:
            out = torch.zeros(self.input_size, device=x.device)
            out[:, :, spatial_mask] = x
        else:
            out = x
        return out


class FeedbackSampler(torch.nn.Module):
    __alg_list__ = {"topk", "uniform", "gtopk"}
    __mode_list__ = {"linear", "conv"}

    def __init__(
        self,
        forward_sparsity: float,
        backward_sparsity: float,
        alg: str = "topk",
        normalize: str = "none",
        mode: str = "linear",
        random_state: Optional[int] = None,
    ):
        super().__init__()
        self.forward_sparsity = forward_sparsity
        self.backward_sparsity = backward_sparsity
        self.alg = alg
        assert alg in self.__alg_list__, logger.error(
            f"Only support {self.__alg_list__}, but got {alg}."
        )
        self.normalize = normalize
        self.mode = mode
        assert mode in self.__mode_list__, logger.error(
            f"Only support {self.__mode_list__}, but got {mode}."
        )

        self.random_state = random_state
        self.mask = None

    def set_sparsity(
        self,
        forward_sparsity: float,
        backward_sparsity: float,
        random_state: Optional[int] = None,
    ) -> None:
        assert 0 <= forward_sparsity <= 1, logger.error(
            f"Illegal forward_sparsity, must within [0,1] but got {forward_sparsity}."
        )
        self.forward_sparsity = forward_sparsity
        assert 0 <= backward_sparsity <= 1, logger.error(
            f"Illegal backward_sparsity, must within [0,1] but got {backward_sparsity}."
        )
        self.backward_sparsity = backward_sparsity
        self.random_state = random_state

    def sample_topk(self, x: Tensor, mask: Tensor, forward: bool = False) -> Tensor:
        # we prefer uniform column-wise sparsity in W, i.e., row-wise sparsity in W^T
        ## x: [p, q, k, k]
        # forward: topk along q dimension, backward: topk along p dimension
        if mask is None:
            mask = self.mask = torch.ones(
                x.size(0), x.size(1), device=x.device, dtype=torch.bool
            )
        sparsity = self.forward_sparsity if forward else self.backward_sparsity
        if sparsity < 1e-9:
            return x.clone()
        # pruned blocks has small total singular value
        norm = x.flatten(2).norm(p=2, dim=-1)  # [p, q]
        if forward:
            thres = torch.quantile(
                norm, q=sparsity, dim=1, keepdim=True
            )  # forward: [p, 1]
        else:
            thres = torch.quantile(
                norm, q=sparsity, dim=0, keepdim=True
            )  # backward: [1, q]
        mask.masked_fill_(norm < thres, 0)
        x = x.clone()
        x[~mask, :, :] = 0
        if self.normalize == "exp":  # expectation maintained (unbiased)
            x = x.mul_(1 / (mask.float().sum() / mask.numel() + 1e-12))
        elif self.normalize == "var":  # variance maintained
            x = x.mul_(1 / (mask.float().sum() / mask.numel() + 1e-12).sqrt())

        return x

    def sample_topk_(self, x: Tensor, mask: Tensor, forward: bool = False) -> Tensor:
        # we prefer uniform column-wise sparsity in W, i.e., row-wise sparsity in W^T
        ## x: [p, q, k, k]
        if mask is None:
            mask = self.mask = torch.ones(
                x.size(0), x.size(1), device=x.device, dtype=torch.bool
            )
        sparsity = self.forward_sparsity if forward else self.backward_sparsity
        if sparsity < 1e-9:
            return x
        # pruned blocks has small total singular value
        norm = x.flatten(2).norm(p=2, dim=-1)
        if forward:
            thres = torch.quantile(
                norm, q=sparsity, dim=1, keepdim=True
            )  # forward: [p, 1]
        else:
            thres = torch.quantile(
                norm, q=sparsity, dim=0, keepdim=True
            )  # backward: [1, q]
        mask.masked_fill_(norm < thres, 0)
        x[~mask, :, :] = 0
        if self.normalize == "exp":
            x = x.mul_(1 / (mask.float().sum() / mask.numel() + 1e-12))
        elif self.normalize == "var":
            x = x.mul_(1 / (mask.float().sum() / mask.numel() + 1e-12).sqrt())
        return x

    def sample_gtopk(self, x: Tensor, mask: Tensor, forward: bool = False) -> Tensor:
        # global top k without load balancing
        ## x: [p, q, k, k]
        if mask is None:
            mask = self.mask = torch.ones(
                x.size(0), x.size(1), device=x.device, dtype=torch.bool
            )
        sparsity = self.forward_sparsity if forward else self.backward_sparsity
        if sparsity < 1e-9:
            return x.clone()
        # pruned blocks has small total singular value
        norm = x.flatten(2).norm(p=2, dim=-1)  # [p,q]
        thres = torch.quantile(norm, q=sparsity)  # [1]
        mask.masked_fill_(norm < thres, 0)
        x = x.clone()
        x[~mask, :, :] = 0
        if self.normalize == "exp":
            x = x.mul_(1 / (mask.float().sum() / mask.numel() + 1e-12))
        elif self.normalize == "var":
            x = x.mul_(1 / (mask.float().sum() / mask.numel() + 1e-12).sqrt())
        return x

    def sample_gtopk_(self, x: Tensor, mask: Tensor, forward: bool = False) -> Tensor:
        # global top k without load balancing
        ## x: [p, q, k, k]
        if mask is None:
            mask = self.mask = torch.ones(
                x.size(0), x.size(1), device=x.device, dtype=torch.bool
            )
        sparsity = self.forward_sparsity if forward else self.backward_sparsity
        if sparsity < 1e-9:
            return x
        # pruned blocks has small total singular value
        norm = x.flatten(2).norm(p=2, dim=-1)
        thres = torch.quantile(norm, q=sparsity)
        mask.masked_fill_(norm < thres, 0)
        x[~mask, :, :] = 0
        if self.normalize == "exp":
            x = x.mul_(1 / (mask.float().sum() / mask.numel() + 1e-12))
        elif self.normalize == "var":
            x = x.mul_(1 / (mask.float().sum() / mask.numel() + 1e-12).sqrt())
        return x

    def sample_uniform(self, x: Tensor, mask: Tensor, forward: bool = False) -> Tensor:
        sparsity = self.forward_sparsity if forward else self.backward_sparsity
        if mask is None:
            mask = self.mask = gen_boolean_mask(
                size=(x.size(0), x.size(1)),
                true_prob=1 - sparsity,
                random_state=self.random_state,
                device=x.device,
            )
        if sparsity < 1e-9:
            return x.clone()
        x = x.clone()
        x[~mask, :, :] = 0
        if self.normalize == "exp":
            x = x.mul_(1 / (mask.float().sum() / mask.numel() + 1e-12))
        elif self.normalize == "var":
            x = x.mul_(1 / (mask.float().sum() / mask.numel() + 1e-12).sqrt())
        return x

    def sample_uniform_(self, x: Tensor, mask: Tensor, forward: bool = False) -> Tensor:
        sparsity = self.forward_sparsity if forward else self.backward_sparsity
        if mask is None:
            mask = self.mask = gen_boolean_mask(
                size=(x.size(0), x.size(1)),
                true_prob=1 - sparsity,
                random_state=self.random_state,
                device=x.device,
            )

        if sparsity < 1e-9:
            return x
        x[~mask, :, :] = 0
        if self.normalize == "exp":
            x = x.mul_(1 / (mask.float().sum() / mask.numel() + 1e-12))
        elif self.normalize == "var":
            x = x.mul_(1 / (mask.float().sum() / mask.numel() + 1e-12).sqrt())
        return x

    def sample(self, x: Tensor, mask: Tensor = None, forward: bool = False) -> Tensor:
        """sample the weight matrix.

        Args:
            x (Tensor): weight matrix W [p, q, k, k]
            forward (bool): whether use forward sparsity or feedback sparsity in topk algorithm.

        Raises:
            NotImplementedError

        Returns:
            Tensor: sparse weight matrix
        """
        ## x [p, q, k, k]
        if not self.training:  # DO NOT sampling during inference
            return x
        if self.alg == "uniform":
            return self.sample_uniform(x, mask, forward)
        elif self.alg == "topk":
            return self.sample_topk(x, mask, forward)
        elif self.alg == "gtopk":
            return self.sample_gtopk(x, mask, forward)
        else:
            raise NotImplementedError(
                f"Only support {self.__alg_list__}, but got {self.alg}."
            )

    def sample_(self, x: Tensor, mask: Tensor = None, forward: bool = False) -> Tensor:
        ## x [p, q, k, k]
        if not self.training:  # DO NOT sampling during inference
            return x
        if self.alg == "uniform":
            return self.sample_uniform_(x, mask, forward)
        elif self.alg == "topk":
            return self.sample_topk_(x, mask, forward)
        elif self.alg == "gtopk":
            return self.sample_gtopk_(x, mask, forward)
        else:
            raise NotImplementedError(
                f"Only support {self.__alg_list__}, but got {self.alg}."
            )


class SingularValueGradientSampler(torch.nn.Module):
    __alg_list__ = {"topk", "uniform"}

    def __init__(
        self,
        rank: int,
        alg: str = "topk",
        sign: bool = False,
        random_state: Optional[int] = None,
    ):
        super().__init__()
        self.rank = rank
        self.alg = alg
        assert alg in self.__alg_list__, logger.error(
            f"Only support {self.__alg_list__}, but got {alg}."
        )
        self.sign = sign
        self.random_state = random_state
        self.mask = None

    def set_rank(self, rank: int, random_state: Optional[int] = None) -> None:
        self.rank = rank
        self.random_state = random_state

    def uniform_mask(self, x: Tensor) -> Tensor:
        ## x [p, q, k]
        if self.rank < x.size(-1):
            indices = (
                torch.ones(x.size(0) * x.size(1), x.size(2), device=x.device)
                .multinomial(num_samples=self.rank)
                .view(x.size(0), x.size(1), -1)
            )
            self.mask = torch.zeros_like(x, dtype=torch.bool).scatter_(-1, indices, 1)
        else:
            self.mask = None
        return self.mask

    def topk_mask(self, x: Tensor) -> Tensor:
        if self.rank < x.size(-1):
            indices = torch.topk(
                x.abs(), k=self.rank, dim=-1, largest=True, sorted=False
            )[1]
            self.mask = torch.zeros_like(x, dtype=torch.bool).scatter_(-1, indices, 1)
        else:
            self.mask = None
        return self.mask

    def sample(
        self,
        u: Tensor,
        s: Tensor,
        v: Tensor,
        grad_weight: Tensor,
        I_U: Optional[Tensor] = None,
        I_V: Optional[Tensor] = None,
        grad_scaling_factor: float = None,
    ):
        if self.alg == "uniform":
            mask = self.uniform_mask(s)
        elif self.alg == "topk":
            mask = self.topk_mask(s)
        else:
            raise NotImplementedError(
                f"Only support {self.__alg_list__}, but got {self.alg}."
            )
        p, q, k = s.size()
        if I_V is not None:
            # u [p,q,k,k] x [p,q,k,k/rank] => [p,q,k,k/rank]
            if mask is not None:
                I_V = I_V.masked_select(mask.unsqueeze(-2)).view(p, q, k, self.rank)
            u = u.matmul(I_V)

        grad_sigma_by_v = u.permute([0, 1, 3, 2]).matmul(grad_weight)

        del grad_weight
        if I_U is not None:
            if mask is not None:
                I_U = I_U[mask].view(p, q, self.rank, k)
            v = I_U.matmul(v)

        grad_sigma = grad_sigma_by_v.mul_(v).sum(dim=-1)  # [p, q, k] or [p, q, bp_rank]
        if mask is not None:
            grad_sigma = torch.zeros(
                p, q, k, device=grad_sigma.device, dtype=grad_sigma.dtype
            ).masked_scatter_(mask, grad_sigma)
        if grad_scaling_factor is not None:
            grad_sigma.mul_(grad_scaling_factor)
        if self.sign:
            grad_sigma = grad_sigma.sign()
        return grad_sigma


class LearningProfiler(torch.nn.Module):
    def __init__(self, _enable: bool = True) -> None:
        super().__init__()
        self.report = None
        self._enable = _enable
        self.reset()

    def reset(self):
        self.report = {
            "forward_core_call": 0,
            "forward_accum_step": 0,  # addition only
            "backward_weight_core_call": 0,
            "backward_weight_accum_step": 0,  # addition and multiplication, doubles the cost
            "backward_input_core_call": 0,
            "backward_input_accum_step": 0,  # addition only
        }

    def enable(self):
        self._enable = True

    def disable(self):
        self._enable = False

    def update_forward(
        self,
        x: Tensor,
        weight: Tensor,
        output: Tensor,
        feedback_sampler: FeedbackSampler,
    ) -> None:
        # important assumption:
        # p x k adders, k adders for each block row
        # k wavelength for parallel PTC forward and backward
        # x [bs, inc] or [bs, inc, h, w]
        # weight: [p, q, k, k]
        # output [bs, outc] or [bs, outc, h', w']
        p, q, k = weight.size(0), weight.size(1), weight.size(2)
        if not self._enable or not self.training:
            return
        # forward
        if (
            feedback_sampler.forward_sparsity > 1e-9
            and feedback_sampler.mask is not None
        ):
            active_weight_block = feedback_sampler.mask.float().sum().item()
            max_accum_step = max(0, feedback_sampler.mask.sum(1).max().item() - 1)
        else:
            active_weight_block = weight.size(0) * weight.size(1)
            max_accum_step = weight.size(1) - 1
        batch = x.size(0) if x.dim() == 2 else int(output.numel() / output.size(1))
        self.report["forward_core_call"] += (
            active_weight_block * batch
        )  # p*q*sparsity*batch
        # do not forget we do im2col sequentially
        self.report["forward_accum_step"] += max_accum_step * batch + np.ceil(
            batch / k
        )  # PTC forward also counted as  1 step/cycle

    def update_backward(
        self,
        x: Tensor,
        weight: Tensor,
        grad_output: Tensor,
        x_requires_grad: bool,
        weight_requires_grad: bool,
        feature_sampler: Union[LinearFeatureSampler, Conv2dFeatureSampler],
        feedback_sampler: FeedbackSampler,
        rank_sampler: SingularValueGradientSampler,
    ) -> None:
        p, q, k = weight.size(0), weight.size(1), weight.size(2)
        if not self._enable or not self.training:
            return
        # weight backward
        if weight_requires_grad:
            if isinstance(feature_sampler, Conv2dFeatureSampler):
                if (
                    feature_sampler.kernel_size == 1
                ):  # conv1x1: spatial sampling=column sampling
                    if feature_sampler.spatial_sparsity > 1e-9:
                        # conv1x1 can have stride>1, padding is always 0
                        # in real X_col, the real sparsity can be obtained from the stride spatial mask
                        stride = feature_sampler.stride
                        active_column = feature_sampler.spatial_mask[
                            ::stride, ::stride
                        ].float().sum().item() * x.size(0)
                    else:
                        active_column = int(grad_output.numel() / grad_output.size(1))
                else:  # conv3x3
                    if feature_sampler.column_sparsity > 1e-9:
                        active_column = (
                            feature_sampler.column_mask.float().sum().item() * x.size(0)
                        )  # column mask is shared between batch
                    else:
                        active_column = int(grad_output.numel() / grad_output.size(1))
                self.report["backward_weight_core_call"] += (
                    weight.size(0) * weight.size(1) * 2 * active_column
                )
                # MAC doubles the cost. lowrank does not impact this.
                # self.report["backward_weight_accum_step"] += max(0,active_column - 1) * 2 * rank_sampler.rank / weight.size(-1)
                self.report["backward_weight_accum_step"] += active_column * 4
            elif isinstance(feature_sampler, LinearFeatureSampler):
                mask = feature_sampler.mask.float()  # [bs, q]
                # nonzero * p
                self.report["backward_weight_core_call"] += (
                    mask.sum().item() * weight.size(0)
                )
                self.report["backward_weight_accum_step"] += mask.sum(
                    0
                ).max().item() * x.size(0)
        # input backward
        if x_requires_grad:
            # share sparsity with feedback or no forward weight sampling, has backward feedback sampling
            if (
                feedback_sampler.forward_sparsity > 1e-9
                or feedback_sampler.backward_sparsity > 1e-9
            ):
                active_weight_block = feedback_sampler.mask.float().sum().item()
                if x.dim() == 2:  # fully-connected
                    max_accum_step = max(
                        0, feedback_sampler.mask.sum(0).max().item() - 1
                    )  # p
                elif x.dim() == 4 and feature_sampler.kernel_size != 1:  # CONV3x3
                    # max_accum_step = max(0, feedback_sampler.mask.sum(1).max().item() - 1)  # q ## this is wrong!!
                    max_accum_step = (
                        np.ceil(x.size(1) / p)
                        * np.ceil(np.log2(2 * k))
                        * np.ceil(feedback_sampler.mask.sum(0).max().item() / 2)
                    )
                elif x.dim() == 4 and feature_sampler.kernel_size == 1:  # conv1x1
                    max_accum_step = max(
                        0, feedback_sampler.mask.sum(0).max().item() - 1
                    )  # p
            else:
                active_weight_block = weight.size(0) * weight.size(1)
                if x.dim() == 2:  # fully-connected
                    max_accum_step = max(0, weight.size(0) - 1)  # p-1
                elif x.dim() == 4 and feature_sampler.kernel_size != 1:  # CONV3x3
                    # max_accum_step = weight.size(1) - 1  # q-1 # wrong
                    max_accum_step = (
                        np.ceil(x.size(1) / p)
                        * np.ceil(np.log2(2 * k))
                        * np.ceil(p / 2)
                    )
                elif x.dim() == 4 and feature_sampler.kernel_size == 1:  # conv1x1
                    max_accum_step = max(0, weight.size(0) - 1)  # p-1
            if x.dim() == 2:  # linear
                batch = x.size(0)
            elif (
                x.dim() == 4 and feature_sampler.kernel_size == 1
            ):  # CONV1x1, maybe have stride
                batch = int(grad_output.numel() / grad_output.size(1))
            elif x.dim() == 4 and feature_sampler.kernel_size > 1:  # CONV3x3 or 5X5
                batch = int(x.numel() / x.size(1))
            else:
                raise NotImplementedError
            self.report["backward_input_core_call"] += (
                active_weight_block * batch
            )  # int(grad_output.numel()/grad_output.size(1))
            # do not forget we do matrix-matrix mul via sequential mat-vec mul
            self.report["backward_input_accum_step"] += (
                max_accum_step * batch
            )  # int(grad_output.numel() / grad_output.size(1))

    def update(
        self,
        x: Tensor,
        weight: Tensor,
        grad_output: Tensor,
        x_requires_grad: bool,
        weight_requires_grad: bool,
        feature_sampler: Union[LinearFeatureSampler, Conv2dFeatureSampler],
        feedback_sampler: FeedbackSampler,
        rank_sampler: SingularValueGradientSampler,
    ) -> None:
        # important assumption:
        # p x k adders, k adders for each block row
        # k wavelength for parallel PTC forward and backward
        # x [bs, inc] or [bs, inc, h, w]
        # weight: [p, q, k, k]
        # grad_output [bs, outc] or [bs, outc, h', w']
        p, q, k = weight.size(0), weight.size(1), weight.size(2)
        if not self._enable or not self.training:
            return
        # forward
        if feedback_sampler.forward_sparsity > 1e-9:
            active_weight_block = feedback_sampler.mask.float().sum().item()
            max_accum_step = max(0, feedback_sampler.mask.sum(1).max().item() - 1)
        else:
            active_weight_block = weight.size(0) * weight.size(1)
            max_accum_step = weight.size(1) - 1
        batch = (
            x.size(0)
            if x.dim() == 2
            else int(grad_output.numel() / grad_output.size(1))
        )
        self.report["forward_core_call"] += (
            active_weight_block * batch
        )  # p*q*sparsity*batch
        # do not forget we do im2col sequentially
        self.report["forward_accum_step"] += max_accum_step * batch + np.ceil(
            batch / k
        )  # PTC forward also counted as  1 step/cycle

        # weight backward
        if weight_requires_grad:
            if isinstance(feature_sampler, Conv2dFeatureSampler):
                if (
                    feature_sampler.kernel_size == 1
                ):  # conv1x1: spatial sampling=column sampling
                    if feature_sampler.spatial_sparsity > 1e-9:
                        # conv1x1 can have stride>1, padding is always 0
                        # in real X_col, the real sparsity can be obtained from the stride spatial mask
                        stride = feature_sampler.stride
                        active_column = feature_sampler.spatial_mask[
                            ::stride, ::stride
                        ].float().sum().item() * x.size(0)
                    else:
                        active_column = int(grad_output.numel() / grad_output.size(1))
                else:  # conv3x3
                    if feature_sampler.column_sparsity > 1e-9:
                        active_column = (
                            feature_sampler.column_mask.float().sum().item() * x.size(0)
                        )  # column mask is shared between batch
                    else:
                        active_column = int(grad_output.numel() / grad_output.size(1))
                self.report["backward_weight_core_call"] += (
                    weight.size(0) * weight.size(1) * 2 * active_column
                )
                # MAC doubles the cost. lowrank does not impact this.
                # self.report["backward_weight_accum_step"] += max(0,active_column - 1) * 2 * rank_sampler.rank / weight.size(-1)
                self.report["backward_weight_accum_step"] += active_column * 4
            elif isinstance(feature_sampler, LinearFeatureSampler):
                mask = feature_sampler.mask.float()  # [bs, q]
                # nonzero * p
                self.report["backward_weight_core_call"] += (
                    mask.sum().item() * weight.size(0)
                )
                self.report["backward_weight_accum_step"] += mask.sum(
                    0
                ).max().item() * x.size(0)
        # input backward
        if x_requires_grad:
            # share sparsity with feedback or no forward weight sampling, has backward feedback sampling
            if (
                feedback_sampler.forward_sparsity > 1e-9
                or feedback_sampler.backward_sparsity > 1e-9
            ):
                active_weight_block = feedback_sampler.mask.float().sum().item()
                if x.dim() == 2:  # fully-connected
                    max_accum_step = max(
                        0, feedback_sampler.mask.sum(0).max().item() - 1
                    )  # p
                elif x.dim() == 4 and feature_sampler.kernel_size != 1:  # CONV3x3
                    # max_accum_step = max(0, feedback_sampler.mask.sum(1).max().item() - 1)  # q ## this is wrong!!
                    max_accum_step = (
                        np.ceil(x.size(1) / p)
                        * np.ceil(np.log2(2 * k))
                        * np.ceil(feedback_sampler.mask.sum(0).max().item() / 2)
                    )
                elif x.dim() == 4 and feature_sampler.kernel_size == 1:  # conv1x1
                    max_accum_step = max(
                        0, feedback_sampler.mask.sum(0).max().item() - 1
                    )  # p
            else:
                active_weight_block = weight.size(0) * weight.size(1)
                if x.dim() == 2:  # fully-connected
                    max_accum_step = weight.size(0) - 1  # p-1
                elif x.dim() == 4 and feature_sampler.kernel_size != 1:  # CONV3x3
                    # max_accum_step = weight.size(1) - 1  # q-1 # wrong
                    max_accum_step = (
                        np.ceil(x.size(1) / p)
                        * np.ceil(np.log2(2 * k))
                        * np.ceil(p / 2)
                    )
                elif x.dim() == 4 and feature_sampler.kernel_size == 1:  # conv1x1
                    max_accum_step = weight.size(0) - 1  # p-1
            self.report["backward_input_core_call"] += active_weight_block * int(
                grad_output.numel() / grad_output.size(1)
            )
            # do not forget we do matrix-matrix mul via sequential mat-vec mul
            self.report["backward_input_accum_step"] += max_accum_step * int(
                grad_output.numel() / grad_output.size(1)
            )

    def __add__(self, other):
        out = LearningProfiler()
        for k in out.report:
            out.report[k] = self.report[k] + other.report[k]
        return out

    def __radd__(self, other):
        if other == 0:
            return self
        else:
            return self.__add__(other)

    def __sub__(self, other):
        out = LearningProfiler()
        for k in out.report:
            out.report[k] = self.report[k] - other.report[k]
        return out

    def __rsub__(self, other):
        return other.__sub__(self)

    def __truediv__(self, other):
        out = LearningProfiler()
        for k in out.report:
            out.report[k] = self.report[k] / other.report[k]
        return out

    def __rtruediv__(self, other):
        return other.__truediv__(self)


def test_crosstalk_simulator():
    device = "cuda"
    simu = ThermalCrosstalkSimulator(filter_size=5, device=device)
    phase = torch.randn((4 * 3) // 2, device=device) % (2 * np.pi)
    phase_c = simu.simple_simulate(phase, None, mode="rectangle")
    print(vector_to_checkerboard(phase))
    print(vector_to_checkerboard(phase_c))


def test_linear_feature_sampler():
    sampler = LinearFeatureSampler(0.5, 4, normalize="none", random_state=42)
    x = torch.randn(4, 4 * 4)
    xs, _ = sampler.sample(x)
    xr = sampler.reconstruct(xs)
    print(x, "\n", xr)


def test_conv2d_feature_sampler():
    sampler = Conv2dFeatureSampler(0.5, 3, normalize="none", random_state=42)
    x = torch.randn(2, 2, 4, 4)
    xs, _, _ = sampler.sample(x)
    xr = sampler.reconstruct(xs)
    print(x, "\n", xr)


def test_feedback_sampler():
    sampler = FeedbackSampler(0.5, "topk", random_state=43)
    w = torch.randn(2, 2, 2, 2)
    ws = sampler.sample(w)
    print(w)
    print(ws)


def test_singularvalue_sampler():
    p, q, k, r = 1, 1, 4, 4
    sampler = SingularValueGradientSampler(r, alg="uniform")

    s = torch.randn(p, q, k)
    u = torch.nn.init.orthogonal_(torch.randn(p, q, k, k))
    v = torch.nn.init.orthogonal_(torch.randn(p, q, k, k))
    I_U = torch.diag_embed(torch.ones(p, q, k))
    I_V = torch.diag_embed(torch.ones(p, q, k))
    grad_weight = torch.randn(p, q, k, k)
    grad_s = sampler.sample(u, s, v, grad_weight, I_U, I_V)
    print(grad_s)
    print(
        u.transpose(-1, -2).matmul(grad_weight.matmul(v.transpose(-1, -2)))[
            ..., :: k + 1
        ]
    )


def calculate_grad_hessian(
    model, train_loader, criterion, mode, num_samples=10, device="cuda:0"
):
    ## average gradients and second order gradients will be stored in weight._first_grad and weight._second_grad
    is_train = model.training
    model.train()
    ## freeze BN stat is important
    bn_state = None
    for m in model.modules():
        if isinstance(m, torch.nn.BatchNorm2d):
            bn_state = m.training
            m.eval()
    params = []
    for m in model.modules():
        if isinstance(m, model._conv_linear):
            # print(m)
            m.weight._first_grad = 0
            # Added the initailization here 0409
            m.weight._second_grad = 0
            m.weight.gftrv = 0
            params.append(m.weight)
    generator = torch.Generator(params[0].device).manual_seed(0)

    for idx, (data, target) in enumerate(tqdm.tqdm(train_loader)):
        data, target = data.to(device), target.to(device)
        output = model(data)
        loss = criterion(output, target)
        loss.backward(create_graph=True)
        ## record the gradient
        grads = []
        for p in params:
            if p.grad is not None:
                ## accumulate gradients and average across all batches
                p._first_grad += p.grad.data / len(train_loader)
                grads.append(p.grad)

        # compute second order gradient
        for _ in range(num_samples):
            zs = [
                torch.randint(0, 2, p.size(), generator=generator, device=p.device)
                * 2.0
                - 1.0
                for p in params
            ]  # Rademacher distribution {-1.0, 1.0}
            h_zs = torch.autograd.grad(
                grads,
                params,
                grad_outputs=zs,
                only_inputs=True,
                retain_graph=num_samples - 1,
            )
            for h_z, z, p in zip(h_zs, zs, params):
                ## accumulate second order gradients
                p._second_grad += h_z * z / (num_samples * len(train_loader))
        model.zero_grad()
        if idx == 3:
            break
        # Added mode identification for attackers and defenders
        if mode == "attacker":
            break
        elif mode == "defender":
            pass
        else:
            raise NotImplementedError("Not implemented")
    # print(params[0]._first_grad, params[0]._first_grad.shape)
    # print(params[0]._second_grad, params[0]._second_grad.shape)
    # print(params[0].shape)
    for m in model.modules():
        if isinstance(m, torch.nn.BatchNorm2d):
            m.train(bn_state)
    model.train(is_train)


if __name__ == "__main__":
    # test_crosstalk_simulator()
    # test_linear_feature_sampler()
    # test_conv2d_feature_sampler()
    # test_feedback_sampler()
    test_singularvalue_sampler()
