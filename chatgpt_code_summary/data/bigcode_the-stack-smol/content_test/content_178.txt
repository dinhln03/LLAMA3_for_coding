"""
"""
from __future__ import division
from torch.optim.optimizer import Optimizer, required

import numpy as np
import torch

from typing import NamedTuple, List
from dataclasses import dataclass
from enum import Enum
from typing import Union, Tuple
# from scipy.sparse.linalg import svds
from scipy.optimize import minimize_scalar


class LayerType(Enum):
    CONV = 1
    FC = 2
    NON_CONV = 3


@dataclass
class LayerMetrics:
    rank: float
    KG: float
    condition: float


@dataclass
class ConvLayerMetrics:
    input_channel: LayerMetrics
    output_channel: LayerMetrics


class LRMetrics(NamedTuple):
    rank_velocity: List[float]
    r_conv: List[float]


def EVBMF(Y, sigma2=None, H=None):
    """Implementation of the analytical solution to Empirical Variational
        Bayes Matrix Factorization.

    This function can be used to calculate the analytical solution to
        empirical VBMF.
    This is based on the paper and MatLab code by Nakajima et al.:
    "Global analytic solution of fully-observed variational Bayesian matrix
        factorization."

    Notes
    -----
        If sigma2 is unspecified, it is estimated by minimizing the free
            energy.
        If H is unspecified, it is set to the smallest of the sides of the
            input Y.

    Attributes
    ----------
    Y : numpy-array
        Input matrix that is to be factorized. Y has shape (L,M), where L<=M.

    sigma2 : int or None (default=None)
        Variance of the noise on Y.

    H : int or None (default = None)
        Maximum rank of the factorized matrices.

    Returns
    -------
    U : numpy-array
        Left-singular vectors.

    S : numpy-array
        Diagonal matrix of singular values.

    V : numpy-array
        Right-singular vectors.

    post : dictionary
        Dictionary containing the computed posterior values.


    References
    ----------
    .. [1] Nakajima, Shinichi, et al. "Global analytic solution of
        fully-observed variational Bayesian matrix factorization." Journal of
        Machine Learning Research 14.Jan (2013): 1-37.

    .. [2] Nakajima, Shinichi, et al. "Perfect dimensionality recovery by
        variational Bayesian PCA." Advances in Neural Information Processing
        Systems. 2012.
    """
    L, M = Y.shape  # has to be L<=M

    if H is None:
        H = L

    alpha = L / M
    tauubar = 2.5129 * np.sqrt(alpha)

    # SVD of the input matrix, max rank of H
    # U, s, V = np.linalg.svd(Y)
    U, s, V = torch.svd(Y)
    U = U[:, :H]
    s = s[:H]
    V = V[:H].T

    # Calculate residual
    residual = 0.
    if H < L:
        # residual = np.sum(np.sum(Y**2)-np.sum(s**2))
        residual = torch.sum(np.sum(Y**2) - np.sum(s**2))

    # Estimation of the variance when sigma2 is unspecified
    if sigma2 is None:
        xubar = (1 + tauubar) * (1 + alpha / tauubar)
        eH_ub = int(np.min([np.ceil(L / (1 + alpha)) - 1, H])) - 1
        # upper_bound = (np.sum(s**2)+residual)/(L*M)
        # lower_bound = np.max(
        #     [s[eH_ub+1]**2/(M*xubar), np.mean(s[eH_ub+1:]**2)/M])
        upper_bound = (torch.sum(s**2) + residual) / (L * M)
        lower_bound = torch.max(torch.stack(
            [s[eH_ub + 1]**2 / (M * xubar), torch.mean(s[eH_ub + 1:]**2) / M], dim=0))

        scale = 1.  # /lower_bound
        s = s * np.sqrt(scale)
        residual = residual * scale
        lower_bound = lower_bound * scale
        upper_bound = upper_bound * scale

        sigma2_opt = minimize_scalar(
            EVBsigma2, args=(L, M, s.cpu().numpy(), residual, xubar),
            bounds=[lower_bound.cpu().numpy(), upper_bound.cpu().numpy()],
            method='Bounded')
        sigma2 = sigma2_opt.x

    # Threshold gamma term
    threshold = np.sqrt(M * sigma2 * (1 + tauubar) * (1 + alpha / tauubar))
    # pos = np.sum(s > threshold)
    pos = torch.sum(s > threshold)

    # Formula (15) from [2]
    # d = torch.multiply(s[:pos]/2,
    #                    1-torch.divide(
    #                        torch.tensor((L+M)*sigma2, device=s.device),
    #     s[:pos]**2) + torch.sqrt((1-torch.divide(
    #         torch.tensor(
    #             (L+M)*sigma2, device=s.device),
    #         s[:pos]**2))**2 -
    #     4*L*M*sigma2**2/s[:pos]**4))
    # d = np.multiply(s[:pos]/2, 1-np.divide((L+M)*sigma2, s[:pos]**2) + np.sqrt(
    #     (1-np.divide((L+M)*sigma2, s[:pos]**2))**2 - 4*L*M*sigma2**2/s[:pos]**4))
    d = (s[:pos] / 2) * (1 - (L + M) * sigma2 / s[:pos]**2
                         + torch.sqrt((1 -
                                       (L + M) * sigma2 / s[:pos]**2)**2 - 4 * L * M * sigma2**2 / s[:pos]**4))

    # Computation of the posterior
    # post = {}
    # post['ma'] = np.zeros(H)
    # post['mb'] = np.zeros(H)
    # post['sa2'] = np.zeros(H)
    # post['sb2'] = np.zeros(H)
    # post['cacb'] = np.zeros(H)

    # tau = np.multiply(d, s[:pos])/(M*sigma2)
    # delta = np.multiply(np.sqrt(np.divide(M*d, L*s[:pos])), 1+alpha/tau)

    # post['ma'][:pos] = np.sqrt(np.multiply(d, delta))
    # post['mb'][:pos] = np.sqrt(np.divide(d, delta))
    # post['sa2'][:pos] = np.divide(sigma2*delta, s[:pos])
    # post['sb2'][:pos] = np.divide(sigma2, np.multiply(delta, s[:pos]))
    # post['cacb'][:pos] = np.sqrt(np.multiply(d, s[:pos])/(L*M))
    # post['sigma2'] = sigma2
    # post['F'] = 0.5*(L*M*np.log(2*np.pi*sigma2) +
    #                  (residual+np.sum(s**2))/sigma2 + np.sum(
    #                      M*np.log(tau+1) + L*np.log(tau/alpha + 1) - M*tau))

    return U[:, :pos], torch.diag(d), V[:, :pos]  # , post


def EVBsigma2(sigma2, L, M, s, residual, xubar):
    H = len(s)

    alpha = L / M
    x = s**2 / (M * sigma2)

    z1 = x[x > xubar]
    z2 = x[x <= xubar]
    tau_z1 = tau(z1, alpha)

    term1 = np.sum(z2 - np.log(z2))
    term2 = np.sum(z1 - tau_z1)
    term3 = np.sum(np.log(np.divide(tau_z1 + 1, z1)))
    term4 = alpha * np.sum(np.log(tau_z1 / alpha + 1))

    obj = term1 + term2 + term3 + term4 + residual / (M * sigma2) + (L - H) * np.log(sigma2)

    return obj


def phi0(x):
    return x - np.log(x)


def phi1(x, alpha):
    return np.log(tau(x, alpha) + 1) + alpha * np.log(tau(x, alpha) / alpha + 1
                                                      ) - tau(x, alpha)


def tau(x, alpha):
    return 0.5 * (x - (1 + alpha) + np.sqrt((x - (1 + alpha))**2 - 4 * alpha))


class Metrics:
    def __init__(self, params, linear: bool = False) -> None:
        '''
        parameters: list of torch.nn.Module.parameters()
        '''
        self.params = params
        self.history = list()
        mask = list()
        for param_idx, param in enumerate(params):
            param_shape = param.shape
            if not linear:
                if len(param_shape) != 4:
                    mask.append(param_idx)
            else:
                if len(param_shape) != 4 and len(param_shape) != 2:
                    mask.append(param_idx)
        self.mask = set(mask)

    def compute_low_rank(self,
                         tensor: torch.Tensor,
                         normalizer: float) -> torch.Tensor:
        if tensor.requires_grad:
            tensor = tensor.detach()
        try:
            tensor_size = tensor.shape
            if tensor_size[0] > tensor_size[1]:
                tensor = tensor.T
            U_approx, S_approx, V_approx = EVBMF(tensor)
        except RuntimeError:
            return None, None, None
        rank = S_approx.shape[0] / tensor_size[0]  # normalizer
        low_rank_eigen = torch.diag(S_approx).data.cpu().numpy()
        if len(low_rank_eigen) != 0:
            condition = low_rank_eigen[0] / low_rank_eigen[-1]
            sum_low_rank_eigen = low_rank_eigen / \
                max(low_rank_eigen)
            sum_low_rank_eigen = np.sum(sum_low_rank_eigen)
        else:
            condition = 0
            sum_low_rank_eigen = 0
        KG = sum_low_rank_eigen / tensor_size[0]  # normalizer
        return rank, KG, condition

    def KG(self, epoch: int) -> np.ndarray:
        KG_list = list()
        for i, (index, metric) in enumerate(self.history[epoch]):
            if isinstance(metric, ConvLayerMetrics):
                KG_list.append((metric.input_channel.KG
                                + metric.output_channel.KG) / 2)
            elif isinstance(metric, LayerMetrics):
                KG_list.append(metric.KG)
        return np.array(KG_list)

    def __call__(self) -> List[Tuple[int, Union[LayerMetrics,
                                                ConvLayerMetrics]]]:
        '''
        Computes the knowledge gain (S) and mapping condition (condition)
        '''
        metrics: List[Tuple[int, Union[LayerMetrics,
                                       ConvLayerMetrics]]] = list()
        for layer_index, layer in enumerate(self.params):
            if layer_index in self.mask:
                metrics.append((layer_index, None))
                continue
            # if np.less(np.prod(layer.shape), 10_000):
            #     metrics.append((layer_index, None))
            if len(layer.shape) == 4:
                layer_tensor = layer.data
                tensor_size = layer_tensor.shape
                mode_3_unfold = layer_tensor.permute(1, 0, 2, 3)
                mode_3_unfold = torch.reshape(
                    mode_3_unfold, [tensor_size[1], tensor_size[0]
                                    * tensor_size[2] * tensor_size[3]])
                mode_4_unfold = layer_tensor
                mode_4_unfold = torch.reshape(
                    mode_4_unfold, [tensor_size[0], tensor_size[1]
                                    * tensor_size[2] * tensor_size[3]])
                in_rank, in_KG, in_condition = self.compute_low_rank(
                    mode_3_unfold, tensor_size[1])
                if in_rank is None and in_KG is None and in_condition is None:
                    if len(self.history) > 0:
                        in_rank = self.history[-1][
                            layer_index][1].input_channel.rank
                        in_KG = self.history[-1][
                            layer_index][1].input_channel.KG
                        in_condition = self.history[-1][
                            layer_index][1].input_channel.condition
                    else:
                        in_rank = in_KG = in_condition = 0.
                out_rank, out_KG, out_condition = self.compute_low_rank(
                    mode_4_unfold, tensor_size[0])
                if out_rank is None and out_KG is None and out_condition is None:
                    if len(self.history) > 0:
                        out_rank = self.history[-1][
                            layer_index][1].output_channel.rank
                        out_KG = self.history[-1][
                            layer_index][1].output_channel.KG
                        out_condition = self.history[-1][
                            layer_index][1].output_channel.condition
                    else:
                        out_rank = out_KG = out_condition = 0.
                metrics.append((layer_index, ConvLayerMetrics(
                    input_channel=LayerMetrics(
                        rank=in_rank,
                        KG=in_KG,
                        condition=in_condition),
                    output_channel=LayerMetrics(
                        rank=out_rank,
                        KG=out_KG,
                        condition=out_condition))))
            elif len(layer.shape) == 2:
                rank, KG, condition = self.compute_low_rank(
                    layer, layer.shape[0])
                if rank is None and KG is None and condition is None:
                    if len(self.history) > 0:
                        rank = self.history[-1][layer_index][1].rank
                        KG = self.history[-1][layer_index][1].KG
                        condition = self.history[-1][layer_index][1].condition
                    else:
                        rank = KG = condition = 0.
                metrics.append((layer_index, LayerMetrics(
                    rank=rank,
                    KG=KG,
                    condition=condition)))
            else:
                metrics.append((layer_index, None))
        self.history.append(metrics)
        return metrics


class Adas(Optimizer):
    """
    Vectorized SGD from torch.optim.SGD
    """

    def __init__(self,
                 params,
                 lr: float = required,
                 beta: float = 0.8,
                 step_size: int = None,
                 linear: bool = True,
                 gamma: float = 1,
                 momentum: float = 0,
                 dampening: float = 0,
                 weight_decay: float = 0,
                 nesterov: bool = False):
        if lr is not required and lr < 0.0:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if momentum < 0.0:
            raise ValueError("Invalid momentum value: {}".format(momentum))
        if weight_decay < 0.0:
            raise ValueError(
                "Invalid weight_decay value: {}".format(weight_decay))

        defaults = dict(lr=lr, momentum=momentum, dampening=dampening,
                        weight_decay=weight_decay, nesterov=nesterov)
        if nesterov and (momentum <= 0 or dampening != 0):
            raise ValueError(
                "Nesterov momentum requires a momentum and zero dampening")
        super(Adas, self).__init__(params[:2], defaults)

        # Adas Specific stuff (not SGD)
        if np.less(beta, 0) or np.greater_equal(beta, 1):
            raise ValueError(f'Invalid beta: {beta}')
        if np.less(gamma, 0):
            raise ValueError(f'Invalid gamma: {gamma}')
        if step_size is not None:
            if np.less_equal(step_size, 0):
                raise ValueError(f'Invalid step_size: {step_size}')
        self.step_size = step_size
        self.gamma = gamma
        self.beta = beta
        self.metrics = metrics = Metrics(params=params[2]["all_params"], linear=linear)
        self.lr_vector = np.repeat(a=lr, repeats=len(metrics.params))
        self.velocity = np.zeros(
            len(self.metrics.params) - len(self.metrics.mask))
        self.not_ready = list(range(len(self.velocity)))
        self.init_lr = lr
        self.zeta = 1.
        self.KG = 0.

    def __setstate__(self, state):
        super(Adas, self).__setstate__(state)
        for group in self.param_groups:
            group.setdefault('nesterov', False)

    def epoch_step(self, epoch: int) -> None:
        self.metrics()
        if epoch == 0:
            velocity = self.init_lr * np.ones(len(self.velocity))
            self.KG = self.metrics.KG(epoch)
        else:
            KG = self.metrics.KG(epoch)
            velocity = KG - self.KG
            self.KG = KG
            for idx in self.not_ready:
                if np.isclose(KG[idx], 0.):
                    velocity[idx] = self.init_lr - \
                        self.beta * self.velocity[idx]
                else:
                    self.not_ready.remove(idx)

        if self.step_size is not None:
            if epoch % self.step_size == 0 and epoch > 0:
                self.lr_vector *= self.gamma
                self.zeta *= self.gamma

        self.velocity = np.maximum(
            self.beta * self.velocity + self.zeta * velocity, 0.)
        count = 0
        for i in range(len(self.metrics.params)):
            if i in self.metrics.mask:
                self.lr_vector[i] = self.lr_vector[i - (1 if i > 0 else 0)]
            else:
                self.lr_vector[i] = self.velocity[count]
                count += 1

    def step(self, closure: callable = None):
        """Performs a single optimization step.

        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        iteration_group = 0
        for group in self.param_groups:
            iteration_group += 1
            weight_decay = group['weight_decay']
            momentum = group['momentum']
            dampening = group['dampening']
            nesterov = group['nesterov']

            for p_index, p in enumerate(group['params']):
                if p.grad is None:
                    continue
                d_p = p.grad.data
                if weight_decay != 0:
                    d_p.add_(p.data, alpha=weight_decay)
                if momentum != 0:
                    param_state = self.state[p]
                    if 'momentum_buffer' not in param_state:
                        buf = param_state['momentum_buffer'] = torch.clone(
                            d_p).detach()
                    else:
                        buf = param_state['momentum_buffer']
                        buf.mul_(momentum).add_(d_p, alpha=1 - dampening)
                    if nesterov:
                        d_p = d_p.add(momentum, buf)
                    else:
                        d_p = buf

                # p.data.add_(-group['lr'], d_p)
                p.data.add_(d_p, alpha=-self.lr_vector[p_index])

        return loss
