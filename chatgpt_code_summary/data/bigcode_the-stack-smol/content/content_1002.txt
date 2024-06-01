from models.flowpp_cifar import CifarFlowPP
from models.rvae import RVAE
from models.modules import init_mode


model_registry = {
    # MNIST
    'rvae_mnist': lambda **kwargs: RVAE(z_size=16, h_size=40, kl_min=0.1,
                                        x_channels=1, **kwargs),

    # CIFAR
    'tiny_rvae': lambda **kwargs: RVAE(z_size=2, h_size=2, kl_min=0.1, **kwargs),
    'rvae': lambda **kwargs: RVAE(z_size=8, h_size=256, kl_min=0.1, **kwargs),

    'tiny_flow_pp': lambda **kwargs: CifarFlowPP(hdim=4, blocks=1, dequant_blocks=1, mix_components=1,
                                                 attn_version=False, force_float32_cond=False, **kwargs),
    'flow_pp': lambda **kwargs: CifarFlowPP(hdim=120, blocks=10, dequant_blocks=2, mix_components=8, attn_version=False,
                                            force_float32_cond=False, dropout=0.2, **kwargs),
    'flow_pp_wide': lambda **kwargs: CifarFlowPP(hdim=180, blocks=10, dequant_blocks=2, mix_components=8, attn_version=False,
                                                 force_float32_cond=False, dropout=0.2, **kwargs)

}
