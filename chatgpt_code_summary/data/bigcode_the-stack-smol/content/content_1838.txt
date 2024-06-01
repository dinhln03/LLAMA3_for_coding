import random
import torch
from torch.utils.tensorboard import SummaryWriter
from flowtron_plotting_utils import plot_alignment_to_numpy
from flowtron_plotting_utils import plot_gate_outputs_to_numpy


class FlowtronLogger(SummaryWriter):
    def __init__(self, logdir):
        super(FlowtronLogger, self).__init__(logdir)

    def log_training(self, loss, learning_rate, iteration):
            self.add_scalar("training/loss", loss, iteration)
            self.add_scalar("learning_rate", learning_rate, iteration)

    def log_validation(self, loss, loss_nll, loss_gate, attns, gate_pred,
                       gate_out, iteration):
        self.add_scalar("validation/loss", loss, iteration)
        self.add_scalar("validation/loss_nll", loss_nll, iteration)
        self.add_scalar("validation/loss_gate", loss_gate, iteration)

        # batch里随机抽一条看看效果
        idx = random.randint(0, len(gate_out) - 1)
        for i in range(len(attns)):
            self.add_image(
                'attention_weights_{}'.format(i),
                plot_alignment_to_numpy(attns[i][idx].data.cpu().numpy().T),
                iteration,
                dataformats='HWC')

        if gate_pred is not None:
            gate_pred = gate_pred.transpose(0, 1)[:, :, 0]
            self.add_image(
                "gate",
                plot_gate_outputs_to_numpy(
                    gate_out[idx].data.cpu().numpy(),
                    torch.sigmoid(gate_pred[idx]).data.cpu().numpy()),
                iteration, dataformats='HWC')
