import time

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.utils.data
from torch import optim
from torch.utils import data
from tqdm import tqdm

from networks import LATENT_CODE_SIZE, device
# from train_autoencoder import SDFSampleDataset, save_checkpoint, SIGMA

class SDFNet(nn.Module):
    def __init__(self, latent_code_size=LATENT_CODE_SIZE, dropout_prob=0.2, point_dim=3):
        super(SDFNet, self).__init__()
        SDF_NET_BREADTH = latent_code_size * 2
        # the decoder should only have xyz information, without sdf values
        self.layers1 = nn.Sequential(
            nn.utils.weight_norm(nn.Linear(point_dim + latent_code_size, SDF_NET_BREADTH)),
            nn.ReLU(),
            nn.Dropout(dropout_prob),

            nn.utils.weight_norm(nn.Linear(SDF_NET_BREADTH, SDF_NET_BREADTH)),
            nn.ReLU(),
            nn.Dropout(dropout_prob),

            nn.utils.weight_norm(nn.Linear(SDF_NET_BREADTH, SDF_NET_BREADTH)),
            nn.ReLU(),
            nn.Dropout(dropout_prob),

            nn.utils.weight_norm(nn.Linear(SDF_NET_BREADTH, SDF_NET_BREADTH - latent_code_size - point_dim)),
            nn.ReLU(),
            nn.Dropout(dropout_prob),
        )

        self.layers2 = nn.Sequential(
            nn.utils.weight_norm(nn.Linear(SDF_NET_BREADTH, SDF_NET_BREADTH)),
            nn.ReLU(),
            nn.Dropout(dropout_prob),

            nn.utils.weight_norm(nn.Linear(SDF_NET_BREADTH, SDF_NET_BREADTH)),
            nn.ReLU(),
            nn.Dropout(dropout_prob),

            nn.utils.weight_norm(nn.Linear(SDF_NET_BREADTH, SDF_NET_BREADTH)),
            nn.ReLU(),
            nn.Dropout(dropout_prob),

            nn.utils.weight_norm(nn.Linear(SDF_NET_BREADTH, SDF_NET_BREADTH)),
            nn.ReLU(),
            nn.Dropout(dropout_prob),

            nn.Linear(SDF_NET_BREADTH, 1),
            nn.Tanh()
        )

    def forward(self, input):
        """
        input: [B, N, latent_size + point_dim]
        :param latent_codes: [B, N, LATENT_CODE_DIM]
        :param points: [B, N, 3]
        :return: sdf_pred: [B, N]
        """
        x = self.layers1(input)
        x = torch.cat((x, input), dim=-1)
        x = self.layers2(x)
        return x

# if __name__ == '__main__':
#     experiment = '5_samples_latent_128_no_reg'
#     num_epochs = 500
#
#     decoder = SDFNet()
#
#     optimiser = optim.Adam(decoder.parameters(), lr=1e-5)
#     # model, optimiser, start_epoch, training_loss = load_or_init_model(experiment)
#     dataset = SDFSampleDataset('data/SdfSamples/ShapeNetV2/03001627/', '5_sample.json')
#     batch_size = 5
#     normal_distribution = torch.distributions.normal.Normal(0, 0.0001)
#     latent_codes = normal_distribution.sample((MODEL_COUNT, LATENT_CODE_SIZE)).to(device)
#     latent_codes.requires_grad = True
#     train_data = data.DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4)
#
#     # training loop starts
#     training_loss = []
#     for epoch in range(1, num_epochs + 1):
#         start_time = time.time()
#         running_loss = []
#
#         for i_batch, batch in tqdm(enumerate(train_data)):
#             optimiser.zero_grad()
#             batch = batch.to(device)  # [B, point_dim, N]
#             sdf_pred, input_trans, latent_code = model(batch)
#             sdf_gt = batch[:, -1, :].squeeze()
#
#             loss = l1_loss(sdf_gt, sdf_pred) # TODO: experiment with only the l1 loss
#             loss += SIGMA**2 * min(1, epoch / 100) * torch.mean(torch.norm(latent_code, dim=1))
#             loss.backward()
#             optimiser.step()
#             running_loss.append(loss.item())
#
#         epoch_duration = time.time() - start_time
#         epoch_loss = np.mean(running_loss)
#         training_loss.append(epoch_loss)
#
#         print("Epoch {:d}, {:.1f}s. Loss: {:.8f}".format(epoch, epoch_duration, epoch_loss))
#
#         if epoch_loss < 0.02:
#             save_checkpoint(epoch, model, optimiser, training_loss, experiment, filename='sub002')
#
#         # always save the latest snapshot
#         save_checkpoint(epoch, model, optimiser, training_loss, experiment)
#         if epoch % 100 == 0:
#             save_checkpoint(epoch, model, optimiser, training_loss, experiment, filename=str(epoch))