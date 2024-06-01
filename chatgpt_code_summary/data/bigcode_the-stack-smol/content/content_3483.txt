import os
import numpy as np
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import argparse
import cv2

import config
from utils import Mesh
from models import CMR
from models.smpl_from_lib import SMPL
from utils.pose_utils import compute_similarity_transform_batch, \
    scale_and_translation_transform_batch
from utils.cam_utils import orthographic_project_torch, undo_keypoint_normalisation
from datasets.my_3dpw_eval_dataset import PW3DEvalDataset


def evaluate_3dpw(model,
                  eval_dataset,
                  metrics,
                  device,
                  vis_save_path,
                  num_workers=4,
                  pin_memory=True,
                  vis_every_n_batches=1000):

    eval_dataloader = DataLoader(eval_dataset,
                                 batch_size=1,
                                 shuffle=False,
                                 drop_last=True,
                                 num_workers=num_workers,
                                 pin_memory=pin_memory)

    smpl = SMPL(config.SMPL_MODEL_DIR, batch_size=1)
    smpl_male = SMPL(config.SMPL_MODEL_DIR, batch_size=1, gender='male')
    smpl_female = SMPL(config.SMPL_MODEL_DIR, batch_size=1, gender='female')
    smpl.to(device)
    smpl_male.to(device)
    smpl_female.to(device)

    J_regressor = torch.from_numpy(np.load(config.JOINT_REGRESSOR_H36M)).float()
    J_regressor_batch = J_regressor[None, :].to(device)

    if 'pve' in metrics:
        pve_smpl_sum = 0.0
        pve_graph_sum = 0.0
        pve_smpl_per_frame = []
        pve_graph_per_frame = []

    if 'pve_scale_corrected' in metrics:
        pve_scale_corrected_smpl_sum = 0.0
        pve_scale_corrected_graph_sum = 0.0
        pve_scale_corrected_smpl_per_frame = []
        pve_scale_corrected_graph_per_frame = []

    if 'pve_pa' in metrics:
        pve_pa_smpl_sum = 0.0
        pve_pa_graph_sum = 0.0
        pve_pa_smpl_per_frame = []
        pve_pa_graph_per_frame = []

    if 'pve-t' in metrics:
        pvet_sum = 0.0
        pvet_per_frame = []

    if 'pve-t_scale_corrected' in metrics:
        pvet_scale_corrected_sum = 0.0
        pvet_scale_corrected_per_frame = []

    if 'mpjpe' in metrics:
        mpjpe_smpl_sum = 0.0
        mpjpe_graph_sum = 0.0
        mpjpe_smpl_per_frame = []
        mpjpe_graph_per_frame = []

    if 'mpjpe_scale_corrected' in metrics:
        mpjpe_scale_corrected_smpl_sum = 0.0
        mpjpe_scale_corrected_graph_sum = 0.0
        mpjpe_scale_corrected_smpl_per_frame = []
        mpjpe_scale_corrected_graph_per_frame = []

    if 'j3d_rec_err' in metrics:
        j3d_rec_err_smpl_sum = 0.0
        j3d_rec_err_graph_sum = 0.0
        j3d_rec_err_smpl_per_frame = []
        j3d_rec_err_graph_per_frame = []

    if 'pve_2d' in metrics:
        pve_2d_smpl_sum = 0.0
        pve_2d_graph_sum = 0.0

    if 'pve_2d_scale_corrected' in metrics:
        pve_2d_scale_corrected_smpl_sum = 0.0
        pve_2d_scale_corrected_graph_sum = 0.0

    if 'pve_2d_pa' in metrics:
        pve_2d_pa_smpl_sum = 0.0
        pve_2d_pa_graph_sum = 0.0

    num_samples = 0
    num_vertices = 6890
    num_joints3d = 14

    model.eval()
    for batch_num, samples_batch in enumerate(tqdm(eval_dataloader)):
        # ------------------------------- TARGETS and INPUTS -------------------------------
        input = samples_batch['input']
        input = input.to(device)

        target_pose = samples_batch['pose'].to(device)
        target_shape = samples_batch['shape'].to(device)
        target_gender = samples_batch['gender'][0]

        if target_gender == 'm':
            target_smpl_output = smpl_male(body_pose=target_pose[:, 3:],
                                           global_orient=target_pose[:, :3],
                                           betas=target_shape)
            target_vertices = target_smpl_output.vertices
            target_reposed_smpl_output = smpl_male(betas=target_shape)
            target_reposed_vertices = target_reposed_smpl_output.vertices
            target_joints_h36m = torch.matmul(J_regressor_batch, target_vertices)
            target_joints_h36mlsp = target_joints_h36m[:, config.H36M_TO_J14, :]
        elif target_gender == 'f':
            target_smpl_output = smpl_female(body_pose=target_pose[:, 3:],
                                             global_orient=target_pose[:, :3],
                                             betas=target_shape)
            target_vertices = target_smpl_output.vertices
            target_reposed_smpl_output = smpl_female(betas=target_shape)
            target_reposed_vertices = target_reposed_smpl_output.vertices
            target_joints_h36m = torch.matmul(J_regressor_batch, target_vertices)
            target_joints_h36mlsp = target_joints_h36m[:, config.H36M_TO_J14, :]

        # ------------------------------- PREDICTIONS -------------------------------
        pred_vertices, pred_vertices_smpl, pred_camera, pred_rotmat, pred_betas = model(input)
        pred_vertices_projected2d = orthographic_project_torch(pred_vertices, pred_camera)
        pred_vertices_projected2d = undo_keypoint_normalisation(pred_vertices_projected2d, input.shape[-1])
        pred_vertices_smpl_projected2d = orthographic_project_torch(pred_vertices_smpl, pred_camera)
        pred_vertices_smpl_projected2d = undo_keypoint_normalisation(pred_vertices_smpl_projected2d, input.shape[-1])
        pred_reposed_smpl_output = smpl(betas=pred_betas)
        pred_reposed_vertices = pred_reposed_smpl_output.vertices

        pred_joints_h36m = torch.matmul(J_regressor_batch, pred_vertices)
        pred_joints_h36mlsp = pred_joints_h36m[:, config.H36M_TO_J14, :]

        pred_joints_smpl_h36m = torch.matmul(J_regressor_batch, pred_vertices_smpl)
        pred_joints_smpl_h36mlsp = pred_joints_smpl_h36m[:, config.H36M_TO_J14, :]

        # Numpy-fying
        target_vertices = target_vertices.cpu().detach().numpy()
        target_reposed_vertices = target_reposed_vertices.cpu().detach().numpy()
        target_joints_h36mlsp = target_joints_h36mlsp.cpu().detach().numpy()

        pred_vertices = pred_vertices.cpu().detach().numpy()
        pred_vertices_smpl = pred_vertices_smpl.cpu().detach().numpy()
        pred_vertices_projected2d = pred_vertices_projected2d.cpu().detach().numpy()
        pred_vertices_smpl_projected2d = pred_vertices_smpl_projected2d.cpu().detach().numpy()
        pred_reposed_vertices = pred_reposed_vertices.cpu().detach().numpy()
        pred_joints_h36mlsp = pred_joints_h36mlsp.cpu().detach().numpy()
        pred_joints_smpl_h36mlsp = pred_joints_smpl_h36mlsp.cpu().detach().numpy()

        # ------------------------------- METRICS -------------------------------

        if 'pve' in metrics:
            pve_smpl_batch = np.linalg.norm(pred_vertices_smpl - target_vertices, axis=-1)  # (1, 6890)
            pve_graph_batch = np.linalg.norm(pred_vertices - target_vertices, axis=-1)
            pve_smpl_sum += np.sum(pve_smpl_batch)  # scalar
            pve_graph_sum += np.sum(pve_graph_batch)
            pve_smpl_per_frame.append(np.mean(pve_smpl_batch, axis=-1))
            pve_graph_per_frame.append(np.mean(pve_graph_batch, axis=-1))

        # Scale and translation correction
        if 'pve_scale_corrected' in metrics:
            pred_vertices_smpl_sc = scale_and_translation_transform_batch(pred_vertices_smpl,
                                                                          target_vertices)
            pred_vertices_sc = scale_and_translation_transform_batch(pred_vertices,
                                                                     target_vertices)
            pve_sc_smpl_batch = np.linalg.norm(pred_vertices_smpl_sc - target_vertices,
                                               axis=-1)  # (1, 6890)
            pve_sc_graph_batch = np.linalg.norm(pred_vertices_sc - target_vertices,
                                                axis=-1)  # (1, 6890)
            pve_scale_corrected_smpl_sum += np.sum(pve_sc_smpl_batch)  # scalar
            pve_scale_corrected_graph_sum += np.sum(pve_sc_graph_batch)  # scalar
            pve_scale_corrected_smpl_per_frame.append(np.mean(pve_sc_smpl_batch, axis=-1))
            pve_scale_corrected_graph_per_frame.append(np.mean(pve_sc_graph_batch, axis=-1))

        # Procrustes analysis
        if 'pve_pa' in metrics:
            pred_vertices_smpl_pa = compute_similarity_transform_batch(pred_vertices_smpl, target_vertices)
            pred_vertices_pa = compute_similarity_transform_batch(pred_vertices, target_vertices)
            pve_pa_smpl_batch = np.linalg.norm(pred_vertices_smpl_pa - target_vertices, axis=-1)  # (1, 6890)
            pve_pa_graph_batch = np.linalg.norm(pred_vertices_pa - target_vertices, axis=-1)  # (1, 6890)
            pve_pa_smpl_sum += np.sum(pve_pa_smpl_batch)  # scalar
            pve_pa_graph_sum += np.sum(pve_pa_graph_batch)  # scalar
            pve_pa_smpl_per_frame.append(np.mean(pve_pa_smpl_batch, axis=-1))
            pve_pa_graph_per_frame.append(np.mean(pve_pa_graph_batch, axis=-1))

        if 'pve-t' in metrics:
            pvet_batch = np.linalg.norm(pred_reposed_vertices - target_reposed_vertices, axis=-1)
            pvet_sum += np.sum(pvet_batch)
            pvet_per_frame.append(np.mean(pvet_batch, axis=-1))

        # Scale and translation correction
        if 'pve-t_scale_corrected' in metrics:
            pred_reposed_vertices_sc = scale_and_translation_transform_batch(pred_reposed_vertices,
                                                                             target_reposed_vertices)
            pvet_scale_corrected_batch = np.linalg.norm(pred_reposed_vertices_sc - target_reposed_vertices,
                                                        axis=-1)  # (bs, 6890)
            pvet_scale_corrected_sum += np.sum(pvet_scale_corrected_batch)  # scalar
            pvet_scale_corrected_per_frame.append(np.mean(pvet_scale_corrected_batch, axis=-1))

        if 'mpjpe' in metrics:
            mpjpe_smpl_batch = np.linalg.norm(pred_joints_smpl_h36mlsp - target_joints_h36mlsp, axis=-1)  # (bs, 14)
            mpjpe_graph_batch = np.linalg.norm(pred_joints_h36mlsp - target_joints_h36mlsp, axis=-1)  # (bs, 14)
            mpjpe_smpl_sum += np.sum(mpjpe_smpl_batch)
            mpjpe_graph_sum += np.sum(mpjpe_graph_batch)
            mpjpe_smpl_per_frame.append(np.mean(mpjpe_smpl_batch, axis=-1))
            mpjpe_graph_per_frame.append(np.mean(mpjpe_graph_batch, axis=-1))

        # Scale and translation correction
        if 'mpjpe_scale_corrected' in metrics:
            pred_joints_smpl_h36mlsp_sc = scale_and_translation_transform_batch(pred_joints_smpl_h36mlsp,
                                                                                target_joints_h36mlsp)
            pred_joints_h36mlsp_sc = scale_and_translation_transform_batch(pred_joints_h36mlsp,
                                                                           target_joints_h36mlsp)
            mpjpe_scale_corrected_smpl_batch = np.linalg.norm(pred_joints_smpl_h36mlsp_sc - target_joints_h36mlsp,
                                                              axis=-1)  # (bs, 14)
            mpjpe_scale_corrected_graph_batch = np.linalg.norm(pred_joints_h36mlsp_sc - target_joints_h36mlsp,
                                                               axis=-1)  # (bs, 14)
            mpjpe_scale_corrected_smpl_sum += np.sum(mpjpe_scale_corrected_smpl_batch)
            mpjpe_scale_corrected_graph_sum += np.sum(mpjpe_scale_corrected_graph_batch)
            mpjpe_scale_corrected_smpl_per_frame.append(np.mean(mpjpe_scale_corrected_smpl_batch, axis=-1))
            mpjpe_scale_corrected_graph_per_frame.append(np.mean(mpjpe_scale_corrected_graph_batch, axis=-1))

        # Procrustes analysis
        if 'j3d_rec_err' in metrics:
            pred_joints_smpl_h36mlsp_pa = compute_similarity_transform_batch(pred_joints_smpl_h36mlsp,
                                                                             target_joints_h36mlsp)
            pred_joints_h36mlsp_pa = compute_similarity_transform_batch(pred_joints_h36mlsp, target_joints_h36mlsp)
            j3d_rec_err_smpl_batch = np.linalg.norm(pred_joints_smpl_h36mlsp_pa - target_joints_h36mlsp, axis=-1)  # (bs, 14)
            j3d_rec_err_graph_batch = np.linalg.norm(pred_joints_h36mlsp_pa - target_joints_h36mlsp, axis=-1)  # (bs, 14)
            j3d_rec_err_smpl_sum += np.sum(j3d_rec_err_smpl_batch)
            j3d_rec_err_graph_sum += np.sum(j3d_rec_err_graph_batch)
            j3d_rec_err_smpl_per_frame.append(np.mean(j3d_rec_err_smpl_batch, axis=-1))
            j3d_rec_err_graph_per_frame.append(np.mean(j3d_rec_err_graph_batch, axis=-1))

        if 'pve_2d' in metrics:
            pred_vertices_smpl_2d = pred_vertices_smpl[:, :, :2]
            pred_vertices_2d = pred_vertices[:, :, :2]
            target_vertices_2d = target_vertices[:, :, :2]
            pve_2d_smpl_batch = np.linalg.norm(pred_vertices_smpl_2d - target_vertices_2d, axis=-1)  # (bs, 6890)
            pve_2d_graph_batch = np.linalg.norm(pred_vertices_2d - target_vertices_2d, axis=-1)  # (bs, 6890)
            pve_2d_smpl_sum += np.sum(pve_2d_smpl_batch)
            pve_2d_graph_sum += np.sum(pve_2d_graph_batch)

        # Scale and translation correction
        if 'pve_2d_scale_corrected' in metrics:
            pred_vertices_smpl_sc = scale_and_translation_transform_batch(pred_vertices_smpl,
                                                                          target_vertices)
            pred_vertices_sc = scale_and_translation_transform_batch(pred_vertices,
                                                                     target_vertices)
            pred_vertices_smpl_2d_sc = pred_vertices_smpl_sc[:, :, :2]
            pred_vertices_2d_sc = pred_vertices_sc[:, :, :2]
            target_vertices_2d = target_vertices[:, :, :2]
            pve_2d_sc_smpl_batch = np.linalg.norm(pred_vertices_smpl_2d_sc - target_vertices_2d,
                                                  axis=-1)  # (bs, 6890)
            pve_2d_sc_graph_batch = np.linalg.norm(pred_vertices_2d_sc - target_vertices_2d,
                                                   axis=-1)  # (bs, 6890)
            pve_2d_scale_corrected_smpl_sum += np.sum(pve_2d_sc_smpl_batch)
            pve_2d_scale_corrected_graph_sum += np.sum(pve_2d_sc_graph_batch)

        # Procrustes analysis
        if 'pve_2d_pa' in metrics:
            pred_vertices_smpl_pa = compute_similarity_transform_batch(pred_vertices_smpl, target_vertices)
            pred_vertices_pa = compute_similarity_transform_batch(pred_vertices, target_vertices)
            pred_vertices_smpl_2d_pa = pred_vertices_smpl_pa[:, :, :2]
            pred_vertices_2d_pa = pred_vertices_pa[:, :, :2]
            target_vertices_2d = target_vertices[:, :, :2]
            pve_2d_pa_smpl_batch = np.linalg.norm(pred_vertices_smpl_2d_pa - target_vertices_2d, axis=-1)  # (bs, 6890)
            pve_2d_pa_graph_batch = np.linalg.norm(pred_vertices_2d_pa - target_vertices_2d, axis=-1)  # (bs, 6890)
            pve_2d_pa_smpl_sum += np.sum(pve_2d_pa_smpl_batch)
            pve_2d_pa_graph_sum += np.sum(pve_2d_pa_graph_batch)

        num_samples += target_pose.shape[0]

        # ------------------------------- VISUALISE -------------------------------
        if vis_every_n_batches is not None:
            if batch_num % vis_every_n_batches == 0:
                vis_imgs = samples_batch['vis_img'].numpy()
                vis_imgs = np.transpose(vis_imgs, [0, 2, 3, 1])

                fnames = samples_batch['fname']

                plt.figure(figsize=(16, 12))
                plt.subplot(341)
                plt.imshow(vis_imgs[0])

                plt.subplot(342)
                plt.imshow(vis_imgs[0])
                plt.scatter(pred_vertices_projected2d[0, :, 0], pred_vertices_projected2d[0, :, 1], s=0.1, c='r')

                plt.subplot(343)
                plt.imshow(vis_imgs[0])
                plt.scatter(pred_vertices_smpl_projected2d[0, :, 0], pred_vertices_smpl_projected2d[0, :, 1], s=0.1, c='r')

                plt.subplot(345)
                plt.scatter(target_vertices[0, :, 0], target_vertices[0, :, 1], s=0.1, c='b')
                plt.scatter(pred_vertices[0, :, 0], pred_vertices[0, :, 1], s=0.1, c='r')
                plt.gca().invert_yaxis()
                plt.gca().set_aspect('equal', adjustable='box')

                plt.subplot(346)
                plt.scatter(target_vertices[0, :, 0], target_vertices[0, :, 1], s=0.1, c='b')
                plt.scatter(pred_vertices_smpl[0, :, 0], pred_vertices_smpl[0, :, 1], s=0.1, c='r')
                plt.gca().invert_yaxis()
                plt.gca().set_aspect('equal', adjustable='box')

                plt.subplot(347)
                plt.scatter(target_vertices[0, :, 0], target_vertices[0, :, 1], s=0.1, c='b')
                plt.scatter(pred_vertices_pa[0, :, 0], pred_vertices_pa[0, :, 1], s=0.1, c='r')
                plt.gca().invert_yaxis()
                plt.gca().set_aspect('equal', adjustable='box')

                plt.subplot(348)
                plt.scatter(target_vertices[0, :, 0], target_vertices[0, :, 1], s=0.1, c='b')
                plt.scatter(pred_vertices_smpl_pa[0, :, 0], pred_vertices_smpl_pa[0, :, 1], s=0.1, c='r')
                plt.gca().invert_yaxis()
                plt.gca().set_aspect('equal', adjustable='box')

                plt.subplot(349)
                plt.scatter(target_reposed_vertices[0, :, 0], target_reposed_vertices[0, :, 1], s=0.1, c='b')
                plt.scatter(pred_reposed_vertices_sc[0, :, 0], pred_reposed_vertices_sc[0, :, 1], s=0.1, c='r')
                plt.gca().set_aspect('equal', adjustable='box')

                plt.subplot(3, 4, 10)
                for j in range(num_joints3d):
                    plt.scatter(pred_joints_h36mlsp[0, j, 0], pred_joints_h36mlsp[0, j, 1], c='r')
                    plt.scatter(target_joints_h36mlsp[0, j, 0], target_joints_h36mlsp[0, j, 1], c='b')
                    plt.text(pred_joints_h36mlsp[0, j, 0], pred_joints_h36mlsp[0, j, 1], s=str(j))
                    plt.text(target_joints_h36mlsp[0, j, 0], target_joints_h36mlsp[0, j, 1], s=str(j))
                plt.gca().invert_yaxis()
                plt.gca().set_aspect('equal', adjustable='box')

                plt.subplot(3, 4, 11)
                for j in range(num_joints3d):
                    plt.scatter(pred_joints_h36mlsp_pa[0, j, 0], pred_joints_h36mlsp_pa[0, j, 1], c='r')
                    plt.scatter(target_joints_h36mlsp[0, j, 0], target_joints_h36mlsp[0, j, 1], c='b')
                    plt.text(pred_joints_h36mlsp_pa[0, j, 0], pred_joints_h36mlsp_pa[0, j, 1], s=str(j))
                    plt.text(target_joints_h36mlsp[0, j, 0], target_joints_h36mlsp[0, j, 1], s=str(j))
                plt.gca().invert_yaxis()
                plt.gca().set_aspect('equal', adjustable='box')

                plt.subplot(3, 4, 12)
                for j in range(num_joints3d):
                    plt.scatter(pred_joints_smpl_h36mlsp_pa[0, j, 0], pred_joints_smpl_h36mlsp_pa[0, j, 1], c='r')
                    plt.scatter(target_joints_h36mlsp[0, j, 0], target_joints_h36mlsp[0, j, 1], c='b')
                    plt.text(pred_joints_smpl_h36mlsp_pa[0, j, 0], pred_joints_smpl_h36mlsp_pa[0, j, 1], s=str(j))
                    plt.text(target_joints_h36mlsp[0, j, 0], target_joints_h36mlsp[0, j, 1], s=str(j))
                plt.gca().invert_yaxis()
                plt.gca().set_aspect('equal', adjustable='box')

                # plt.show()
                save_fig_path = os.path.join(vis_save_path, fnames[0])
                plt.savefig(save_fig_path, bbox_inches='tight')
                plt.close()

    if 'pve' in metrics:
        pve_smpl = pve_smpl_sum / (num_samples * num_vertices)
        print('PVE SMPL: {:.5f}'.format(pve_smpl))
        pve_graph = pve_graph_sum / (num_samples * num_vertices)
        print('PVE GRAPH: {:.5f}'.format(pve_graph))
        pve_smpl_per_frame = np.concatenate(pve_smpl_per_frame, axis=0)
        pve_graph_per_frame = np.concatenate(pve_graph_per_frame, axis=0)
        np.save(os.path.join(save_path, 'pve_per_frame.npy'), pve_smpl_per_frame)
        np.save(os.path.join(save_path, 'pve_graph_per_frame.npy'), pve_graph_per_frame)

    if 'pve_scale_corrected' in metrics:
        pve_sc_smpl = pve_scale_corrected_smpl_sum / (num_samples * num_vertices)
        print('PVE SC SMPL: {:.5f}'.format(pve_sc_smpl))
        pve_sc_graph = pve_scale_corrected_graph_sum / (num_samples * num_vertices)
        print('PVE SC GRAPH: {:.5f}'.format(pve_sc_graph))
        pve_scale_corrected_smpl_per_frame = np.concatenate(pve_scale_corrected_smpl_per_frame, axis=0)
        pve_scale_corrected_graph_per_frame = np.concatenate(pve_scale_corrected_graph_per_frame, axis=0)
        np.save(os.path.join(save_path, 'pve_scale_corrected_per_frame.npy'),
                pve_scale_corrected_smpl_per_frame)
        np.save(os.path.join(save_path, 'pve_scale_corrected_graph_per_frame.npy'),
                pve_scale_corrected_graph_per_frame)

    if 'pve_pa' in metrics:
        pve_pa_smpl = pve_pa_smpl_sum / (num_samples * num_vertices)
        print('PVE PA SMPL: {:.5f}'.format(pve_pa_smpl))
        pve_pa_graph = pve_pa_graph_sum / (num_samples * num_vertices)
        print('PVE PA GRAPH: {:.5f}'.format(pve_pa_graph))
        pve_pa_smpl_per_frame = np.concatenate(pve_pa_smpl_per_frame, axis=0)
        pve_pa_graph_per_frame = np.concatenate(pve_pa_graph_per_frame, axis=0)
        np.save(os.path.join(save_path, 'pve_pa_per_frame.npy'), pve_pa_smpl_per_frame)
        np.save(os.path.join(save_path, 'pve_pa_graph_per_frame.npy'), pve_pa_graph_per_frame)

    if 'pve-t' in metrics:
        pvet = pvet_sum / (num_samples * num_vertices)
        print('PVE-T: {:.5f}'.format(pvet))
        pvet_per_frame = np.concatenate(pvet_per_frame, axis=0)
        np.save(os.path.join(save_path, 'pvet_per_frame.npy'), pvet_per_frame)

    if 'pve-t_scale_corrected' in metrics:
        pvet_sc = pvet_scale_corrected_sum / (num_samples * num_vertices)
        print('PVE-T SC: {:.5f}'.format(pvet_sc))
        pvet_scale_corrected_per_frame = np.concatenate(pvet_scale_corrected_per_frame, axis=0)
        np.save(os.path.join(save_path, 'pvet_scale_corrected_per_frame.npy'),
                pvet_scale_corrected_per_frame)

    if 'mpjpe' in metrics:
        mpjpe_smpl = mpjpe_smpl_sum / (num_samples * num_joints3d)
        print('MPJPE SMPL: {:.5f}'.format(mpjpe_smpl))
        mpjpe_graph = mpjpe_graph_sum / (num_samples * num_joints3d)
        print('MPJPE GRAPH: {:.5f}'.format(mpjpe_graph))
        mpjpe_smpl_per_frame = np.concatenate(mpjpe_smpl_per_frame, axis=0)
        mpjpe_graph_per_frame = np.concatenate(mpjpe_graph_per_frame, axis=0)
        np.save(os.path.join(save_path, 'mpjpe_per_frame.npy'), mpjpe_smpl_per_frame)
        np.save(os.path.join(save_path, 'mpjpe_graph_per_frame.npy'), mpjpe_graph_per_frame)

    if 'mpjpe_scale_corrected' in metrics:
        mpjpe_sc_smpl = mpjpe_scale_corrected_smpl_sum / (num_samples * num_joints3d)
        print('MPJPE SC SMPL: {:.5f}'.format(mpjpe_sc_smpl))
        mpjpe_sc_graph = mpjpe_scale_corrected_graph_sum / (num_samples * num_joints3d)
        print('MPJPE SC GRAPH: {:.5f}'.format(mpjpe_sc_graph))
        mpjpe_scale_corrected_smpl_per_frame = np.concatenate(
            mpjpe_scale_corrected_smpl_per_frame, axis=0)
        mpjpe_scale_corrected_graph_per_frame = np.concatenate(
            mpjpe_scale_corrected_graph_per_frame, axis=0)
        np.save(os.path.join(save_path, 'mpjpe_scale_corrected_per_frame.npy'),
                mpjpe_scale_corrected_smpl_per_frame)
        np.save(os.path.join(save_path, 'mpjpe_scale_corrected_graph_per_frame.npy'),
                mpjpe_scale_corrected_graph_per_frame)

    if 'j3d_rec_err' in metrics:
        j3d_rec_err_smpl = j3d_rec_err_smpl_sum / (num_samples * num_joints3d)
        print('Rec Err SMPL: {:.5f}'.format(j3d_rec_err_smpl))
        j3d_rec_err_graph = j3d_rec_err_graph_sum / (num_samples * num_joints3d)
        print('Rec Err GRAPH: {:.5f}'.format(j3d_rec_err_graph))
        j3d_rec_err_smpl_per_frame = np.concatenate(j3d_rec_err_smpl_per_frame, axis=0)
        j3d_rec_err_graph_per_frame = np.concatenate(j3d_rec_err_graph_per_frame, axis=0)
        np.save(os.path.join(save_path, 'j3d_rec_err_per_frame.npy'),
                j3d_rec_err_smpl_per_frame)
        np.save(os.path.join(save_path, 'j3d_rec_err_graph_per_frame.npy'),
                j3d_rec_err_graph_per_frame)

    if 'pve_2d' in metrics:
        pve_2d_smpl = pve_2d_smpl_sum / (num_samples * num_vertices)
        print('PVE 2D SMPL: {:.5f}'.format(pve_2d_smpl))
        pve_2d_graph = pve_2d_graph_sum / (num_samples * num_vertices)
        print('PVE 2D GRAPH: {:.5f}'.format(pve_2d_graph))

    if 'pve_2d_scale_corrected' in metrics:
        pve_2d_sc_smpl = pve_2d_scale_corrected_smpl_sum / (num_samples * num_vertices)
        print('PVE 2D SC SMPL: {:.5f}'.format(pve_2d_sc_smpl))
        pve_2d_sc_graph = pve_2d_scale_corrected_graph_sum / (num_samples * num_vertices)
        print('PVE 2D SC GRAPH: {:.5f}'.format(pve_2d_sc_graph))

    if 'pve_2d_pa' in metrics:
        pve_2d_pa_smpl = pve_2d_pa_smpl_sum / (num_samples * num_vertices)
        print('PVE 2D PA SMPL: {:.5f}'.format(pve_2d_pa_smpl))
        pve_2d_pa_graph = pve_2d_pa_graph_sum / (num_samples * num_vertices)
        print('PVE 2D PA GRAPH: {:.5f}'.format(pve_2d_pa_graph))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', default=None, help='Path to network checkpoint')
    parser.add_argument('--gpu', default="0", type=str, help='GPU')
    args = parser.parse_args()

    # Device
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # see issue #152
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    # Load model
    mesh = Mesh(device=device)
    # Our pretrained networks have 5 residual blocks with 256 channels.
    # You might want to change this if you use a different architecture.
    model = CMR(mesh, 5, 256, pretrained_checkpoint=args.checkpoint, device=device)
    model.to(device)
    model.eval()

    # Setup evaluation dataset
    dataset_path = '/scratch2/as2562/datasets/3DPW/test'
    dataset = PW3DEvalDataset(dataset_path, img_wh=config.INPUT_RES)
    print("Eval examples found:", len(dataset))

    # Metrics
    metrics = ['pve', 'pve-t', 'pve_pa', 'pve-t_pa', 'mpjpe', 'j3d_rec_err',
               'pve_2d', 'pve_2d_pa', 'pve_2d_scale_corrected',
               'pve_scale_corrected', 'pve-t_scale_corrected', 'mpjpe_scale_corrected']

    save_path = '/data/cvfs/as2562/GraphCMR/evaluations/3dpw'
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    # Run evaluation
    evaluate_3dpw(model=model,
                  eval_dataset=dataset,
                  metrics=metrics,
                  device=device,
                  vis_save_path=save_path,
                  num_workers=4,
                  pin_memory=True,
                  vis_every_n_batches=1000)







