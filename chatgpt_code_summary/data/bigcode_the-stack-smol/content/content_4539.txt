import argparse
import re

####
#  # Box 1
####

import sys,os,imageio,lpips
root = '/home/youngsun/documents/mvs/mvsnerf_timing'
os.chdir(root)
sys.path.append(root)

from opt_src import config_parser
from data import dataset_dict
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt


# models
from models_src import *
from renderer_src import *
from data.ray_utils import get_rays

from tqdm import tqdm


from skimage.metrics import structural_similarity

# pytorch-lightning
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning import LightningModule, Trainer, loggers


from data.ray_utils import ray_marcher

import torch


torch.cuda.set_device(0)
os.environ['CUDA_VISIBLE_DEVICES'] = '0'


####
#  # Box 2
####
def decode_batch(batch):
    rays = batch['rays']  # (B, 8)
    rgbs = batch['rgbs']  # (B, 3)
    return rays, rgbs

def unpreprocess(data, shape=(1,1,3,1,1)):
    # to unnormalize image for visualization
    # data N V C H W
    device = data.device
    mean = torch.tensor([-0.485 / 0.229, -0.456 / 0.224, -0.406 / 0.225]).view(*shape).to(device)
    std = torch.tensor([1 / 0.229, 1 / 0.224, 1 / 0.225]).view(*shape).to(device)

    return (data - mean) / std

def read_depth(filename):
    depth_h = np.array(read_pfm(filename)[0], dtype=np.float32) # (800, 800)
    depth_h = cv2.resize(depth_h, None, fx=0.5, fy=0.5,
                       interpolation=cv2.INTER_NEAREST)  # (600, 800)
    depth_h = depth_h[44:556, 80:720]  # (512, 640)
#     depth = cv2.resize(depth_h, None, fx=0.5, fy=0.5,interpolation=cv2.INTER_NEAREST)#!!!!!!!!!!!!!!!!!!!!!!!!!
    mask = depth>0
    return depth_h,mask

loss_fn_vgg = lpips.LPIPS(net='vgg') 
mse2psnr = lambda x : -10. * np.log(x) / np.log(10.)


####
#  # Box 3
####

# create function for returning dense, sparse, far views

def get_source_imgs(source_dataset, target_position, N_views, device, view_type='nearest', 
                    fixed_idxs=None,
                    is_source_target_overlap=False):
    
    pair_idx = get_pair_idx(source_dataset, target_position, N_views, view_type, fixed_idxs, is_source_target_overlap)
    
    imgs_source, proj_mats, near_far_source, pose_source = source_dataset.read_source_views(pair_idx=pair_idx,device=device)
    
    return imgs_source, proj_mats, near_far_source, pose_source
    

def get_pair_idx(source_dataset, target_position, N_views, view_type='nearest', 
                 fixed_idxs=None,
                 is_source_target_overlap=False):
    
    positions = source_dataset.poses[:,:3,3]
    dis = np.sum(np.abs(positions - target_position), axis=-1)
    
    dis_sort = np.argsort(dis)
    
    if is_source_target_overlap:
        dis_sort = dis_sort[1:]
            
    
    if view_type == 'nearest': # or "as dense as possible ㅎㅎ"
        pair_idx = dis_sort[:N_views]
        pair_idx = [source_dataset.img_idx[item] for item in pair_idx]
                
    if view_type == 'dense':
        idxs = torch.randperm(int(np.rint(N_views*1.5)))[:N_views].sort()[0]
        pair_idx = dis_sort[idxs]
        pair_idx = [source_dataset.img_idx[item] for item in pair_idx]
    
    if view_type == 'random': # i know its unnecessarily long... 
        idxs = torch.randperm(len(dis_sort))[:N_views]
        pair_idx = dis_sort[idxs]
        pair_idx = [source_dataset.img_idx[item] for item in pair_idx]
    
    if view_type == 'sparse':
        idxs = torch.linspace(0, len(dis_sort), steps=N_views+1).round()
        idxs = [np.random.choice(range(int(idxs[i]), int(idxs[i+1]))) for i in range(len(idxs)-1)]
        pair_idx = dis_sort[idxs]
        pair_idx = [source_dataset.img_idx[item] for item in pair_idx]
        
    if view_type == 'far':
        idxs = torch.randperm(int(np.rint(N_views*1.5)))[:N_views].sort(descending=True)[0]
        pair_idx = dis_sort[::-1][idxs]
        pair_idx = [source_dataset.img_idx[item] for item in pair_idx]
        
    if view_type == 'farthest':
        pair_idx = dis_sort[::-1][:N_views]
        pair_idx = [source_dataset.img_idx[item] for item in pair_idx]
        
    # return index for the case of 'fixed'
    if view_type == 'fixed':
        pair_idx = fixed_idxs
    
    return pair_idx

####
#  # Box 4
####

def render_blender(view_type='nearest', 
                   scenes=['ficus'], 
                   num_src_views=3, 
                   ckpt='base-3src-dense.tar', 
                   source_split='train',
                   target_split='val',
                   select_index=None, 
                   is_fixed=False, 
                   is_source_target_overlap=False
                  ):
    
    psnr_all,ssim_all,LPIPS_vgg_all = [],[],[]
    # for i_scene, scene in enumerate(['ship','mic','chair','lego','drums','ficus','materials','hotdog']):#
    for i_scene, scene in enumerate(scenes):#
        psnr,ssim,LPIPS_vgg = [],[],[]
        cmd = f'--datadir /mnt/hdd/mvsnerf_data/nerf_synthetic/{scene}  \
         --dataset_name blender_src --white_bkgd \
        --net_type v0 --ckpt ./ckpts/{ckpt} --num_src_views {num_src_views}'
        
        save_dir = f'/mnt/hdd/youngsun/mvsnerf_timing/results/{ckpt[:-4]}/blender-{num_src_views}-'

        if is_fixed:
            save_dir += 'fixed-'
        
        save_dir += f'{view_type}-'

        save_dir += f'{source_split}-{target_split}/{scene}'

        args = config_parser(cmd.split())
        args.use_viewdirs = True

        args.N_samples = 128
        # args.feat_dim =  8+12
        args.feat_dim = 8+4*num_src_views

        # create models
        if 0==i_scene:
            render_kwargs_train, render_kwargs_test, start, grad_vars = create_nerf_mvs(args, use_mvs=True, dir_embedder=False, pts_embedder=True)
            filter_keys(render_kwargs_train)

            MVSNet = render_kwargs_train['network_mvs']
            render_kwargs_train.pop('network_mvs')


        datadir = args.datadir
        datatype = 'train'
        pad = 16
        args.chunk = 5120


        print('============> rendering dataset <===================')
        dataset_source = dataset_dict[args.dataset_name](args, split=source_split)
        dataset_target = dataset_dict[args.dataset_name](args, split=target_split, select_index=select_index)
        target_idx = dataset_target.img_idx

        save_as_image = True

        os.makedirs(save_dir, exist_ok=True)
        MVSNet.train()
        MVSNet = MVSNet.cuda()

        with torch.no_grad():

            try:
                tqdm._instances.clear() 
            except Exception:     
                pass

            for i, batch in enumerate(tqdm(dataset_target)):
                torch.cuda.empty_cache()

                rays, img = decode_batch(batch)
                rays = rays.squeeze().to(device)  # (H*W, 3)
                img = img.squeeze().cpu().numpy()  # (H, W, 3)
                
                
                if is_fixed:
                    
                    if i == 0:
                        if select_index is not None:
                            pair_idx = get_pair_idx(source_dataset=dataset_source,
                                                    target_position=dataset_target.poses[[len(select_index)//2],:3,3],
                                                    N_views=args.num_src_views, 
                                                    view_type=view_type)
                        else:
                            pair_idx = get_pair_idx(source_dataset=dataset_source,
                                                    target_position=dataset_target.poses[[50],:3,3],
                                                    N_views=args.num_src_views, 
                                                    view_type=view_type)
                                                    
                    imgs_source, proj_mats, near_far_source, pose_source = dataset_source.read_source_views(pair_idx=pair_idx,
                                                                                                           device=device)
                    
                else:
                # created fixed image_source
                    imgs_source, proj_mats, near_far_source, pose_source = get_source_imgs(source_dataset=dataset_source, 
                                                                                           target_position=dataset_target.poses[[i],:3,3], 
                                                                                           N_views=args.num_src_views, device=device, 
                                                                                           view_type=view_type)
                    

                volume_feature, _, _ = MVSNet(imgs_source, proj_mats, near_far_source, pad=pad)
                imgs_source = unpreprocess(imgs_source)

                N_rays_all = rays.shape[0]
                rgb_rays, depth_rays_preds = [],[]
                for chunk_idx in range(N_rays_all//args.chunk + int(N_rays_all%args.chunk>0)):

                    xyz_coarse_sampled, rays_o, rays_d, z_vals = ray_marcher(rays[chunk_idx*args.chunk:(chunk_idx+1)*args.chunk],
                                                        N_samples=args.N_samples)

                    # Converting world coordinate to ndc coordinate
                    H, W = img.shape[:2]
                    inv_scale = torch.tensor([W - 1, H - 1]).to(device)
                    w2c_ref, intrinsic_ref = pose_source['w2cs'][0], pose_source['intrinsics'][0].clone()
                    intrinsic_ref[:2] *= args.imgScale_test/args.imgScale_train
                    xyz_NDC = get_ndc_coordinate(w2c_ref, intrinsic_ref, xyz_coarse_sampled, inv_scale,
                                                 near=near_far_source[0], far=near_far_source[1], pad=pad*args.imgScale_test)


                    # rendering
                    rgb, disp, acc, depth_pred, alpha, extras = rendering(args, pose_source, xyz_coarse_sampled,
                                                                           xyz_NDC, z_vals, rays_o, rays_d,
                                                                           volume_feature,imgs_source, **render_kwargs_train)


                    rgb, depth_pred = torch.clamp(rgb.cpu(),0,1.0).numpy(), depth_pred.cpu().numpy()
                    rgb_rays.append(rgb)
                    depth_rays_preds.append(depth_pred)


                depth_rays_preds = np.concatenate(depth_rays_preds).reshape(H, W)
                depth_rays_preds, _ = visualize_depth_numpy(depth_rays_preds, near_far_source)

                rgb_rays = np.concatenate(rgb_rays).reshape(H, W, 3)
                img_vis = np.concatenate((img*255,rgb_rays*255,depth_rays_preds),axis=1)

                img_vis = np.concatenate((torch.cat(torch.split(imgs_source*255, [1]*num_src_views, dim=1),-1).squeeze().permute(1,2,0).cpu().numpy(),img_vis),axis=1)

                if save_as_image:
                    imageio.imwrite(f'{save_dir}/{scene}_{target_idx[i]:03d}.png', img_vis.astype('uint8'))
                else:
                    rgbs.append(img_vis.astype('uint8'))

                # quantity
                # center crop 0.8 ratio
                H_crop, W_crop = np.array(rgb_rays.shape[:2])//10
                img = img[H_crop:-H_crop,W_crop:-W_crop]
                rgb_rays = rgb_rays[H_crop:-H_crop,W_crop:-W_crop]

                psnr.append( mse2psnr(np.mean((rgb_rays-img)**2)))
                ssim.append( structural_similarity(rgb_rays, img, multichannel=True))

                img_tensor = torch.from_numpy(rgb_rays)[None].permute(0,3,1,2).float()*2-1.0 # image should be RGB, IMPORTANT: normalized to [-1,1]
                img_gt_tensor = torch.from_numpy(img)[None].permute(0,3,1,2).float()*2-1.0
                LPIPS_vgg.append( loss_fn_vgg(img_tensor, img_gt_tensor).item())

            print(f'=====> scene: {scene} mean psnr {np.mean(psnr)} ssim: {np.mean(ssim)} lpips: {np.mean(LPIPS_vgg)}')   
            psnr_all.append(psnr);ssim_all.append(ssim);LPIPS_vgg_all.append(LPIPS_vgg)

        if not save_as_image:
            imageio.mimwrite(f'{save_dir}/{scene}_spiral.mp4', np.stack(rgbs), fps=20, quality=10)

    print(f'=====> all mean psnr {np.mean(psnr_all)} ssim: {np.mean(ssim_all)} lpips: {np.mean(LPIPS_vgg_all)}') 

####
#  # Box 5
####

def render_blender_all_settings(scenes=['lego'], num_src_views=3, ckpt='base-3src-dense.tar',source_split='train', target_split='val', select_index=[30,60,90], view_types=[1]):
    
    if 1 in view_types:    
        render_blender('nearest', scenes, num_src_views, ckpt, source_split, target_split, select_index, is_fixed=None)

    if 2 in view_types:    
        render_blender('dense', scenes, num_src_views, ckpt, source_split, target_split, select_index, is_fixed=None)

    if 3 in view_types:    
        render_blender('sparse', scenes, num_src_views, ckpt, source_split, target_split, select_index, is_fixed=None)

    if 4 in view_types:    
        render_blender('far', scenes, num_src_views, ckpt, source_split, target_split, select_index, is_fixed=None)

    if 5 in view_types:    
        render_blender('random', scenes, num_src_views, ckpt, source_split, target_split, select_index, is_fixed=None)
    

    if 6 in view_types:    
        render_blender('nearest', scenes, num_src_views, ckpt, source_split, target_split, select_index, is_fixed=True)

    if 7 in view_types:    
        render_blender('sparse', scenes, num_src_views, ckpt, source_split, target_split, select_index, is_fixed=True)


    if 8 in view_types:    
        render_blender('nearest', scenes, num_src_views, ckpt, source_split, target_split, select_index, is_fixed=None, is_source_target_overlap=True)

    if 9 in view_types:    
        render_blender('sparse', scenes, num_src_views, ckpt, source_split, target_split, select_index, is_fixed=None, is_source_target_overlap=True)

    return None

####
#  # Box 6
####

####
#  # Box 7
####


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--view_types', nargs="+", type=int,
                         help=  'Enter list of view types to render:' \
                                ' 1 - nearest, 2 - dense, 3 - sparse, 4 - far, 5 - random, ' \
                                '6 - fixed nearset, 7 - fixed sparse, 8 - unseen nearest, 9 - unseen sparse')

    parser.add_argument('--view_indexes', nargs="+", type=int, const=None, default=None, 
                        help=  'default - all views (100)')

    parser.add_argument('--scenes', nargs='+', default=[])

    parser.add_argument('--ckpts', nargs='+', default=[])

    parser.add_argument('--source', type=str, default='train')

    parser.add_argument('--target', type=str, default='val')

    args = parser.parse_args()
    
    for ckpt in args.ckpts:

        num_src_views = int(re.findall('[0-9]+', ckpt)[0])

        render_blender_all_settings(scenes=args.scenes, 
                                    num_src_views=num_src_views, 
                                    ckpt=ckpt,
                                    source_split=args.source,
                                    target_split=args.target,
                                    select_index=args.view_indexes, 
                                    view_types=args.view_types)
        torch.cuda.empty_cache()