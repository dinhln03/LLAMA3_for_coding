import os
import re
import struct
import glob
import numpy as np
import frame_utils
import skimage
import skimage.io

import torch
from torch.utils.data import Dataset


class KLens(Dataset):
    #def __init__(self,raft_path="/data2/opticalflow/rnd/opticalflow/RAFT/out_klens_raft_chairs", root_path="/data2/opticalflow/KLENS/images/",root_path2="/data2/opticalflow/KLENS/pins/",filenumberlist=["0030","1106","1113","1132","1134","1167","1173"],split="train",ref="",meas=""):
    def __init__(self,raft_path="/data2/opticalflow/algo_comp/flownet2/out/", root_path="/data2/opticalflow/KLENS/images/",root_path2="/data2/opticalflow/KLENS/pins/",filenumberlist=["0030","1106","1113","1132","1134","1167","1173"],split="train",ref="",meas=""):
        super(KLens, self).__init__()
        self.split = split
        raftflowpaths = glob.glob(os.path.join(raft_path,"*.flo"))
        file_list = {}
        file_list['train'] = []
        file_list['valid'] = []
        file_list['test'] = []
        file_list['train+valid'] = []
        
        for filenum in filenumberlist:
            for raftflowpath in raftflowpaths:
                #print(raftflowpath)
                if "KLE_"+filenum in raftflowpath:
                    file_list['train'].append([os.path.join(root_path,"KLE_"+filenum+".jpg3.png"),os.path.join(root_path,"KLE_"+filenum+".jpg5.png"),raftflowpath])
        file_list["train"].extend([[os.path.join(root_path,"KLE_0309_exp_sub5.jpg"),os.path.join(root_path,"KLE_0309_exp_sub6.jpg")],[os.path.join(root_path,"KLE_0730_sub5.jpg"),os.path.join(root_path,"KLE_0730_sub6.jpg")],[os.path.join(root_path,"KLE_0747_sub5.jpg"),os.path.join(root_path,"KLE_0747_sub6.jpg")],[os.path.join(root_path,"KLE_9797clean_sub5.jpg"),os.path.join(root_path,"KLE_9797clean_sub6.jpg")],[os.path.join(root_path,"KLE_9803clean_sub5.jpg"),os.path.join(root_path,"KLE_9803clean_sub6.jpg")],[os.path.join(root_path,"NKM_0063_sub5.jpg"),os.path.join(root_path,"NKM_0063_sub6.jpg")],[os.path.join(root_path,"NKM_0109_sub5.jpg"),os.path.join(root_path,"NKM_0109_sub6.jpg")],[os.path.join(root_path,"scene_1_sub5.jpg"),os.path.join(root_path,"scene_1_sub6.jpg")]])
        file_list["valid"].extend([[os.path.join(root_path,"KLE_0309_exp_sub5.jpg"),os.path.join(root_path,"KLE_0309_exp_sub6.jpg")],[os.path.join(root_path,"KLE_0730_sub5.jpg"),os.path.join(root_path,"KLE_0730_sub6.jpg")],[os.path.join(root_path,"KLE_0747_sub5.jpg"),os.path.join(root_path,"KLE_0747_sub6.jpg")],[os.path.join(root_path,"KLE_9797clean_sub5.jpg"),os.path.join(root_path,"KLE_9797clean_sub6.jpg")],[os.path.join(root_path,"KLE_9803clean_sub5.jpg"),os.path.join(root_path,"KLE_9803clean_sub6.jpg")],[os.path.join(root_path,"NKM_0063_sub5.jpg"),os.path.join(root_path,"NKM_0063_sub6.jpg")],[os.path.join(root_path,"NKM_0109_sub5.jpg"),os.path.join(root_path,"NKM_0109_sub6.jpg")],[os.path.join(root_path,"scene_1_sub5.jpg"),os.path.join(root_path,"scene_1_sub6.jpg")]])
        file_list["test"].extend([[os.path.join(root_path,"KLE_0309_exp_sub5.jpg"),os.path.join(root_path,"KLE_0309_exp_sub6.jpg")],[os.path.join(root_path,"KLE_0730_sub5.jpg"),os.path.join(root_path,"KLE_0730_sub6.jpg")],[os.path.join(root_path,"KLE_0747_sub5.jpg"),os.path.join(root_path,"KLE_0747_sub6.jpg")],[os.path.join(root_path,"KLE_9797clean_sub5.jpg"),os.path.join(root_path,"KLE_9797clean_sub6.jpg")],[os.path.join(root_path,"KLE_9803clean_sub5.jpg"),os.path.join(root_path,"KLE_9803clean_sub6.jpg")],[os.path.join(root_path,"NKM_0063_sub5.jpg"),os.path.join(root_path,"NKM_0063_sub6.jpg")],[os.path.join(root_path,"NKM_0109_sub5.jpg"),os.path.join(root_path,"NKM_0109_sub6.jpg")],[os.path.join(root_path,"scene_1_sub5.jpg"),os.path.join(root_path,"scene_1_sub6.jpg")]])
        file_list["train+valid"].extend([[os.path.join(root_path,"KLE_0309_exp_sub5.jpg"),os.path.join(root_path,"KLE_0309_exp_sub6.jpg")],[os.path.join(root_path,"KLE_0730_sub5.jpg"),os.path.join(root_path,"KLE_0730_sub6.jpg")],[os.path.join(root_path,"KLE_0747_sub5.jpg"),os.path.join(root_path,"KLE_0747_sub6.jpg")],[os.path.join(root_path,"KLE_9797clean_sub5.jpg"),os.path.join(root_path,"KLE_9797clean_sub6.jpg")],[os.path.join(root_path,"KLE_9803clean_sub5.jpg"),os.path.join(root_path,"KLE_9803clean_sub6.jpg")],[os.path.join(root_path,"NKM_0063_sub5.jpg"),os.path.join(root_path,"NKM_0063_sub6.jpg")],[os.path.join(root_path,"NKM_0109_sub5.jpg"),os.path.join(root_path,"NKM_0109_sub6.jpg")],[os.path.join(root_path,"scene_1_sub5.jpg"),os.path.join(root_path,"scene_1_sub6.jpg")]])
        #file_list["train"].extend([[os.path.join(root_path2,"9-AIT_pins_2.jpg"),os.path.join(root_path2,"9-AIT_pins_3.jpg")],[os.path.join(root_path2,"10-Hela_2.jpg"),os.path.join(root_path2,"10-Hela_3.jpg")],[os.path.join(root_path2,"11-Hela_1_2.jpg"),os.path.join(root_path2,"11-Hela_1_3.jpg")],])
        #file_list["train"].extend([[os.path.join(root_path2,"9-AIT_pins_2.jpg"),os.path.join(root_path2,"9-AIT_pins_0.jpg")],[os.path.join(root_path2,"10-Hela_2.jpg"),os.path.join(root_path2,"10-Hela_0.jpg")],[os.path.join(root_path2,"11-Hela_1_2.jpg"),os.path.join(root_path2,"11-Hela_1_0.jpg")],])
        #file_list["train"].extend([[os.path.join(root_path2,"9-AIT_pins_2.jpg"),os.path.join(root_path2,"9-AIT_pins_1.jpg")],[os.path.join(root_path2,"10-Hela_2.jpg"),os.path.join(root_path2,"10-Hela_1.jpg")],[os.path.join(root_path2,"11-Hela_1_2.jpg"),os.path.join(root_path2,"11-Hela_1_1.jpg")],])
        #file_list["train"].extend([[os.path.join(root_path2,"9-AIT_pins_2.jpg"),os.path.join(root_path2,"9-AIT_pins_4.jpg")],[os.path.join(root_path2,"10-Hela_2.jpg"),os.path.join(root_path2,"10-Hela_4.jpg")],[os.path.join(root_path2,"11-Hela_1_2.jpg"),os.path.join(root_path2,"11-Hela_1_4.jpg")],])
        self.dataset = file_list

    def __len__(self):
        return len(self.dataset[self.split])

    def __getitem__(self, idx):
        try:
            im0_path, im1_path, raftflow_path = self.dataset[self.split][idx]
            raftflow = frame_utils.readFlow(raftflow_path)
        except:
            im0_path, im1_path = self.dataset[self.split][idx]
            raftflow = np.array([])
        img0 = skimage.io.imread(im0_path)
        img1 = skimage.io.imread(im1_path)
        img0 = torch.tensor(img0/255.).float()
        img1 = torch.tensor(img1/255.).float()

        return img0, img1,np.array([]),np.array([]), [im0_path , im1_path],raftflow


class Flo:
    def __init__(self, w, h):
        self.__floec1__ = float(202021.25)
        self.__floec2__ = int(w)
        self.__floec3__ = int(h)
        self.__floheader__ = struct.pack('fii', self.__floec1__, self.__floec2__, self.__floec3__)
        self.__floheaderlen__ = len(self.__floheader__)
        self.__flow__ = w
        self.__floh__ = h
        self.__floshape__ = [self.__floh__, self.__flow__, 2]

        if self.__floheader__[:4] != b'PIEH':
            raise Exception('Expect machine to be LE.')

    def load(self, file):
        with open(file, 'rb') as fp:
            if fp.read(self.__floheaderlen__) != self.__floheader__:
                raise Exception('Bad flow header: ' + file)
            result = np.ndarray(shape=self.__floshape__,
                                dtype=np.float32,
                                buffer=fp.read(),
                                order='C')
            return result

    def save(self, arr, fname):
        with open(fname, 'wb') as fp:
            fp.write(self.__floheader__)
            fp.write(arr.astype(np.float32).tobytes())

