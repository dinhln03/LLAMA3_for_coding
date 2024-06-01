import torch.nn as nn
import torch.nn.functional as F
import torch
from mmcv.cnn import ConvModule
from mmcv.runner import force_fp32

from mmdet.models.builder import HEADS, build_loss
from mmdet.models.losses import accuracy
from .bbox_head import BBoxHead

from mmdet.core import multi_apply, multiclass_nms

from mmdet.core.bbox.iou_calculators.builder import build_iou_calculator

@HEADS.register_module()
class ConvFCBBoxHead(BBoxHead):
    r"""More general bbox head, with shared conv and fc layers and two optional
    separated branches.

    .. code-block:: none

                                    /-> cls convs -> cls fcs -> cls
        shared convs -> shared fcs
                                    \-> reg convs -> reg fcs -> reg
                                    
                                   (\-> dis convs -> dis fcs -> dis)
    """  # noqa: W605

    def __init__(self,
                 num_shared_convs=0,
                 num_shared_fcs=0,
                 num_cls_convs=0,
                 num_cls_fcs=0,
                 num_reg_convs=0,
                 num_reg_fcs=0,
                 conv_out_channels=256,
                 fc_out_channels=1024,
                 conv_cfg=None,
                 norm_cfg=None,
                 with_dis=False, #for leaves
                 num_dis_convs=0,
                 num_dis_fcs=0,
                 *args,
                 **kwargs):
        super(ConvFCBBoxHead, self).__init__(*args, **kwargs)
        #only for leaves
        self.with_dis = with_dis
        self.num_dis_convs = num_dis_convs
        self.num_dis_fcs = num_dis_fcs
        
        assert (num_shared_convs + num_shared_fcs + num_cls_convs +
                num_cls_fcs + num_reg_convs + num_reg_fcs > 0)
        if num_cls_convs > 0 or num_reg_convs > 0:
            assert num_shared_fcs == 0
        if not self.with_cls:
            assert num_cls_convs == 0 and num_cls_fcs == 0
        if not self.with_reg:
            assert num_reg_convs == 0 and num_reg_fcs == 0
        if not self.with_dis:
            assert num_dis_convs == 0 and num_dis_fcs == 0
        self.num_shared_convs = num_shared_convs
        self.num_shared_fcs = num_shared_fcs
        self.num_cls_convs = num_cls_convs
        self.num_cls_fcs = num_cls_fcs
        self.num_reg_convs = num_reg_convs
        self.num_reg_fcs = num_reg_fcs
        self.conv_out_channels = conv_out_channels
        self.fc_out_channels = fc_out_channels
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        
        # add shared convs and fcs
        self.shared_convs, self.shared_fcs, last_layer_dim = \
            self._add_conv_fc_branch(
                self.num_shared_convs, self.num_shared_fcs, self.in_channels,
                True)
        self.shared_out_channels = last_layer_dim

        # add cls specific branch
        self.cls_convs, self.cls_fcs, self.cls_last_dim = \
            self._add_conv_fc_branch(
                self.num_cls_convs, self.num_cls_fcs, self.shared_out_channels)

        # add reg specific branch
        self.reg_convs, self.reg_fcs, self.reg_last_dim = \
            self._add_conv_fc_branch(
                self.num_reg_convs, self.num_reg_fcs, self.shared_out_channels)
        
        #add dis branch(only for leaves)
        if self.with_dis:
            self.dis_convs, self.dis_fcs, self.dis_last_dim = \
                self._add_conv_fc_branch(
                    self.num_dis_convs, self.num_dis_fcs, self.shared_out_channels)
            
        if self.num_shared_fcs == 0 and not self.with_avg_pool:
            if self.num_cls_fcs == 0:
                self.cls_last_dim *= self.roi_feat_area
            if self.num_reg_fcs == 0:
                self.reg_last_dim *= self.roi_feat_area

        self.relu = nn.ReLU(inplace=True)
        # reconstruct fc_cls and fc_reg since input channels are changed
        if self.with_cls:
            self.fc_cls = nn.Linear(self.cls_last_dim, self.num_classes + 1)
        if self.with_reg:
            out_dim_reg = (4 if self.reg_class_agnostic else 4 *
                           self.num_classes)
            self.fc_reg = nn.Linear(self.reg_last_dim, out_dim_reg)
        if self.with_dis:
            if self.dis_selector == 0 or self.dis_selector == 1: 
                self.fc_dis = nn.Linear(self.cls_last_dim, 1)
            elif self.dis_selector == 2:
                self.fc_dis = nn.Linear(self.cls_last_dim, 4)

    def _add_conv_fc_branch(self,
                            num_branch_convs,
                            num_branch_fcs,
                            in_channels,
                            is_shared=False):
        """Add shared or separable branch.

        convs -> avg pool (optional) -> fcs
        """
        last_layer_dim = in_channels
        # add branch specific conv layers
        branch_convs = nn.ModuleList()
        if num_branch_convs > 0:
            for i in range(num_branch_convs):
                conv_in_channels = (
                    last_layer_dim if i == 0 else self.conv_out_channels)
                branch_convs.append(
                    ConvModule(
                        conv_in_channels,
                        self.conv_out_channels,
                        3,
                        padding=1,
                        conv_cfg=self.conv_cfg,
                        norm_cfg=self.norm_cfg))
            last_layer_dim = self.conv_out_channels
        # add branch specific fc layers
        branch_fcs = nn.ModuleList()
        if num_branch_fcs > 0:
            # for shared branch, only consider self.with_avg_pool
            # for separated branches, also consider self.num_shared_fcs
            if (is_shared
                    or self.num_shared_fcs == 0) and not self.with_avg_pool:
                last_layer_dim *= self.roi_feat_area
            for i in range(num_branch_fcs):
                fc_in_channels = (
                    last_layer_dim if i == 0 else self.fc_out_channels)
                branch_fcs.append(
                    nn.Linear(fc_in_channels, self.fc_out_channels))
            last_layer_dim = self.fc_out_channels
        return branch_convs, branch_fcs, last_layer_dim

    def init_weights(self):
        super(ConvFCBBoxHead, self).init_weights()
        # conv layers are already initialized by ConvModule
        if self.with_dis:
            for module_list in [self.shared_fcs, self.cls_fcs, self.reg_fcs, self.dis_fcs]:
                for m in module_list.modules():
                    if isinstance(m, nn.Linear):
                        nn.init.xavier_uniform_(m.weight)
                        nn.init.constant_(m.bias, 0)
        else:
            for module_list in [self.shared_fcs, self.cls_fcs, self.reg_fcs]:
                for m in module_list.modules():
                    if isinstance(m, nn.Linear):
                        nn.init.xavier_uniform_(m.weight)
                        nn.init.constant_(m.bias, 0)

    def forward(self, x):
        # shared part
        if self.num_shared_convs > 0:
            for conv in self.shared_convs:
                x = conv(x)

        if self.num_shared_fcs > 0:
            if self.with_avg_pool:
                x = self.avg_pool(x)

            x = x.flatten(1)

            for fc in self.shared_fcs:
                x = self.relu(fc(x))
        # separate branches
        x_cls = x
        x_reg = x
        if self.with_dis:
            x_dis = x
            
            for conv in self.dis_convs:
                x_dis = conv(x_dis)
            if x_dis.dim() > 2:
                if self.with_avg_pool:
                    x_dis = self.avg_pool(x_dis)
                x_dis = x_dis.flatten(1)
            for fc in self.dis_fcs:
                x_dis = self.relu(fc(x_dis))

        for conv in self.cls_convs:
            x_cls = conv(x_cls)
        if x_cls.dim() > 2:
            if self.with_avg_pool:
                x_cls = self.avg_pool(x_cls)
            x_cls = x_cls.flatten(1)
        for fc in self.cls_fcs:
            x_cls = self.relu(fc(x_cls))

        for conv in self.reg_convs:
            x_reg = conv(x_reg)
        if x_reg.dim() > 2:
            if self.with_avg_pool:
                x_reg = self.avg_pool(x_reg)
            x_reg = x_reg.flatten(1)
        for fc in self.reg_fcs:
            x_reg = self.relu(fc(x_reg))

        cls_score = self.fc_cls(x_cls) if self.with_cls else None
        bbox_pred = self.fc_reg(x_reg) if self.with_reg else None
        dis_pred = self.fc_dis(x_dis) if self.with_dis else None
        return cls_score, bbox_pred, dis_pred


@HEADS.register_module()
class Shared2FCBBoxHead(ConvFCBBoxHead):

    def __init__(self, fc_out_channels=1024, *args, **kwargs):
        super(Shared2FCBBoxHead, self).__init__(
            num_shared_convs=0,
            num_shared_fcs=2,
            num_cls_convs=0,
            num_cls_fcs=0,
            num_reg_convs=0,
            num_reg_fcs=0,
            fc_out_channels=fc_out_channels,
            *args,
            **kwargs)

@HEADS.register_module()
class Shared2FCBBoxHeadLeaves(ConvFCBBoxHead):

    def __init__(self, fc_out_channels=1024, *args, **kwargs):
        loss_dis = kwargs['loss_dis']
        self.reference_labels = kwargs['reference_labels']
        self.classes = kwargs['classes']
        self.dis_selector = kwargs['dis_selector']
        assert self.dis_selector in (0, 1, 2)
        kwargs.pop('loss_dis')
        kwargs.pop('reference_labels')
        kwargs.pop('classes')
        kwargs.pop('dis_selector')
        
        super(Shared2FCBBoxHeadLeaves, self).__init__(
            num_shared_convs=0,
            num_shared_fcs=2,
            num_cls_convs=0,
            num_cls_fcs=0,
            num_reg_convs=0,
            num_reg_fcs=0,
            fc_out_channels=fc_out_channels,
            with_dis=True, #only for leaves
            num_dis_convs=0,
            num_dis_fcs=0,
            *args,
            **kwargs)
        
        if self.dis_selector == 0 or self.dis_selector == 1:
            assert loss_dis['use_sigmoid'], "used invalid loss_dis"
        elif self.dis_selector == 2:
            assert not loss_dis['use_sigmoid'], "used invalid loss_dis"
        self.loss_dis = build_loss(loss_dis)
        #DEBUG
        #loss_dis_py =dict(type='py_FocalLoss',
        #                  alpha=torch.tensor(self.dis_weights, device=torch.device('cpu')),
        #                  gamma = 2.0,
        #                  reduction = 'mean')
        #self.loss_dis_py = build_loss(loss_dis_py)
 
            
    #Override
    def get_targets(self,
                    sampling_results,
                    gt_bboxes,
                    gt_labels,
                    rcnn_train_cfg,
                    reference_labels,
                    classes,
                    concat=True):
        """Calculate the ground truth for all samples in a batch according to
        the sampling_results.
        Almost the same as the implementation in bbox_head, we passed
        additional parameters pos_inds_list and neg_inds_list to
        `_get_target_single` function.
        Args:
            sampling_results (List[obj:SamplingResults]): Assign results of
                all images in a batch after sampling.
            gt_bboxes (list[Tensor]): Gt_bboxes of all images in a batch,
                each tensor has shape (num_gt, 4),  the last dimension 4
                represents [tl_x, tl_y, br_x, br_y].
            gt_labels (list[Tensor]): Gt_labels of all images in a batch,
                each tensor has shape (num_gt,).
            rcnn_train_cfg (obj:ConfigDict): `train_cfg` of RCNN.
            concat (bool): Whether to concatenate the results of all
                the images in a single batch.
        Returns:
            Tuple[Tensor]: Ground truth for proposals in a single image.
            Containing the following list of Tensors:
                - labels (list[Tensor],Tensor): Gt_labels for all
                  proposals in a batch, each tensor in list has
                  shape (num_proposals,) when `concat=False`, otherwise
                  just a single tensor has shape (num_all_proposals,).
                - label_weights (list[Tensor]): Labels_weights for
                  all proposals in a batch, each tensor in list has
                  shape (num_proposals,) when `concat=False`, otherwise
                  just a single tensor has shape (num_all_proposals,).
                - bbox_targets (list[Tensor],Tensor): Regression target
                  for all proposals in a batch, each tensor in list
                  has shape (num_proposals, 4) when `concat=False`,
                  otherwise just a single tensor has shape
                  (num_all_proposals, 4), the last dimension 4 represents
                  [tl_x, tl_y, br_x, br_y].
                - bbox_weights (list[tensor],Tensor): Regression weights for
                  all proposals in a batch, each tensor in list has shape
                  (num_proposals, 4) when `concat=False`, otherwise just a
                  single tensor has shape (num_all_proposals, 4).
                - dis_targets (list[tensor], Tensor): Gt_dis for all
                  proposal in a batch, each tensor in list has
                  shape (num_proposal,) when 'concat=False`, otherwise
                  just a single tensor has shape (num_all_proposals,).
        """
        
        pos_bboxes_list = [res.pos_bboxes for res in sampling_results]
        neg_bboxes_list = [res.neg_bboxes for res in sampling_results]
        pos_gt_bboxes_list = [res.pos_gt_bboxes for res in sampling_results]
        pos_gt_labels_list = [res.pos_gt_labels for res in sampling_results]
        labels, label_weights, bbox_targets, bbox_weights = multi_apply(
            self._get_target_single,
            pos_bboxes_list,
            neg_bboxes_list,
            pos_gt_bboxes_list,
            pos_gt_labels_list,
            cfg=rcnn_train_cfg)
            
        #processing for dis_target
        iou_calculator=dict(type='BboxOverlaps2D')                                                                                                        
        iou_calculator = build_iou_calculator(iou_calculator)
        isolation_thr = 0.45	#TODO da mettere come arg
        #retrive the gt_superclass bboxes
        dis_targets = []
        for i, res in enumerate(sampling_results):
            ref_grap_list =[]
            ref_leav_list =[]
            ref_grap_dis_list =[]
            ref_leav_dis_list =[]
            for j, bbox in enumerate(gt_bboxes[i]):
            
                if self.dis_selector == 0:
                    if 'grappolo' in classes[gt_labels[i][j]] and gt_labels[i][j] != reference_labels['grappolo_vite']:
                        ref_grap_dis_list.append(bbox)
                        
                    elif (('foglia' in classes[gt_labels[i][j]] or classes[gt_labels[i][j]] == 'malattia_esca'\
                        or classes[gt_labels[i][j]] == 'virosi_pinot_grigio')
                        and gt_labels[i][j] != reference_labels['foglia_vite']):
                            ref_leav_dis_list.append(bbox)
                            
                elif self.dis_selector == 1:
                    if gt_labels[i][j] == reference_labels['grappolo_vite']:
                        ref_grap_list.append(bbox)
                    elif gt_labels[i][j] == reference_labels['foglia_vite']:
                        ref_leav_list.append(bbox)
                
                elif self.dis_selector == 2:
                    if gt_labels[i][j] == reference_labels['grappolo_vite']:
                        ref_grap_list.append(bbox)
                    elif gt_labels[i][j] == reference_labels['foglia_vite']:
                        ref_leav_list.append(bbox)
                    elif 'grappolo' in classes[gt_labels[i][j]]:
                        ref_grap_dis_list.append(bbox)
                    elif 'foglia' in classes[gt_labels[i][j]] or classes[gt_labels[i][j]] == 'malattia_esca'\
                        or classes[gt_labels[i][j]] == 'virosi_pinot_grigio':
                            ref_leav_dis_list.append(bbox)
                '''
                if 'grappolo' in classes[gt_labels[i][j]] and gt_labels[i][j] != reference_labels['grappolo_vite']:
                    ref_grap_dis_list.append(bbox)
                elif (('foglia' in classes[gt_labels[i][j]] or classes[gt_labels[i][j]] == 'malattia_esca'\
                    or classes[gt_labels[i][j]] == 'virosi_pinot_grigio')
                    and gt_labels[i][j] != reference_labels['foglia_vite']):
                        ref_leav_dis_list.append(bbox)
                '''           
            if len(ref_grap_list) > 0:
                ref_grap_tensor = torch.cat(ref_grap_list)
                ref_grap_tensor = torch.reshape(ref_grap_tensor, (len(ref_grap_list), 4))
            
            if len(ref_leav_list) > 0:
                ref_leav_tensor = torch.cat(ref_leav_list)
                ref_leav_tensor = torch.reshape(ref_leav_tensor, (len(ref_leav_list), 4))
                
            if len(ref_grap_dis_list) > 0:
                ref_grap_dis_tensor = torch.cat(ref_grap_dis_list)
                ref_grap_dis_tensor = torch.reshape(ref_grap_dis_tensor, (len(ref_grap_dis_list), 4))
            
            if len(ref_leav_dis_list) > 0:
                ref_leav_dis_tensor = torch.cat(ref_leav_dis_list)
                ref_leav_dis_tensor = torch.reshape(ref_leav_dis_tensor, (len(ref_leav_dis_list), 4))

            num_pos = res.pos_bboxes.size(0)
            num_neg = res.neg_bboxes.size(0)
            num_samples = num_pos + num_neg
            dis_tensor= res.pos_bboxes.new_full((num_samples, ), -1, dtype=torch.long)
            dis_list = []
            for j, bbox in enumerate(res.pos_bboxes):
                #trick for using the iof calculator
                bbox = bbox.unsqueeze(0)
                
                if res.pos_gt_labels[j] == reference_labels['grappolo_vite']:
                    if self.dis_selector == 0:
                       dis_list.append(-1)    #the grape is not considered
                    
                    elif self.dis_selector == 1 or self.dis_selector == 2:
                        if len(ref_grap_dis_list) > 0:
                            overlaps = iou_calculator(ref_grap_dis_tensor, bbox, mode='iof')
                            overlaps = overlaps < isolation_thr
                            if overlaps.all():
                                dis_list.append(0)    #the grape is healthy
                            else:
                                dis_list.append(1)    #the grape is affected by a disease
                        else:
                            dis_list.append(0)    #the grape is healthy
                            
                elif res.pos_gt_labels[j] == reference_labels['foglia_vite']:
                    if self.dis_selector == 0:
                        dis_list.append(-1)    #the leaf is not considered
                    
                    elif self.dis_selector == 1 or self.dis_selector == 2:
                        if len(ref_leav_dis_list) > 0:
                            overlaps = iou_calculator(ref_leav_dis_tensor, bbox, mode='iof')
                            overlaps = overlaps < isolation_thr
                            if overlaps.all():
                                dis_list.append(0)    #the leaf is healthy
                            else:
                                dis_list.append(1)    #the leaf is affected by a disease
                        else:
                            dis_list.append(0)    #the leaf is healthy
                    
                elif 'grappolo' in classes[res.pos_gt_labels[j]] and res.pos_gt_labels[j] != reference_labels['grappolo_vite']:
                    if self.dis_selector == 1:
                        dis_list.append(-1)    #the disease is not considered
                    
                    elif self.dis_selector == 0:
                        if len(ref_grap_list) > 0:
                            overlaps = iou_calculator(bbox, ref_grap_tensor, mode='iof')
                            overlaps = overlaps < isolation_thr
                            if overlaps.all():
                                dis_list.append(0)    #the disease is isolated
                            else:
                                dis_list.append(1)    #the disease is inside a leaf or grape
                        else:
                            dis_list.append(0)    #the disease is isolated
                    
                    elif self.dis_selector == 2:
                        if len(ref_grap_list) > 0:
                            overlaps = iou_calculator(bbox, ref_grap_tensor, mode='iof')
                            overlaps = overlaps < isolation_thr
                            if overlaps.all():
                                dis_list.append(2)    #the disease is isolated
                            else:
                                dis_list.append(3)    #the disease is inside a leaf or grape
                        else:
                            dis_list.append(2)    #the disease is isolated
                
                elif (('foglia' in classes[res.pos_gt_labels[j]] or classes[res.pos_gt_labels[j]] == 'malattia_esca'
                    or classes[res.pos_gt_labels[j]] == 'virosi_pinot_grigio')
                    and res.pos_gt_labels[j] != reference_labels['foglia_vite']):
                        if self.dis_selector == 1:
                            dis_list.append(-1)    #the disease is not considered
                    
                        elif self.dis_selector == 0:
                            if len(ref_leav_list) > 0:
                                overlaps = iou_calculator(bbox, ref_leav_tensor, mode='iof')
                                overlaps = overlaps < isolation_thr
                                if overlaps.all():
                                    dis_list.append(0)    #the disease is isolated
                                else:
                                    dis_list.append(1)    #the disease is inside a leaf or grape
                            else:
                                dis_list.append(0)    #the disease is isolated
                        
                        elif self.dis_selector == 2:
                            if len(ref_leav_list) > 0:
                                overlaps = iou_calculator(bbox, ref_leav_tensor, mode='iof')
                                overlaps = overlaps < isolation_thr
                                if overlaps.all():
                                    dis_list.append(2)    #the disease is isolated
                                else:
                                    dis_list.append(3)    #the disease is inside a leaf or grape
                            else:
                                dis_list.append(2)    #the disease is isolated
                    
                #elif res.pos_gt_labels[j] == reference_labels['oidio_tralci']:
                #    dis_list.append(-1)    #the disease is not considered
            
            dis_tensor[:num_pos] = torch.tensor(dis_list)
            dis_targets.append(dis_tensor)         
            
        if concat:
            labels = torch.cat(labels, 0)
            label_weights = torch.cat(label_weights, 0)
            bbox_targets = torch.cat(bbox_targets, 0)
            bbox_weights = torch.cat(bbox_weights, 0)
            dis_targets = torch.cat(dis_targets, 0)
            
        #del dis_tensor
        #torch.cuda.empty_cache()
        return labels, label_weights, bbox_targets, bbox_weights, dis_targets
    
    #Override
    @force_fp32(apply_to=('cls_score', 'bbox_pred', 'dis_pred'))
    def loss(self,
             cls_score,
             bbox_pred,
             dis_pred,
             rois,
             labels,
             label_weights,
             bbox_targets,
             bbox_weights,
             dis_targets,
             reduction_override=None):
        losses = dict()
        if cls_score is not None:
            avg_factor = max(torch.sum(label_weights > 0).float().item(), 1.)
            if cls_score.numel() > 0:
                losses['loss_cls'] = self.loss_cls(
                    cls_score,
                    labels,
                    label_weights,
                    avg_factor=avg_factor,
                    reduction_override=reduction_override)
                losses['acc'] = accuracy(cls_score, labels)
        if bbox_pred is not None:
            bg_class_ind = self.num_classes
            # 0~self.num_classes-1 are FG, self.num_classes is BG
            pos_inds = (labels >= 0) & (labels < bg_class_ind)
            # do not perform bounding box regression for BG anymore.
            if pos_inds.any():
                if self.reg_decoded_bbox:
                    # When the regression loss (e.g. `IouLoss`,
                    # `GIouLoss`, `DIouLoss`) is applied directly on
                    # the decoded bounding boxes, it decodes the
                    # already encoded coordinates to absolute format.
                    bbox_pred = self.bbox_coder.decode(rois[:, 1:], bbox_pred)
                if self.reg_class_agnostic:
                    pos_bbox_pred = bbox_pred.view(
                        bbox_pred.size(0), 4)[pos_inds.type(torch.bool)]
                else:
                    pos_bbox_pred = bbox_pred.view(
                        bbox_pred.size(0), -1,
                        4)[pos_inds.type(torch.bool),
                           labels[pos_inds.type(torch.bool)]]
                losses['loss_bbox'] = self.loss_bbox(
                    pos_bbox_pred,
                    bbox_targets[pos_inds.type(torch.bool)],
                    bbox_weights[pos_inds.type(torch.bool)],
                    avg_factor=bbox_targets.size(0),
                    reduction_override=reduction_override)
            else:
                losses['loss_bbox'] = bbox_pred[pos_inds].sum()
        if dis_pred is not None:
            pos_inds = dis_targets != -1
            if pos_inds.any():
                pos_dis_pred = dis_pred[pos_inds.type(torch.bool)]
                pos_dis_targets = dis_targets[pos_inds.type(torch.bool)]
                avg_factor = dis_pred.size(0)
                
                losses['loss_dis'] = self.loss_dis(
                    pos_dis_pred,
                    pos_dis_targets,
                    avg_factor=avg_factor,
                    reduction_override=reduction_override)
                    
                #DEBUG
                #loss_py = self.loss_dis_py(pos_dis_pred,
                #                           pos_dis_targets)
                
                #from mmcv.utils import print_log
                #import logging
                #logger = logging.getLogger(__name__)
                #print_log("loss_dis:{:0.4f},    loss_dis_py:{:0.4f}".format(losses['loss_dis'], loss_py), logger = logger)
        
        return losses
        
    #Override
    @force_fp32(apply_to=('cls_score', 'bbox_pred', 'dis_pred'))
    def get_bboxes(self,
                   rois,
                   cls_score,
                   bbox_pred,
                   dis_pred,
                   img_shape,
                   scale_factor,
                   rescale=False,
                   cfg=None):
        if isinstance(cls_score, list):
            cls_score = sum(cls_score) / float(len(cls_score))
        scores = F.softmax(cls_score, dim=1) if cls_score is not None else None

        if bbox_pred is not None:
            bboxes = self.bbox_coder.decode(
                rois[:, 1:], bbox_pred, max_shape=img_shape)
        else:
            bboxes = rois[:, 1:].clone()
            if img_shape is not None:
                bboxes[:, [0, 2]].clamp_(min=0, max=img_shape[1])
                bboxes[:, [1, 3]].clamp_(min=0, max=img_shape[0])

        if rescale and bboxes.size(0) > 0:
            if isinstance(scale_factor, float):
                bboxes /= scale_factor
            else:
                scale_factor = bboxes.new_tensor(scale_factor)
                bboxes = (bboxes.view(bboxes.size(0), -1, 4) /
                          scale_factor).view(bboxes.size()[0], -1)
        if dis_pred is not None:
            if self.dis_selector == 0 or self.dis_selector == 1:
                diseases = F.sigmoid(dis_pred)
            elif self.dis_selector == 2:
                diseases = F.softmax(dis_pred, dim=1)
        
        if cfg is None:
            return bboxes, scores, diseases
        else:
            det_bboxes, det_labels, inds = multiclass_nms(bboxes, scores,
                                                    cfg.score_thr, cfg.nms,
                                                    cfg.max_per_img,
                                                    return_inds=True)

            if self.dis_selector == 0 or self.dis_selector == 1:
                diseases = diseases.expand(bboxes.size(0), scores.size(1) - 1)
                diseases = diseases.reshape(-1)
            elif self.dis_selector == 2:
                diseases = diseases[:, None].expand(bboxes.size(0), scores.size(1) - 1, 4) 
                diseases = diseases.reshape(-1, 4)
            
            det_dis = diseases[inds]
            return det_bboxes, det_labels, det_dis


@HEADS.register_module()
class Shared4Conv1FCBBoxHead(ConvFCBBoxHead):

    def __init__(self, fc_out_channels=1024, *args, **kwargs):
        super(Shared4Conv1FCBBoxHead, self).__init__(
            num_shared_convs=4,
            num_shared_fcs=1,
            num_cls_convs=0,
            num_cls_fcs=0,
            num_reg_convs=0,
            num_reg_fcs=0,
            fc_out_channels=fc_out_channels,
            *args,
            **kwargs)
