from dataloaders.datasets import cityscapes, coco, combine_dbs, pascal, sbd, rip
from torch.utils.data import DataLoader


def make_data_loader(args, **kwargs):

    if args.dataset == 'pascal':
        train_set = pascal.VOCSegmentation(args, split='train')
        val_set = pascal.VOCSegmentation(args, split='val')
        if args.use_sbd:
            sbd_train = sbd.SBDSegmentation(args, split=['train', 'val'])
            train_set = combine_dbs.CombineDBs([train_set, sbd_train], excluded=[val_set])

        num_class = train_set.NUM_CLASSES
        train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, **kwargs)
        val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, **kwargs)
        test_loader = None

        return train_loader, val_loader, test_loader, num_class

    elif args.dataset == 'cityscapes':
        train_set = cityscapes.CityscapesSegmentation(args, split='train')
        val_set = cityscapes.CityscapesSegmentation(args, split='val')
        test_set = cityscapes.CityscapesSegmentation(args, split='test')
        num_class = train_set.NUM_CLASSES
        train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, **kwargs)
        val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, **kwargs)
        test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False, **kwargs)

        return train_loader, val_loader, test_loader, num_class

    elif args.dataset == 'coco':
        train_set = coco.COCOSegmentation(args, split='train')
        val_set = coco.COCOSegmentation(args, split='val')
        num_class = train_set.NUM_CLASSES
        train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, **kwargs)
        val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, **kwargs)
        test_loader = None
        return train_loader, val_loader, test_loader, num_class

    elif args.dataset == 'rip':
        classes = {'full': 7, 'level1': 2, 'level2': 3, 'level3': 5}
        import os
        from mypath import Path
        data_root = Path.db_root_dir(args.dataset)
        root = os.path.join(data_root, 'RipTrainingAllData')

        patches, level = args.rip_mode.split('-')
        if patches == 'patches':
            patches = 'COCOJSONPatches'
        elif patches == 'patches_v1':
            patches = 'COCOJSONPatches_v1'
        else:
            patches = 'COCOJSONs'
        # patches = 'COCOJSONPatches' if patches == 'patches' else 'COCOJSONs'
        train_ann_file =os.path.join(data_root, patches, level, 'cv_5_fold', 'train_1.json')
        val_ann_file =os.path.join(data_root, patches, level, 'cv_5_fold', 'val_1.json')

        train_set = rip.RIPSegmentation(args, split='train', root=root, ann_file=train_ann_file)
        val_set = rip.RIPSegmentation(args, split='val', root=root, ann_file=val_ann_file)
        num_classes = classes[level]
        # NOTE: drop_last=True here to avoid situation when batch_size=1 which causes BatchNorm2d errors
        train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, drop_last=True, **kwargs)
        val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, **kwargs)
        test_loader = None
        return train_loader, val_loader, test_loader, num_classes

    else:
        raise NotImplementedError

