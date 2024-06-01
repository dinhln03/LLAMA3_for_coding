import os

import torch

import tkdet.utils.comm as comm
from tkdet.checkpoint import DetectionCheckpointer
from tkdet.config import get_cfg
from tkdet.data import MetadataCatalog
from tkdet.engine import DefaultTrainer
from tkdet.engine import default_argument_parser
from tkdet.engine import default_setup
from tkdet.engine import launch
from tkdet.evaluation import CityscapesInstanceEvaluator
from tkdet.evaluation import COCOEvaluator
from tkdet.evaluation import DatasetEvaluators
from tkdet.evaluation import LVISEvaluator
from tkdet.evaluation import verify_results

from point_rend import add_pointrend_config


class Trainer(DefaultTrainer):
    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        if output_folder is None:
            output_folder = os.path.join(cfg.OUTPUT_DIR, "inference")
        evaluator_list = []
        evaluator_type = MetadataCatalog.get(dataset_name).evaluator_type
        if evaluator_type == "lvis":
            return LVISEvaluator(dataset_name, cfg, True, output_folder)
        if evaluator_type == "coco":
            return COCOEvaluator(dataset_name, cfg, True, output_folder)
        if evaluator_type == "cityscapes":
            assert torch.cuda.device_count() >= comm.get_rank(), \
                "CityscapesEvaluator currently do not work with multiple machines."
            return CityscapesInstanceEvaluator(dataset_name)
        if len(evaluator_list) == 0:
            raise NotImplementedError(
                f"no Evaluator for the dataset {dataset_name} with the type {evaluator_type}"
            )
        if len(evaluator_list) == 1:
            return evaluator_list[0]
        return DatasetEvaluators(evaluator_list)


def setup(args):
    cfg = get_cfg()
    add_pointrend_config(cfg)
    cfg.merge_from_file(args.config)
    cfg.merge_from_list(args.opts)
    cfg.freeze()
    default_setup(cfg, args)
    return cfg


def main(args):
    cfg = setup(args)

    if args.eval_only:
        model = Trainer.build_model(cfg)
        DetectionCheckpointer(model, save_dir=cfg.OUTPUT_DIR).resume_or_load(
            cfg.MODEL.WEIGHTS, resume=args.resume
        )
        res = Trainer.test(cfg, model)
        if comm.is_main_process():
            verify_results(cfg, res)
        return res

    trainer = Trainer(cfg)
    trainer.resume_or_load(resume=args.resume)
    return trainer.train()


if __name__ == "__main__":
    args = default_argument_parser().parse_args()
    print("Command Line Args:", args)
    launch(
        main,
        args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        args=(args,),
    )
