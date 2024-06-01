import os
import json
import logging
from dataclasses import dataclass, field
from typing import Dict, Optional, Callable

import torch
import wandb
import numpy as np
from tqdm.auto import tqdm
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.dataset import Dataset
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data.sampler import RandomSampler
from transformers import (
    Trainer, 
    TrainingArguments, 
    EvalPrediction, 
    DataCollator,
    DefaultDataCollator,
)
from transformers.trainer_utils import PredictionOutput
from transformers.training_args import is_tpu_available

from src.data.task_data_processors import task_output_modes

from src.data.data_utils import compute_task_metrics

if is_tpu_available():
    import torch_xla.core.xla_model as xm
    import torch_xla.debug.metrics as met
    import torch_xla.distributed.parallel_loader as pl
    
logger = logging.getLogger(__name__)


@dataclass
class MultiTaskTrainingArguments(TrainingArguments):
    use_mt_uncertainty: bool = field(
        default=False,
        metadata={"help": "Use MT-Uncertainty sampling method"},
    )
    uniform_mt_sampling: bool = field(
        default=False,
        metadata={"help": "Sample each task an equal amount to times per epoch."},
    )
    percent_of_max_data_size: float = field(
        default=1.0,
        metadata={
            "help": "If uniform_mt_sampling=True, specify the samples per task per "
            "epoch based on the maximum dataset length. If below 0.0 or above 1.0,"
            "it will be set to the closest of 0.0 or 1.0."
        },
    )


class MultiTaskTrainer(Trainer):
    def __init__(
        self,
        tokenizer,
        data_args,
        eval_datasets=None,
        test_datasets=None,
        *args,
        **kwargs,
    ):
        super(MultiTaskTrainer, self).__init__(*args, **kwargs)
        self.tokenizer = tokenizer
        self.data_args = data_args
        self.eval_datasets = eval_datasets
        self.test_datasets = test_datasets
#         self.data_collator = DefaultDataCollator()

    def get_train_dataloader(self) -> DataLoader:
        if self.args.use_mt_uncertainty:
            return self._create_custom_dataloader()
        else:
            return super().get_train_dataloader()

    def _create_custom_dataloader(self):
        class MtUcertaintyIterator:
            """Sample tasks using uncertainty measure."""

            def __init__(self, my_loader):
                self.my_loader = my_loader
                self.loader_iters = [iter(loader) for loader in self.my_loader.loaders]
                self.loader_iter_sizes = [len(i) for i in self.loader_iters]
                self.max_count = len(self.my_loader)
                self.batch_count = 0

            def __iter__(self):
                return self

            def __next__(self):
                if self.batch_count == self.max_count:
                    self.batch_count = 0
                    raise StopIteration()

                test_batch = {}
                for idx, loader_iter in enumerate(self.loader_iters):
                    try:
                        batch = loader_iter.__next__()
                    except StopIteration:
                        new_loader_iter = iter(self.my_loader.loaders[idx])
                        self.loader_iters[idx] = new_loader_iter
                        batch = new_loader_iter.__next__()

                    test_batch = self.batchify_data(batch, test_batch)

                inputs = {}
                for k, v in test_batch.items():
                    if k not in ["labels"]:
                        inputs[k] = v.detach().to(self.my_loader.args.device)

                with torch.no_grad():
                    model.select_batch_mode = True
                    outputs = model(**inputs)
                    model.select_batch_mode = False

                (
                    test_batch_entropy,
                    test_batch_entropy_mean,
                    max_mean_batch_entropy,
                ) = outputs[-3:]

                for _, v in inputs.items():
                    del v  # free GPU mem
                del inputs

                test_batch_entropy_mean = (
                    test_batch_entropy_mean / max_mean_batch_entropy
                )
                test_batch_entropy = test_batch_entropy * test_batch_entropy_mean

                select_size = min(
                    self.my_loader.args.train_batch_size,
                    test_batch["input_ids"].shape[0],
                )  # Handled the last batch if it is lower than the batch size

                top_entropy = torch.topk(test_batch_entropy, select_size)

                for k, v in test_batch.items():
                    test_batch[k] = torch.index_select(v, 0, top_entropy.indices)

                self.batch_count += 1

                return test_batch

            @staticmethod
            def batchify_data(data, curr_batch):
                for k in data.keys():
                    if k in curr_batch.keys():
                        curr_batch[k] = torch.cat((curr_batch[k], data[k]), dim=0)
                    else:
                        curr_batch[k] = data[k]
                return curr_batch

        class CustomLoader:
            def __init__(self, loaders, datasets, loader_args):
                self.loaders = loaders
                self.dataset = datasets
                self.args = loader_args
                self.current_epoch = 0

            def __iter__(self):
                iterator = MtUcertaintyIterator(self)

                # for determinism across runs
                # https://github.com/pytorch/examples/issues/501
                for l in self.loaders:
                    if isinstance(l.sampler, DistributedSampler):
                        l.sampler.set_epoch(self.current_epoch)
                self.current_epoch += 1
                return iterator

            def __len__(self):
                loader_len = [len(loader) for loader in self.loaders]
                if self.args.uniform_mt_sampling:
                    return int(
                        self.args.percent_of_max_data_size
                        * max(loader_len)
                        * len(self.loaders)
                        / self.args.train_batch_size
                    )
                elif self.args.uncert_batch:
                    return int(
                        max(loader_len)
                        * len(self.loaders)
                        * self.args.percent_of_max_data_size
                    )
                else:
                    return sum(loader_len)

        model = self.model
        tasks = self.data_args.tasks

        data_loaders = []
        for dataset in self.train_dataset.datasets:
            train_sampler = (
                RandomSampler(dataset)
                if self.args.local_rank == -1
                else DistributedSampler(dataset)
            )

            data_loader = DataLoader(
                dataset,
                batch_size=self.args.train_batch_size,
                sampler=train_sampler,
                collate_fn=self.data_collator.collate_batch,
            )
            data_loaders.append(data_loader)

        return CustomLoader(data_loaders, self.train_dataset, self.args)

    def evaluate(
        self,
        eval_dataset: Optional[Dataset] = None,
        prediction_loss_only: Optional[bool] = None,
        context: str = None,
        do_test_if_needed: bool = True,
    ):
        datasets = eval_dataset or self.eval_datasets
        logger.info("*** Evaluate on dev ***")
        for task_name, eval_dataset in datasets.items():
            logger.info(task_name)
            self.compute_metrics = self.build_compute_metrics_fn(eval_dataset)
            eval_dataloader = self.get_eval_dataloader(eval_dataset)
            
            eval_result = self._prediction_loop(
                eval_dataloader, description="Evaluation", task_name=task_name, 
                mode=eval_dataset.mode)
            
            self._log(eval_result.metrics)

            for key, value in eval_result.metrics.items():
                logger.info("  %s = %s", key, value)

        if self.args.tpu_metrics_debug:
            # tpu-comment: Logging debug metrics for PyTorch/XLA (compile, execute times, ops, etc.)
            xm.master_print(met.metrics_report())
            
            
    def predict(
        self,
        eval_dataset: Optional[Dataset] = None,
        prediction_loss_only: Optional[bool] = None,
        scoring_model: Optional[str] = None
    ):
        logging.info("*** Test ***")
        datasets = eval_dataset or self.test_datasets
        for task_name, test_dataset in datasets.items():
            logger.info(task_name)
            
            test_dataloader = self.get_test_dataloader(test_dataset)
            test_result = self._prediction_loop(
                test_dataloader, description="Prediction", task_name=task_name, 
                mode=test_dataset.mode)
            
            self._log(test_result.metrics)
            for key, value in test_result.metrics.items():
                logger.info("  %s = %s", key, value)
                
            softmax = torch.nn.Softmax(dim=1)
            probs = softmax(torch.Tensor(test_result.predictions)).numpy().astype('float64')
            logits = test_result.predictions.astype('float64')
            output_mode = task_output_modes[task_name] 
            if output_mode == "classification":
                predictions = np.argmax(logits, axis=1)
            
            self.run_name = wandb.run.name
            output_test_file = os.path.join(
                self.args.output_dir,
                f"{task_name}_test_iter_{self.run_name}.tsv",
            )
            if scoring_model is None:
                scoring_model = self.run_name
            if self.is_world_master():
                with open(output_test_file, "w") as writer:
                    logger.info("***** Test results {} *****".format(task_name))
                    logger.info("***** Writing as {} *****".format(self.run_name))
                    if output_mode == "regression":
                        writer.write("index\tprediction\n")
                    else:
                        writer.write("index\tscoring_model\tprediction\tprobability\tlogits\n")
                    for index, item in enumerate(predictions):
                        if output_mode == "regression":
                            writer.write("%d\t%3.3f\n" % (index, item))
                        else:
                            i_probs = probs[index,:]
                            i_logits = logits[index,:]
                            i_logits = json.dumps(dict(zip(test_dataset.get_labels(), i_logits)))
                            writer.write(
                                "%d\t%s\t%s\t%3.6f\t%s\n" % (
                                    index, scoring_model, test_dataset.get_labels()[item], 
                                    i_probs[item], i_logits)
                            )
                            
    def _prediction_loop(
        self, dataloader: DataLoader, description: str, task_name: str, mode: str,
        prediction_loss_only: Optional[bool] = None, 
    ) -> PredictionOutput:
        """
        Prediction/evaluation loop, shared by `evaluate()` and `predict()`.
        Works both with or without labels.
        """

        prediction_loss_only = prediction_loss_only if prediction_loss_only is not None else self.prediction_loss_only

        model = self.model
        # multi-gpu eval
        if self.args.n_gpu > 1:
            model = torch.nn.DataParallel(model)
        else:
            model = self.model
        # Note: in torch.distributed mode, there's no point in wrapping the model
        # inside a DistributedDataParallel as we'll be under `no_grad` anyways.

        batch_size = dataloader.batch_size
        logger.info("***** Running %s *****", description)
        logger.info("  Num examples = %d", self.num_examples(dataloader))
        logger.info("  Batch size = %d", batch_size)
        eval_losses: List[float] = []
        preds: torch.Tensor = None
        label_ids: torch.Tensor = None
        model.eval()

        if is_tpu_available():
            dataloader = pl.ParallelLoader(dataloader,
                                           [self.args.device]).per_device_loader(self.args.device)

        for inputs in tqdm(dataloader, desc=description):
            has_labels = any(
                inputs.get(k) is not None for k in ["labels", "lm_labels", "masked_lm_labels"])

            for k, v in inputs.items():
                inputs[k] = v.to(self.args.device)

            with torch.no_grad():
                outputs = model(**inputs)
                if has_labels:
                    step_eval_loss, logits = outputs[:2]
                    eval_losses += [step_eval_loss.mean().item()]
                else:
                    logits = outputs[0]

            if not prediction_loss_only:
                if preds is None:
                    preds = logits.detach()
                else:
                    preds = torch.cat((preds, logits.detach()), dim=0)
                if inputs.get("labels") is not None:
                    if label_ids is None:
                        label_ids = inputs["labels"].detach()
                    else:
                        label_ids = torch.cat((label_ids, inputs["labels"].detach()), dim=0)

        if self.args.local_rank != -1:
            # In distributed mode, concatenate all results from all nodes:
            if preds is not None:
                preds = self.distributed_concat(preds,
                                                num_total_examples=self.num_examples(dataloader))
            if label_ids is not None:
                label_ids = self.distributed_concat(label_ids,
                                                    num_total_examples=self.num_examples(dataloader))
        elif is_tpu_available():
            # tpu-comment: Get all predictions and labels from all worker shards of eval dataset
            if preds is not None:
                preds = xm.mesh_reduce("eval_preds", preds, torch.cat)
            if label_ids is not None:
                label_ids = xm.mesh_reduce("eval_label_ids", label_ids, torch.cat)

        # Finally, turn the aggregated tensors into numpy arrays.
        if preds is not None:
            preds = preds.cpu().numpy()
        if label_ids is not None:
            label_ids = label_ids.cpu().numpy()

        if self.compute_metrics is not None and preds is not None and label_ids is not None:
            metrics = self.compute_metrics(EvalPrediction(predictions=preds, label_ids=label_ids))
        else:
            metrics = {}
        if len(eval_losses) > 0:
            metrics[f"{task_name}_{mode}_loss"] = np.mean(eval_losses)

        # Prefix all keys with {task_name}_{model}_
        for key in list(metrics.keys()):
            if not key.startswith(f"{task_name}_{mode}_"):
                metrics[f"{task_name}_{mode}_{key}"] = metrics.pop(key)

        return PredictionOutput(predictions=preds, label_ids=label_ids, metrics=metrics)
                            
                            
                       
                            
    @staticmethod
    def build_compute_metrics_fn(
        eval_dataset
    ) -> Callable[[EvalPrediction], Dict]:
        def compute_metrics_fn(p: EvalPrediction):
            return compute_task_metrics(eval_dataset.task_name, p)

        return compute_metrics_fn
