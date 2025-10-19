# Copyright (c) Facebook, Inc. and its affiliates.
from typing import List, Union
from collections import OrderedDict, abc
import logging
import time
import datetime
from contextlib import ExitStack, contextmanager
import itertools

import numpy as np
import torch
from torch import nn
import einops
import utils3d

from .logger import log_every_n_seconds
from . import dist_utils as comm
from . import frame_utils
from ..dataloader import build_val_loader
from ..utils.alignment import align_depth_affine, mask_aware_nearest_resize, align_affine_lstsq


def print_csv_format(results):
    """
    Print main metrics in a format similar to Detectron2,
    so that they are easy to copypaste into a spreadsheet.


    Args:
        results (OrderedDict[dict]): task_name -> {metric -> score}
            unordered dict can also be printed, but in arbitrary order
    """
    assert isinstance(results, abc.Mapping) or not len(results), results
    logger = logging.getLogger("bridgedepth")
    for task, res in results.items():
        if isinstance(res, abc.Mapping):
            important_res = [(k, v) for k, v in res.items()]
            logger.info("copypaste: Task: {}".format(task))
            logger.info("copypaste: " + ",".join([k[0] for k in important_res]))
            logger.info("copypaste: " + ",".join(["{0:.4f}".format(k[1]) for k in important_res]))
        else:
            logger.info(f"copypaste: {task}={res}")


class DatasetEvaluator:
    """
    Base class for a dataset evaluator.

    The function :func:`inference_on_dataset` runs the model over
    all samples in the dataset, and have a DatasetEvaluator to process the inputs/outputs.

    This class will accumulate information of the inputs/outputs (by:meth:`process`),
    add produce evaluation results in the end (by :meth:`evaluate`).
    """

    def reset(self):
        """
        Preparation for a new round of evaluation.
        Should be called before starting a round of evaluation.
        """
        pass

    def process(self, inputs, outputs):
        """
        Process the pair of inputs and outputs.
        If they contain batches, the pairs can be consumed one-by-one using `zip`:

        .. code-block:: python

            for input_, output in zip(inputs, outputs):
                # do evaluation on single input/output pair
                ...

        Args:
            inputs (list): the inputs that' used to call the model.
            outputs (list): the return value of `model(inputs)`
        """
        pass

    def evaluate(self):
        """
        Evaluate/summarize the performance, after processing all input/output pairs.

        Returns:
            dict:
                A new evaluator class can return a dict of arbitrary format
                as long as the user can process the results.
                In our train_net.py, we expect the following format:

                * key: the name of the task (e.g., bbox)
                * value: a dict of {metric name: score}, e.g.: {"AP50": 80}
        """
        pass


class DatasetEvaluators(DatasetEvaluator):
    """
    Wrapper class to combine multiple :class:`DatasetEvaluator` instances.

    This class dispatches every evaluation call to
    all of its :class:`DatasetEvaluator`.
    """

    def __init__(self, evaluators):
        """
        Args:
            evaluators (list): the evaluators to combine.
        """
        super().__init__()
        self._evaluators = evaluators

    def reset(self):
        for evaluator in self._evaluators:
            evaluator.reset()

    def process(self, inputs, outputs):
        for evaluator in self._evaluators:
            evaluator.process(inputs, outputs)

    def evaluate(self):
        results = OrderedDict()
        for evaluator in self._evaluators:
            result = evaluator.evaluate()
            if comm.is_main_process() and result is not None:
                for k, v in result.items():
                    assert (
                            k not in results
                    ), "Different evaluators produce results with the same key {}".format(k)
                    results[k] = v
        return results


def inference_on_dataset(
        model, data_loader, evaluator: Union[DatasetEvaluator, List[DatasetEvaluator], None], **kwargs
):
    """
    Run model on the data_loader and evaluate the metrics with evaluator.
    Also benchmark the inference speed of `model.__call__` accurately.
    The model will be used in eval mode.

    Args:
        model (callable): a callable which takes an object from
            `data_loader` and returns some outputs.

            If it's an nn.Module, it will be temporarily set to `eval` mode.
            If you wish to evaluate a model in `training` model instead, you can
            wrap the given model and override it behavior of `.eval()` and `.train()`.

        data_loader: an iterable objects with a length.
            The elements it generates will be the inputs to the model.
        evaluator: the evaluator(s) to run. Use `None` if you only want to benchmark,
            but don't want to do any evaluation.
        kwargs: additional arguments to model

    Returns:
        The return value of `evaluator.evaluate()`
    """
    num_devices = comm.get_dist_info()[1]
    logger = logging.getLogger("bridgedepth")
    logger.info("Start inference on {} batches".format(len(data_loader)))

    total = len(data_loader)  # inference data loader must have a fixed length
    if evaluator is None:
        # create a no-op evaluator
        evaluator = DatasetEvaluators([])
    if isinstance(evaluator, abc.MutableSequence):
        evaluator = DatasetEvaluators(evaluator)
    evaluator.reset()

    num_warmup = min(5, total - 1)
    start_time = time.perf_counter()
    total_data_time = 0
    total_compute_time = 0
    total_eval_time = 0
    with ExitStack() as stack:
        if isinstance(model, nn.Module):
            stack.enter_context(inference_context(model))
        stack.enter_context(torch.no_grad())

        start_data_time = time.perf_counter()
        for idx, inputs in enumerate(data_loader):
            total_data_time += time.perf_counter() - start_data_time
            if idx == num_warmup:
                start_time = time.perf_counter()
                total_data_time = 0
                total_compute_time = 0
                total_eval_time = 0

            start_compute_time = time.perf_counter()
            outputs = model(inputs, **kwargs)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            total_compute_time += time.perf_counter() - start_compute_time

            start_eval_time = time.perf_counter()
            evaluator.process(inputs, outputs)
            total_eval_time += time.perf_counter() - start_eval_time

            iters_after_start = idx + 1 - num_warmup * int(idx >= num_warmup)
            data_seconds_per_iter = total_data_time / iters_after_start
            compute_seconds_per_iter = total_compute_time / iters_after_start
            eval_seconds_per_iter = total_eval_time / iters_after_start
            total_seconds_per_iter = (time.perf_counter() - start_time) / iters_after_start
            if idx >= num_warmup * 2 or compute_seconds_per_iter > 5:
                eta = datetime.timedelta(seconds=int(total_seconds_per_iter * (total - idx - 1)))
                log_every_n_seconds(
                    logging.INFO,
                    (
                        f"Inference done {idx + 1}/{total}. "
                        f"Dataloading: {data_seconds_per_iter:.4f} s/iter. "
                        f"Inference: {compute_seconds_per_iter:.4f} s/iter. "
                        f"Eval: {eval_seconds_per_iter:.4f} s/iter. "
                        f"Total: {total_seconds_per_iter:.4f} s/iter. "
                        f"ETA={eta}"
                    ),
                    n=5,
                    name="bridgedepth"
                )
            start_data_time = time.perf_counter()

    # Measure the time only for this worker (before synchronization barrier)
    total_time = time.perf_counter() - start_time
    total_time_str = str(datetime.timedelta(seconds=total_time))
    # NOTE this format is parsed by grep
    logger.info(
        "Total inference time: {} ({:.6f} s / iter per device, on {} devices)".format(
            total_time_str, total_time / (total - num_warmup), num_devices
        )
    )
    total_compute_time_str = str(datetime.timedelta(seconds=int(total_compute_time)))
    logger.info(
        "Total inference pure compute time: {} ({:.6f} s / iter per device, on {} devices)".format(
            total_compute_time_str, total_compute_time / (total - num_warmup), num_devices
        )
    )

    results = evaluator.evaluate()
    # An evaluator may return None when not in main process.
    # Replace it by an empty dict instead to make it easier for downstream code
    if results is None:
        results = {}
    return results


@contextmanager
def inference_context(model):
    """
    A context where the model is temporarily changed to eval mode,
    and restored to previous mode afterwards,

    Args:
        model: a torch Module
    """
    training_mode = model.training
    model.eval()
    yield
    model.train(training_mode)


class DispEvaluator(DatasetEvaluator):
    """Evaluate disparity accuracy using metrics."""

    def __init__(self, thres, only_valid, max_disp=None, eval_prop=True, divis_by=8):
        """
        Args:
            thres (list[str] or None): threshold for outlier
            only_valid (bool): whether invalid pixels are excluded from evaluation
            max_disp (int or None): If None, maximum disparity will be regarded as infinity
            eval_prop (bool): whether evaluate the proposal quality.
        """
        # If True, will collect results from all ranks and return evaluation
        # in the main process. Otherwise, will evaluate the results in the current
        # process.
        self._distributed = comm.get_world_size() > 1
        self._max_disp = np.inf if max_disp is None else max_disp
        self._thres = thres
        self._only_valid = only_valid
        self._eval_prop = eval_prop
        self._divis_by = divis_by

    def reset(self):
        self._epe = []
        self._thres_metric = OrderedDict()
        self._d1 = []

        if self._thres is not None:
            for t in self._thres:
                self._thres_metric[t] = []

        self._prop_epe = []
        self._prop_recall_3 = []
        self._prop_recall_8 = []

    def process(self, inputs, outputs):
        """
        Args:
            inputs: the inputs to the model. It is a dict, which contains keys like "img1",
                "ïmg2", "disp".
            outputs: the outputs of the model. It is a dict.
        """
        inputs = [dict(zip(inputs, t)) for t in zip(*inputs.values())]
        outputs = [dict(zip(outputs, t)) for t in zip(*outputs.values())]
        for input, output in zip(inputs, outputs):
            disp_pr = output['disp_pred']
            disp_mono_pr = output['disp_mono']
            disp_gt = input['disp'].to(disp_pr.device)
            valid_gt = input['valid'].to(disp_pr.device)
            if self._only_valid:
                valid = valid_gt & (disp_gt < self._max_disp)
            else:
                valid = disp_gt < self._max_disp
            assert disp_pr.shape == disp_gt.shape, (disp_pr.shape, disp_gt.shape)

            if not torch.any(valid).item():
                continue

            # align
            _, lr_mask, lr_index = mask_aware_nearest_resize(None, valid, (64, 64), return_index=True)
            disp_mono_lr_masked, disp_gt_lr_masked = disp_mono_pr[lr_index][lr_mask], disp_gt[lr_index][lr_mask]
            scale, shift = align_depth_affine(disp_mono_lr_masked, disp_gt_lr_masked, torch.ones_like(disp_mono_lr_masked))
            disp_pr = torch.nn.functional.relu(scale * disp_mono_pr + shift)

            epe = torch.abs(disp_pr - disp_gt)
            epe = epe.flatten()
            val = valid.flatten()

            if (np.isnan(epe[val].mean().item())):
                continue

            self._epe.append(epe[val].mean().item())
            self._d1.append(((epe[val] > 3) & (epe[val] / disp_gt.flatten()[val] > 0.05)).float().mean().item())

            if len(self._thres_metric) > 0:
                for t in self._thres:
                    tf = float(t)
                    out = (epe > tf)
                    self._thres_metric[t].append(out[val].float().mean().item())

            if self._eval_prop:
                proposal = output['proposal'] * 8
                disp_gt_clone = disp_gt.clone()
                disp_gt_clone[~valid_gt] = 0
                padder = frame_utils.InputPadder(disp_gt_clone.shape, mode='nmrf', divis_by=self._divis_by)
                disp_gt_clone = padder.pad(disp_gt_clone[None][None])[0]
                disp_gt = einops.rearrange(disp_gt_clone, '1 1 (h m) (w n) -> (h w) (m n)', m=8, n=8)
                epe = torch.cdist(disp_gt[..., None], proposal[..., None], p=1)
                epe, _ = torch.min(epe, dim=-1, keepdim=False)
                mask = (disp_gt > 0) & (disp_gt < self._max_disp)
                if np.isnan(epe[mask].mean().item()):
                    continue
                self._prop_epe.append(epe[mask].mean().item())
                self._prop_recall_3.append((epe[mask] <= 3).float().mean().item())
                self._prop_recall_8.append((epe[mask] <= 8).float().mean().item())

    def evaluate(self):
        if self._distributed:
            comm.synchronize()
            epe = list(itertools.chain(*comm.gather(self._epe, dst=0)))
            d1 = list(itertools.chain(*comm.gather(self._d1, dst=0)))
            thres_metric = OrderedDict()
            for k, v in self._thres_metric.items():
                thres_metric[k] = list(itertools.chain(*comm.gather(v, dst=0)))
            prop_epe = list(itertools.chain(*comm.gather(self._prop_epe, dst=0)))
            prop_recall_3 = list(itertools.chain(*comm.gather(self._prop_recall_3, dst=0)))
            prop_recall_8 = list(itertools.chain(*comm.gather(self._prop_recall_8, dst=0)))
            if not comm.is_main_process():
                return {}
        else:
            epe = self._epe
            d1 = self._d1
            thres_metric = self._thres_metric
            prop_epe = self._prop_epe
            prop_recall_3 = self._prop_recall_3
            prop_recall_8 = self._prop_recall_8

        epe = torch.tensor(epe).mean().item()
        d1 = torch.tensor(d1).mean().item() * 100
        res = {'epe': epe, 'd1': d1}
        for k, v in thres_metric.items():
            res[f'bad {k}'] = torch.tensor(v).mean().item() * 100
        if self._eval_prop:
            res['prop_epe'] = torch.tensor(prop_epe).mean().item()
            res['prop_recall_3'] = torch.tensor(prop_recall_3).mean().item() * 100
            res['prop_recall_8'] = torch.tensor(prop_recall_8).mean().item() * 100

        results = {'disp': res}
        return results
    

def eval_disp(model, cfg):
    logger = logging.getLogger("bridgedepth")
    results = OrderedDict()
    for idx, dataset_name in enumerate(cfg.DATASETS.TEST):
        data_loader = build_val_loader(cfg, dataset_name)
        # build evaluator for this dataset
        evaluator = DispEvaluator(thres=cfg.TEST.EVAL_THRESH[idx], only_valid=cfg.TEST.EVAL_ONLY_VALID[idx],
                                  max_disp=cfg.TEST.EVAL_MAX_DISP[idx], eval_prop=cfg.TEST.EVAL_PROP[idx],
                                  divis_by=cfg.DATASETS.DIVIS_BY)
        results_i = inference_on_dataset(model, data_loader, evaluator)
        results[dataset_name] = results_i
        if comm.is_main_process():
            assert isinstance(
                results_i, dict
            ), "Evaluator must return a dict on the main process. Got {} instead.".format(
                results_i
            )
            logger.info("Evaluation results for {} in csv format:".format(dataset_name))
            print_csv_format(results_i)

    if len(results) == 1:
        results = list(results.values())[0]
    return results