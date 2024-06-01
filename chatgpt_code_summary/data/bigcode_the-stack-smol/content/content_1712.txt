import hydra
import os

import logging
import json

import numpy as np
import torch
import matplotlib.pyplot as plt
from collections import defaultdict
import json

from IPython import embed
# from AD_models import AD_Time_Series
# from AD_utils import AD_report, AD_dataset, plot_AD_dataset, AD_preprocessing
# import T_models, A_models


import stric.datasets as datasets
import stric.detection_models.time_series_models as models
import stric.detection_models.detector_models as detectors

from stric.detection_models.time_series_models.stric import InterpretableTCNFading
import stric.detection_models.detector_models.likelihood_ratio_estimators as likelihood_ratio_estimators
from stric.detection_models.detector_models.base_detector import Detector

@hydra.main(config_name="config/config_interpretable_model")
def main(cfg):
    data_path = os.path.join(hydra.utils.get_original_cwd(), 'data')

    dataset = datasets.__dict__[cfg.dataset.info.name](
                                                            past_len=cfg.t_model.info.memory_length,
                                                            fut_len=cfg.t_model.info.pred_length,
                                                            data_path=data_path,
                                                            dataset_subset=cfg.dataset.info.subname,
                                                            dataset_index=cfg.dataset.info.index,
                                                            normalize=cfg.dataset.preprocessing.normalize,
                                                          )

    linear_kernel_sizes = cfg.t_model.info.linear_kernel_sizes
    interpretable_kernel_sizes = cfg.t_model.info.memory_length if linear_kernel_sizes is None else linear_kernel_sizes

    ############# Trend parameters  ################
    HP_lams = np.logspace(8, 10, cfg.t_model.info.num_trends_filters)  # Range of values of regularization parameter for HP filter (regulates the regularity of the trend component)
    HP_Ts = [interpretable_kernel_sizes] * cfg.t_model.info.num_trends_filters  # Lenght of the HP filter (here we could choose large numbers if we want to increase the memory of the HP filter)

    ############# Periodic part parameters  ################
    theta = np.random.uniform(2 * np.pi / 20, 2 * np.pi / 10, cfg.t_model.info.n_periodic_poles).reshape(-1, 1)
    r = np.random.uniform(1, 1, cfg.t_model.info.n_periodic_poles).reshape(-1, 1)
    purely_periodic_poles = np.concatenate((r, theta), 1)

    ############# Linear part parameters  ################
    real_poles = np.random.uniform(-1, 1, cfg.t_model.info.n_complex_poles).reshape(-1, 1)
    theta = np.random.uniform(2 * np.pi / 20, 2 * np.pi / 10, cfg.t_model.info.n_complex_poles).reshape(-1, 1)
    r = np.random.uniform(0, 1, cfg.t_model.info.n_complex_poles).reshape(-1, 1)
    complex_poles = np.concatenate((r, theta), 1)

    model = InterpretableTCNFading(data=dataset, test_portion=cfg.t_model.info.test_portion,
                              memory_length=cfg.t_model.info.memory_length, pred_length=cfg.t_model.info.pred_length,
                              input_channels=dataset.n_timeseries, output_channels=dataset.n_timeseries,

                              linear_kernel_sizes=interpretable_kernel_sizes,

                              HP_lams=HP_lams, HP_Ts=HP_Ts,

                              purely_periodic_poles=purely_periodic_poles,

                              real_poles=real_poles,
                              complex_poles=complex_poles,

                              num_channels_TCN=cfg.t_model.info.num_channels_TCN,
                              kernel_size_TCN=cfg.t_model.info.kernel_size_TCN,
                              dropout_TCN=cfg.t_model.info.dropout_TCN,
                              learnable_filters=False, random_init=False,
                              ).to(cfg.device)

    model.train_model(bs=cfg.t_model.info.bs, lr=cfg.t_model.info.lr, epochs=cfg.t_model.info.epochs)

    # To visualize predictions per time-series (this plots all the available time-series)
    model.visualize(save=cfg.save_images)

    # Test predictive performance of the trained_model: see prediction errors across time-series for training and test
    ind = 4
    train_residuals, test_residuals = model.get_residuals(ind=ind)

    # Save results
    predictions_logs = defaultdict(list)
    predictions_logs['train_residuals'] = train_residuals.tolist()
    predictions_logs['test_residuals'] = test_residuals.tolist()
    predictions_logs['train_residuals_stds'] = train_residuals.std(0).tolist()
    predictions_logs['test_residuals_stds'] = test_residuals.std(0).tolist()
    predictions_logs['train_residuals_stds_mean'] = train_residuals.std(0).mean().item()
    predictions_logs['test_residuals_stds_mean'] = test_residuals.std(0).mean().item()

    with open('predictions_logs.json', 'w') as file:
        json.dump(predictions_logs, file)

    # Plot Interepretable decomposition
    _ = model.get_components(ind=None, save=cfg.save_images)

    # Anomaly detection
    ####### Detector' HPs ########
    kernel_length_scale = cfg.a_model.info.kernel_length_scale * test_residuals.std()
    kernel_type = cfg.a_model.info.kernel_type
    kernel_hps = {'length_scales': torch.tensor(kernel_length_scale), 'train_length_scales': False,
                  'scale_factor': torch.tensor(1.), 'train_scale_factor': False}
    ones = np.ones(dataset.n_timeseries)
    ####### Detector' HPs ########
    a_model = Detector(test_residuals, detectors.__dict__[cfg.a_model.type],
                       cfg.a_model.info.kernel_type, kernel_hps,  win_length=cfg.a_model.info.k, n=cfg.a_model.info.n,
                       device=cfg.device)
    a_model.fit()

    log_lik = a_model.get_future_log_lik()
    a_labels = a_model.get_anomaly_labels(cfg.a_model.info.threshold * ones)

    a_model.visualize_anomaly_scores(save=cfg.save_images)
    a_model.visualize_anomaly_labels(thresholds=cfg.a_model.info.threshold * ones, save=cfg.save_images)

    # Save results
    anomaly_logs = defaultdict(list)
    anomaly_logs['log_lik'] = log_lik.tolist()
    anomaly_logs['a_labels'] = a_labels.tolist()

    with open('anomaly_logs.json', 'w') as file:
        json.dump(anomaly_logs, file)


if __name__ == "__main__":
    main()