import pylab as pl
from get_fish_info import get_fish_info
from fit_integrator_model import get_model_result, get_target_result
import numpy as np
from pathlib import Path
import gmm_model_fit
import pandas as pd
from pymoo.factory import get_problem, get_visualization, get_decomposition

# import random
#
# for dt in [0.001, 0.002, 0.005, 0.01, 0.1]:
#
#     tau = 4
#     Is = np.arange(0, 30, dt)
#     xs = np.empty_like(Is)
#     xs[0]
#
#     for i in range(1, len(Is)):
#         dx = random.gauss(0.2, 5) - xs[i - 1]
#         xs[i] = xs[i - 1] + dx * dt / tau
#     pl.plot(Is, xs)
# pl.show()
# sdf

root_path = Path("/Users/arminbahl/Desktop/mutant_behavior_data/surrogate_fish1")
#root_path = Path("/Users/arminbahl/Desktop/mutant_behavior_data/scn1lab_NIBR")
#root_path = Path("/Users/arminbahl/Desktop/mutant_behavior_data/disc1_hetinx")
df = pd.read_hdf(root_path / "all_data.h5", key="all_bouts")
#
# df_extracted_features, df_extracted_binned_features, \
# df_extracted_binned_features_same_direction, \
# df_extracted_binned_features_heading_angle_change_histograms, \
# df_extracted_binned_features_inter_bout_interval_histograms = get_mean_fish_info(df)
#
# print(df_extracted_features)
# pl.plot(df_extracted_features.loc["wt", :]["correctness"])
# pl.plot(df_extracted_features.loc["het", :]["correctness"])
# pl.plot(df_extracted_features.loc["hom", :]["correctness"])
#
# pl.figure()
# pl.plot(df_extracted_features.loc["wt", :]["inter_bout_interval"])
# pl.plot(df_extracted_features.loc["het", :]["inter_bout_interval"])
# pl.plot(df_extracted_features.loc["hom", :]["inter_bout_interval"])
#
# pl.figure()
# pl.plot(df_extracted_binned_features.loc["wt", 0])
# pl.plot(df_extracted_binned_features.loc["wt", 1])
# pl.plot(df_extracted_binned_features.loc["wt", 2])
# pl.plot(df_extracted_binned_features.loc["wt", 3])
#
# pl.figure()
# pl.plot(df_extracted_binned_features_same_direction.loc["wt"])
# pl.plot(df_extracted_binned_features_same_direction.loc["het"])
# pl.plot(df_extracted_binned_features_same_direction.loc["hom"])
#
#
# pl.figure()
# pl.plot(df_extracted_binned_features_heading_angle_change_histograms.loc["wt", 0])
# pl.plot(df_extracted_binned_features_heading_angle_change_histograms.loc["wt", 1])
# pl.plot(df_extracted_binned_features_heading_angle_change_histograms.loc["wt", 2])
# pl.plot(df_extracted_binned_features_heading_angle_change_histograms.loc["wt", 3])
#
# pl.show()
#
#
# pl.show()
#
#
# print(df_extracted_features)
# gg
# sdf

genotype = "hom"

target_df_correctness_as_function_of_coherence, \
target_df_inter_bout_interval_as_function_of_coherence, \
target_df_binned_correctness, \
target_df_binned_same_direction, \
target_df_binned_features_heading_angle_change_histograms, \
target_df_binned_features_inter_bout_interval_histograms, \
target_df_gmm_fitting_results = get_target_result(root_path, genotype)


# colors = ["#000000", "#330000",  "#990000", "#CC3333"]
#
# for i in range(4):
#     pl.plot(target_df_binned_features_heading_angle_change_histograms.loc[i, :].droplevel("stim"), label=f"Coherence {i*25}%", color=colors[i], linewidth=2)
#
# pl.xlabel("Heading angle change (deg)")
# pl.ylabel("Probability")
# pl.legend()
#
# fig = pl.figure()
# fig.suptitle("Target functions")
# pl.subplot(211)
# pl.plot(target_df_correctness_as_function_of_coherence, 'o-', color='black')
# pl.xlabel("Coherence (%)")
# pl.ylabel("Correctness (%)")
# pl.subplot(212)
# pl.plot(target_df_inter_bout_interval_as_function_of_coherence, 'o-', color='black')
# pl.xlabel("Coherence (%)")
# pl.ylabel("Inter-bout interval (s)")
#

medianprops = dict(linestyle='-.', linewidth=2.5, color='firebrick')

errornames = ["Error: 'Correctness as function of coherence'",
              "Error: 'Inter-bout interval as function of coherence'",
              "Error: 'Binned correctness at 25, 50, 100 %'",
              "Error: 'Binned same direction'",
              "Error: 'Histogram weights'"]

#errornames = ["Mixed"]

repeat = 1
X = np.load(root_path / f"leaky_integrator_model2_X_{genotype}_{repeat}.npy")
F = np.load(root_path / f"leaky_integrator_model2_F_{genotype}_{repeat}.npy")
#
#
# for i in range(7):
#     F[-1, :, i] = F[-1, :, i] / np.max(F[-1, :, i])
# print(F.shape)
#
# i6 = np.argmin(F[-1, :, 0] + F[-1, :, 1] + F[-1, :, 2] + F[-1, :, 3] + F[-1, :, 4] + F[-1, :, 5] + F[-1, :, 6])
# print(F[-1, i6, 0])
# dd
#get_decomposition("asf").do(F[-1], [1, 1, 1, 1, 1, 1, 1]).argmin()
#print(I)
#sdfsdf
#X = np.load(root_path / f"leaky_integrator_model2_X_{genotype}_{repeat}_single_error.npy")
#F = np.load(root_path / f"leaky_integrator_model2_F_{genotype}_{repeat}_single_error.npy")

# from pymoo.factory import get_decision_making, get_reference_directions
#
# ref_dirs = get_reference_directions("das-dennis", 4, n_partitions=12)
# F = get_problem("dtlz1").pareto_front(ref_dirs)
#
# weights = np.array([10.25, 10.25, 0.25, 0.25])
# a, pseudo_weights = get_decision_making("pseudo-weights", weights).do(F, return_pseudo_weights=True)
# pl.plot(F[:, 0], F[:,1], 'o')
# pl.plot(F[a, 0], F[a,1], 'o')
# pl.show()
#
# print(a, pseudo_weights, F.shape)
# ghj
from pymoo.factory import get_decision_making, get_reference_directions


#weights = [1000, 1000, 1000, 0, 0, 0, 0]
#a, pseudo_weights = get_decision_making("pseudo-weights", weights).do(F[-1], return_pseudo_weights=True)
#print(pseudo_weights[0])
#print(a, pseudo_weights)
#dfg
for i in range(5):
    #pl.hist(F[-1, :, i])
    #pl.show()

    #print(np.percentile(F[-1, :, i], 75))
    #print(np.max(F[-1, :, i]) - np.min(F[-1, :, i]))
    F[-1, :, i] = F[-1, :, i] / np.percentile(F[-1, :, i], 75)
# print(F.shape)
#

#i6 = a

#i1 = np.argmin(F[-1, :, 0])
# i2 = np.argmin(F[-1, :, 1])
# i3 = np.argmin(F[-1, :, 0] + F[-1, :, 1]*500)
# i4 = np.argmin(F[-1, :, 0] + F[-1, :, 1]*500 + F[-1, :, 3])
# i5 = np.argmin(F[-1, :, 0] + F[-1, :, 1]*500 + F[-1, :, 3] + F[-1, :, 5]*0.25)
# #i6 = np.argmin(F[-1, :, 0] + F[-1, :, 1]*500 + F[-1, :, 3] + F[-1, :, 5]*0.25 + F[-1, :, 6]*5800)
# #i6 = np.argmin(F[-1, :, 0] + F[-1, :, 1] * 2500 + F[-1, :, 3] * 5 + F[-1, :, 5] * 0.5 + F[-1, :, 6] * 6800)
# i6 = np.argmin(F[-1, :, 0]*500 + F[-1, :, 1]*2500 + F[-1, :, 3]*50 + F[-1, :, 5]*0.5 + F[-1, :, 6]*4500)

i6 = np.argmin(F[-1, :, 0] + 3*F[-1, :, 1] + F[-1, :, 2] + F[-1, :, 3] + F[-1, :, 4])


#from pymoo.factory import get_decision_making
#dm = get_decision_making("high-tradeoff")

#I = dm.do(pf)

# print(F.shape)
# np.set_printoptions(precision=4, suppress=True)
# print((X[-1, i]))
# #gdfgh
# for error_i in range(len(errornames)):
#     pl.figure()
#     pl.title(errornames[error_i])
#     bp = pl.boxplot(F[:, :, error_i].T, whis=[5, 95], showfliers=False, medianprops=medianprops)
#     for gen in range(50):
#         sc = pl.scatter([gen+1], [F[gen, :, error_i].min()], s=5, marker='.', c='firebrick')
#     pl.yscale("log")
#     pl.xlabel("Generation")
#     pl.ylabel("Log Error")
# pl.show()
# dd
#
# pl.figure()
# pl.title("Compromise between all error functions")
# #error = F[:, :, 0] + F[:, :, 1]*500 + F[:, :, 3] + F[:, :, 5]*0.25 + F[:, :, 6]*500
# error = F[:, :, 0] + F[:, :, 1]*2500 + F[:, :, 3]*5 + F[:, :, 5]*0.5 + F[:, :, 6]*1500
#
# bp = pl.boxplot(error.T, whis=[5, 95], showfliers=False, medianprops=medianprops)
# for gen in range(50):
#     sc = pl.scatter([gen + 1], [error[gen].min()], s=10, marker='.', c='firebrick')
# pl.yscale("log")
# pl.xlabel("Generation")
# pl.ylabel("Log Error")
# pl.show()

# pl.figure()
# pl.scatter(F[-1, :, 0], F[-1, :, 1], s=10, marker='.', c='C0', label='Individual')
# pl.scatter(F[-1, i1, 0], F[-1, i1, 1], s=15, marker='o', c='C1', label="Best for 'Correctness as function of coherence'")
# pl.scatter(F[-1, i2, 0], F[-1, i2, 1], s=15, marker='o', c='C2', label="Best for 'Inter-bout interval as function of coherence'")
# pl.scatter(F[-1, i3, 0], F[-1, i3, 1], s=15, marker='o', c='C3', label="Compromise")
# pl.legend()
# pl.xlabel(errornames[0])
# pl.ylabel(errornames[1])
#
#
# pl.figure()
# pl.scatter(F[-1, :, 0] + F[-1, :, 1]*500, F[-1, :, 3], s=10, marker='.', c='C0', label='Individual')
# pl.scatter(F[-1, i1, 0] + F[-1, i1, 1]*500, F[-1, i1, 3], s=15, marker='o', c='C1', label="Best for 'Correctness as function of coherence'")
# pl.scatter(F[-1, i2, 0] + F[-1, i2, 1]*500, F[-1, i2, 3], s=15, marker='o', c='C2', label="Best for 'Inter-bout interval as function of coherence'")
# pl.scatter(F[-1, i3, 0] + F[-1, i3, 1]*500, F[-1, i3, 3], s=15, marker='o', c='C3', label="Compromise between 1 and 2")
# pl.scatter(F[-1, i4, 0] + F[-1, i4, 1]*500, F[-1, i4, 3], s=15, marker='o', c='C4', label="Compromise between all")
# pl.legend()
# pl.xlabel("Compromise between 1 and 2")
# pl.ylabel(errornames[3])
#
# pl.figure()
# pl.scatter(F[-1, :, 0] + F[-1, :, 1]*500 + F[-1, :, 3], F[-1, :, 5], s=10, marker='.', c='C0', label='Individual')
# pl.scatter(F[-1, i1, 0] + F[-1, i1, 1]*500 + F[-1, i1, 3], F[-1, i1, 5], s=15, marker='o', c='C1', label="Best for 'Correctness as function of coherence'")
# pl.scatter(F[-1, i2, 0] + F[-1, i2, 1]*500 + F[-1, i2, 3], F[-1, i2, 5], s=15, marker='o', c='C2', label="Best for 'Inter-bout interval as function of coherence'")
# pl.scatter(F[-1, i3, 0] + F[-1, i3, 1]*500 + F[-1, i3, 3], F[-1, i3, 5], s=15, marker='o', c='C3', label="Compromise between 1 and 2")
# pl.scatter(F[-1, i4, 0] + F[-1, i4, 1]*500 + F[-1, i4, 3], F[-1, i4, 5], s=15, marker='o', c='C4', label="Compromise between 1, 2, and 3")
# pl.scatter(F[-1, i5, 0] + F[-1, i5, 1]*500 + F[-1, i5, 3], F[-1, i5, 5], s=15, marker='o', c='C5', label="Compromise between all")
# pl.legend()
# pl.xlabel("Compromise between 1, 2, and 3")
# pl.ylabel(errornames[5])
#
#
# pl.figure()
# pl.scatter(F[-1, :, 0] + F[-1, :, 1]*500 + F[-1, :, 3] + F[-1, :, 5]*0.25, F[-1, :, 6], s=10, marker='.', c='C0', label='Individual')
# pl.scatter(F[-1, i1, 0] + F[-1, i1, 1]*500 + F[-1, i1, 3] + F[-1, i1, 5]*0.25, F[-1, i1, 6], s=15, marker='o', c='C1', label="Best for 'Correctness as function of coherence'")
# pl.scatter(F[-1, i2, 0] + F[-1, i2, 1]*500 + F[-1, i2, 3] + F[-1, i2, 5]*0.25, F[-1, i2, 6], s=15, marker='o', c='C2', label="Best for 'Inter-bout interval as function of coherence'")
# pl.scatter(F[-1, i3, 0] + F[-1, i3, 1]*500 + F[-1, i3, 3] + F[-1, i3, 5]*0.25, F[-1, i3, 6], s=15, marker='o', c='C3', label="Compromise between 1 and 2")
# pl.scatter(F[-1, i4, 0] + F[-1, i4, 1]*500 + F[-1, i4, 3] + F[-1, i4, 5]*0.25, F[-1, i4, 6], s=15, marker='o', c='C4', label="Compromise between 1, 2, and 3")
# pl.scatter(F[-1, i5, 0] + F[-1, i5, 1]*500 + F[-1, i5, 3] + F[-1, i5, 5]*0.25, F[-1, i5, 6], s=15, marker='o', c='C5', label="Compromise between 1, 2, 3, and 4")
# pl.scatter(F[-1, i6, 0] + F[-1, i6, 1]*500 + F[-1, i6, 3] + F[-1, i6, 5]*0.25, F[-1, i6, 6], s=15, marker='o', c='C6', label="Compromise between all")
# pl.legend()
# pl.xlabel("Compromise between 1, 2, 3, and 4")
# pl.ylabel(errornames[6])
#
# fig = pl.figure()
# model_df_correctness_as_function_of_coherence, \
# model_df_inter_bout_interval_as_function_of_coherence, \
# model_df_binned_correctness, \
# model_df_binned_same_direction, \
# model_df_binned_features_heading_angle_change_histograms, \
# model_df_binned_features_inter_bout_interval_histograms, \
# model_df_gmm_fitting_results = get_model_result(X[-1, i1])
# fig.suptitle("Best for 'Correctness as function of coherence'")
# pl.subplot(211)
# pl.plot([0, 25, 50, 100], target_df_correctness_as_function_of_coherence.values, 'o-', color='black')
# pl.plot([0, 25, 50, 100], model_df_correctness_as_function_of_coherence.values, 'o--', color='C1')
# pl.xlabel("Coherence (%)")
# pl.ylabel("Correctness (%)")
# pl.subplot(212)
# pl.plot([0, 25, 50, 100], target_df_inter_bout_interval_as_function_of_coherence.values, 'o-', color='black')
# pl.plot([0, 25, 50, 100], model_df_inter_bout_interval_as_function_of_coherence.values, 'o--', color='C1')
# pl.xlabel("Coherence (%)")
# pl.ylabel("Inter-bout interval (s)")
#
# fig = pl.figure()
# model_df_correctness_as_function_of_coherence, \
# model_df_inter_bout_interval_as_function_of_coherence, \
# model_df_binned_correctness, \
# model_df_binned_same_direction, \
# model_df_binned_features_heading_angle_change_histograms, \
# model_df_binned_features_inter_bout_interval_histograms, \
# model_df_gmm_fitting_results = get_model_result(X[-1, i2])
# fig.suptitle("Best for 'Inter-bout interval as function of coherence'")
# pl.subplot(211)
# pl.plot([0, 25, 50, 100], target_df_correctness_as_function_of_coherence.values, 'o-', color='black')
# pl.plot([0, 25, 50, 100], model_df_correctness_as_function_of_coherence.values, 'o--', color='C2')
# pl.xlabel("Coherence (%)")
# pl.ylabel("Correctness (%)")
# pl.subplot(212)
# pl.plot([0, 25, 50, 100], target_df_inter_bout_interval_as_function_of_coherence.values, 'o-', color='black')
# pl.plot([0, 25, 50, 100], model_df_inter_bout_interval_as_function_of_coherence.values, 'o--', color='C2')
# pl.xlabel("Coherence (%)")
# pl.ylabel("Inter-bout interval (s)")
#
# fig = pl.figure()
# model_df_correctness_as_function_of_coherence, \
# model_df_inter_bout_interval_as_function_of_coherence, \
# model_df_binned_correctness, \
# model_df_binned_same_direction, \
# model_df_binned_features_heading_angle_change_histograms, \
# model_df_binned_features_inter_bout_interval_histograms, \
# model_df_gmm_fitting_results = get_model_result(X[-1, i3])
# fig.suptitle("Compromise between 'Correctness and inter-bout interval as function of coherence'")
# pl.subplot(211)
# pl.plot([0, 25, 50, 100], target_df_correctness_as_function_of_coherence.values, 'o-', color='black')
# pl.plot([0, 25, 50, 100], model_df_correctness_as_function_of_coherence.values, 'o--', color='C3')
# pl.xlabel("Coherence (%)")
# pl.ylabel("Correctness (%)")
# pl.subplot(212)
# pl.plot([0, 25, 50, 100], target_df_inter_bout_interval_as_function_of_coherence.values, 'o-', color='black')
# pl.plot([0, 25, 50, 100], model_df_inter_bout_interval_as_function_of_coherence.values, 'o--', color='C3')
# pl.xlabel("Coherence (%)")
# pl.ylabel("Inter-bout interval (s)")
#
# fig = pl.figure()
# model_df_correctness_as_function_of_coherence, \
# model_df_inter_bout_interval_as_function_of_coherence, \
# model_df_binned_correctness, \
# model_df_binned_same_direction, \
# model_df_binned_features_heading_angle_change_histograms, \
# model_df_binned_features_inter_bout_interval_histograms, \
# model_df_gmm_fitting_results = get_model_result(X[-1, i3])
# fig.suptitle("Compromise between 'Correctness and inter-bout interval as function of coherence'")
# pl.subplot(221)
# pl.plot([0, 25, 50, 100], target_df_correctness_as_function_of_coherence.values, 'o-', color='black')
# pl.plot([0, 25, 50, 100], model_df_correctness_as_function_of_coherence.values, 'o--', color='C3')
# pl.xlabel("Coherence (%)")
# pl.ylabel("Correctness (%)")
# pl.subplot(222)
# pl.plot([0, 25, 50, 100], target_df_inter_bout_interval_as_function_of_coherence.values, 'o-', color='black')
# pl.plot([0, 25, 50, 100], model_df_inter_bout_interval_as_function_of_coherence.values, 'o--', color='C3')
# pl.xlabel("Coherence (%)")
# pl.ylabel("Inter-bout interval (s)")
# pl.subplot(223)
# for i in range(4):
#     pl.plot(target_df_binned_correctness.loc[i, :].droplevel("stim"), 'o-', color='black')
#     pl.plot(model_df_binned_correctness.loc[i, :].droplevel("stim"), 'o--', color='C3')
# pl.xlabel("Correctness (%)")
# pl.ylabel("Time (s)")
#
#
# fig = pl.figure()
# model_df_correctness_as_function_of_coherence, \
# model_df_inter_bout_interval_as_function_of_coherence, \
# model_df_binned_correctness, \
# model_df_binned_same_direction, \
# model_df_binned_features_heading_angle_change_histograms, \
# model_df_binned_features_inter_bout_interval_histograms, \
# model_df_gmm_fitting_results = get_model_result(X[-1, i4])
# fig.suptitle("Compromise between all three error functions")
# pl.subplot(221)
# pl.plot([0, 25, 50, 100], target_df_correctness_as_function_of_coherence.values, 'o-', color='black')
# pl.plot([0, 25, 50, 100], model_df_correctness_as_function_of_coherence.values, 'o--', color='C4')
# pl.xlabel("Coherence (%)")
# pl.ylabel("Correctness (%)")
# pl.subplot(222)
# pl.plot([0, 25, 50, 100], target_df_inter_bout_interval_as_function_of_coherence.values, 'o-', color='black')
# pl.plot([0, 25, 50, 100], model_df_inter_bout_interval_as_function_of_coherence.values, 'o--', color='C4')
# pl.xlabel("Coherence (%)")
# pl.ylabel("Inter-bout interval (s)")
# pl.subplot(223)
# for i in range(4):
#     pl.plot(target_df_binned_correctness.loc[i, :].droplevel("stim"), 'o-', color='black')
#     pl.plot(model_df_binned_correctness.loc[i, :].droplevel("stim"), 'o--', color='C4')
# pl.xlabel("Correctness (%)")
# pl.ylabel("Time (s)")
#
#
# fig = pl.figure()
# model_df_correctness_as_function_of_coherence, \
# model_df_inter_bout_interval_as_function_of_coherence, \
# model_df_binned_correctness, \
# model_df_binned_same_direction, \
# model_df_binned_features_heading_angle_change_histograms, \
# model_df_binned_features_inter_bout_interval_histograms, \
# model_df_gmm_fitting_results = get_model_result(X[-1, i5])
# fig.suptitle("Compromise between all four error functions")
# pl.subplot(221)
# pl.plot([0, 25, 50, 100], target_df_correctness_as_function_of_coherence.values, 'o-', color='black')
# pl.plot([0, 25, 50, 100], model_df_correctness_as_function_of_coherence.values, 'o--', color='C5')
# pl.xlabel("Coherence (%)")
# pl.ylabel("Correctness (%)")
# pl.subplot(222)
# pl.plot([0, 25, 50, 100], target_df_inter_bout_interval_as_function_of_coherence.values, 'o-', color='black')
# pl.plot([0, 25, 50, 100], model_df_inter_bout_interval_as_function_of_coherence.values, 'o--', color='C5')
# pl.xlabel("Coherence (%)")
# pl.ylabel("Inter-bout interval (s)")
# pl.subplot(223)
# for i in range(4):
#     pl.plot(target_df_binned_correctness.loc[i, :].droplevel("stim"), 'o-', color='black')
#     pl.plot(model_df_binned_correctness.loc[i, :].droplevel("stim"), 'o--', color='C5')
# pl.xlabel("Correctness (%)")
# pl.ylabel("Time (s)")
# pl.subplot(224)
# pl.plot(target_df_binned_same_direction, 'o-', color='black')
# pl.plot(model_df_binned_same_direction, 'o--', color='C5')
# pl.xlabel("Time since last bout (s)")
# pl.ylabel("Correctness (%)")

fig = pl.figure()
model_df_correctness_as_function_of_coherence, \
model_df_inter_bout_interval_as_function_of_coherence, \
model_df_binned_correctness, \
model_df_binned_same_direction, \
model_df_binned_features_heading_angle_change_histograms, \
model_df_binned_features_inter_bout_interval_histograms, \
model_df_gmm_fitting_results = get_model_result(X[-1, i6])
fig.suptitle("Compromise between all five error functions")
pl.subplot(231)
pl.plot([0, 25, 50, 100], target_df_correctness_as_function_of_coherence.values, 'o-', color='black')
pl.plot([0, 25, 50, 100], model_df_correctness_as_function_of_coherence.values, 'o--', color='C6')
pl.xlabel("Coherence (%)")
pl.ylabel("Correctness (%)")
pl.subplot(232)
pl.plot([0, 25, 50, 100], target_df_inter_bout_interval_as_function_of_coherence.values, 'o-', color='black')
pl.plot([0, 25, 50, 100], model_df_inter_bout_interval_as_function_of_coherence.values, 'o--', color='C6')
pl.xlabel("Coherence (%)")
pl.ylabel("Inter-bout interval (s)")
pl.subplot(233)
for i in range(4):
    pl.plot(target_df_binned_correctness.loc[i, :].droplevel("stim"), 'o-', color='black')
    pl.plot(model_df_binned_correctness.loc[i, :].droplevel("stim"), 'o--', color='C6')
pl.xlabel("Time (s)")
pl.ylabel("Correctness (%)")
pl.subplot(234)
pl.plot(target_df_binned_same_direction, 'o-', color='black')
pl.plot(model_df_binned_same_direction, 'o--', color='C6')
pl.xlabel("Time since last bout (s)")
pl.ylabel("Correctness (%)")
# pl.subplot(235)
# pl.plot(target_df_gmm_fitting_results.index*25, target_df_gmm_fitting_results["w_left"].values, '-o', color='black', label='s_left')
# pl.plot(target_df_gmm_fitting_results.index*25, target_df_gmm_fitting_results["w_center"].values, '-o', color='black', label='s_center')
# pl.plot(target_df_gmm_fitting_results.index*25, target_df_gmm_fitting_results["w_right"].values, '-o', color='black', label='s_right')
#
# pl.plot(model_df_gmm_fitting_results.index*25, model_df_gmm_fitting_results["w_left"].values, '--o', color='C6', label='s_left')
# pl.plot(model_df_gmm_fitting_results.index*25, model_df_gmm_fitting_results["w_center"].values, '--o', color='C6', label='s_center')
# pl.plot(model_df_gmm_fitting_results.index*25, model_df_gmm_fitting_results["w_right"].values, '--o', color='C6', label='s_right')
# pl.xlabel("Coherence (%)")
# pl.ylabel("Weight")
# pl.legend()
pl.subplot(235)
for i in range(4):
    pl.plot(target_df_binned_features_heading_angle_change_histograms.loc[i, :].droplevel("stim"), color=f"black")
    pl.plot(model_df_binned_features_heading_angle_change_histograms.loc[i, :].droplevel("stim"), color=f"C6", linestyle='--')
pl.xlabel("Heading angle change")
pl.ylabel("Probability")
pl.show()


found_parameters = []
for repeat in range(12):
    for genotype in ["wt", "het", "hom"]:
        X = np.load(root_path / f"leaky_integrator_model2_X_{genotype}_{repeat}.npy")
        F = np.load(root_path / f"leaky_integrator_model2_F_{genotype}_{repeat}.npy")

        for i in range(5):
            #F[-1, :, i] = F[-1, :, i] / np.median(F[-1, :, i])
            F[-1, :, i] = F[-1, :, i] / np.percentile(F[-1, :, i], 75)
        #i6 = np.argmin(F[-1, :, 0] + F[-1, :, 1] + 5 * F[-1, :, 3] + F[-1, :, 5] + 5 * F[-1, :, 6])
        #i6 = np.argmin(F[-1, :, 0] + 5 * F[-1, :, 1] + 20 * F[-1, :, 4] + F[-1, :, 5] + 5 * F[-1, :, 6])
        i6 = np.argmin(F[-1, :, 0] + 3 * F[-1, :, 1] + F[-1, :, 2] + F[-1, :, 3] + F[-1, :, 4])
        #i6 = np.argmin(F[-1, :, 0] + 2 * F[-1, :, 1] + F[-1, :, 2] + 3 * F[-1, :, 3] + F[-1, :, 5] + F[-1, :, 6])
        #i6 = np.argmin(F[-1, :, 0] + F[-1, :, 1] * 500 + F[-1, :, 3] + F[-1, :, 5] * 0.25 + F[-1, :, 6] * 500)
        #i6 = np.argmin(F[-1, :, 0] + F[-1, :, 1] * 2500 + F[-1, :, 3] * 5 + F[-1, :, 5] * 0.5 + F[-1, :, 6] * 1500)
        #i6 = np.argmin(F[-1, :, 0]*500 + F[-1, :, 1]*2500 + F[-1, :, 3]*50 + F[-1, :, 5]*0.5 + F[-1, :, 6]*4500)

        found_parameters.append([genotype, repeat, 49] + list(X[-1, i6, :]))

df = pd.DataFrame(found_parameters,
                  columns=["genotype",
                           "repeat",
                           "gen",
                           "tau",
                           "sigma",
                           "T",
                           "p_below",
                           "p_above"]).astype(dtype={"repeat": "int64", "gen": "int64"}, copy=False)

df.set_index(["genotype", 'repeat', 'gen'], inplace=True)
df.sort_index(inplace=True)
df.to_hdf(root_path / "found_parameters.h5", key="parameters", complevel=9)
