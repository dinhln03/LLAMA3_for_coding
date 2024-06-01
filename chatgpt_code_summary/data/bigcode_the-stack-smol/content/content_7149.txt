# libraries imported
import os
import pathlib
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties

# uses predict folder code to run predictor on entire folders
from predict_folder import predict_dir


def predict_dirs(model_type, target_prediction):
    directory_name = str(pathlib.Path(__file__).parent.resolve()) + "/data/"
    # list holding all folders to use
    folder_names = [
        "Novice Pointing",
        "Novice Tracing",
        "Surgeon Pointing",
        "Surgeon Tracing"
    ]

    # finds the directories
    directories = []
    for folder_name in folder_names:
        os.fsencode(directory_name + folder_name + "/")

    # puts all txt files' names in a list
    file_names = []
    for directory in directories:
        for file in os.listdir(directory):
            file_names.append(os.fsdecode(file))

    if model_type == "SVM":
        C_parameters = []
        epsilon_parameters = []
    elif model_type == "Random Forest":
        num_trees = []

    all_r2 = []
    all_tremor_r2 = []
    all_rmse = []
    all_tremor_rmse = []
    all_training_times = []
    all_prediction_times = []
    # runs the prediction code for each folder
    for folder_name in folder_names:
        [hyperparameters, r2_scores, tremor_r2_scores, rmses, tremor_rmses, training_times, prediction_times] \
            = predict_dir(directory_name + folder_name, model_type, target_prediction)

        if model_type == "SVM":
            C_parameters.extend(hyperparameters[:2])
            epsilon_parameters.extend(hyperparameters[2:])
        elif model_type == "Random Forest":
            num_trees.extend(hyperparameters)

        all_r2.append(r2_scores)
        all_tremor_r2.append(tremor_r2_scores)
        all_rmse.append(rmses)
        all_tremor_rmse.append(tremor_rmses)
        all_training_times.append(training_times)
        all_prediction_times.append(prediction_times)

    if model_type == "SVM":
        maxmin_hyperparameters = [
            np.max(C_parameters), np.min(C_parameters),
            np.max(epsilon_parameters), np.min(epsilon_parameters)
        ]
        print("\nHyperparameters of the model [C_max, C_min, epsilon_max, epsilon_min]:", maxmin_hyperparameters)
    elif model_type == "Random Forest":
        maxmin_hyperparameters = [np.max(num_trees), np.min(num_trees)]
        print("\nHyperparameters of the model [n_estimators_max, n_estimators_min]:", maxmin_hyperparameters)
    # prints the average metrics for all datasets
    print(
        "\nAverage R2 score of the model:", str(np.mean(all_r2)) + "%",
        "\nAverage R2 score of the tremor component:", str(np.mean(all_tremor_r2)) + "%",
        "\nAverage RMS error of the model:", str(np.mean(all_rmse)) + "mm",
        "\nAverage RMS error of the tremor component:", str(np.mean(all_tremor_rmse)) + "mm",
        "\nAverage time taken to train:", str(np.mean(all_training_times)) + "s",
        "\nAverage time taken to make a prediction:", str(np.mean(all_prediction_times)) + "s"
    )

    fig, axes = plt.subplots(2, figsize=(10, 10))
    # bar chart properties
    bar_width = 0.1
    labels = folder_names
    x_axis = np.arange(len(labels))

    # data for plotting bar chart
    bar_xr2 = []
    bar_yr2 = []
    bar_zr2 = []
    bar_xtremor_r2 = []
    bar_ytremor_r2 = []
    bar_ztremor_r2 = []
    bar_training_times = np.round(np.multiply(all_training_times, 1000), 2)
    bar_xpredict_times = []
    bar_ypredict_times = []
    bar_zpredict_times = []
    # formats the lists above for use in generating bars in the chart
    for i in range(len(folder_names)):
        bar_xr2.append(np.round(all_r2[i][0]))
        bar_yr2.append(np.round(all_r2[i][1]))
        bar_zr2.append(np.round(all_r2[i][2]))
        bar_xtremor_r2.append(np.round(all_tremor_r2[i][0]))
        bar_ytremor_r2.append(np.round(all_tremor_r2[i][1]))
        bar_ztremor_r2.append(np.round(all_tremor_r2[i][2]))
        bar_xpredict_times.append(round(1000 * all_prediction_times[i][0], 2))
        bar_ypredict_times.append(round(1000 * all_prediction_times[i][1], 2))
        bar_zpredict_times.append(round(1000 * all_prediction_times[i][2], 2))

    # bars for each result
    bar1 = axes[0].bar(x_axis - (5 * bar_width / 2), bar_xr2, width=bar_width, label="R2 (X)")
    bar2 = axes[0].bar(x_axis - (3 * bar_width / 2), bar_yr2, width=bar_width, label="R2 (Y)")
    bar3 = axes[0].bar(x_axis - (bar_width / 2), bar_zr2, width=bar_width, label="R2 (Z)")
    bar4 = axes[0].bar(x_axis + (bar_width / 2), bar_xtremor_r2, width=bar_width, label="Tremor R2 (X)")
    bar5 = axes[0].bar(x_axis + (3 * bar_width / 2), bar_ytremor_r2, width=bar_width, label="Tremor R2 (Y)")
    bar6 = axes[0].bar(x_axis + (5 * bar_width / 2), bar_ztremor_r2, width=bar_width, label="Tremor R2 (Z)")
    bar7 = axes[1].bar(x_axis - (3 * bar_width / 2), bar_training_times, width=bar_width, label="Training time")
    bar8 = axes[1].bar(x_axis - (bar_width / 2), bar_xpredict_times, width=bar_width, label="Prediction time (X)")
    bar9 = axes[1].bar(x_axis + (bar_width / 2), bar_ypredict_times, width=bar_width, label="Prediction time (Y)")
    bar10 = axes[1].bar(x_axis + (3 * bar_width / 2), bar_zpredict_times, width=bar_width, label="Prediction time (Z)")
    # displays bar value above the bar
    axes[0].bar_label(bar1)
    axes[0].bar_label(bar2)
    axes[0].bar_label(bar3)
    axes[0].bar_label(bar4)
    axes[0].bar_label(bar5)
    axes[0].bar_label(bar6)
    axes[1].bar_label(bar7)
    axes[1].bar_label(bar8)
    axes[1].bar_label(bar9)
    axes[1].bar_label(bar10)

    # axis labels + title
    axes[0].set_title("Accuracy", fontweight="bold")
    axes[0].set_xlabel("R2 score metrics")
    axes[0].set_ylabel("Accuracy (%)")
    axes[1].set_title("Speed", fontweight="bold")
    axes[1].set_xlabel("Time metrics")
    axes[1].set_ylabel("Time (ms)")
    # setting ticks and tick params
    axes[0].set_xticks(x_axis)
    axes[0].set_xticklabels(labels)
    axes[1].set_xticks(x_axis)
    axes[1].set_xticklabels(labels)
    axes[0].tick_params(axis="x", which="both")
    axes[0].tick_params(axis="y", which="both")
    axes[1].tick_params(axis="x", which="both")
    axes[1].tick_params(axis="y", which="both")

    # legend
    font_prop = FontProperties()
    font_prop.set_size("small")  # font size of the legend content
    legend1 = axes[0].legend(prop=font_prop)
    legend2 = axes[1].legend(prop=font_prop)
    # font size of the legend title
    plt.setp(legend1.get_title(), fontsize="medium")
    plt.setp(legend2.get_title(), fontsize="medium")
    # figure title
    fig.suptitle((model + " results based on multiple datasets"), fontweight="bold", fontsize="x-large")

    plt.show()


if __name__ == '__main__':
    # choose what ML regression algorithm to use
    # model = "SVM"
    model = "Random Forest"

    # choose what output to predict
    # prediction_target = "voluntary motion"
    prediction_target = "tremor component"

    predict_dirs(model, prediction_target)
