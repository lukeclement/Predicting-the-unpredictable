import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from mpl_toolkits.axes_grid1 import make_axes_locatable


def get_size(series):
    min_val = np.min(series)
    max_val = np.max(series)
    series_range = [min_val-0.5, max_val+0.5]
    size = max_val - min_val + 1
    return [series_range, size, min_val]


def plot(series_a_name, series_b_name, dataset):
    fig, ax_a = plt.subplots()
    encode_size_size = get_size(dataset[series_a_name].tolist())
    max_transpose_layers_size = get_size(dataset[series_b_name].tolist())

    h, x_edge, y_edge = np.histogram2d(
        dataset[series_a_name], dataset[series_b_name],
        range=[encode_size_size[0], max_transpose_layers_size[0]],
        bins=(encode_size_size[1], max_transpose_layers_size[1])
    )
    h_1, x_edge, y_edge = np.histogram2d(
        dataset[series_a_name], dataset[series_b_name],
        weights=dataset["Trainable parameters"],
        range=[encode_size_size[0], max_transpose_layers_size[0]],
        bins=(encode_size_size[1], max_transpose_layers_size[1])
    )
    h_average = h_1 / h
    xticks = np.asarray(range(0, int(max_transpose_layers_size[1])))
    yticks = np.asarray(range(0, int(encode_size_size[1])))
    ax_a.set_xticks(xticks)
    ax_a.set_yticks(yticks)
    xtickslabels = np.asarray(
        range(int(max_transpose_layers_size[0][0])+1,
              int(max_transpose_layers_size[0][1])+1)
    )
    ytickslabels = np.asarray(
        range(int(encode_size_size[0][0])+1,
              int(encode_size_size[0][1])+1)
    )
    ax_a.set_xticklabels(xtickslabels)
    ax_a.set_yticklabels(ytickslabels)
    print(xticks)

    img_0 = ax_a.imshow(h_average)
    ax_a.set_ylabel(series_a_name)
    ax_a.set_xlabel(series_b_name)
    divider = make_axes_locatable(ax_a)
    cax = divider.append_axes('right', size='5%', pad=0.05)
    col_0 = plt.colorbar(img_0, cax=cax, orientation='vertical')
    col_0.set_label("Average number of parameters")
    fig.savefig("Parameter_tests,{}_vs_{}.png".format(series_a_name, series_b_name), dpi=500)
    plt.close('all')


def main():
    dataset = pd.read_csv("Trainable_parameters_for_S64-Tany-F1-D0_00_.csv")
    pairs = sns.pairplot(dataset, hue="allow_pooling")

    fig = pairs.fig
    fig.savefig("Parameter_space_tests.png")
    plot("encode_size", "max_transpose_layers", dataset)
    plot("kernel_size", "max_transpose_layers", dataset)
    plot("encode_size", "kernel_size", dataset)


if __name__ == "__main__":
    main()