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

    img_0 = ax_a.imshow(h_average, cmap="Greys")
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
    #pairs = sns.pairplot(dataset, hue="allow_pooling")

    #fig = pairs.fig
    #fig.savefig("Parameter_space_tests.png")
    plot("encode_size", "max_transpose_layers", dataset)
    plot("kernel_size", "max_transpose_layers", dataset)
    plot("encode_size", "kernel_size", dataset)

    sizes = np.linspace(0, 130, 130)
    frames = np.linspace(1, 4, 4)
    print(frames)
    m = 0
    n = 0
    N = 14977
    D = 16
    T = 20
    k = 3.53*10**-8
    sense_lim_x = np.array([10, 10])
    sense_lim_y = np.array([0, 8])
    hard_lim_x = np.array([0, 130])
    hard_lim_y = np.array([7, 7])
    plt.plot(sense_lim_x, sense_lim_y)
    plt.plot(hard_lim_x, hard_lim_y)
    for F in frames:
        memory = np.square(sizes)*(n-m+1)*(N - D*T*F)*(F+1)*k
        plt.plot(sizes, memory, "--", label="{:.0f} Frames".format(F))
    plt.legend()
    plt.grid()
    plt.xlabel("Image size / pixel")
    plt.ylabel("Memory usage / GB")
    plt.ylim((0, 8))
    plt.xlim((0, 90))
    plt.savefig("Memory_per_frame(m0,n0,T20).png", dpi=500)
    #plt.show()
    plt.close('all')

    frames = np.linspace(1, 10, 100)
    allowed_M = np.linspace(4, 10, 7)
    print(allowed_M)
    for M in allowed_M:
        size = np.sqrt(M / ((n-m+1)*(N-D*T*frames)*(frames+1)*k))
        plt.plot(frames, size, label="{:.0f}GB allowed".format(M))
    plt.grid()
    plt.legend()
    plt.xlabel("Frames")
    plt.ylabel("Pixel size")
    plt.xlim(1, 10)
    plt.savefig("Allowed_memory(m0,n0,T20).png", dpi=500)

if __name__ == "__main__":
    main()