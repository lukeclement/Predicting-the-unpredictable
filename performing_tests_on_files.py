import create_network
import matplotlib.pyplot as plt
from tensorflow.keras import losses


def main():
    input_frames = []
    encode_sizes = []
    allowing_pool = []
    allowing_upsampling = []
    max_transpose_layers = []
    kernel_sizes = []
    parameters = []

    plt.xlabel("Maximum number of transpose layers")
    plt.ylabel("Kernel size")
    for i in range(1, 10):
        for j in range(2, 10):
            model = create_network.create_neural_network('LeakyReLU', 'adam', losses.BinaryCrossentropy, 4,
                                                         image_size=64, encode_size=3,
                                                         allow_pooling=True, allow_upsampling=True,
                                                         kernel_size=j, max_transpose_layers=i)
            encode_sizes.append(i)
            kernel_sizes.append(j)
            parameters_line = create_network.interpret_model_summary(model)
            number = float(parameters_line.split(":")[1].replace(",", ""))
            parameters.append(number)
    lowest_parameter = min(parameters)
    target = parameters.index(lowest_parameter)
    plt.hist2d(encode_sizes, kernel_sizes, weights=parameters, range=[[1, 10], [2, 10]], bins=(9, 8))
    cbar = plt.colorbar()
    cbar.set_label("Number of trainable parameters")
    plt.scatter([encode_sizes[target] + 0.5], [kernel_sizes[target] + 0.5], c="red", marker="x")
    print(lowest_parameter)
    plt.savefig("Transpose_layers_vs_Kernel_size.png", dpi=500)
    plt.close()

    parameters = []
    for i in [True, False]:
        for j in [True, False]:
            model = create_network.create_neural_network('LeakyReLU', 'adam', losses.BinaryCrossentropy, 4,
                                                         image_size=64, encode_size=3,
                                                         allow_pooling=i, allow_upsampling=j,
                                                         kernel_size=3, max_transpose_layers=3)
            allowing_pool.append(float(i))
            allowing_upsampling.append(float(j))
            parameters_line = create_network.interpret_model_summary(model)
            number = float(parameters_line.split(":")[1].replace(",", ""))
            parameters.append(number)
    plt.xlabel("Allow max pooling")
    plt.ylabel("Allow upsampling")
    lowest_parameter = min(parameters)
    target = parameters.index(lowest_parameter)
    plt.hist2d(allowing_pool, allowing_upsampling, weights=parameters, range=[[0, 2], [0, 2]], bins=(2, 2))
    cbar = plt.colorbar()
    cbar.set_label("Number of trainable parameters")
    plt.scatter([allowing_pool[target] + 0.5], [allowing_upsampling[target] + 0.5], c="red", marker="x")
    print(lowest_parameter)
    plt.savefig("Upsample_vs_pooling.png", dpi=500)
    plt.close()


if __name__ == "__main__":
    main()
