import bubble_prediction
import dat_to_training
import loss_functions
import create_network
from tensorflow.keras import layers

GLOBAL_ACTIVATION = layers.LeakyReLU()
GLOBAL_OPTIMISATION = "adam"


def send_request(options):
    for index, option in enumerate(options):
        print("{}: {}".format(index, option))
    invalid = True
    while invalid:
        try:
            response = int(input(">>"))
            if response >= len(options):
                raise ValueError
            invalid = False
        except ValueError:
            print("Invalid response!")
    return response


def ranged_request(allowed_range):
    while True:
        try:
            output = int(input(">>"))
            if not allowed_range[0] <= output <= allowed_range[1]:
                raise ValueError
            break
        except ValueError:
            print("Invalid input! Allowed range is [{}, {}] inclusive".format(allowed_range[0], allowed_range[1]))
    return output


def ranged_float_request(allowed_range):
    while True:
        try:
            output = float(input(">>"))
            if not allowed_range[0] <= output <= allowed_range[1]:
                raise ValueError
            break
        except ValueError:
            print("Invalid input! Allowed range is [{}, {}] inclusive".format(allowed_range[0], allowed_range[1]))
    return output


def manufacture_network():
    print("Create a new network from scratch or based on a saved network?")
    user = send_request(["From scratch", "Based on a saved network"])
    parameters = [
        loss_functions.UBERLOSS, 0, 0, 0, True, True, 0, 0, 0.0, [0], True, [0], 0, "None", 0
    ]
    print("What shall this network be called?")
    naming = send_request(["Custom manual name", "Automatic name"])
    net_name = "Test_please_overwrite"
    if naming == 0:
        print("The neural network created today will henceforth be called...")
        name = input(">>").replace(" ", "_")
        print("{}!".format(name))
        net_name = name
    parameters[13] = net_name
    if user == 0:
        print("Section A: Properties of the inputs and outputs")
        print("Image size to use:")
        parameters[2] = ranged_request([1, 1000])
        print("Number of image frames to pass as input:")
        parameters[1] = ranged_request([1, 1000])
        print("The number of 'in-between' frames between input and output images:")
        parameters[12] = ranged_request([1, 1000])
        print("Section B: Properties of the network architecture")
        print("Shall pooling layers be enabled?")
        parameters[4] = send_request(["Yes", "No"]) == 0
        print("Shall upsampling layers be enabled?")
        parameters[5] = send_request(["Yes", "No"]) == 0
        print("Size of encoded image within network:")
        parameters[3] = ranged_request([1, parameters[2]-parameters[1]])
        print("Maximum number of transpose layers:")
        parameters[6] = ranged_request([1, 1000])
        print("Size of the convolutional kernel:")
        parameters[7] = ranged_request([2, parameters[2]])
        print("Section C: Properties of training the network")
        print("Which loss function would you like to use in training?")
        options = [
            "UBERLOSS",
            "Cubic loss",
            "Relative difference",
            "Absolute difference",
            "SSIM",
            "Mean squared error (with intersection over union)",
            "Binary cross-entropy (with intersection over union)"
        ]
        losses = [
            loss_functions.UBERLOSS,
            loss_functions.cubic_loss,
            loss_functions.relative_diff,
            loss_functions.absolute_diff,
            loss_functions.ssim_loss,
            loss_functions.mse_dice,
            loss_functions.bce_dice
        ]
        parameters[0] = losses[send_request(options)]
        print("What resolution should the images be interpolated to?")
        parameters[8] = ranged_float_request([0.000000001, 1])
        print("By how much should images be offset by in training?")
        options_a = ["None", "A range of amounts", "Some specific amounts"]
        user_response = send_request(options_a)
        if user_response == 0:
            parameters[9] = [0]
        elif user_response == 1:
            print("What is the maximum amount:")
            maxi = ranged_request([-99999, 99999])
            print("What is the minimum amount:")
            mini = ranged_request([-99999, 99999])
            parameters[9] = []
            for i in range(mini, maxi+1):
                parameters[9].append(i)
        else:
            inputting = True
            print("Please input offset amounts. Once finished, type X.")
            num_input = 0
            parameters[9] = []
            while inputting:
                try:
                    print("Input {}:".format(num_input))
                    val = input(">>")
                    parameters[9].append(int(val))
                    num_input += 1
                except ValueError:
                    if val == "X":
                        inputting = False
                    else:
                        print("Invalid input! To exit, enter X.")
        print("Should inverted images be used in training?")
        parameters[10] = send_request(["Yes", "No"]) == 0
        print("Which simulations be excluded from training?")
        print("If none should be excluded, type X immediately. Once finished, type X.")
        inputting = True
        num_input = 0
        parameters[11] = []
        while inputting:
            try:
                print("Input {}:".format(num_input))
                val = input(">>")
                parameters[11].append(int(val))
                num_input += 1
            except ValueError:
                if val == "X":
                    inputting = False
                else:
                    print("Invalid input! To exit, enter X.")
        print("How many epochs?")
        parameters[14] = ranged_request([1, 10000])
        model = create_network.create_neural_network(
            GLOBAL_ACTIVATION, GLOBAL_OPTIMISATION,
            parameters[0], parameters[1],
            image_size=parameters[2], encode_size=parameters[3], max_transpose_layers=parameters[6],
            allow_pooling=parameters[4], allow_upsampling=parameters[5], kernel_size=parameters[7],
        )
    return parameters, model


def main():
    print("Welcome to the basic neural network generator program!")
    while True:
        print("What would you like to do?")
        responses = [
            "Create a new network",
            "Load a network",
            "Generate data for networks",
            "Look at currently existing networks"
        ]
        user = send_request(responses)
        if user == 0:
            parameters, model = manufacture_network()

        training_data = dat_to_training.create_training_data(
            parameters[1], parameters[12],
            image_size=parameters[2], variants=parameters[9], resolution=parameters[8],
            excluded_sims=parameters[11], flips_allowed=parameters[10]
        )
        model, history = create_network.train_model(model, training_data, epochs=parameters[14])
        model.save("Generated_models/{}".format(parameters[13]))


if __name__ == "__main__":
    main()