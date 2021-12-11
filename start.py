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
            output = input(">>")
            if not allowed_range[0] <= output <= allowed_range[1]:
                raise ValueError
            break
        except ValueError:
            print("Invalid input! Allowed range is [{}, {}] inclusive".format(allowed_range[0], allowed_range[1]))


def manufacture_network():
    print("Create a new network from scratch or based on a saved network?")
    user = send_request(["From scratch", "Based on a saved network"])
    parameters = [
        loss_functions.UBERLOSS,
        0, 0, 0,
        True, True,
        0, 0,
        0.0,
        [0],
        True,
        [0]
    ]
    print("What shall this network be called?")
    naming = send_request(["Custom manual name", "Automatic name"])
    net_name = "Test_please_overwrite"
    if naming == 0:
        print("The neural network created today will henceforth be called...")
        name = input(">>").replace(" ", "_")
        print("{}!".format(name))
    if user == 0:

        dat_to_training.create_training_data(1, 1, variants=[0], resolution=0.001, excluded_sims=[13])


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
            manufacture_network()



if __name__ == "__main__":
    main()