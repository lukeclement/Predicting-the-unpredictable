import bubble_prediction as bp
import dat_to_training as dtt
import loss_functions as lf
import create_network as cn


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


def manufacture_network():
    print("Create a new network from scratch or based on a saved network?")
    user = send_request(["From scratch", "Based on a saved network"])
    """
    activation_function
    optimizer
    loss_function
    image_frames
    image_size
    encode_size
    allow_pooling
    allow_upsampling
    max_transpose_layers
    kernel_size
    resolution
    variants to use 
    include flips
    simulations to use
    focus level
    
    """
    parameters = [

    ]
    print("What shall this network be called?")
    naming = send_request(["Custom manual name", "Automatic name"])
    net_name = "Test_please_overwrite"
    if naming == 0:
        print("The neural network created today will henceforth be called...")
        name = input(">>").replace(" ", "_")
        print("{}!".format(name))
    if user == 0:
        dtt.create_training_data(1, 1, variants=[0], resolution=0.001)


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