import matplotlib.pyplot as plt
import bubble_prediction
from tensorflow.keras import models, losses
import loss_functions
import numpy as np
import keras.backend as k

FUTURE_DISTANCE = 200


def get_model(name):
    model = models.load_model(
        "models/{}".format(name),
        custom_objects={
            "mean_squared_logarithmic_error": losses.mean_squared_logarithmic_error,
            "binary_crossentropy": losses.binary_crossentropy,
            # "ssim_loss": loss_functions.ssim_loss,
            "UBERLOSS": loss_functions.UBERLOSS,
            "UBERLOSS_minus_dice": loss_functions.UBERLOSS_minus_dice,
            "mse_dice": loss_functions.mse_dice,
            "bce_dice": loss_functions.bce_dice
        })
    return model


def read_parameters(model_name):
    parameter_options = [
        [loss_functions.mse_dice, 60, 2, 3, True, True, 20, 3, 0.001, [0], True, [0], 5, "MSEDICE_parallel", 20, True],
        [losses.binary_crossentropy, 60, 2, 10, True, True, 10, 3, 0.001, [0], True, [0], 5, "Parallel", 5, True],
        [losses.binary_crossentropy, 60, 2, 10, True, True, 10, 3, 0.001, [0], True, [0], 5, "Linear", 5, False],
        [loss_functions.bce_dice, 60, 2, 3, True, True, 20, 3, 0.001, [0], True, [0], 5, "BCEDICE_parallel", 20, True],
        [loss_functions.UBERLOSS, 60, 2, 3, True, True, 10, 3, 0.001, [0], True, [0], 5, "Trans_Original_parallel", 20, True],
        # [loss_functions.UBERLOSS, 60, 2, 3, True, True, 10, 3, 0.001, [0], True, [0], 5, "Original_long_epochs_parallel", 50],
        [loss_functions.UBERLOSS, 60, 2, 3, True, True, 5, 3, 0.001, [0], True, [0], 5, "Trans_Trans_parallel", 20, True],
        [loss_functions.UBERLOSS, 60, 1, 3, True, True, 10, 3, 0.001, [0], True, [0], 5, "Trans_Large_parallel", 20, True],
        [loss_functions.UBERLOSS, 30, 4, 3, True, True, 10, 3, 0.001, [0], True, [0], 5, "Trans_Long_parallel", 20, True],
        [loss_functions.UBERLOSS_minus_dice, 60, 2, 3, True, True, 10, 3, 0.001, [0], True, [0], 5, "Trans_Original-_parallel", 20, True],
        [losses.mean_squared_logarithmic_error, 60, 2, 3, True, True, 10, 3, 0.001, [0], True, [0], 5, "Trans_OriginalMSE_parallel", 20, True],
        [losses.binary_crossentropy, 60, 2, 3, True, True, 10, 3, 0.001, [0], True, [0], 5, "Trans_OriginalBCE_parallel", 20, True],
        [loss_functions.UBERLOSS, 60, 2, 3, True, True, 10, 2, 0.001, [0], True, [0], 5, "Trans_Sanders_parallel", 20, True],
        [loss_functions.UBERLOSS, 60, 2, 3, True, True, 10, 7, 0.001, [0], True, [0], 5, "Trans_Johnson_parallel", 20, True],
        [loss_functions.UBERLOSS, 60, 2, 1, True, True, 10, 3, 0.001, [0], True, [0], 5, "Trans_Thin_parallel", 20, True],
        [loss_functions.UBERLOSS, 60, 2, 10, True, True, 10, 3, 0.001, [0], True, [0], 5, "Trans_Thick_parallel", 20, True],
        [loss_functions.UBERLOSS, 60, 2, 10, True, True, 10, 7, 0.001, [0], True, [0], 5, "Trans_ThickJohnson_parallel", 20, True],
        [loss_functions.UBERLOSS, 60, 2, 1, True, True, 10, 2, 0.001, [0], True, [0], 5, "Trans_ThinSanders_parallel", 20, True],
        [loss_functions.UBERLOSS, 60, 2, 10, True, True, 10, 2, 0.001, [0], True, [0], 5, "Trans_ThickSanders_parallel", 20, True],
        [loss_functions.UBERLOSS, 60, 2, 1, True, True, 10, 7, 0.001, [0], True, [0], 5, "Trans_ThinJohnson_parallel", 20, True],
        [loss_functions.UBERLOSS, 60, 2, 3, True, True, 10, 3, 0.1, [0], True, [0], 5, "Trans_Low_parallel", 20, True],
        [loss_functions.UBERLOSS, 60, 2, 3, True, True, 10, 3, 0.01, [0], True, [0], 5, "Trans_Medium_parallel", 20, True],
        [loss_functions.UBERLOSS, 60, 2, 3, True, True, 10, 3, 0.001, [0], True, [0], 5, "Trans_High_parallel", 20, True],
        [loss_functions.UBERLOSS, 60, 2, 3, True, True, 10, 3, 0.001, [0], True, [0], 20, "Trans_Fast_parallel", 20, True],
        [loss_functions.UBERLOSS, 60, 2, 3, True, True, 10, 3, 0.001, [0], True, [0], 1, "Trans_Slow_parallel", 20, True],
        [loss_functions.mse_dice, 60, 2, 3, True, True, 10, 3, 0.001, [0], True, [0], 5, "Trans_MSEDICE_parallel", 20, True],
        [loss_functions.bce_dice, 60, 2, 3, True, True, 10, 3, 0.001, [0], True, [0], 5, "Trans_BCEDICE_parallel", 20, True],


        [loss_functions.mse_dice, 60, 2, 3, True, True, 20, 3, 0.001, [0], True, [0], 5, "MSEDICE", 20, False],
        [loss_functions.bce_dice, 60, 2, 3, True, True, 20, 3, 0.001, [0], True, [0], 5, "BCEDICE", 20, False],
        [loss_functions.UBERLOSS, 60, 2, 3, True, True, 10, 3, 0.001, [0], True, [0], 5, "Trans_Original", 20, False],
        # [loss_functions.UBERLOSS, 60, 2, 3, True, True, 10, 3, 0.001, [0], True, [0], 5, "Original_long_epochs", 50],
        [loss_functions.UBERLOSS, 60, 2, 3, True, True, 5, 3, 0.001, [0], True, [0], 5, "Trans_Trans", 20, False],
        [loss_functions.UBERLOSS, 60, 1, 3, True, True, 10, 3, 0.001, [0], True, [0], 5, "Trans_Large", 20, False],
        [loss_functions.UBERLOSS, 30, 4, 3, True, True, 10, 3, 0.001, [0], True, [0], 5, "Trans_Long", 20, False],
        [loss_functions.UBERLOSS_minus_dice, 60, 2, 3, True, True, 10, 3, 0.001, [0], True, [0], 5, "Trans_Original-", 20, False],
        [losses.mean_squared_logarithmic_error, 60, 2, 3, True, True, 10, 3, 0.001, [0], True, [0], 5, "Trans_OriginalMSE", 20, False],
        [losses.binary_crossentropy, 60, 2, 3, True, True, 10, 3, 0.001, [0], True, [0], 5, "Trans_OriginalBCE", 20, False],
        [loss_functions.UBERLOSS, 60, 2, 3, True, True, 10, 2, 0.001, [0], True, [0], 5, "Trans_Sanders", 20, False],
        [loss_functions.UBERLOSS, 60, 2, 3, True, True, 10, 7, 0.001, [0], True, [0], 5, "Trans_Johnson", 20, False],
        [loss_functions.UBERLOSS, 60, 2, 1, True, True, 10, 3, 0.001, [0], True, [0], 5, "Trans_Thin", 20, False],
        [loss_functions.UBERLOSS, 60, 2, 10, True, True, 10, 3, 0.001, [0], True, [0], 5, "Trans_Thick", 20, False],
        [loss_functions.UBERLOSS, 60, 2, 10, True, True, 10, 7, 0.001, [0], True, [0], 5, "Trans_ThickJohnson", 20, False],
        [loss_functions.UBERLOSS, 60, 2, 1, True, True, 10, 2, 0.001, [0], True, [0], 5, "Trans_ThinSanders", 20, False],
        [loss_functions.UBERLOSS, 60, 2, 10, True, True, 10, 2, 0.001, [0], True, [0], 5, "Trans_ThickSanders", 20, False],
        [loss_functions.UBERLOSS, 60, 2, 1, True, True, 10, 7, 0.001, [0], True, [0], 5, "Trans_ThinJohnson", 20, False],
        [loss_functions.UBERLOSS, 60, 2, 3, True, True, 10, 3, 0.1, [0], True, [0], 5, "Trans_Low", 20, False],
        [loss_functions.UBERLOSS, 60, 2, 3, True, True, 10, 3, 0.01, [0], True, [0], 5, "Trans_Medium", 20, False],
        [loss_functions.UBERLOSS, 60, 2, 3, True, True, 10, 3, 0.001, [0], True, [0], 5, "Trans_High", 20, False],
        [loss_functions.UBERLOSS, 60, 2, 3, True, True, 10, 3, 0.001, [0], True, [0], 20, "Trans_Fast", 20, False],
        [loss_functions.UBERLOSS, 60, 2, 3, True, True, 10, 3, 0.001, [0], True, [0], 1, "Trans_Slow", 20, False],
        [loss_functions.mse_dice, 60, 2, 3, True, True, 10, 3, 0.001, [0], True, [0], 5, "Trans_MSEDICE", 20, False],
        [loss_functions.bce_dice, 60, 2, 3, True, True, 10, 3, 0.001, [0], True, [0], 5, "Trans_BCEDICE", 20, False]

        [loss_functions.UBERLOSS, 45, 3, 5, True, True, 5, 3, 0.001, [0], True, [0], 5, "Alpha", 20, True],
        [loss_functions.UBERLOSS, 45, 3, 5, True, True, 5, 3, 0.001, [0], True, [0], 5, "Aberdeen", 20, False],
        #
        [loss_functions.UBERLOSS, 45, 3, 10, True, True, 5, 3, 0.001, [0], True, [0], 5, "Bravo", 20, True],
        [loss_functions.UBERLOSS, 45, 3, 10, True, True, 5, 3, 0.001, [0], True, [0], 5, "Bristol", 20, False],
        ##
        [loss_functions.UBERLOSS, 45, 3, 5, True, True, 1, 3, 0.001, [0], True, [0], 5, "Charlie", 20, True],
        [loss_functions.UBERLOSS, 45, 3, 5, True, True, 1, 3, 0.001, [0], True, [0], 5, "Castlebay", 20, False],
        #
        [loss_functions.UBERLOSS, 45, 3, 10, True, True, 1, 3, 0.001, [0], True, [0], 5, "Delta", 20, True],
        [loss_functions.UBERLOSS, 45, 3, 10, True, True, 1, 3, 0.001, [0], True, [0], 5, "Derby", 20, False],
        ###
        [loss_functions.UBERLOSS, 45, 3, 5, True, True, 5, 2, 0.001, [0], True, [0], 5, "Echo", 20, True],
        [loss_functions.UBERLOSS, 45, 3, 5, True, True, 5, 2, 0.001, [0], True, [0], 5, "Exeter", 20, False],
        #
        [loss_functions.UBERLOSS, 45, 3, 10, True, True, 5, 2, 0.001, [0], True, [0], 5, "Foxtrot", 20, True],
        [loss_functions.UBERLOSS, 45, 3, 10, True, True, 5, 2, 0.001, [0], True, [0], 5, "Fleet", 20, False],
        ##
        [loss_functions.UBERLOSS, 45, 3, 5, True, True, 1, 2, 0.001, [0], True, [0], 5, "Golf", 20, True],
        [loss_functions.UBERLOSS, 45, 3, 5, True, True, 1, 2, 0.001, [0], True, [0], 5, "Glasgow", 20, False],
        #
        [loss_functions.UBERLOSS, 45, 3, 10, True, True, 1, 2, 0.001, [0], True, [0], 5, "Hotel", 20, True],
        [loss_functions.UBERLOSS, 45, 3, 10, True, True, 1, 2, 0.001, [0], True, [0], 5, "Hartlepool", 20, False],
        ####
        [loss_functions.UBERLOSS, 45, 3, 5, True, True, 5, 5, 0.001, [0], True, [0], 5, "India", 20, True],
        [loss_functions.UBERLOSS, 45, 3, 5, True, True, 5, 5, 0.001, [0], True, [0], 5, "Inverness", 20, False],
        #
        [loss_functions.UBERLOSS, 45, 3, 10, True, True, 5, 5, 0.001, [0], True, [0], 5, "Juliet", 20, True],
        [loss_functions.UBERLOSS, 45, 3, 10, True, True, 5, 5, 0.001, [0], True, [0], 5, "JohnOGroats", 20, False],
        ##
        [loss_functions.UBERLOSS, 45, 3, 5, True, True, 1, 5, 0.001, [0], True, [0], 5, "Kilo", 20, True],
        [loss_functions.UBERLOSS, 45, 3, 5, True, True, 1, 5, 0.001, [0], True, [0], 5, "Kingston", 20, False],
        #
        [loss_functions.UBERLOSS, 45, 3, 10, True, True, 1, 5, 0.001, [0], True, [0], 5, "Lima", 20, True],
        [loss_functions.UBERLOSS, 45, 3, 10, True, True, 1, 5, 0.001, [0], True, [0], 5, "London", 20, False],
        ###
        [loss_functions.UBERLOSS, 45, 3, 5, True, True, 5, 4, 0.001, [0], True, [0], 5, "Mancy", 20, True],
        [loss_functions.UBERLOSS, 45, 3, 5, True, True, 5, 4, 0.001, [0], True, [0], 5, "Manchester", 20, False],
        #
        [loss_functions.UBERLOSS, 45, 3, 10, True, True, 5, 4, 0.001, [0], True, [0], 5, "November", 20, True],
        [loss_functions.UBERLOSS, 45, 3, 10, True, True, 5, 4, 0.001, [0], True, [0], 5, "Nottingham", 20, False],
        ##
        [loss_functions.UBERLOSS, 45, 3, 5, True, True, 1, 4, 0.001, [0], True, [0], 5, "Oscar", 20, True],
        [loss_functions.UBERLOSS, 45, 3, 5, True, True, 1, 4, 0.001, [0], True, [0], 5, "Oxford", 20, False],
        #
        [loss_functions.UBERLOSS, 45, 3, 10, True, True, 1, 4, 0.001, [0], True, [0], 5, "Papa", 20, True],
        [loss_functions.UBERLOSS, 45, 3, 10, True, True, 1, 4, 0.001, [0], True, [0], 5, "Portsmouth", 20, False],
        #####
        [loss_functions.UBERLOSS, 45, 3, 3, True, True, 5, 3, 0.001, [0], True, [0], 5, "Quebec", 20, True],
        [loss_functions.UBERLOSS, 45, 3, 3, True, True, 5, 3, 0.001, [0], True, [0], 5, "Quedgeley", 20, False],
        #
        [loss_functions.UBERLOSS, 45, 3, 7, True, True, 5, 3, 0.001, [0], True, [0], 5, "Romeo", 20, True],
        [loss_functions.UBERLOSS, 45, 3, 7, True, True, 5, 3, 0.001, [0], True, [0], 5, "Reading", 20, False],
        ##
        [loss_functions.UBERLOSS, 45, 3, 3, True, True, 1, 3, 0.001, [0], True, [0], 5, "Sierra", 20, True],
        [loss_functions.UBERLOSS, 45, 3, 3, True, True, 1, 3, 0.001, [0], True, [0], 5, "Salisbury", 20, False],
        #
        [loss_functions.UBERLOSS, 45, 3, 7, True, True, 1, 3, 0.001, [0], True, [0], 5, "Tango", 20, True],
        [loss_functions.UBERLOSS, 45, 3, 7, True, True, 1, 3, 0.001, [0], True, [0], 5, "Taunton", 20, False],
        ###
        [loss_functions.UBERLOSS, 45, 3, 3, True, True, 5, 2, 0.001, [0], True, [0], 5, "Uniform", 20, True],
        [loss_functions.UBERLOSS, 45, 3, 3, True, True, 5, 2, 0.001, [0], True, [0], 5, "Uxbridge", 20, False],
        #
        [loss_functions.UBERLOSS, 45, 3, 7, True, True, 5, 2, 0.001, [0], True, [0], 5, "Victor", 20, True],
        [loss_functions.UBERLOSS, 45, 3, 7, True, True, 5, 2, 0.001, [0], True, [0], 5, "Verwood", 20, False],
        ##
        [loss_functions.UBERLOSS, 45, 3, 3, True, True, 1, 2, 0.001, [0], True, [0], 5, "Whiskey", 20, True],
        [loss_functions.UBERLOSS, 45, 3, 3, True, True, 1, 2, 0.001, [0], True, [0], 5, "Woking", 20, False],
        #
        [loss_functions.UBERLOSS, 45, 3, 7, True, True, 1, 2, 0.001, [0], True, [0], 5, "Xray", 20, True],
        [loss_functions.UBERLOSS, 45, 3, 7, True, True, 1, 2, 0.001, [0], True, [0], 5, "Xfuckonlyknows", 20, False],
        ####
        [loss_functions.UBERLOSS, 45, 3, 3, True, True, 5, 5, 0.001, [0], True, [0], 5, "Yankee", 20, True],
        [loss_functions.UBERLOSS, 45, 3, 3, True, True, 5, 5, 0.001, [0], True, [0], 5, "Yateley", 20, False],
        #
        [loss_functions.UBERLOSS, 45, 3, 7, True, True, 5, 5, 0.001, [0], True, [0], 5, "Zulu", 20, True],
        [loss_functions.UBERLOSS, 45, 3, 7, True, True, 5, 5, 0.001, [0], True, [0], 5, "Zetland", 20, False],
        ##
        [loss_functions.UBERLOSS, 45, 3, 3, True, True, 1, 5, 0.001, [0], True, [0], 5, "0_Amsterdam", 20, True],
        [loss_functions.UBERLOSS, 45, 3, 3, True, True, 1, 5, 0.001, [0], True, [0], 5, "0_Andover", 20, False],
        #
        [loss_functions.UBERLOSS, 45, 3, 7, True, True, 1, 5, 0.001, [0], True, [0], 5, "0_Brussels", 20, True],
        [loss_functions.UBERLOSS, 45, 3, 7, True, True, 1, 5, 0.001, [0], True, [0], 5, "0_Birmingham", 20, False],
        ###
        [loss_functions.UBERLOSS, 45, 3, 3, True, True, 5, 4, 0.001, [0], True, [0], 5, "0_Copenhagen", 20, True],
        [loss_functions.UBERLOSS, 45, 3, 3, True, True, 5, 4, 0.001, [0], True, [0], 5, "0_Cardiff", 20, False],
        #
        [loss_functions.UBERLOSS, 45, 3, 7, True, True, 5, 4, 0.001, [0], True, [0], 5, "0_Dublin", 20, True],
        [loss_functions.UBERLOSS, 45, 3, 7, True, True, 5, 4, 0.001, [0], True, [0], 5, "0_Dundee", 20, False],
        ##
        [loss_functions.UBERLOSS, 45, 3, 3, True, True, 1, 4, 0.001, [0], True, [0], 5, "0_ElAaiun", 20, True],
        [loss_functions.UBERLOSS, 45, 3, 3, True, True, 1, 4, 0.001, [0], True, [0], 5, "0_Edinburgh", 20, False],
        #
        [loss_functions.UBERLOSS, 45, 3, 7, True, True, 1, 4, 0.001, [0], True, [0], 5, "0_Freetown", 20, True],
        [loss_functions.UBERLOSS, 45, 3, 7, True, True, 1, 4, 0.001, [0], True, [0], 5, "0_Frimley", 20, False],

    ]

    for i in range(len(parameter_options)):
        temp_s = parameter_options[i][1]
        temp_l = parameter_options[i][2]
        parameter_options[i][1] = temp_l
        parameter_options[i][2] = temp_s

    for p in parameter_options:
        if model_name == p[13]:
            return p
    return [loss_functions.UBERLOSS, 2, 60, 5, True, True, 10, 4, 0.001, [0], True, [12], 5, "Test_collection", 20]


def generate_predictions(model_name, initial_conditions, dry_run=False):
    parameters = read_parameters(model_name)
    return bubble_prediction.long_term_prediction(
        get_model(model_name),
        initial_conditions[0], initial_conditions[1],
        parameters[2], parameters[12], parameters[1], FUTURE_DISTANCE, parameters[8],
        round_result=False, extra=True, dry_run=dry_run
    )


def cross_check(model_name, initial_conditions):
    parameters = read_parameters(model_name)
    guesses = np.asarray(generate_predictions(model_name, initial_conditions))[:, :, :, 1]
    guesses_2 = np.asarray(generate_predictions(model_name, [13, 20]))[:, :, :, 1]
    actual = np.asarray(generate_predictions(model_name, initial_conditions, dry_run=True))[:, :, :, 1]
    length_actual = np.shape(actual)[0]
    guesses = guesses[:min(length_actual, FUTURE_DISTANCE)]
    guesses_2 = guesses_2[:min(length_actual, FUTURE_DISTANCE)]
    difference = actual.astype(np.float32) - guesses.astype(np.float32)
    difference = np.abs(difference).astype(np.uint8)
    # bubble_prediction.make_gif(difference, "model_performance/{}_raw_difference".format(model_name))
    composite = np.zeros((np.shape(guesses)[0], parameters[2], parameters[2], 3), np.uint8)
    composite[:, :, :, 2] = guesses
    composite[:, :, :, 0] = actual
    bubble_prediction.make_gif(composite, "model_performance/{}_composite".format(model_name))
    difference_sums = np.zeros(np.shape(guesses)[0])
    bce_values = np.zeros(np.shape(guesses)[0])
    UBERLOSS_values = np.zeros(np.shape(guesses)[0])
    mse_values = np.zeros(np.shape(guesses)[0])
    ssim_values = np.zeros(np.shape(guesses)[0])
    raw_data_actual = np.zeros((1, np.shape(guesses)[0], 1, parameters[2], parameters[2], 1))
    raw_data_guess = np.zeros((1, np.shape(guesses)[0], 1, parameters[2], parameters[2], 1))
    raw_data_guess_2 = np.zeros((1, np.shape(guesses)[0], 1, parameters[2], parameters[2], 1))
    raw_data_actual[0, :, 0, :, :, 0] = actual.astype(np.float32) / 255
    raw_data_guess[0, :, 0, :, :, 0] = guesses.astype(np.float32) / 255
    raw_data_guess_2[0, :, 0, :, :, 0] = guesses_2.astype(np.float32) / 255

    for i in range(0, np.shape(guesses)[0]):
        difference_sums[i] = np.sum(difference[i, :, :])
        looking_act = k.constant(raw_data_actual[:, i, :, :, :, :])
        looking_gue = k.constant(raw_data_guess[:, i, :, :, :, :])
        bce_values[i] = np.mean(losses.binary_crossentropy(looking_act, looking_gue).numpy())
        # print(loss_functions.UBERLOSS(looking_act, looking_gue))
        UBERLOSS_values[i] = np.mean(loss_functions.UBERLOSS(looking_act, looking_gue).numpy())
        mse_values[i] = np.mean(losses.mean_squared_logarithmic_error(looking_act, looking_gue).numpy())
        # ssim_values[i] = np.mean(loss_functions.ssim_loss(looking_act, looking_gue).numpy())
    looking_act = k.constant(raw_data_actual[:, 0, :, :, :, :])
    looking_gue = k.constant(raw_data_guess[:, 0, :, :, :, :])
    # print(losses.binary_crossentropy(looking_act, looking_gue).numpy())
    plt.clf()
    plt.close()
    plt.grid()
    plt.ylabel("Value/AU")
    plt.xlabel("Step")
    plt.plot(UBERLOSS_values, label="UBERLOSS")
    plt.plot(mse_values, label="MSE")
    plt.plot(bce_values, label="BCE")
    # plt.plot(ssim_values, label="SSIM")
    plt.legend()
    plt.savefig("model_performance/{}_losses.png".format(model_name), dpi=500)

    actual_com = []
    actual_angles = []
    actual_y = []
    for data in raw_data_actual[0, :, 0, :, :, :]:
        zz = np.zeros((parameters[2], parameters[2], 2))
        zz[:, :, 1] = data[:, :, 0]
        actual_com.append(bubble_prediction.calculate_com(zz))
        point_x, point_y = bubble_prediction.calculate_com(zz, True)
        actual_angles.append(np.arctan2(point_x, point_y))
        actual_y.append(point_x)
    guess_com = []
    guess_angles = []
    guess_y = []
    for data in raw_data_guess[0, :, 0, :, :, :]:
        zz = np.zeros((parameters[2], parameters[2], 2))
        zz[:, :, 1] = data[:, :, 0]
        guess_com.append(bubble_prediction.calculate_com(zz))
        point_x, point_y = bubble_prediction.calculate_com(zz, True)
        guess_angles.append(np.arctan2(point_x, point_y))
        guess_y.append(point_x)
    guess_2_y = []
    for data in raw_data_guess_2[0, :, 0, :, :, :]:
        zz = np.zeros((parameters[2], parameters[2], 2))
        zz[:, :, 1] = data[:, :, 0]
        point_x, point_y = bubble_prediction.calculate_com(zz, True)
        guess_2_y.append(point_x)
    plt.clf()
    plt.grid()
    plt.ylabel("Centre of mass distance/AU")
    plt.xlabel("Step")
    plt.plot(actual_com, label="Actual")
    plt.plot(guess_com, label="Predictions")
    plt.legend()
    # plt.savefig("model_performance/{}_centre_of_mass.png".format(model_name), dpi=500)
    plt.clf()
    plt.grid()
    plt.ylabel("Angle from centre/Radians")
    plt.xlabel("Step")

    plt.plot(actual_angles, label="Actual")
    plt.plot(guess_angles, label="Predictions")
    plt.legend()
    # plt.savefig("model_performance/{}_centre_of_mass_angle.png".format(model_name), dpi=500)
    plt.clf()
    plt.grid()
    plt.ylabel("Value/Radians")
    plt.xlabel("Step")
    plt.plot(np.asarray(actual_angles) - np.asarray(guess_angles), label="Actual")
    # plt.savefig("model_performance/{}_centre_of_mass_angle_diff.png".format(model_name), dpi=500)
    plt.clf()
    plt.grid()
    plt.ylabel("Centre of mass difference")
    plt.xlabel("Step")
    plt.plot(np.asarray(actual_com) - np.asarray(guess_com), label="Actual")
    # plt.savefig("model_performance/{}_centre_of_mass_diff.png".format(model_name), dpi=500)
    plt.clf()
    plt.grid()
    plt.ylabel("Y position")
    plt.xlabel("Step")
    plt.plot(actual_y, label="Actual")
    plt.plot(guess_y, label="Prediction")
    plt.legend()
    plt.savefig("model_performance/{}_y_pos.png".format(model_name), dpi=500)
    plt.clf()
    plt.grid()
    plt.ylabel("Y position")
    plt.xlabel("Step")
    plt.plot(guess_2_y, label="Prediction from 13")
    plt.plot(guess_y, label="Prediction from 12")
    plt.legend()
    plt.savefig("model_performance/{}_y_pos_div.png".format(model_name), dpi=500)


def main():
    to_analyse = [
        [loss_functions.UBERLOSS, 45, 3, 5, True, True, 5, 3, 0.001, [0], True, [0], 5, "Alpha", 20, True],
        [loss_functions.UBERLOSS, 45, 3, 5, True, True, 5, 3, 0.001, [0], True, [0], 5, "Aberdeen", 20, False],
        #
        [loss_functions.UBERLOSS, 45, 3, 10, True, True, 5, 3, 0.001, [0], True, [0], 5, "Bravo", 20, True],
        [loss_functions.UBERLOSS, 45, 3, 10, True, True, 5, 3, 0.001, [0], True, [0], 5, "Bristol", 20, False],
        ##
        [loss_functions.UBERLOSS, 45, 3, 5, True, True, 1, 3, 0.001, [0], True, [0], 5, "Charlie", 20, True],
        [loss_functions.UBERLOSS, 45, 3, 5, True, True, 1, 3, 0.001, [0], True, [0], 5, "Castlebay", 20, False],
        #
        [loss_functions.UBERLOSS, 45, 3, 10, True, True, 1, 3, 0.001, [0], True, [0], 5, "Delta", 20, True],
        [loss_functions.UBERLOSS, 45, 3, 10, True, True, 1, 3, 0.001, [0], True, [0], 5, "Derby", 20, False],
        ###
        [loss_functions.UBERLOSS, 45, 3, 5, True, True, 5, 2, 0.001, [0], True, [0], 5, "Echo", 20, True],
        [loss_functions.UBERLOSS, 45, 3, 5, True, True, 5, 2, 0.001, [0], True, [0], 5, "Exeter", 20, False],
        #
        [loss_functions.UBERLOSS, 45, 3, 10, True, True, 5, 2, 0.001, [0], True, [0], 5, "Foxtrot", 20, True],
        [loss_functions.UBERLOSS, 45, 3, 10, True, True, 5, 2, 0.001, [0], True, [0], 5, "Fleet", 20, False],
        ##
        [loss_functions.UBERLOSS, 45, 3, 5, True, True, 1, 2, 0.001, [0], True, [0], 5, "Golf", 20, True],
        [loss_functions.UBERLOSS, 45, 3, 5, True, True, 1, 2, 0.001, [0], True, [0], 5, "Glasgow", 20, False],
        #
        [loss_functions.UBERLOSS, 45, 3, 10, True, True, 1, 2, 0.001, [0], True, [0], 5, "Hotel", 20, True],
        [loss_functions.UBERLOSS, 45, 3, 10, True, True, 1, 2, 0.001, [0], True, [0], 5, "Hartlepool", 20, False],
        ####
        [loss_functions.UBERLOSS, 45, 3, 5, True, True, 5, 5, 0.001, [0], True, [0], 5, "India", 20, True],
        [loss_functions.UBERLOSS, 45, 3, 5, True, True, 5, 5, 0.001, [0], True, [0], 5, "Inverness", 20, False],
        #
        [loss_functions.UBERLOSS, 45, 3, 10, True, True, 5, 5, 0.001, [0], True, [0], 5, "Juliet", 20, True],
        [loss_functions.UBERLOSS, 45, 3, 10, True, True, 5, 5, 0.001, [0], True, [0], 5, "JohnOGroats", 20, False],
        ##
        [loss_functions.UBERLOSS, 45, 3, 5, True, True, 1, 5, 0.001, [0], True, [0], 5, "Kilo", 20, True],
        [loss_functions.UBERLOSS, 45, 3, 5, True, True, 1, 5, 0.001, [0], True, [0], 5, "Kingston", 20, False],
        #
        [loss_functions.UBERLOSS, 45, 3, 10, True, True, 1, 5, 0.001, [0], True, [0], 5, "Lima", 20, True],
        [loss_functions.UBERLOSS, 45, 3, 10, True, True, 1, 5, 0.001, [0], True, [0], 5, "London", 20, False],
        ###
        [loss_functions.UBERLOSS, 45, 3, 5, True, True, 5, 4, 0.001, [0], True, [0], 5, "Mancy", 20, True],
        [loss_functions.UBERLOSS, 45, 3, 5, True, True, 5, 4, 0.001, [0], True, [0], 5, "Manchester", 20, False],
        #
        [loss_functions.UBERLOSS, 45, 3, 10, True, True, 5, 4, 0.001, [0], True, [0], 5, "November", 20, True],
        [loss_functions.UBERLOSS, 45, 3, 10, True, True, 5, 4, 0.001, [0], True, [0], 5, "Nottingham", 20, False],
        ##
        [loss_functions.UBERLOSS, 45, 3, 5, True, True, 1, 4, 0.001, [0], True, [0], 5, "Oscar", 20, True],
        [loss_functions.UBERLOSS, 45, 3, 5, True, True, 1, 4, 0.001, [0], True, [0], 5, "Oxford", 20, False],
        #
        [loss_functions.UBERLOSS, 45, 3, 10, True, True, 1, 4, 0.001, [0], True, [0], 5, "Papa", 20, True],
        [loss_functions.UBERLOSS, 45, 3, 10, True, True, 1, 4, 0.001, [0], True, [0], 5, "Portsmouth", 20, False],
        #####
        [loss_functions.UBERLOSS, 45, 3, 3, True, True, 5, 3, 0.001, [0], True, [0], 5, "Quebec", 20, True],
        [loss_functions.UBERLOSS, 45, 3, 3, True, True, 5, 3, 0.001, [0], True, [0], 5, "Quedgeley", 20, False],
        #
        [loss_functions.UBERLOSS, 45, 3, 7, True, True, 5, 3, 0.001, [0], True, [0], 5, "Romeo", 20, True],
        [loss_functions.UBERLOSS, 45, 3, 7, True, True, 5, 3, 0.001, [0], True, [0], 5, "Reading", 20, False],
        ##
        [loss_functions.UBERLOSS, 45, 3, 3, True, True, 1, 3, 0.001, [0], True, [0], 5, "Sierra", 20, True],
        [loss_functions.UBERLOSS, 45, 3, 3, True, True, 1, 3, 0.001, [0], True, [0], 5, "Salisbury", 20, False],
        #
        [loss_functions.UBERLOSS, 45, 3, 7, True, True, 1, 3, 0.001, [0], True, [0], 5, "Tango", 20, True],
        [loss_functions.UBERLOSS, 45, 3, 7, True, True, 1, 3, 0.001, [0], True, [0], 5, "Taunton", 20, False],
        ###
        [loss_functions.UBERLOSS, 45, 3, 3, True, True, 5, 2, 0.001, [0], True, [0], 5, "Uniform", 20, True],
        [loss_functions.UBERLOSS, 45, 3, 3, True, True, 5, 2, 0.001, [0], True, [0], 5, "Uxbridge", 20, False],
        #
        [loss_functions.UBERLOSS, 45, 3, 7, True, True, 5, 2, 0.001, [0], True, [0], 5, "Victor", 20, True],
        [loss_functions.UBERLOSS, 45, 3, 7, True, True, 5, 2, 0.001, [0], True, [0], 5, "Verwood", 20, False],
        ##
        [loss_functions.UBERLOSS, 45, 3, 3, True, True, 1, 2, 0.001, [0], True, [0], 5, "Whiskey", 20, True],
        [loss_functions.UBERLOSS, 45, 3, 3, True, True, 1, 2, 0.001, [0], True, [0], 5, "Woking", 20, False],
        #
        [loss_functions.UBERLOSS, 45, 3, 7, True, True, 1, 2, 0.001, [0], True, [0], 5, "Xray", 20, True],
        [loss_functions.UBERLOSS, 45, 3, 7, True, True, 1, 2, 0.001, [0], True, [0], 5, "Xfuckonlyknows", 20, False],
        ####
        [loss_functions.UBERLOSS, 45, 3, 3, True, True, 5, 5, 0.001, [0], True, [0], 5, "Yankee", 20, True],
        [loss_functions.UBERLOSS, 45, 3, 3, True, True, 5, 5, 0.001, [0], True, [0], 5, "Yateley", 20, False],
        #
        [loss_functions.UBERLOSS, 45, 3, 7, True, True, 5, 5, 0.001, [0], True, [0], 5, "Zulu", 20, True],
        [loss_functions.UBERLOSS, 45, 3, 7, True, True, 5, 5, 0.001, [0], True, [0], 5, "Zetland", 20, False],
        ##
        [loss_functions.UBERLOSS, 45, 3, 3, True, True, 1, 5, 0.001, [0], True, [0], 5, "0_Amsterdam", 20, True],
        [loss_functions.UBERLOSS, 45, 3, 3, True, True, 1, 5, 0.001, [0], True, [0], 5, "0_Andover", 20, False],
        #
        [loss_functions.UBERLOSS, 45, 3, 7, True, True, 1, 5, 0.001, [0], True, [0], 5, "0_Brussels", 20, True],
        [loss_functions.UBERLOSS, 45, 3, 7, True, True, 1, 5, 0.001, [0], True, [0], 5, "0_Birmingham", 20, False],
        ###
        [loss_functions.UBERLOSS, 45, 3, 3, True, True, 5, 4, 0.001, [0], True, [0], 5, "0_Copenhagen", 20, True],
        [loss_functions.UBERLOSS, 45, 3, 3, True, True, 5, 4, 0.001, [0], True, [0], 5, "0_Cardiff", 20, False],
        #
        [loss_functions.UBERLOSS, 45, 3, 7, True, True, 5, 4, 0.001, [0], True, [0], 5, "0_Dublin", 20, True],
        [loss_functions.UBERLOSS, 45, 3, 7, True, True, 5, 4, 0.001, [0], True, [0], 5, "0_Dundee", 20, False],
        ##
        [loss_functions.UBERLOSS, 45, 3, 3, True, True, 1, 4, 0.001, [0], True, [0], 5, "0_ElAaiun", 20, True],
        [loss_functions.UBERLOSS, 45, 3, 3, True, True, 1, 4, 0.001, [0], True, [0], 5, "0_Edinburgh", 20, False],
        #
        [loss_functions.UBERLOSS, 45, 3, 7, True, True, 1, 4, 0.001, [0], True, [0], 5, "0_Freetown", 20, True],
        [loss_functions.UBERLOSS, 45, 3, 7, True, True, 1, 4, 0.001, [0], True, [0], 5, "0_Frimley", 20, False],
    ]
    for model in to_analyse:
        print(model[13])
        cross_check(model[13], [12, 20])

if __name__ == "__main__":
    main()
