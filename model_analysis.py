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
        [loss_functions.mse_dice, 60, 2, 3, True, True, 20, 3, 0.001, [0], True, [0], 5, "MSEDICE", 20],
        [loss_functions.bce_dice, 60, 2, 3, True, True, 20, 3, 0.001, [0], True, [0], 5, "BCEDICE", 20],
        [loss_functions.UBERLOSS, 60, 2, 3, True, True, 10, 3, 0.001, [0], True, [0], 5, "Trans_Original", 20],
        [loss_functions.UBERLOSS, 60, 4, 10, True, True, 10, 3, 0.001, [0], True, [0], 5, "Parallel", 5, True],
        [loss_functions.UBERLOSS, 60, 4, 10, True, True, 10, 3, 0.001, [0], True, [0], 5, "Linear", 5, False],
        # [loss_functions.UBERLOSS, 60, 2, 3, True, True, 10, 3, 0.001, [0], True, [0], 5, "Original_long_epochs", 50],
        [loss_functions.UBERLOSS, 60, 2, 3, True, True, 5, 3, 0.001, [0], True, [0], 5, "Trans_Trans", 20],
        [loss_functions.UBERLOSS, 60, 1, 3, True, True, 10, 3, 0.001, [0], True, [0], 5, "Trans_Large", 20],
        [loss_functions.UBERLOSS, 30, 4, 3, True, True, 10, 3, 0.001, [0], True, [0], 5, "Trans_Long", 20],
        [loss_functions.UBERLOSS_minus_dice, 60, 2, 3, True, True, 10, 3, 0.001, [0], True, [0], 5, "Trans_Original-", 20],
        [losses.mean_squared_logarithmic_error, 60, 2, 3, True, True, 10, 3, 0.001, [0], True, [0], 5, "Trans_OriginalMSE", 20],
        [losses.binary_crossentropy, 60, 2, 3, True, True, 10, 3, 0.001, [0], True, [0], 5, "Trans_OriginalBCE", 20],
        # [loss_functions.ssim_loss, 60, 2, 3, True, True, 10, 3, 0.001, [0], True, [0], 5, "Trans_OriginalSSIM", 20],
        [loss_functions.UBERLOSS, 60, 2, 3, True, True, 10, 2, 0.001, [0], True, [0], 5, "Trans_Sanders", 20],
        [loss_functions.UBERLOSS, 60, 2, 3, True, True, 10, 7, 0.001, [0], True, [0], 5, "Trans_Johnson", 20],
        [loss_functions.UBERLOSS, 60, 2, 1, True, True, 10, 3, 0.001, [0], True, [0], 5, "Trans_Thin", 20],
        [loss_functions.UBERLOSS, 60, 2, 10, True, True, 10, 3, 0.001, [0], True, [0], 5, "Trans_Thick", 20],
        [loss_functions.UBERLOSS, 60, 2, 10, True, True, 10, 7, 0.001, [0], True, [0], 5, "Trans_ThickJohnson", 20],
        [loss_functions.UBERLOSS, 60, 2, 1, True, True, 10, 2, 0.001, [0], True, [0], 5, "Trans_ThinSanders", 20],
        [loss_functions.UBERLOSS, 60, 2, 10, True, True, 10, 2, 0.001, [0], True, [0], 5, "Trans_ThickSanders", 20],
        [loss_functions.UBERLOSS, 60, 2, 1, True, True, 10, 7, 0.001, [0], True, [0], 5, "Trans_ThinJohnson", 20],
        [loss_functions.UBERLOSS, 60, 2, 3, True, True, 10, 3, 0.1, [0], True, [0], 5, "Trans_Low", 20],
        [loss_functions.UBERLOSS, 60, 2, 3, True, True, 10, 3, 0.01, [0], True, [0], 5, "Trans_Medium", 20],
        [loss_functions.UBERLOSS, 60, 2, 3, True, True, 10, 3, 0.001, [0], True, [0], 5, "Trans_High", 20],
        [loss_functions.UBERLOSS, 60, 2, 3, True, True, 10, 3, 0.001, [0], True, [0], 20, "Trans_Fast", 20],
        [loss_functions.UBERLOSS, 60, 2, 3, True, True, 10, 3, 0.001, [0], True, [0], 1, "Trans_Slow", 20],
        [loss_functions.mse_dice, 60, 2, 3, True, True, 10, 3, 0.001, [0], True, [0], 5, "Trans_MSEDICE", 20],
        [loss_functions.bce_dice, 60, 2, 3, True, True, 10, 3, 0.001, [0], True, [0], 5, "Trans_BCEDICE", 20],
        [loss_functions.UBERLOSS, 60, 2,  3, True, True, 20, 3, 0.001, [0], True, [0], 5, "Original", 20],
        [loss_functions.UBERLOSS, 60, 2, 3, True, True, 20, 3, 0.001, [0], True, [0], 5, "Original_long_epochs", 150],
        [loss_functions.UBERLOSS, 60, 2, 3, True, True, 10, 3, 0.001, [0], True, [0], 5, "Trans", 20],
        [loss_functions.UBERLOSS, 60, 1, 3, True, True, 20, 3, 0.001, [0], True, [0], 5, "Large", 20],
        [loss_functions.UBERLOSS, 30, 4, 3, True, True, 20, 3, 0.001, [0], True, [0], 5, "Long", 20],
        [loss_functions.UBERLOSS_minus_dice, 60, 2, 3, True, True, 20, 3, 0.001, [0], True, [0], 5, "Original-", 20],
        [losses.mean_squared_logarithmic_error, 60, 2, 3, True, True, 20, 3, 0.001, [0], True, [0], 5, "OriginalMSE", 20],
        [losses.binary_crossentropy, 60, 2, 3, True, True, 20, 3, 0.001, [0], True, [0], 5, "OriginalBCE", 20],
        # [loss_functions.ssim_loss, 60, 2, 3, True, True, 20, 3, 0.001, [0], True, [0], 5, "OriginalSSIM", 20],
        [loss_functions.UBERLOSS, 60, 2, 3, True, True, 20, 2, 0.001, [0], True, [0], 5, "Sanders", 20],
        [loss_functions.UBERLOSS, 60, 2, 3, True, True, 20, 7, 0.001, [0], True, [0], 5, "Johnson", 20],
        [loss_functions.UBERLOSS, 60, 2, 1, True, True, 20, 3, 0.001, [0], True, [0], 5, "Thin", 20],
        [loss_functions.UBERLOSS, 60, 2, 10, True, True, 20, 3, 0.001, [0], True, [0], 5, "Thick", 20],
        [loss_functions.UBERLOSS, 60, 2, 10, True, True, 20, 7, 0.001, [0], True, [0], 5, "ThickJohnson", 20],
        [loss_functions.UBERLOSS, 60, 2, 1, True, True, 20, 2, 0.001, [0], True, [0], 5, "ThinSanders", 20],
        [loss_functions.UBERLOSS, 60, 2, 10, True, True, 20, 2, 0.001, [0], True, [0], 5, "ThickSanders", 20],
        [loss_functions.UBERLOSS, 60, 2, 1, True, True, 20, 7, 0.001, [0], True, [0], 5, "ThinJohnson", 20],
        [loss_functions.UBERLOSS, 60, 2, 3, True, True, 20, 3, 0.1, [0], True, [0], 5, "Low", 20],
        [loss_functions.UBERLOSS, 60, 2, 3, True, True, 20, 3, 0.01, [0], True, [0], 5, "Medium", 20],
        [loss_functions.UBERLOSS, 60, 2, 3, True, True, 20, 3, 0.001, [0], True, [0], 5, "High", 20],
        [loss_functions.UBERLOSS, 60, 2, 3, True, True, 20, 3, 0.001, [0], True, [0], 20, "Fast", 20],
        [loss_functions.UBERLOSS, 60, 2, 3, True, True, 20, 3, 0.001, [0], True, [0], 1, "Slow", 20]
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
    bubble_prediction.make_gif(difference, "model_performance/{}_raw_difference".format(model_name))
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
    plt.savefig("model_performance/{}_centre_of_mass.png".format(model_name), dpi=500)
    plt.clf()
    plt.grid()
    plt.ylabel("Angle from centre/Radians")
    plt.xlabel("Step")

    plt.plot(actual_angles, label="Actual")
    plt.plot(guess_angles, label="Predictions")
    plt.legend()
    plt.savefig("model_performance/{}_centre_of_mass_angle.png".format(model_name), dpi=500)
    plt.clf()
    plt.grid()
    plt.ylabel("Value/Radians")
    plt.xlabel("Step")
    plt.plot(np.asarray(actual_angles) - np.asarray(guess_angles), label="Actual")
    plt.savefig("model_performance/{}_centre_of_mass_angle_diff.png".format(model_name), dpi=500)
    plt.clf()
    plt.grid()
    plt.ylabel("Centre of mass difference")
    plt.xlabel("Step")
    plt.plot(np.asarray(actual_com) - np.asarray(guess_com), label="Actual")
    plt.savefig("model_performance/{}_centre_of_mass_diff.png".format(model_name), dpi=500)
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
    cross_check("Parallel", [12, 20])
    cross_check("Linear", [12, 20])
    # print("0")
    # cross_check("Original", [12, 20])
    # print("1")
    # cross_check("Large", [12, 20])
    # print("2")
    # cross_check("Long", [12, 20])
    # print("3")
    # cross_check("Original-", [12, 20])
    # print("4")
    # cross_check("OriginalMSE", [12, 20])
    # print("5")
    # cross_check("OriginalBCE", [12, 20])
    # print("6")
    # cross_check("OriginalSSIM", [12, 20])
    # print("7")
    # cross_check("Sanders", [12, 20])
    # print("8")
    # cross_check("Johnson", [12, 20])
    # print("9")
    # cross_check("Thin", [12, 20])
    # print("10")
    # cross_check("Thick", [12, 20])
    # print("11")
    # cross_check("ThickJohnson", [12, 20])
    # print("12")
    # cross_check("ThinSanders", [12, 20])
    # print("13")
    # cross_check("ThickSanders", [12, 20])
    # print("14")
    # cross_check("ThinJohnson", [12, 20])
    # print("15")
    # cross_check("Low", [12, 20])
    # print("16")
    # cross_check("Medium", [12, 20])
    # print("17")
    # cross_check("High", [12, 20])
    # print("18")
    # cross_check("Fast", [12, 20])
    # print("19")
    # cross_check("Slow", [12, 20])
    # print("20")
    # cross_check("Trans", [12, 20])
    # print("21")
    # cross_check("MSEDICE", [12, 20])
    # print("22")
    # cross_check("BCEDICE", [12, 20])
    # print("23--")
    # print("0")
    # cross_check("Trans_Original", [12, 20])
    # print("1")
    # cross_check("Trans_Large", [12, 20])
    # print("2")
    # cross_check("Trans_Long", [12, 20])
    # print("3")
    # cross_check("Trans_Original-", [12, 20])
    # print("4")
    # cross_check("Trans_OriginalMSE", [12, 20])
    # print("5")
    # cross_check("Trans_OriginalBCE", [12, 20])
    # print("6")
    # cross_check("Trans_OriginalSSIM", [12, 20])
    # print("7")
    # cross_check("Trans_Sanders", [12, 20])
    # print("8")
    # cross_check("Trans_Johnson", [12, 20])
    # print("9")
    # cross_check("Trans_Thin", [12, 20])
    # print("10")
    # cross_check("Trans_Thick", [12, 20])
    # print("11")
    # cross_check("Trans_ThickJohnson", [12, 20])
    # print("12")
    # cross_check("Trans_ThinSanders", [12, 20])
    # print("13")
    # cross_check("Trans_ThickSanders", [12, 20])
    # print("14")
    # cross_check("Trans_ThinJohnson", [12, 20])
    # print("15")
    # cross_check("Trans_Low", [12, 20])
    # print("16")
    # cross_check("Trans_Medium", [12, 20])
    # print("17")
    # cross_check("Trans_High", [12, 20])
    # print("18")
    # cross_check("Trans_Fast", [12, 20])
    # print("19")
    # cross_check("Trans_Slow", [12, 20])
    # print("20")
    # cross_check("Trans_Trans", [12, 20])
    # print("21")
    # cross_check("Trans_MSEDICE", [12, 20])
    # print("22")
    # cross_check("Trans_BCEDICE", [12, 20])
    # print("23--")

if __name__ == "__main__":
    main()
