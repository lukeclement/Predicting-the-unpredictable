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
            "ssim_loss": loss_functions.ssim_loss,
            "UBERLOSS": loss_functions.UBERLOSS
        })
    return model


def read_parameters(model_name):
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
    actual = np.asarray(generate_predictions(model_name, initial_conditions, dry_run=True))[:, :, :, 1]
    length_actual = np.shape(actual)[0]
    guesses = guesses[:min(length_actual, FUTURE_DISTANCE)]
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
    raw_data_actual[0, :, 0, :, :, 0] = actual.astype(np.float32) / 255
    raw_data_guess[0, :, 0, :, :, 0] = guesses.astype(np.float32) / 255

    for i in range(0, np.shape(guesses)[0]):
        difference_sums[i] = np.sum(difference[i, :, :])
        looking_act = k.constant(raw_data_actual[:, i, :, :, :, :])
        looking_gue = k.constant(raw_data_guess[:, i, :, :, :, :])
        bce_values[i] = np.mean(losses.binary_crossentropy(looking_act, looking_gue).numpy())
        # print(loss_functions.UBERLOSS(looking_act, looking_gue))
        UBERLOSS_values[i] = np.mean(loss_functions.UBERLOSS(looking_act, looking_gue).numpy())
        mse_values[i] = np.mean(losses.mean_squared_logarithmic_error(looking_act, looking_gue).numpy())
        ssim_values[i] = np.mean(loss_functions.ssim_loss(looking_act, looking_gue).numpy())
    looking_act = k.constant(raw_data_actual[:, 0, :, :, :, :])
    looking_gue = k.constant(raw_data_guess[:, 0, :, :, :, :])
    print(losses.binary_crossentropy(looking_act, looking_gue).numpy())
    plt.grid()
    plt.xlabel("Step")
    plt.plot(UBERLOSS_values, label="UBERLOSS")
    plt.plot(mse_values, label="MSE")
    plt.plot(bce_values, label="BCE")
    plt.plot(ssim_values, label="SSIM")
    plt.legend()
    plt.savefig("model_performance/{}_losses.png".format(model_name), dpi=500)

    actual_com = []
    actual_angles = []
    for data in raw_data_actual[0, :, 0, :, :, :]:
        zz = np.zeros((parameters[2], parameters[2], 2))
        zz[:, :, 1] = data[:, :, 0]
        actual_com.append(bubble_prediction.calculate_com(zz))
        point_x, point_y = bubble_prediction.calculate_com(zz, True)
        actual_angles.append(np.arctan2(point_x, point_y))
    guess_com = []
    guess_angles = []
    for data in raw_data_guess[0, :, 0, :, :, :]:
        zz = np.zeros((parameters[2], parameters[2], 2))
        zz[:, :, 1] = data[:, :, 0]
        guess_com.append(bubble_prediction.calculate_com(zz))
        point_x, point_y = bubble_prediction.calculate_com(zz, True)
        guess_angles.append(np.arctan2(point_x, point_y))
    plt.clf()
    plt.grid()
    plt.plot(actual_com, label="Actual")
    plt.plot(guess_com, label="Predictions")
    plt.legend()
    plt.savefig("model_performance/{}_centre_of_mass.png".format(model_name), dpi=500)
    plt.clf()
    plt.grid()
    plt.plot(actual_angles, label="Actual")
    plt.plot(guess_angles, label="Predictions")
    plt.legend()
    plt.savefig("model_performance/{}_centre_of_mass_angle.png".format(model_name), dpi=500)
    plt.clf()
    plt.grid()
    plt.plot(np.asarray(actual_angles) - np.asarray(guess_angles), label="Actual")
    plt.savefig("model_performance/{}_centre_of_mass_angle_diff.png".format(model_name), dpi=500)
    plt.clf()
    plt.grid()
    plt.plot(np.asarray(actual_com) - np.asarray(guess_com), label="Actual")
    plt.savefig("model_performance/{}_centre_of_mass_diff.png".format(model_name), dpi=500)


def main():
    cross_check("Original", [12, 20])
    cross_check("Large", [12, 20])
    cross_check("Long", [12, 20])
    cross_check("Original-", [12, 20])
    cross_check("OriginalMSE", [12, 20])
    cross_check("OriginalBCE", [12, 20])
    cross_check("OriginalSSIM", [12, 20])
    cross_check("Sanders", [12, 20])
    cross_check("Johnson", [12, 20])
    cross_check("Thin", [12, 20])
    cross_check("Thick", [12, 20])
    cross_check("ThickJohnson", [12, 20])
    cross_check("ThinSanders", [12, 20])
    cross_check("ThickSanders", [12, 20])
    cross_check("ThinJohnson", [12, 20])
    cross_check("Low", [12, 20])
    cross_check("High", [12, 20])
    cross_check("Fast", [12, 20])
    cross_check("Slow", [12, 20])


if __name__ == "__main__":
    main()
