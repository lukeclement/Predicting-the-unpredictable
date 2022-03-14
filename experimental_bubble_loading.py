import bubble_prediction
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow.keras import losses, models
import dat_to_training
import create_network
import loss_functions
import glob


def get_parameters(filename):
    elements = filename.split("_")
    parameters = []
    for element in elements:
        if ";" in element:
            target = element.split(";")[0]
            parameters.append(int(target))
    return parameters


def main():
    model_list = glob.glob("desktop_models/models/Proper-*")
    primary_key = []
    # Formatted Proper-frame_1;size_45;time_16;drop_0;encode_10;maxTrans_3;kernel_4;focus_1;
    frames = []
    sizes = []
    times = []
    encodes = []
    max_transpose = []
    kernel = []
    multis = []

    trainable_parameters = []

    loss_values = []

    index = 0
    for model_file in model_list:
        primary_key.append(index)
        parameters = get_parameters(model_file)
        frames.append(parameters[0])
        sizes.append(parameters[1])
        times.append(parameters[2])
        encodes.append(parameters[4])
        max_transpose.append(parameters[5])
        kernel.append(parameters[6])
        multis.append(parameters[7])
        model = models.load_model(model_file, custom_objects={"bce_dice": loss_functions.bce_dice})
        print(model_file)
        # dat_to_training.convert_dat_files([0, 0], parameters[1], 3, 7)
        data = dat_to_training.create_training_data(
            parameters[0], parameters[2], validation_split=0.5, image_size=parameters[1], focus=parameters[7]
        )

        loss = model.evaluate(data[1])
        parameters_line = create_network.interpret_model_summary(model)
        print(parameters_line)
        number = int(parameters_line.split(":")[1].replace(",", ""))
        trainable_parameters.append(number)
        make_gifs(index, model, parameters)
        loss_values.append(loss)
        index += 1
    model_data = pd.DataFrame({
        "Primary key": primary_key,
        "Image frames": frames,
        "Image size": sizes,
        "Timestep": times,
        "Encode size": encodes,
        "Max transpose layers": max_transpose,
        "Kernel size": kernel,
        "Focus": multis,
        "Loss": loss_values,
        "Trainable parameters": trainable_parameters
    })
    # model_data.to_csv("Model_quality.csv")
    model_data.to_csv("Model_quality_proper.csv")


def make_gifs(index, model, parameters):
    testing = bubble_prediction.long_term_prediction(
        model, 3, 20, parameters[1], parameters[2], parameters[0], 400, round_result=False, extra=True, focus=parameters[7]
    )
    bubble_prediction.make_gif(testing, 'desktop_models/model_gifs_fixed_thicc/without_rounding_with_extras_{}'.format(index))
    testing = bubble_prediction.long_term_prediction(
        model, 3, 20, parameters[1], parameters[2], parameters[0], 400, round_result=False, extra=False, focus=parameters[7]
    )
    bubble_prediction.make_gif(testing, 'desktop_models/model_gifs_fixed_thicc/without_rounding_without_extras_{}'.format(index))
    testing = bubble_prediction.long_term_prediction(
        model, 3, 20, parameters[1], parameters[2], parameters[0], 400, round_result=False, extra=True, jump=True, focus=parameters[7]
    )
    bubble_prediction.make_gif(testing, 'desktop_models/model_gifs_fixed_thicc/without_rounding_with_jump_{}'.format(index))
    testing = bubble_prediction.long_term_prediction(
        model, 3, 20, parameters[1], parameters[2], parameters[0], 400, round_result=True, extra=True, jump=True, focus=parameters[7]
    )
    bubble_prediction.make_gif(testing, 'desktop_models/model_gifs_fixed_thicc/with_rounding_with_jump_{}'.format(index))
    testing = bubble_prediction.long_term_prediction(
        model, 3, 20, parameters[1], parameters[2], parameters[0], 400, round_result=True, extra=True, focus=parameters[7]
    )
    bubble_prediction.make_gif(testing, 'desktop_models/model_gifs_fixed_thicc/with_rounding_with_extras_{}'.format(index))
    testing = bubble_prediction.long_term_prediction(
        model, 3, 20, parameters[1], parameters[2], parameters[0], 400, round_result=True, extra=False, focus=parameters[7]
    )
    bubble_prediction.make_gif(testing, 'desktop_models/model_gifs_fixed_thicc/with_rounding_without_extras_{}'.format(index))
    testing = bubble_prediction.long_term_prediction(
        model, 3, 20, parameters[1], parameters[2], parameters[0], 400, round_result=True, extra=False, dry_run=True, focus=parameters[7]
    )
    bubble_prediction.make_gif(testing, 'desktop_models/model_gifs_fixed_thicc/raw_sim_{}'.format(index))



if __name__ == "__main__":
    main()