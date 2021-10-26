import numpy as np
import glob
import os
import time


def get_question(sim, file, frames, loc, step_number):
    start_array = []
    for i in range(frames, 0, -1):
        source_array = np.load("{}/img_{}.npy".format(sim, step_number - i))
        start_array.append(source_array)
    return start_array


def get_source_arrays_2(sims, timestep_size=5, frames=4, size=64, channels=3):
    files = []
    file_nums = []
    print("Checking sims...")
    for sim in sims:
        sim_files = glob.glob("{}/*.npy".format(sim))
        for i in range(0, len(sim_files)-timestep_size*frames):
            files.append("{}/img_{}.npy".format(sim, i))
            file_nums.append([sim, i])
    print("Generating arrays of size {}...".format(len(files)))
    questions_array = np.zeros((len(files), frames, size, size, channels))
    answers_array = np.zeros((len(files), size, size, 1))
    print("Running...")
    print(np.shape(questions_array))
    print(np.shape(answers_array))
    for index, file in enumerate(files):
        for frame in range(0, frames*timestep_size, timestep_size):
            questions_array[index, int(frame/timestep_size), :, :, :] = np.load(
                "{}/img_{}.npy".format(file_nums[index][0], file_nums[index][1]+frame)
            )
        answers_array[index, :, :, 0] = np.load(
            "{}/img_{}.npy".format(file_nums[index][0], file_nums[index][1] + timestep_size*frames)
        )[:, :, 1]
    print("Saving...")
    np.save("Questions", questions_array)
    np.save("Answers", answers_array)
    return [np.load("Questions.npy"), np.load("Answers.npy")]


def get_source_arrays(sims, timestep_size=5, frames=4):
    """Get the arrays from simulated data.
    Input:
        sims:              list of simulations (list of strings)
        timestep_size:      int of the timestep to use (1 is minimum, default is 5)
    Output:
        training_images:    source files of training images
            a 2d array containing training set and solutions
    """
    training_questions = []
    training_solutions = []
    in_use = 0
    os.system("rm Questions.npy")
    os.system("rm Answers.npy")
    run_before = False
    for sim in sims:
        time_data = []
        print("Running {}...".format(sim))
        files = glob.glob("{}/*.npy".format(sim))
        number_of_steps = np.size(files)
        for file in files:
            loc = file.find("/img_") + 5
            step_number = int(file[loc:-4])
            t_1 = time.time() * 1000
            if step_number + timestep_size < number_of_steps and step_number - frames > 0:
                if not run_before:
                    start_array = get_question(sim, file, frames, loc, step_number)
                    current_questions = np.array([np.stack([x.tolist() for x in start_array])])
                    current_answers = np.array([np.load("{}/img_{}.npy".format(sim, step_number + timestep_size))])
                    run_before = True
                else:
                    # current_questions = np.load("Questions.npy")
                    # current_answers = np.load("Answers.npy")
                    new_question = get_question(sim, file, frames, loc, step_number)
                    new_answer = np.load("{}/img_{}.npy".format(sim, step_number + timestep_size))
                    current_answers = np.append(current_answers, np.array([new_answer]), axis=0)
                    current_questions = np.append(current_questions, np.array([
                        np.stack([x.tolist() for x in new_question])
                    ]), axis=0)
                # print(np.shape(current_questions))
                # np.save("Questions", current_questions)
                # np.save("Answers", current_answers)
            t_2 = time.time() * 1000
            time_data.append(t_2 - t_1)
        print("Mean time of {:.3f} mins".format(np.mean(time_data) / (60 * 1000)))
        print("Initial time of {:.3f} mins".format(time_data[0] / (60 * 1000)))
        print("Final time of {:.3f} mins".format(time_data[-1:][0] / (60 * 1000)))
        print("Total time of {:.2f} mins".format(np.sum(time_data) / (60 * 1000)))
        print(np.shape(current_questions))
        print(np.shape(current_answers))
        np.save("Questions", current_questions)
        np.save("Answers", current_answers)

    return [np.load("Questions.npy"), np.load("Answers.npy")]


def main():
    sources = glob.glob("Simulation_images/*")
    results = get_source_arrays_2(sources)
    print(np.shape(results[0]))
    print(np.shape(results[1]))


if __name__ == "__main__":
    main()