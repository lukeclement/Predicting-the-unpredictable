import imageio
import requests
import numpy as np
import matplotlib.pyplot as plt
import tqdm
from astropy.io import fits
from datetime import datetime
from datetime import timedelta
import sunpy.map
from astropy.time import Time
import astropy.units as u
import astropy.table as t
from sunpy.net import Fido
from sunpy.net import attrs as a
import skimage.measure
from tqdm import tqdm
from multiprocessing import Pool
from itertools import repeat
import tensorflow as tf

# def main():
#     # print("Howdy!")
#     metadata = "http://ssa.esac.esa.int/ssa/aio/metadata-action"
#     data = "http://ssa.esac.esa.int/ssa/aio/product-action"
#     samples = 10
#     query = ''
#     query += 'OBSERVATION.BEGINDATE>\'1997-06-17 23:00:00\''
#     query += ' AND '
#     query += 'OBSERVATION.BEGINDATE<\'2019-06-18 21:59:59\''
#     query += ' AND '
#     query += 'OBSERVATION.OBSERVATIONTYPE==\'CME WATCH 195\''
#     payload = {
#         "SELECTED_FIELDS": "OBSERVATION",
#         "PAGE_SIZE": str(samples),
#         'PAGE': '1',
#         'RESOURCE_CLASS': 'OBSERVATION',
#         'INSTRUMENT.NAME': 'EIT',
#         # 'INSTRUMENT.NAME': 'LASCO',
#         # 'INSTRUMENT.NAME': 'MDI',
#         'RETURN_TYPE': 'JSON',
#         'QUERY': query
#     }
#     print("Sending request...")
#     r = requests.get(metadata, params=payload)
#     print(r.url)
#     # print(r.json())
#     # print(r.json()['data'][0]['OBSERVATION.ID'])
#     json_data = r.json()
#     max_sol = []
#     min_sol = []
#     avg_sol = []
#     dates = []
#     # solar_data = []
#     print(np.shape(json_data['data']))
#     # exit()
#     # solar_data = np.zeros((np.shape(json_data['data'])[0], 1024, 1024, 3), dtype=np.uint8)
#     for i in range(np.shape(json_data['data'])[0]):
#         print(i)
#         # print(json_data['data'][i])
#         print(json_data['data'][i]['OBSERVATION.ID'])
#         print(json_data['data'][i]['OBSERVATION.BEGINDATE'])
#         print(json_data['data'][i]['OBSERVATION.OBSERVATIONTYPE'])
#         payload = {
#             'OBSERVATION.ID': json_data['data'][i]['OBSERVATION.ID']
#         }
#
#         # r = requests.get(data, params=payload)
#         # with open("testing.fits", 'wb') as f:
#         #     f.write(r.content)
#         # solar_fit = fits.open('testing.fits')
#         # solar_fit.info()
#         # sol = solar_fit[0].data
#         # try:
#         #     print(np.mean(sol))
#         #     avg_sol.append(np.mean(sol))
#         #     print(np.min(sol))
#         #     min_sol.append(np.min(sol))
#         #     print(np.max(sol))
#         #     max_sol.append(np.max(sol))
#         #     solar_data[i, :, :, 2] = (np.tanh(sol/2000) * 255).astype(np.uint8)
#         # except ValueError:
#         #     print("Incorrect size")
#         # solar_fit.close()
#         dates.append(json_data['data'][i]['OBSERVATION.BEGINDATE'])
#
#     dates = np.asarray(dates)
#     index = dates.argsort()
#     print(dates[index])
#     actual_dates = []
#     for date in dates[index]:
#         actual = datetime.strptime(date, "%Y-%m-%d %H:%M:%S.%f")
#         actual_dates.append(actual)
#     deltas = []
#     for i in range(len(actual_dates)-1):
#         deltas.append((actual_dates[i+1]-actual_dates[i]).seconds)
#     plt.hist(deltas, bins=50)
#     plt.show()
#     solar_data = solar_data[index]
#     images = []
#     for i in solar_data:
#         images.append(i)
#     imageio.mimsave("SOHO_EDI.gif", images)
#     print(np.shape(solar_data))
#     # plt.imshow(solar_data)
#     # plt.show()


def get_metadata(years, months, instrument, wavelength: int):
    observation_times = []
    reference_codes = []
    sizes = []
    search_wavelength = a.Wavelength(wavelength * u.Angstrom)
    for year in years:
        for month in months:
            if month == 12:
                time_range = a.Time('{:04d}/{:02d}/01 00:00:00'.format(year, month),
                                    '{:04d}/{:02d}/01 00:00:00'.format(year+1, 1))
            else:
                time_range = a.Time('{:04d}/{:02d}/01 00:00:00'.format(year, month),
                                    '{:04d}/{:02d}/01 00:00:00'.format(year, month+1))
            print('Requesting data for {:02d}/{:04d} on wavelength {}'.format(month, year, wavelength))
            search_results = Fido.search(time_range, instrument, search_wavelength)
            for result in search_results[0]:
                reference_codes.append(result)
            print('Found {} results, total metadata spans {} entries'.format(len(search_results[0]), len(reference_codes)))
            metadata = np.asarray(search_results.show('Start Time', 'Size'))[0]
            bar = tqdm(total=len(search_results[0]))
            for data in metadata:
                observation_times.append(data[0].to_value('unix'))
                sizes.append(data[1])
                bar.update(1)
            bar.close()
    observation_times = np.asarray(observation_times)
    sizes = np.asarray(sizes)
    # reference_codes = np.asarray(reference_codes)
    return observation_times, sizes, reference_codes


def get_valid_data(frames, future_look, observation_times,
                   time_separation=720, separation_window=60, start_window=920000000, window_range=10000000):
    validation_mask = np.zeros(len(observation_times))
    chains = []
    time_chains = []
    for index, time in enumerate(observation_times):
        chain_building = True
        current_frames = 0
        current_chain = [index]
        time_chain = [time]
        ref_point = 0
        while chain_building:
            try:
                future_point = len(current_chain) - frames
                if current_frames < frames:
                    start_chain_length = len(current_chain)
                    ref_point += 1
                    while observation_times[index + ref_point] - time < time_separation*(current_frames + 1) + separation_window/2:
                        if observation_times[index + ref_point] - time > time_separation*(current_frames + 1) - separation_window/2:
                            current_chain.append(index + ref_point)
                            time_chain.append(observation_times[index+ref_point])
                            break
                        ref_point += 1
                    if start_chain_length == len(current_chain):
                        chain_building = False
                    else:
                        current_frames += 1
                elif future_point < 2:
                    start_chain_length = len(current_chain)
                    ref_point = 1
                    while observation_times[index + ref_point] - time < time_separation*(future_look + frames) + separation_window/2:
                        if observation_times[index + ref_point] - time > time_separation*(future_look + frames) - separation_window/2:
                            current_chain.append(index + ref_point)
                            time_chain.append(observation_times[index+ref_point])
                            break
                        ref_point += 1
                    if start_chain_length == len(current_chain):
                        chain_building = False
                else:
                    chain_building = False
            except IndexError:
                chain_building = False
        if len(current_chain) == 6:
            chains.append(current_chain)
            time_chains.append(time_chain)
            # print(current_chain)
            # print(time_chain)

    for chain in chains:
        for index in chain:
            validation_mask[index] = 1
    print(chains)
    return validation_mask, time_chains


def starmap_with_kwargs(pool, fn, args_iter, kwargs_iter):
    args_for_starmap = zip(repeat(fn), args_iter, kwargs_iter)
    return pool.starmap(apply_args_and_kwargs, args_for_starmap)


def apply_args_and_kwargs(fn, args, kwargs):
    return fn(*args, **kwargs)


def download_data(references, validation_mask):
    print("Downloading {} files...".format(np.sum(validation_mask)))
    bar = tqdm(total=np.sum(validation_mask))
    pool = Pool(4)
    valid_refs = []
    for i, data in enumerate(references):
        if validation_mask[i] == 1:
            Fido.fetch(data, path='./sun_data', progress=False, max_conn=1)
            valid_refs.append(data)
            bar.update(1)
    bar.close()
    # starmap_with_kwargs(pool, Fido.fetch, valid_refs, repeat(dict(path='./sun_data', progess=False, max_conn=0)))
    return None


def generate_training_data(time_chains, frames, image_size):
    questions = np.zeros((len(time_chains), frames, image_size, image_size, 1), dtype=np.float32)
    answers = np.zeros((len(time_chains), 2, image_size, image_size, 1), dtype=np.float32)
    for index, chain in enumerate(time_chains):
        for frame, time in enumerate(chain):
            # print(time)
            time_of_obs = datetime.fromtimestamp(time - 60*60)
            # print(time_of_obs)
            data_string = "sun_data/efz{:04d}{:02d}{:02d}.{:02d}{:02d}{:02d}".format(
                time_of_obs.year, time_of_obs.month, time_of_obs.day, time_of_obs.hour, time_of_obs.minute, time_of_obs.second)
            # print(data_string)
            if frame < frames:
                questions[index, frame, :, :, 0] = get_data(data_string, image_size)
            else:
                answers[index, frame-frames, :, :, 0] = get_data(data_string, image_size)

    batch_size = 8
    print("Turning into dataset...")
    testing_data = tf.data.Dataset.from_tensor_slices((questions, answers))
    print("Batching...")
    testing_data = testing_data.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    print("Sending off...")
    return testing_data


def find_data_refs():
    # Looking month by month
    running_total = 0
    frames = 4
    future_look = 10
    gap_positions = []
    downloads = []
    for year in range(2001, 2002):
        for month in range(6, 8):
            if month != 12:
                time_range = a.Time('{}/{:02d}/01 00:00:00'.format(year, month),
                                    '{}/{:02d}/01 00:00:00'.format(year, month+1))
                # time_range = a.Time('{}/{:02d}/01 00:00:00'.format(2001, 1),
                #                     '{}/{:02d}/01 00:00:00'.format(2002, 1))
            else:
                time_range = a.Time('{}/{:02d}/01 00:00:00'.format(year, month),
                                    '{}/{:02d}/01 00:00:00'.format(year+1, 1))

            instrument = a.Instrument.eit
            wavelength = a.Wavelength(195 * u.Angstrom)
            print("Searching month {}/{:02d}".format(year, month))
            search_results = Fido.search(time_range, instrument, wavelength)
            times = np.asarray(search_results.show("Start Time", "Size"))[0]
            to_use = np.zeros(len(times))
            previous = times[0][0]
            to_use[0] = 1
            added_files = 1
            running = 1
            running_start = 0
            great_runs = []
            for index, recording in enumerate(times[1:]):
                # See if gap is 12 minutes
                # print((recording[0] - previous).to_value('sec')/60)
                if 9 * 60 <= (recording[0] - previous).to_value('sec') <= 15 * 60:
                    to_use[index + 1] = 1
                    running += 1
                else:
                    to_use[index + 1] = 0
                    if running < frames + future_look:
                        great_runs.append(running)
                        to_use[running_start:running_start+running] = 0
                    running = 0
                    running_start = index + 2
                    if np.sum(to_use) + running_total not in gap_positions:
                        gap_positions.append(np.sum(to_use) + running_total)
                previous = recording[0]
            print(max(great_runs))
            running_total += np.sum(to_use)
            gap_positions.append(running_total)
            print("{}/{} new files added, currently {} files ({:.2f} Gb)".format(
                int(np.sum(to_use)), len(times), running_total, running_total*2.1/1024))
            # print(to_use[:] == 1)
            # print(search_results[0][to_use[:] == 1])
            # print(gap_positions)
            print("Downloading...")
            downloads += Fido.fetch(search_results[0][to_use[:] == 1], path='./sun_data', progress=False)
    np.save("gap_positions.npy", gap_positions)
    np.save("download_refs.npy", np.asarray(downloads))
    return gap_positions, downloads


def get_data(item, image_size):
    mapped_item = sunpy.map.Map(item)
    # Should return NxN image
    output_maybe = np.asarray(mapped_item.data)
    size = np.shape(output_maybe)[0]
    output = skimage.measure.block_reduce(output_maybe, (size//image_size, size//image_size), np.mean)
    output = np.tanh(output/1500)
    del mapped_item
    del output_maybe
    return output


def files_to_numpy(downloads, gaps, size, frames, future_runs):
    future_look = future_runs
    total = 0
    for index, gap in enumerate(gaps[1:]):
        if gap - gaps[index] > frames + future_look:
            total += int(gap - gaps[index] - (frames + future_look))
            # print(gap - gaps[index] - (frames + future_look))
    print("--")
    print(total)
    questions = np.zeros((total, frames, size, size, 1), dtype=np.float32)
    answers = np.zeros((total, 2, size, size, 1), dtype=np.float32)
    stretch = []
    accessed_index = 0
    # print(gaps)
    # print(downloads[0:28])
    progress = tqdm.tqdm(total=len(gaps[1:]))
    for index, gap in enumerate(gaps[1:]):
        progress.update(1)
        if gap - gaps[index] > frames + future_look:
            for i in range(int(gap - gaps[index] - (frames + future_look))):
                current_index = int(gaps[index]) + i
                for frame in range(frames):
                    questions[accessed_index, frame, :, :, 0] = get_data(downloads[current_index + frame], size)
                if len(stretch) == 0:
                    for j in range(int(gap - gaps[index])):
                        stretch.append(get_data(downloads[current_index + j], size))
                answers[accessed_index, 0, :, :, 0] = get_data(downloads[current_index + frames], size)
                answers[accessed_index, 1, :, :, 0] = get_data(downloads[current_index + frames + future_look], size)
                accessed_index += 1
    progress.close()
    print(np.shape(questions))
    print(np.max(questions))
    print(np.std(questions))
    total_data = np.asarray(stretch)
    print(np.shape(total_data))
    print("Number of data stretches: {}".format(len(gaps)-1))
    # plt.hist(np.reshape(total_data, 84*512*512), bins=50)
    # plt.show()
    # exit()
    image_converts = total_data * 255
    image_converts = image_converts.astype(np.uint8)
    images = []
    for i in image_converts:
        images.append(i)
    imageio.mimsave("SOHO_test.gif", images)
    batch_size = 8
    testing_data = tf.data.Dataset.from_tensor_slices((questions, answers))
    testing_data = testing_data.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return testing_data


def main():
    # gap, down = find_data_refs()

    obs, size, ref = get_metadata([1999], [6], a.Instrument.eit, 195)
    mask, time_chains = get_valid_data(4, 10, obs)
    print("Total data use {:.2f}Gb ({} files), {} training data items".format(
        np.sum(size[mask[:] == 1])/1024, np.sum(mask), len(time_chains)))
    download_data(ref, mask)
    generate_training_data(time_chains, 4, 128)
    exit()
    gap = np.load("gap_positions.npy")
    down = np.sort(np.load("download_refs.npy"))
    files_to_numpy(down, gap)
    exit()
    # time_range = a.Time('2004/06/15 08:00:00', '2004/06/15 09:00:00')
    # instrument = a.Instrument.eit
    # result = Fido.search(time_range, instrument)
    # print(result)
    # downloaded_file = Fido.fetch(result)
    # print(downloaded_file)
    # hmi_map = sunpy.map.Map(downloaded_file[0])
    # fig = plt.figure()
    # hmi_map.plot()
    # print(hmi_map.data)
    # plt.show()
    # print(a.Instrument)
    times = []
    during_day = []
    time_gaps = []
    for i in range(0, 22):
        print('20{:02d}/01/01 00:00:00'.format(i))
        time_range = a.Time('20{:02d}/01/01 00:00:00'.format(i), '20{:02d}/02/01 00:00:00'.format(i))
        instrument = a.Instrument.eit
        # wavelength = a.Physobs.intensity
        wavelength = a.Wavelength(195 * u.Angstrom)
        results = Fido.search(time_range, instrument, wavelength)
        print(results.all_colnames)
        table = np.asarray(results.show("Start Time", "Wavelength", "fileid"))
        if results.file_num > 0:
            print(table[0, :])
        run = 1
        for time in table[0, :]:
            try:
                if len(times) > 0:
                    if np.floor(times[-1]) == np.floor(time[0].to_value("mjd")):
                        run += 1
                    else:
                        run = 1
                    if (time[0].to_value('mjd') - times[-1])*24*60 < 60:
                        time_gaps.append((time[0].to_value('mjd') - times[-1])*24*60)
                    # if run > 4:
                    #     print(time[0])
                times.append(time[0].to_value("mjd"))
                during_day.append(time[0].to_value("mjd") - np.floor(time[0].to_value("mjd")))
            except IndexError:
                print("None found")
        # print(results.all_colnames)
        print(results.file_num)

    # plt.hist(times, bins=8401)
    # plt.hist(times, bins=31)
    # plt.show()
    # plt.clf()
    # plt.hist(np.asarray(during_day)*24*60, bins=24*60)
    # plt.show()
    plt.hist(time_gaps, bins=120, range=(-0.25, 59.75))
    plt.show()



if __name__ == "__main__":
    main()




