import imageio
import requests
import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
from datetime import datetime
from datetime import timedelta
import sunpy.map
from astropy.time import Time
import astropy.units as u
import astropy.table as t
from sunpy.net import Fido
from sunpy.net import attrs as a

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


def find_data_refs():
    # Looking month by month
    running_total = 0
    frames = 4
    future_look = 20
    gap_positions = []
    downloads = []
    for year in range(1999, 2000):
        for month in range(1, 13):
            if month != 12:
                time_range = a.Time('{}/{:02d}/01 00:00:00'.format(year, month),
                                    '{}/{:02d}/01 00:00:00'.format(year, month+1))
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
            for index, recording in enumerate(times[1:]):
                # See if gap is 12 minutes
                if 10 * 60 <= (recording[0] - previous).to_value('sec') <= 14 * 60:
                    to_use[index + 1] = 1
                    running += 1
                else:
                    to_use[index + 1] = 0
                    if running < frames + future_look:
                        to_use[running_start:running_start+running] = 0
                    running = 0
                    running_start = index + 2
                    if np.sum(to_use) + running_total not in gap_positions:
                        gap_positions.append(np.sum(to_use) + running_total)
                previous = recording[0]
            running_total += np.sum(to_use)
            print("{}/{} new files added, currently {} files ({:.2f} Gb)".format(
                int(np.sum(to_use)), len(times), running_total, running_total*2.1/1024))
            # print(to_use[:] == 1)
            # print(search_results[0][to_use[:] == 1])
            # print(gap_positions)
            # downloads += Fido.fetch(search_results[0][to_use[:] == 1], path='./sun_data')

    return gap_positions, downloads


def files_to_numpy(downloads, gaps):
    frames = 4
    future_look = 10
    total = 0
    for index, gap in enumerate(gaps[1:]):
        if gap - gaps[index] > frames + future_look:
            total += gap - gaps[index] - (frames + future_look)
            print(gap - gaps[index] - (frames + future_look))
    print("--")
    print(total)
    return 0


def main():
    gap, down = find_data_refs()
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




