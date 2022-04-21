import imageio
import requests
import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
from datetime import datetime


def main():
    # print("Howdy!")
    metadata = "http://ssa.esac.esa.int/ssa/aio/metadata-action"
    data = "http://ssa.esac.esa.int/ssa/aio/product-action"
    samples = 1000
    query = ''
    query += 'OBSERVATION.BEGINDATE>\'1997-06-17 23:00:00\''
    query += ' AND '
    query += 'OBSERVATION.BEGINDATE<\'2019-06-18 21:59:59\''
    query += ' AND '
    query += 'OBSERVATION.OBSERVATIONTYPE==\'CME WATCH 195\''
    payload = {
        "SELECTED_FIELDS": "OBSERVATION",
        "PAGE_SIZE": str(samples),
        'PAGE': '1',
        'RESOURCE_CLASS': 'OBSERVATION',
        'INSTRUMENT.NAME': 'EIT',
        # 'INSTRUMENT.NAME': 'LASCO',
        # 'INSTRUMENT.NAME': 'MDI',
        'RETURN_TYPE': 'JSON',
        'QUERY': query
    }
    r = requests.get(metadata, params=payload)
    print(r.url)
    # print(r.json())
    # print(r.json()['data'][0]['OBSERVATION.ID'])
    json_data = r.json()
    max_sol = []
    min_sol = []
    avg_sol = []
    dates = []
    # solar_data = []
    print(np.shape(json_data['data']))
    # exit()
    # solar_data = np.zeros((np.shape(json_data['data'])[0], 1024, 1024, 3), dtype=np.uint8)
    for i in range(np.shape(json_data['data'])[0]):
        print(i)
        # print(json_data['data'][i])
        print(json_data['data'][i]['OBSERVATION.ID'])
        print(json_data['data'][i]['OBSERVATION.BEGINDATE'])
        print(json_data['data'][i]['OBSERVATION.OBSERVATIONTYPE'])
        payload = {
            'OBSERVATION.ID': json_data['data'][i]['OBSERVATION.ID']
        }

        # r = requests.get(data, params=payload)
        # with open("testing.fits", 'wb') as f:
        #     f.write(r.content)
        # solar_fit = fits.open('testing.fits')
        # solar_fit.info()
        # sol = solar_fit[0].data
        # try:
        #     print(np.mean(sol))
        #     avg_sol.append(np.mean(sol))
        #     print(np.min(sol))
        #     min_sol.append(np.min(sol))
        #     print(np.max(sol))
        #     max_sol.append(np.max(sol))
        #     solar_data[i, :, :, 2] = (np.tanh(sol/2000) * 255).astype(np.uint8)
        # except ValueError:
        #     print("Incorrect size")
        # solar_fit.close()
        dates.append(json_data['data'][i]['OBSERVATION.BEGINDATE'])

    dates = np.asarray(dates)
    index = dates.argsort()
    print(dates[index])
    actual_dates = []
    for date in dates[index]:
        actual = datetime.strptime(date, "%Y-%m-%d %H:%M:%S.%f")
        actual_dates.append(actual)
    deltas = []
    for i in range(len(actual_dates)-1):
        deltas.append((actual_dates[i+1]-actual_dates[i]).seconds)
    plt.hist(deltas, bins=50)
    plt.show()
    solar_data = solar_data[index]
    images = []
    for i in solar_data:
        images.append(i)
    imageio.mimsave("SOHO_EDI.gif", images)
    print(np.shape(solar_data))
    # plt.imshow(solar_data)
    # plt.show()


if __name__ == "__main__":
    main()




