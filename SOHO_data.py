import requests
import numpy as np
import matplotlib.pyplot as plt
import astropy.io


def main():
    print("Howdy!")
    metadata = "http://ssa.esac.esa.int/ssa/aio/metadata-action"
    payload = {
        "SELECTED_FIELDS": "OBSERVATION",
        "PAGE_SIZE": '1',
        'PAGE': '3',
        'RESOURCE_CLASS': 'OBSERVATION',
        'INSTRUMENT.NAME': 'EIT',
        'RETURN_TYPE': 'JSON'
    }
    r = requests.get(metadata, params=payload)
    print(r.url)
    print(r.json())
    print(r.json()['data'][0]['OBSERVATION.ID'])
    data = "http://ssa.esac.esa.int/ssa/aio/product-action"
    payload = {
        'OBSERVATION.ID': r.json()['data'][0]['OBSERVATION.ID']
    }
    r = requests.get(data, params=payload)
    print(r.url)
    with open("testing.fits", 'wb') as f:
        f.write(r.content)



if __name__ == "__main__":
    main()




