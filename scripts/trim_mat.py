import h5py
import numpy as np

if __name__ == '__main__':

    """
        A small utility to generate a slimmed down version of the order book .mat file
    """

    input_file = 'data/data.mat'
    output_file = 'data/test.mat'
    seq_len = 100

    # load
    series = {}
    with h5py.File(input_file, 'r') as f:
        for name, data in f.items():
            print(f'Loading: {name} ...')
            series[name] = np.array(data)[:, :seq_len]

    # save
    with h5py.File(output_file, 'w') as f:
        for name, np_data in series.items():
            f.create_dataset(name, data=np_data)

    # reload and check
    with h5py.File(output_file, 'r') as f:
        for name, data in f.items():
            print(f'Loading: {name} ...')
            np_data = np.array(data)
            np.testing.assert_equal(np_data, series[name])
