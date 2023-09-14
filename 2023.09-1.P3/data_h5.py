import pandas as pd
import h5py


def savefile(methy_dir, chunk_size, name):
    df_chunks = pd.read_csv(methy_dir,
                            sep=',',
                            index_col=0,
                            chunksize=chunk_size)

    with h5py.File(name, 'w') as file:
        total_cols = 0
        for i, chunk in enumerate(df_chunks):
            chunk = chunk.transpose()
            chunk = chunk.fillna(0)
            # fill nan with 0, you can try other methods
            data_array = chunk.to_numpy()
            chunk_cols = data_array.shape[1]
            if i == 0:
                samples_num = data_array.shape[0]
                dataset = file.create_dataset('data',
                                              shape=data_array.shape,
                                              maxshape=(samples_num, None))

            dataset.resize((dataset.shape[0], total_cols + chunk_cols))

            dataset[:, total_cols:total_cols + chunk_cols] = data_array

            total_cols += chunk_cols  # Update total_cols within the loop

    return None


chunk_size = 5000

methy_train_dir = 'traindata.csv'
savefile(methy_train_dir, chunk_size, 'train.h5')
print('transform traindata over')

methy_test_dir = 'testdata.csv'
savefile(methy_test_dir, chunk_size, 'test.h5')
print('transform testdata over')
