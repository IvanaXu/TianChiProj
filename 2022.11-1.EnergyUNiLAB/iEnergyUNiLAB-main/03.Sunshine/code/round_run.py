import random
from tqdm import tqdm
from multiprocessing import Pool

def run(in_chunk_len, out_chunk_len, max_epochs, imax_epochs, batch_size, rnn_type_or_module):
    try:
        import paddle
        import datetime
        import pandas as pd
        from paddlets import TSDataset
        from sklearn.metrics import mean_squared_error
        from paddlets.models.forecasting import RNNBlockRegressor

        sunshine = pd.read_csv("../data/sunshine.csv")
        print("sunshine", sunshine.describe())

        wind = pd.read_csv("../data/wind.csv")
        print("wind", wind.describe())

        temp = pd.read_csv("../data/temp.csv")
        print("temp", temp.describe())

        print(sunshine.shape, wind.shape, temp.shape)
        assert wind.shape[0] == temp.shape[0]

        print(sunshine.shape[0]/15, wind.shape[0]/24)
        assert sunshine.shape[0]/15 == wind.shape[0]/24 - 10


        def dh2dt(_d, _h, _p=False):
            _r = datetime.datetime.strptime(f"2000-01-01 {int(_h)-1}:00:00", "%Y-%m-%d %H:%M:%S") + datetime.timedelta(days=_d)
            if _p:
                print("dh2dt", _r)
            return _r 


        def dt2dh(_dt, _p=False):
            _d = (_dt - datetime.datetime.strptime(f"2000-01-01 00:00:00", "%Y-%m-%d %H:%M:%S")).days
            _h = _dt.hour + 1
            if _p:
                print("dt2dh", _d, _h)
            return _d, _h


        for _d in range(2):
            for _h in range(1, 25):
                dt2dh(dh2dt(_d, _h, _p=True), _p=True)


        data = pd.merge(wind, temp, on=["Day", "Hour"], how="left")
        data = pd.merge(data, sunshine, on=["Day", "Hour"], how="left")
        data["Day"] = data["Day"].apply(float)
        data["Hour"] = data["Hour"].apply(float)
        data["is_Hour"] = data["Hour"].apply(lambda x: 1.0 if 6 <= x <= 20 else 0.0)

        NNN = 0.25
        print(f'Radiation mean: {NNN}')
        data["Radiation"] = data["Radiation"].fillna(NNN)

        for icol in data.columns:
            data[icol] = data[icol].fillna(data[icol].mean())
        # data.fillna(-999999, inplace=True)
        data["dt"] = [
            dh2dt(_d, _h)
            for _d, _h in zip(data["Day"], data["Hour"])
        ]
        data["para-A"] = 2.0
        data["para-n"] = 0.5
        print(data)


        data_ds = TSDataset.load_from_dataframe(
            data,
            time_col='dt',
            target_cols='Radiation',
            known_cov_cols=['Day', 'Hour', 'is_Hour', 'Dir', 'Spd', 'Temp'],
            static_cov_cols=["para-A", "para-n"],
            freq='1h',

            # max, min, avg, median, pre, back, zero
            # fill_missing_dates=True,
            # fillna_method='max',
        )
        print(data_ds)

        train_ds, testa_ds = data_ds.split(
            pd.Timestamp(dh2dt(_d=299-10, _h=24, _p=True)),
        )
        print(train_ds, testa_ds)
        
        model = RNNBlockRegressor(
            in_chunk_len=in_chunk_len,
            out_chunk_len=out_chunk_len,
            rnn_type_or_module=rnn_type_or_module,
            dropout=0.1,
            max_epochs=max_epochs,
            patience=max_epochs//imax_epochs,
            loss_fn=paddle.nn.functional.mse_loss,
            eval_metrics=['mse'],
            seed=10086,
            verbose=1,
            batch_size=batch_size,
        )
        model.fit(train_tsdataset=train_ds, valid_tsdataset=testa_ds)

        train_pr = model.recursive_predict(
            tsdataset=train_ds, 
            predict_length=20 * 24
        )

        _1 = testa_ds.to_dataframe().head(10*24)["Radiation"].to_numpy()
        _2 = train_pr.to_numpy()[:, 0][:10*24]
        _1_cut = [_1[_k] for _k, _v in enumerate(_1) if _v != NNN]
        _2_cut = [_2[_k] for _k, _v in enumerate(_1) if _v != NNN]
        assert len(_1) == len(_2)
        assert len(_1_cut) == len(_2_cut)
        mse = mean_squared_error(_1, _2)
        print(f"MSE {len(_1)}: {mse:.4f}\n"
            f"MSE {len(_1_cut)}: {mean_squared_error(_1_cut, _2_cut):.4f}")
        
        _result = train_pr.to_dataframe()
        _result["_d"] = [dt2dh(i)[0] for i in _result.index]
        _result["_h"] = [dt2dh(i)[1] for i in _result.index]
        _result = _result[(_result["_d"] >= 300) & (_result["_h"] >= 6) & (_result["_h"] <= 20)]
        _result["Radiation"].to_csv(f"./save/result-{mse:.12f}.csv", index=False)
        
        with open(f"./save/result-{mse:.12f}.csv.log", "w") as f:
            f.write(
                f"in_chunk_len: {in_chunk_len},"
                f"out_chunk_len: {out_chunk_len},"
                f"max_epochs: {max_epochs},"
                f"imax_epochs: {imax_epochs},"
                f"batch_size: {batch_size},"
                f"rnn_type_or_module: {rnn_type_or_module},"
            )
    except:
        mse = 999999
    return mse


if __name__=='__main__':
    p = Pool(12)

    # in_chunk_len, out_chunk_len, max_epochs, imax_epochs, batch_size
    Lpara = []
    for para1 in tqdm(range(1, 12, 1)):
        for para2 in [1, 2, 3]:
            for para3 in range(100, 1000, 100):
                for para4 in [10, 20, 50, 100]:
                    for para5 in [8, 16, 32, 64, 128]:
                        for para6 in ["SimpleRNN", "LSTM", "GRU"]:
                            for _ in range(4):
                                Lpara.append([para1, para2, para3, para4, para5, para6])
    random.shuffle(Lpara)

    for [para1, para2, para3, para4, para5, para6] in Lpara:
        p.apply_async(run, args=(para1, para2, para3, para4, para5, para6, ))
    
    p.close()
    p.join()
