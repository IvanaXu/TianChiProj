import os
import xml
import datetime
import pandas as pd
from tqdm import tqdm
from xml.dom import minidom

day = datetime.datetime.strftime(datetime.datetime.now(), "%Y_%m_%d")
print(day)

os.system(f"rm -rf eqList/eqList-day.xls")
os.system(f"cp /Users/ivan/Downloads/eqList{day}.xls eqList/eqList-day.xls")
print(os.listdir("eqList/"))
aeta = f"eqList/eqList-day.xls"
print(aeta)

N = 7
T = 20
# 经度
lonL, lonH = 98, 107
# 纬度
latL, latH = 22, 34

DOMTree = xml.dom.minidom.parse(aeta).documentElement
Row = DOMTree.getElementsByTagName("Row")

cols, data = [], []
for nRow, iRow in enumerate(Row):
    if nRow == 0:
        for jRow in iRow.getElementsByTagName("Cell"):
            cols.append(jRow.childNodes[0].childNodes[0].data)
    else:
        idata = []
        for jRow in iRow.getElementsByTagName("Cell"):
            idata.append(jRow.childNodes[0].childNodes[0].data)
        data.append([float(dj) if nj in [1, 2, 3, 4] else str(dj) for nj, dj in enumerate(idata)])

data = pd.DataFrame(data, columns=cols)
print(data.dtypes)
print(data.head(T))


def regeo(location):
    try:
        import json
        import requests

        # TODO: appcode !!!
        host = 'https://regeo.market.alicloudapi.com'
        path = '/v3/geocode/regeo'
        appcode = '830fa522a512421eaa202bb80afe8921'
        querys = f'location={location}' # 经度在前，纬度在后
        with requests.get(
            f"{host}{path}?{querys}",
            headers={'Authorization': f'APPCODE {appcode}'}
        ) as r:
            j = json.loads(r.content.decode())["regeocode"]["formatted_address"]
            return j
    except Exception as e:
        print(e)
        return "NaN"


Kdata = []
for STATION in tqdm(data.head(T)["参考位置"]):
    # idata
    idata = data[data["参考位置"] == STATION][["参考位置", "发震时刻"]]
    idata.rename({
        "参考位置": "初次参考位置",
        "发震时刻": "初次发震时刻",
    }, axis=1, inplace=True)
    idata["K"] = 1
    # print(idata.shape)

    # jdata
    jdata = data[
        (data["参考位置"] != STATION) &
        (data["经度(°)"] >= lonL) & (data["经度(°)"] <= lonH) &
        (data["纬度(°)"] >= latL) & (data["纬度(°)"] <= latH) &
        # (data["震级(M)"] >= 3.5) &
        True
    ][["参考位置", "震级(M)", "纬度(°)", "经度(°)", "发震时刻"]]
    jdata["K"] = 1
    # print(jdata.shape)

    # k0data
    k0data = pd.merge(
        idata,
        jdata,
        on="K"
    )
    # print(k0data.shape)

    # k1data
    # 筛选近N天内
    k0data["gap-发震时刻"] = (
        pd.to_datetime(k0data["发震时刻"]) -
        pd.to_datetime(k0data["初次发震时刻"])
    ).dt.days

    k1data = (k0data["gap-发震时刻"] >= 1) & (k0data["gap-发震时刻"] <= N)
    # print(pd.value_counts(k1data))

    k1data = k0data[k1data]
    # print(k1data.shape)

    # k2data
    k2data = k1data.groupby(
        ["初次参考位置", "初次发震时刻", "K"],
        as_index=False
    ).agg({
        "震级(M)": "mean", # 特殊/余震
        "纬度(°)": "mean",
        "经度(°)": "mean",
    }).groupby(
        ["K"],
        as_index=False
    ).agg(
        {
            "K": "sum",
            "震级(M)": "mean",
            "纬度(°)": "mean",
            "经度(°)": "mean",
    })

    k2data["参考位置"] = STATION
    k2data["K%"] = k2data["K"]/idata.shape[0]
    k2data["震级C"] = k2data["震级(M)"].apply(lambda x: "" if x >= 3.5 else "❌")
    k2data["经度C"] = k2data["经度(°)"].apply(lambda x: "" if lonL <= x <= lonH else "❌")
    k2data["纬度C"] = k2data["纬度(°)"].apply(lambda x: "" if latL <= x <= latH else "❌")
    k2data["经纬度解析"] = [regeo(f"{_1},{_2}") for _1, _2 in zip(k2data["经度(°)"], k2data["纬度(°)"])]
    k2data["经度(°)"] = k2data["经度(°)"].apply(lambda x: f"{x:.6f}")
    k2data["纬度(°)"] = k2data["纬度(°)"].apply(lambda x: f"{x:.6f}")

    k2data = k2data[
        ['参考位置', '震级(M)', '震级C', '纬度(°)', '纬度C', '经度(°)', '经度C', 'K', 'K%', '经纬度解析']
    ]
    # print(k2data)
    Kdata.append(k2data)

    #
    del idata, jdata, k0data, k1data, k2data

endata = pd.merge(
    data.head(T).rename(
        {icol: f"#{icol}#" for icol in data.columns if icol != "参考位置"},
        axis=1
    ),
    pd.concat(Kdata),
    on="参考位置"
).sort_values("#发震时刻#", ascending=False)
print(endata.to_string())


#
day1 = datetime.datetime.strftime(datetime.datetime.now() + datetime.timedelta(days=1), "%Y-%m-%d")
dayn = datetime.datetime.strftime(datetime.datetime.now() + datetime.timedelta(days=N), "%Y-%m-%d")
day7 = datetime.datetime.strftime(datetime.datetime.now() + datetime.timedelta(days=7), "%Y-%m-%d")
print(f">>> {day1} To {day7} / {dayn}")

message = f"""
# 有震预测 earthquake prediction
check_my_prediction(myToken, '{day1}', '{dayn}', 1, latitude=$_1, longitude=$_2, magnitude=$_3)
"""
endata = endata[endata["震级C"] != "❌"]
for _1, _2, _3 in zip(endata["纬度(°)"], endata["经度(°)"], endata["震级(M)"]):
    print(message
          .replace("$_1", f"{float(_1):.6f}")
          .replace("$_2", f"{float(_2):.6f}")
          .replace("$_3", f"{float(_3):.1f}")
         )

message = f"""
# 无震预测 No earthquake prediction
check_my_prediction(myToken, '{day1}', '{day7}', 0)
"""
print(message)
print()
