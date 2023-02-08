# 中文

本模块包含台站信息介绍与数据特征介绍



### StationInfo.csv

`StationInfo.csv`  涵盖目标区域（北纬 22~34，东经 98~107）内计159个台站。

**各列说明**

|     列名      |               说明               |
| :-----------: | :------------------------------: |
|    `Title`    |             台站名称             |
|  `StationID`  |              台站ID              |
|  `Longitude`  |         台站经度（东经）         |
|  `Latitude`   |         台站纬度（北纬）         |
|  `MagnData`   |         台站含有电磁数据         |
| `MagnUpdate`  | 台站2020/01/01后仍在更新电磁数据 |
|  `SoundData`  |         台站含有地声数据         |
| `SoundUpdate` | 台站2020/01/01后仍在更新地声数据 |

**备注**

1. 在2017/01/01—2020/11/30近四年时间中，台站安装时间不一，因此台站的数据起始时间不一；

2. 近四年时间中，部分台站换位置或者停机，部分台站数据不再更新；

3. 近四年时间中，部分台站会由于天气、环境原因出现短时间断电重启，因此台站时间戳可能存在不连续情况。

   

### DataFeatureInfo

包含`EM_20170101-20201130.zip`电磁压缩文件及`GA_20170101-20201130.zip`地声压缩文件。

以`EM_20170101-20201130.zip`为例介绍压缩文件内容：

> 19_magn.zip（台站ID_信号类型）
> 
> >19_magn.csv（台站ID_信号类型）
>
> 24_magn.zip
>
> ......
>
> 60251_magn.zip

其中信号类型分为：magn为电磁扰动，sound为地声。

电磁特征共51类，地声特征共44类。

**特征说明**

|             特征类型              |         名称          |         中文含义          | 电磁特征数 | 地声特征数 |
| :-------------------------------: | :-------------------: | :-----------------------: | :--------: | :--------: |
|   `电磁扰动时域`/`地声原始时域`   |          var          |           方差            |     2      |     1      |
|   `电磁扰动时域`/`地声原始时域`   |         power         |           功率            |     2      |     1      |
|  ``电磁扰动时域``/`地声原始时域`  |         skew          |           偏度            |     2      |     1      |
|   `电磁扰动时域`/`地声原始时域`   |         kurt          |           峰度            |     2      |     1      |
|   `电磁扰动时域`/`地声原始时域`   |         mean          |           均值            |     0      |     1      |
|   `电磁扰动时域`/`地声原始时域`   |        abs_max        |      绝对值的最大值       |     2      |     1      |
|   `电磁扰动时域`/`地声原始时域`   |       abs_mean        |       绝对值的均值        |     2      |     1      |
|   `电磁扰动时域`/`地声原始时域`   |       abs_top_x       |     绝对值最大 x%位置     |     4      |     2      |
|   `电磁扰动时域`/`地声原始时域`   |      energy_sstd      |      短时能量标准差       |     2      |     1      |
|   `电磁扰动时域`/`地声原始时域`   |      energy_smax      |      短时能量最大值       |     2      |     1      |
|   `电磁扰动时域`/`地声原始时域`   |      s_zero_rate      |      短时过零率均值       |     0      |     1      |
|   `电磁扰动时域`/`地声原始时域`   |    s_zero_rate_max    |     短时过零率最大值      |     0      |     1      |
|     `电磁扰动频域`/`地声频域`     |    power_rate_atob    |     频谱中 a~bHz 功率     |     11     |     11     |
|     `电磁扰动频域`/`地声频域`     |   frequency_center    |         重心频率          |     1      |     1      |
|     `电磁扰动频域`/`地声频域`     | mean_square_frequency |         均方频率          |     1      |     1      |
|     `电磁扰动频域`/`地声频域`     |  variance_frequency   |         频率方差          |     1      |     1      |
|     `电磁扰动频域`/`地声频域`     |   frequency_entropy   |          频谱熵           |     1      |     1      |
| `电磁扰动小波变换`/`地声小波变换` |    levelx_absmean     | 第 x 层重构后绝对值的均值 |     4      |     4      |
| `电磁扰动小波变换`/`地声小波变换` |     levelx_energy     |     第 x 层重构后能量     |     4      |     4      |
| `电磁扰动小波变换`/`地声小波变换` |  levelx_energy_svar   |  第 x 层重构后能量值方差  |     4      |     4      |
| `电磁扰动小波变换`/`地声小波变换` |  levelx_energy_smax   | 第 x 层重构后能量值最大值 |     4      |     4      |
|               合计                |                       |                           |     51     |     44     |

**备注**

1. 原始信号提取的特征分成三大类：时域相关的特征，频域相关的特征，以及小波变换相关的特征；


# English

This module contains the introduction of station information and data characteristics.



### StationInfo.csv

`StationInfo.csv`  covers a total of 159 stations in the target area (22°N -34°N, 98°E -107°E).

**Columns**

|    Column     |                         Description                          |
| :-----------: | :----------------------------------------------------------: |
|    `Title`    |                         Station name                         |
|  `StationID`  |                          Station ID                          |
|  `Longitude`  |                  East Longitude of station                   |
|  `Latitude`   |                  North Latitude of station                   |
|  `MagnData`   |            Station contains electromagnetic data             |
| `MagnUpdate`  | Station still updating electromagnetic data after Jan 1st, 2020 |
|  `SoundData`  |              Station contains geoacoustic data               |
| `SoundUpdate` | Station still updating geoacoustic data after Jan 1st, 2020  |

**remarks**

1. AETA stations are installed at different time, so the beginning time of data in each station is different.

2. During the past four years, some stations are moved or closed, and these stations no longer update their data.

3. In the past four years, short-term power outages or breakdown due to weather and environmental affection, so the timestamps of data may be discontinuous.

   

### DataFeatureInfo

Contains a compressed file of electromagnetic data ( `EM_20170101-20201130.zip` ) and a compressed file of geoacoustic data ( `GA_20170101-20201130.zip` ).

Take `EM_20170101-20201130.zip` as an example to introduce the contents of compressed files：

> 19_magn.zip（StationID_SignalType）
>
> >19_magn.csv（StationID_SignalType）
>
> 24_magn.zip
>
> ......
>
> 60251_magn.zip

There are two SignalType, magn represents EM signal and sound represents GA signal.

There are 51 types of electromagnetic features and 44 types of geoacoustic features.

**Feature description**

EMDTD : Electromagnetic disturbance time domain

EMDFD : Electromagnetic disturbance frequency domain

EMDWT : Electromagnetic disturbance wavelet transform

GAOTD : Geoacoustic original time domain

GAFD : Geoacoustic frequency domain

GAWT : Geoacoustic wavelet transform

|  Feature type   |        column         |                        description                         | number of EM features | number of GA features |
| :-------------: | :-------------------: | :--------------------------------------------------------: | :-------------------: | :-------------------: |
| `EMDTD`/`GAOTD` |          var          |                          Variance                          |           2           |           1           |
| `EMDTD`/`GAOTD` |         power         |                           Power                            |           2           |           1           |
| `EMDTD`/`GAOTD` |         skew          |                          Skewness                          |           2           |           1           |
| `EMDTD`/`GAOTD` |         kurt          |                          Kurtosis                          |           2           |           1           |
| `EMDTD`/`GAOTD` |         mean          |                            Mean                            |           0           |           1           |
| `EMDTD`/`GAOTD` |        abs_max        |                   Maximum absolute value                   |           2           |           1           |
| `EMDTD`/`GAOTD` |       abs_mean        |                    Mean absolute value                     |           2           |           1           |
| `EMDTD`/`GAOTD` |       abs_top_x       |                Absolute maximum x% position                |           4           |           2           |
| `EMDTD`/`GAOTD` |      energy_sstd      |            Short-term energy standard deviation            |           2           |           1           |
| `EMDTD`/`GAOTD` |      energy_smax      |                 Maximum short-term energy                  |           2           |           1           |
| `EMDTD`/`GAOTD` |      s_zero_rate      |             Mean short-term zero-crossing rate             |           0           |           1           |
| `EMDTD`/`GAOTD` |    s_zero_rate_max    |           Maximum short-time zero-crossing rate            |           0           |           1           |
| `EMDFD`/`GAFD`  |    power_rate_atob    |                        a~bHz power                         |          11           |          11           |
| `EMDFD`/`GAFD`  |   frequency_center    |                Center of gravity frequency                 |           1           |           1           |
| `EMDFD`/`GAFD`  | mean_square_frequency |                   Mean square frequency                    |           1           |           1           |
| `EMDFD`/`GAFD`  |  variance_frequency   |                     Frequency variance                     |           1           |           1           |
| `EMDFD`/`GAFD`  |   frequency_entropy   |                     Frequency entropy                      |           1           |           1           |
| `EMDWT`/`GAWT`  |    levelx_absmean     | Mean absolute value after reconstruction of the xth level  |           4           |           4           |
| `EMDWT`/`GAWT`  |     levelx_energy     |        Energy after reconstruction of the xth level        |           4           |           4           |
| `EMDWT`/`GAWT`  |  levelx_energy_svar   |  Variance of energy after reconstruction of the xth level  |           4           |           4           |
| `EMDWT`/`GAWT`  |  levelx_energy_smax   | Maximum energy value after reconstruction of the xth level |           4           |           4           |
|      total      |                       |                                                            |          51           |          44           |

**remarks**

1. The features extracted from the original signal are divided into three categories: time domain related features, frequency domain related features, and wavelet transform related features.
