import os
import sys
import time
import requests


def check_my_prediction(token, startDay, endDay, whether, longitude=None, latitude=None, magnitude=None):
    url = 'https://www.aeta.cn/aeta_platform/checkPrediction'

    data = {
        'whether': whether,
        'startDay': startDay,
        'endDay': endDay,
        'longitude': longitude,
        'latitude': latitude,
        'magnitude': magnitude
    }

    resp = requests.post(url, data=data, headers={'Authorization': token})
    if resp.ok:
        result = resp.json()['data']
        if 'msg' in result:
            print('Error Message 出错了：')
            print(result['msg'])
        else:
            print()
            print(result['info'])
            # print('\nDo you want to generate the CSV file? (You need to manually submit to tianchi platform)')
            # print('是否生成本次待提交的csv文件？ (需要您手动提交到天池平台)')
            while True:
                choose = str(input('(yes/no): ')).strip().lower() if len(sys.argv) == 1 else sys.argv[1]
                if choose == 'yes' or choose == 'no':
                    break
                else:
                    print("Please enter yes or no: ")
                    raise("请输入yes或no: ")
                    
            if choose == 'yes':
                filepath = os.getcwd() + os.path.sep + "outs" + os.path.sep + str(time.time())[0:10] + '_prediction.csv'
                with open(filepath, 'wb') as fd:
                    fd.write(result['prediction'].encode())
                print(f'The generated CSV file path is {filepath}\n')


if __name__ == '__main__':
    myToken = 'afaf18b81b1c4f0a93f02d7aad795ce5'

    # 请修改示例参数 Please modify the sample parameters
    # latitude 22~34/longitude 98~107
    """
    check_my_prediction(
        myToken, '2022-07-31', '2022-08-06', 1, latitude=29.095263, longitude=102.968099, magnitude=3.6)

    check_my_prediction(myToken, '2022-08-10', '2022-08-16', 0)

    check_my_prediction(
        myToken, '2022-08-17', '2022-08-19', 1, latitude=29.816143, longitude=102.714429, magnitude=3.5)

    check_my_prediction(
        myToken, '2022-08-20', '2022-08-26', 0)

    check_my_prediction(
        myToken, '2022-08-27', '2022-09-02', 0)

    check_my_prediction(
        myToken, '2022-09-03', '2022-09-09', 0)

    check_my_prediction(
        myToken, '2022-09-06', '2022-09-06', 1, latitude=27.817500, longitude=102.898500, magnitude=3.8)

    check_my_prediction(
        myToken, '2022-09-12', '2022-09-18', 0)

    check_my_prediction(
        myToken, '2022-09-20', '2022-09-22', 1, latitude=29.099329, longitude=104.226905, magnitude=3.6)

    check_my_prediction(
        myToken, '2022-09-23', '2022-09-29', 0)

    check_my_prediction(
        myToken, '2022-09-30', '2022-10-02', 1, latitude=24.780000, longitude=98.770000, magnitude=4.5)

    check_my_prediction(
        myToken, '2022-10-03', '2022-10-09', 0)
    
    check_my_prediction(
        myToken, '2022-10-10', '2022-10-16', 0)
        
    check_my_prediction(
        myToken, '2022-10-17', '2022-10-23', 0)
    
    check_my_prediction(
        myToken, '2022-10-24', '2022-10-30', 1, latitude=29.939062, longitude=102.640174, magnitude=3.8)
        
    check_my_prediction(
        myToken, '2022-11-01', '2022-11-07', 0)
    
    check_my_prediction(
        myToken, '2022-11-09', '2022-11-15', 0)
        
    check_my_prediction(
        myToken, '2022-11-16', '2022-11-22', 0)
    
    check_my_prediction(myToken, '2022-11-24', '2022-11-30', 0)
    check_my_prediction(myToken, '2022-12-01', '2022-12-07', 0)
    check_my_prediction(myToken, '2022-12-09', '2022-12-15', 0)
    check_my_prediction(myToken, '2022-12-19', '2022-12-25', 0)
    """
    check_my_prediction(myToken, '2022-12-28', '2022-12-31', 1, latitude=27.606855, longitude=103.138737, magnitude=3.5)
