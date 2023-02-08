import requests
url = "http://www.ceic.ac.cn/ajax/search?page=10&&start=2018-01-01&&end=2022-07-04&&jingdu1=0&&jingdu2=180&&weidu1=0&&weidu2=90&&height1=&&height2=&&zhenji1=3.5&&zhenji2="
res = requests.get(url)
res.content