# 天池AAIG-CUP-推荐攻击识别

### 赛题介绍

第三届 Apache Flink 极客挑战赛暨AAIG CUP已在阿里云天池平台拉开帷幕。大赛由阿里云联手英特尔、Apache Flink社区、阿里巴巴人工智能治理与可持续发展实验室(AAIG)、Occlum共同发起。继往届垃圾分类和实时疫情追踪等热点民生问题之后，本届大赛基于推荐系统的流量攻击实时检测问题进行思考和创新，为行业带来更多实时计算赋能实践的思路。具体详情如下：

实时攻击识别
- 如何准确、高效地识别电商推荐中恶意流量攻击，实时过滤恶意的点击数据是推荐系统中迫切需要解决的问题。
- 输入：图像恶意点击、正常点击及对应的“商品”、“用户”相关的属性信息；
- 输出：实现实时的恶意点击识别分类算法，包括模型训练和模型预测。

赛题链接：https://tianchi.aliyun.com/s/ea4fbcbaadab849b7389354501f38e2e

### baseline介绍

baseline简单定义了TF2下的全连接网络模型的，并通过docker打包可以成功提交。

- 步骤1：安装docker

Docker命令行安装（Ubuntu环境）：
```
sudo apt install docker.io
```

验证：
```
docker info
```
![](https://tianchi-public.oss-cn-hangzhou.aliyuncs.com/public/files/forum/160658242933332501606582428585.png)

- 步骤2：创建阿里云端镜像仓库
  - 步骤参考：https://tianchi.aliyun.com/competition/entrance/231759/tab/174
  - 创建云端镜像仓库：https://cr.console.aliyun.com/
  - 创建命名空间和镜像仓库；

```
git clone https://gitee.com/coggle/tianchi-3rd-AAIG-CUP/
cd tianchi-3rd-AAIG-CUP/

# 用于登录的用户名为阿里云账号全名，密码为开通服务时设置的密码。
sudo docker login --username=xxx@mail.com registry.cn-hangzhou.aliyuncs.com

# 使用本地Dockefile进行构建，使用创建仓库的【公网地址】
docker build -t registry.cn-hangzhou.aliyuncs.com/tianchi-lyz/tianchi-aaig-cpu:1.1 .

sudo docker push registry.cn-hangzhou.aliyuncs.com/tianchi-lyz/tianchi-aaig-cpu:1.1
```

- 步骤3：提交镜像

[比赛提交页面](https://tianchi.aliyun.com/competition/entrance/531925/submission/853)，填写镜像路径+版本号，以及用户名和密码则可以完成提交。

### 贡献者

感谢`ChauncyYao`提供了TF2的baseline，感谢`阿水`撰写提交文档。

![](https://coggle.club/assets/img/coggle_qrcode.jpg)

添加微信`coggle666`拉你进比赛微信群。
