# 更新日志

* <font  color="red">评测主页迁移</font>，当前页面为数据集页面，请参赛选手到算法比赛页面参与评测，<font  color="red">赛制规则和赛程安排以算法比赛页面为准，已在数据集页面提交过评测结果的参赛选手需要在算法比赛页面中重新提交</font>，算法比赛主页链接：   
   [CCL2023-Eval 汉语框架语义解析评测_算法大赛](https://tianchi.aliyun.com/competition/entrance/532083/introduction)

* <font  color="red">报名方式更新</font>：分为以下两个步骤，更多报名细节见算法比赛首页： [CCL2023-Eval 汉语框架语义解析评测](https://tianchi.aliyun.com/competition/entrance/532083/introduction)。

	1.  4月1日阿里天池平台([https://tianchi.aliyun.com/](https://tianchi.aliyun.com/))将开放本次比赛的报名组队、登录比赛官网（[CCL2023-Eval 汉语框架语义解析评测_算法大赛](https://tianchi.aliyun.com/competition/entrance/532083/introduction)），完成个人信息注册，即可报名参赛；选手可以单人参赛，也可以组队参赛。组队参赛的每个团队2-3人，每位选手只能加入一支队伍；选手需确保报名信息准确有效，组委会有权取消不符合条件队伍的参赛资格及奖励；选手报名、组队变更等操作截止时间为5月27日23：59：59；各队伍（包括队长及全体队伍成员）需要在5月27日23：59：59前完成实名认证（认证入口：天池官网-右上角个人中心-认证-支付宝实名认证），未完成认证的参赛团队将无法进行后续的比赛；
	    
	2.  向赛题举办方发送电子邮件进行报名，以获取数据解压密码。邮件标题为：“CCL2023-汉语框架语义解析评测-参赛单位”，例如：“CCL2023-汉语框架语义解析评测-复旦大学”；附件内容为队伍的参赛报名表，报名表[点此下载](https://github.com/SXUNLP/Chinese-Frame-Semantic-Parsing/blob/main/%E6%B1%89%E8%AF%AD%E6%A1%86%E6%9E%B6%E8%AF%AD%E4%B9%89%E8%A7%A3%E6%9E%90%E8%AF%84%E6%B5%8B%E6%8A%A5%E5%90%8D%E8%A1%A8.docx)，同时报名表应更名为“参赛队名+参赛队长信息+参赛单位名称”。请参加评测的队伍发送邮件至  [202122407024@email.sxu.edu.cn](mailto:202122407024@email.sxu.edu.cn)，报名成功后赛题数据解压密码会通过邮件发送给参赛选手，选手在天池平台下载数据即可

	**注意：报名截止前未发送报名邮件者不参与后续的评选。**
* 报名时间由 4月1日-4月31日 更新为 4月1日-5月27日。 
 
 
# 总体概述
  
* CFN 1.0数据集是由山西大学以汉语真实语料为依据构建的框架语义资源，数据由框架知识及标注例句组成，包含了近700个语义框架及20000条标注例句。CFN 1.0数据集遵循[CC BY-NC 4.0协议](https://creativecommons.org/licenses/by-nc/4.0/)。      
* 框架语义解析（Frame Semantic Parsing，FSP）是自然语言处理领域中的一项重要任务，其目标是从句子中提取框架语义结构<sup>[1]</sup>，实现对句子中涉及到的事件或情境的深层理解。FSP在阅读理解<sup>[2-3]</sup>、文本摘要<sup>[4-5]</sup>、关系抽取<sup>[6]</sup>等下游任务有着重要意义。


# 任务介绍

汉语框架语义解析（Chinese FSP，CFSP）是基于汉语框架网(Chinese FrameNet, CFN)的语义解析任务，本次标注数据格式如下：

1. 标注数据的字段信息如下：  
   + sentence_id：例句id
   + cfn_spans：框架元素标注信息
   + frame：例句所激活的框架名称
   + target：目标词的相关信息
      + start：目标词在句中的起始位置
      + end：目标词在句中的结束位置
      + pos：目标词的词性
   + text：标注例句
   + word：例句的分词结果及其词性信息
 
   数据样例：
   ```json
   [{
      "sentence_id": 2611,
      "cfn_spans": [
         { "start": 0, "end": 2, "fe_abbr": "ent_1", "fe_name": "实体1" },
         { "start": 4, "end": 17, "fe_abbr": "ent_2", "fe_name": "实体2" }
      ],
      "frame": "等同",
      "target": { "start": 3, "end": 3, "pos": "v" },
      "text": "餐饮业是天津市在海外投资的重点之一。",
      "word": [
         { "start": 0, "end": 2, "pos": "n" },
         { "start": 3, "end": 3, "pos": "v" },
         { "start": 4, "end": 6, "pos": "nz" },
         { "start": 7, "end": 7, "pos": "p" },
         { "start": 8, "end": 9, "pos": "n" },
         { "start": 10, "end": 11, "pos": "v" },
         { "start": 12, "end": 12, "pos": "u" },
         { "start": 13, "end": 14, "pos": "n" },
         { "start": 15, "end": 16, "pos": "n" },
         { "start": 17, "end": 17, "pos": "wp" }
      ]
   }]
   ```
   
2. 框架信息在`frame_info.json`中，框架数据的字段信息如下：
   + frame_name：框架名称
   + frame_ename：框架英文名称
   + frame_def：框架定义  
   + fes：框架元素信息
      + fe_name：框架元素名称  
      + fe_abbr：框架元素缩写  
      + fe_ename：框架元素英文名称  
      + fe_def：框架元素定义  
   
   数据样例：
   ```json
   [{
   "frame_name": "等同",
   "frame_ename": "Equating",
   "frame_def": "表示两个实体具有相等、相同、同等看待等的关系。",
   "fes": [
      { "fe_name": "实体集", "fe_abbr": "ents", "fe_ename": "Entities", "fe_def": "具有同等关系的两个或多个实体" },
      { "fe_name": "实体1", "fe_abbr": "ent_1", "fe_ename": "Entity_1", "fe_def": "与实体2具有等同关系的实体" },
      { "fe_name": "实体2", "fe_abbr": "ent_2", "fe_ename": "Entity_2", "fe_def": "与实体1具有等同关系的实体" },
      { "fe_name": "施动者", "fe_abbr": "agt", "fe_ename": "Agent", "fe_def": "判断实体集具有同等关系的人。" },
      { "fe_name": "方式", "fe_abbr": "manr", "fe_ename": "Manner", "fe_def": "修饰用来概括无法归入其他更具体的框架元素的任何语义成分，包括认知的修饰（如很可能，大概，神秘地），辅助描述（安静地，大声地），和与事件相比较的一般描述（同样的方式）。" },
      { "fe_name": "时间", "fe_abbr": "time", "fe_ename": "Time", "fe_def": "实体之间具有等同关系的时间" }
   ]
   }]
   ```

本次评测共分为以下三个子任务：

* 子任务1: 框架识别（Frame Identification），识别句子中给定目标词激活的框架。  
* 子任务2: 论元范围识别（Argument Identification），识别句子中给定目标词所支配论元的边界范围。  
* 子任务3: 论元角色识别（Role Identification），预测子任务2所识别论元的语义角色标签。  


## 子任务1: 框架识别（Frame Identification）
### 1. 任务描述

框架识别任务是框架语义学研究中的核心任务，其要求根据给定句子中目标词的上下文语境，为其寻找一个可以激活的框架。框架识别任务是自然语言处理中非常重要的任务之一，它可以帮助计算机更好地理解人类语言，并进一步实现语言处理的自动化和智能化。具体来说，框架识别任务可以帮助计算机识别出句子中的关键信息和语义框架，从而更好地理解句子的含义。这对于自然语言处理中的许多任务都是至关重要的。

### 2. 任务说明

该任务给定一个包含目标词的句子，需要根据目标词语境识别出激活的框架，并给出识别出的框架名称。  

   1. 输入：句子相关信息（id和文本内容）及目标词。
   2. 输出：句子id及目标词所激活框架的识别结果，数据为json格式，<font color="red">所有例句的识别结果需要放在同一list中</font>,样例如下：  

      ```json
      [
         [2611, "事件发生场所停业"],
         [2612, "等同"],
         ...
      ]
      ```

### 3. 评测指标

   &emsp;&emsp;框架识别采用正确率作为评价指标：

   $$task1\_acc = 正确识别的个数 / 总数$$


## 子任务2: 论元范围识别（Argument Identification）

### 1. 任务描述

给定一句汉语句子及目标词，在目标词已知的条件下，从句子中自动识别出目标词所搭配的语义角色的边界。该任务的主要目的是确定句子中目标词所涉及的每个论元在句子中的位置。论元范围识别任务对于框架语义解析任务来说非常重要，因为正确识别谓词和论元的范围可以帮助系统更准确地识别论元的语义角色，并进一步分析句子的语义结构。

### 2. 任务说明

论元范围识别任务是指，在给定包含目标词的句子中，识别出目标词所支配的语义角色的边界。
   
   1. 输入：句子相关信息（id和文本内容）及目标词。
   2. 输出：句子id，及所识别出所有论元角色的范围，每组结果包含例句id：`task_id`, `span`起始位置, `span`结束位置，<font color="red">每句包含的论元数量不定，识别出多个论元需要添加多个元组，所有例句识别出的结果共同放存在一个list中</font>，样例如下：
      ```json
      [
         [ 2611, 0, 2 ],
         [ 2611, 4, 17],
         ...
         [ 2612, 5, 7],
         ...
      ]
      ```

### 3. 评测指标

   论元范围识别采用P、R、F1作为评价指标：
   
   $${\rm{precision}} = \frac{{{\rm{InterSec(gold,pred)}}}}{{{\rm{Len(pred)}}}}$$
   
   $${\rm{recall}} = \frac{{{\rm{InterSec(gold,pred)}}}}{{{\rm{Len(gold)}}}}$$
   
   $${\rm{task2\\_f1}} = \frac{{{\rm{2\*precision\*recall}}}}{{{\rm{precision}} + {\rm{recall}}}}$$    
   
   其中：gold 和 pred 分别表示真实结果与预测结果，InterSec(\*)表示计算二者共有的token数量， Len(\*)表示计算token数量。

## 子任务3: 论元角色识别（Role Identification）

### 1. 任务描述

框架语义解析任务中，论元角色识别任务是非常重要的一部分。该任务旨在确定句子中每个论元对应的框架元素，即每个论元在所属框架中的语义角色。例如，在“我昨天买了一本书”这个句子中，“我”是“商业购买”框架中的“买方”框架元素，“一本书”是“商品”框架元素。论元角色识别任务对于许多自然语言处理任务都是至关重要的，例如信息提取、关系抽取和机器翻译等。它可以帮助计算机更好地理解句子的含义，从而更准确地提取句子中的信息，进而帮助人们更好地理解文本。

### 2. 任务说明

论元角色识别任务是指，在给定包含目标词的句子中，识别出目标词所支配语义角色的角色名称，该任务需要论元角色的边界信息以及目标词所激活框架的信息（即子任务1和子任务2的结果）。  
框架及其框架元素的所属关系在`frame_info.json`文件中。  


   1. 输入：句子相关信息（id和文本内容）、目标词、框架信息以及论元角色范围。
   2. 输出：句子id，及论元角色识别的结果，示例中“实体集”和“施动者”是“等同”框架中的框架元素。<font color="red">注意所有例句识别出的结果应共同放存在一个list中</font>，样例如下：  
   
   ```json
   [
      [ 2611, 0, 2, "实体集" ],
      [ 2611, 4, 17, "施动者" ],
      ...
      [ 2612, 5, 7, "时间" ],
      ...
   ]
   ```

### 3. 评测指标
论元角色识别采用P、R、F1作为评价指标：
   $${\rm{precision}} = \frac{{{\rm{Count(gold \cap pred)}}}} {{{\rm{Count(pred)}}}}$$  
   
   $${\rm{recall}} = \frac{{{\rm{Count(gold \cap pred)}}}} {{{\rm{Count(gold)}}}}$$  
   
   $${\rm{task3\\_f1}} = \frac{{{\rm{2\*precision\*recall}}}}{{{\rm{precision}} + {\rm{recall}}}}$$  

   其中，gold 和 pred 分别表示真实结果与预测结果， Count(\*) 表示计算集合元素的数量。



# 结果提交
本次评测结果在阿里天池平台上进行提交和排名。参赛队伍需要在阿里天池平台的“提交结果”界面提交预测结果，提交的压缩包命名为submit.zip，其中包含三个子任务的预测文件。

   + submit.zip
      + A_task1_test.json
      + A_task2_test.json
      + A_task3_test.json

<font color="red">1. 三个任务的提交结果需严格命名为A_task1_test.json、A_task2_test.json和A_task3_test.json。 2. 请严格使用`zip submit.zip A_task1_test.json A_task2_test.json A_task3_test.json` 进行压缩，即要求解压后的文件不能存在中间目录。</font>  选⼿可以只提交部分任务的结果，如只提交“框架识别”任务：`zip submit.zip A_task1_test.json`，未预测任务的分数默认为0。

# 系统排名

1. 所有评测任务均采用百分制分数显示，小数点后保留2位。
2. 系统排名取各项任务得分的加权和（三个子任务权重依次为 0.3，0.3，0.4），即：
	${\rm{task\_score=0.3*task1\_acc+0.3*task2\_f1+0.4*task3\_f1}} $
3. 如果某项任务未提交，默认分数为0，仍参与到系统最终得分的计算。

# Baseline
Baseline下载地址：[Github](https://github.com/SXUNLP/Chinese-Frame-Semantic-Parsing)    
Baseline表现：
|task1_acc|task2_f1|task3_f1|task_score|
|---------|--------|--------|----------|
|65.1|87.55|54.07|67.42|



# 评测数据
   
   数据由json格式给出，数据集包含以下内容：

   + CFN-train.json: 训练集标注数据，10000条。
   + CFN-dev.json: 验证集标注数据，2000条。
   + CFN-test-A.json: A榜测试集，4000条。
   + CFN-test-B.json: B榜测试集，4000条。B榜开赛前开放下载。
   + frame_info.json: 框架信息。
   + result.zip：提交示例。
      + A_task1_test.json：task1子任务提交示例。
      + A_task2_test.json：task2子任务提交示例。
      + A_task3_test.json：task3子任务提交示例。
   + README.md: 说明文件。

   

# 数据集信息

* 数据集提供方：山西大学智能计算与中文信息处理教育部重点实验室，山西太原 030000。  
* 负责人：谭红叶 tanhongye@sxu.edu.cn。  
* 联系人：闫智超 202022408073@email.sxu.edu.cn、李俊材 202122407024@email.sxu.edu.cn。  


# 赛程安排 

本次大赛分为报名组队、A榜、B榜三个阶段，具体安排和要求如下：  
1.  <font  color="red">报名时间：4月1日-5月27日</font>
2.  训练、验证数据及baseline发布：4月10日
3.  测试A榜数据发布：4月11日
4.  <font  color="red">测试A榜评测截止：5月29日 17:59:59</font>
5.  测试B榜数据发布：5月31日
6.  <font  color="red">测试B榜最终测试结果：6月2日 17:59:59</font>
7.  公布测试结果：6月10日前
8.  提交中文或英文技术报告：6月20日
9.  中文或英文技术报告反馈：6月28日
10.  正式提交中英文评测论文：7月3日
11.  公布获奖名单：7月７日
12.  评测报告及颁奖：8月3-5日
    

**注意：报名组队与实名认证（2023年4月1日—5月27日）**

# 赛事规则

1.  由于版权保护问题，CFN数据集只免费提供给用户用于非盈利性科学研究使用，参赛人员不得将数据用于任何商业用途。如果用于商业产品，请联系柴清华老师，联系邮箱  [charles@sxu.edu.cn](mailto:charles@sxu.edu.cn)。
2.  每名参赛选手只能参加一支队伍，一旦发现某选手以注册多个账号的方式参加多支队伍，将取消相关队伍的参赛资格。
3.  数据集的具体内容、范围、规模及格式以最终发布的真实数据集为准。验证集不可用于模型训练，针对测试集，参赛人员不允许执行任何人工标注。
4.  参赛队伍可在参赛期间随时上传测试集的预测结果，阿里天池平台A榜阶段每天可提交3次、B榜阶段每天可提交5次，系统会实时更新当前最新榜单排名情况，严禁参赛团队注册其它账号多次提交。
5.  允许使用公开的代码、工具、外部数据（从其他渠道获得的标注数据）等，但需要保证参赛结果可以复现。
6.  参赛队伍可以自行设计和调整模型，但需注意模型参数量最多不超过1.5倍BERT-Large（510M）。
7.  算法与系统的知识产权归参赛队伍所有。要求最终结果排名前10的队伍提供算法代码与系统报告（包括方法说明、数据处理、参考文献和使用的开源工具、外部数据等信息）。提交完毕将采用随机交叉检查的方法对各个队伍提交的模型进行检验，如果在排行榜上的结果无法复现，将取消获奖资格。
8.  参赛团队需保证提交作品的合规性，若出现下列或其他重大违规的情况，将取消参赛团队的参赛资格和成绩，获奖团队名单依次递补。重大违规情况如下：  
     a. 使用小号、串通、剽窃他人代码等涉嫌违规、作弊行为；  
     b. 团队提交的材料内容不完整，或提交任何虚假信息；  
     c. 参赛团队无法就作品疑义进行足够信服的解释说明；
9.  <font  color="red">获奖队伍必须注册会议并在线下参加（如遇特殊情况，可申请线上参加）</font>。
10. 评测单位：山西大学、北京大学、南京大学。
11. 评测负责人：谭红叶 tanhongye@sxu.edu.cn；联系人：闫智超 202022408073@email.sxu.edu.cn、李俊材 202122407024@email.sxu.edu.cn。


# 报名方式  
   &emsp;&emsp;本次评测采用电子邮件进行报名，邮件标题为：“CCL2023-汉语框架语义解析评测-参赛单位”，例如：“CCL2023-汉语框架语义解析评测-山西大学”；附件内容为队伍的参赛报名表，报名表[点此下载](https://github.com/SXUNLP/Chinese-Frame-Semantic-Parsing/blob/main/%E6%B1%89%E8%AF%AD%E6%A1%86%E6%9E%B6%E8%AF%AD%E4%B9%89%E8%A7%A3%E6%9E%90%E8%AF%84%E6%B5%8B%E6%8A%A5%E5%90%8D%E8%A1%A8.docx)，同时报名表应更名为“参赛队名+参赛队长信息+参赛 单位名称”。请参加评测的队伍发送邮件至202122407024@email.sxu.edu.cn，并同时在阿里天池平台完成报名，完成报名后需加入评测交流群：22240029459

   * 报名截止前未发送报名邮件者不参与后续的评选。
   * 大赛技术交流群： 请加钉钉群 22240029459 。
   
# 评测网址   
   评测首页：[[CCL2023-Eval 汉语框架语义解析评测_算法大赛](https://tianchi.aliyun.com/competition/entrance/532083/introduction)](https://tianchi.aliyun.com/competition/entrance/532083/introduction)    
   数据集网址：[https://tianchi.aliyun.com/dataset/149079]( https://tianchi.aliyun.com/dataset/149079)    
   GitHub：[https://github.com/SXUNLP/Chinese-Frame-Semantic-Parsing](https://github.com/SXUNLP/Chinese-Frame-Semantic-Parsing)   

# 奖项信息
   本次评测将评选出如下奖项。
   由中国中文信息学会计算语言学专委会（CIPS-CL）为获奖队伍提供荣誉证书。
   
   |奖项|一等奖|二等奖|三等奖|
   |----|----|----|----|
   |数量|1名|待定| 待定 |
   |奖励|荣誉证书|荣誉证书|荣誉证书|

  
# 数据集协议

该数据集遵循协议： [CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/?spm=5176.12282016.0.0.7a0a1517bGbbHL)。

由于版权保护问题，CFN数据集只免费提供给用户用于非盈利性科学研究使用，参赛人员不得将数据用于任何商业用途。如果用于商业产品，请联系柴清华老师，联系邮箱 charles@sxu.edu.cn。


# FAQ

* Q：比赛是否有技术交流群？
* A：请加钉钉群 22240029459 。
* Q：数据集解压密码是什么？
* A：请阅读“如何报名”，发送邮件报名成功后接收解压邮件。
* Q：验证集可否用于模型训练？
* A：不可以。


# 参考文献
[1] Daniel Gildea and Daniel Jurafsky. 2002. Automatic labeling of semantic roles. Computational linguistics,28(3):245–288.  
[2] Shaoru Guo, Ru Li*, Hongye Tan, Xiaoli Li, Yong Guan. A Frame-based Sentence Representation for Machine Reading Comprehension[C]. Proceedings of the 58th Annual Meeting of the Association for Computational Linguistic (ACL), 2020: 891-896.  
[3] Shaoru Guo, Yong Guan, Ru Li*, Xiaoli Li, Hongye Tan. Incorporating Syntax and Frame Semantics in Neural Network for Machine Reading Comprehension[C]. Proceedings of the 28th International Conference on Computational Linguistics (COLING), 2020: 2635-2641.  
[4] Yong Guan, Shaoru Guo, Ru Li*, Xiaoli Li, and Hu Zhang. Integrating Semantic Scenario and Word Relations for Abstractive Sentence Summarization[C]. Proceedings of the 2021 Conference on Empirical Methods in Natural Language Processing (EMNLP) 2021: 2522-2529.  
[5] Yong Guan, Shaoru Guo, Ru Li*, Xiaoli Li, and Hongye Tan, 2021. Frame Semantic-Enhanced Sentence Modeling for Sentence-level Extractive Text Summarization[C]. Proceedings of the 2021 Conference on Empirical Methods in Natural Language Processing (EMNLP) 2021: 404-4052.  
[6] Hongyan Zhao, Ru Li*, Xiaoli Li, Hongye Tan. CFSRE: Context-aware based on frame-semantics for distantly supervised relation extraction[J]. Knowledge-Based Systems, 2020, 210: 106480.  
 


