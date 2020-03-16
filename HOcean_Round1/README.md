> SAS, STATISTICAL ANALYSIS SYSTEM.

根据个人习惯，以下采用叙事格式为：
``` shell
code
# output
```
修订如下：
> C000 ；

#### 〇、SAS介绍
语句简易，**DATA皆PROC**；

玩味一下，丰富天池代码；

相关教程可咨：
> https://github.com/IvanaXu/DataSAS
> 

#### 一、方案明细
配置数据路径
``` SAS
	OPTIONS COMPRESS = YES;
	
/*	00.逻辑库 */
	LIBNAME ANA "C:\Users\IvanXu\Desktop\data\hy_round1";
	%LET CP = C:\Users\IvanXu\Desktop\code\Hy_round1;
	%LET DP = C:\Users\IvanXu\Desktop\data\hy_round1\hy_round1_train_20200102\hy_round1_train_20200102;
	%LET TP = C:\Users\IvanXu\Desktop\data\hy_round1\hy_round1_testA_20200102\hy_round1_testA_20200102;

	%LET PN = 000001;
```
csv数据读取，并打印日志
``` SAS
/*	01.数据读取 */
	PROC PRINTTO LOG = "&CP.\RUN.log" NEW;RUN;
	
		%MACRO R();
			PROC DELETE DATA = ANA.RESULT;RUN;
			DATA ANA.RESULT;
			FORMAT VAR1 VAR2 VAR3 VAR4 VAR5 VAR6 VAR7 $30.;
			STOP;
			RUN;

			%DO I = 0 %TO 6999;
				%PUT &I.;
				
				FILENAME INPT "&DP.\&I..csv" ENCODING = "UTF-8";
				PROC IMPORT 
					DATAFILE = INPT
					OUT = DEMO 
					DBMS = CSV 
					REPLACE;
					GETNAMES = NO;
				RUN;

				PROC APPEND DATA = DEMO BASE = ANA.RESULT;RUN;
				PROC DELETE DATA = DEMO;RUN;
			%END;
			
			DATA ANA.RESULT;
			SET ANA.RESULT;
			WHERE VAR7 ^= "type";
			RUN;
		%MEND;
		

		%MACRO T();
			PROC DELETE DATA = ANA.TEST;RUN;

			%DO I = 7000 %TO 9999;
				%PUT &I.;

				FILENAME INPT "&TP.\&I..csv" ENCODING = "UTF-8";
				PROC IMPORT
					DATAFILE = INPT
					OUT = DEMO
					DBMS = CSV
					REPLACE;
					GETNAMES = NO;
				RUN;

				DATA DEMO;
				SET DEMO(WHERE = (VAR6 ^= "time"));
				FORMAT
					ID $4.
					X_1 X_2 X_1V X_2V 24.12
					X_3 X_3_1 8.2 X_3_2 8.4
					X_4 X_4_1 8.2 X_4_2 8.4 X_3_4 8.2
					X_5_MON X_5_DAY X_5_HOR X_5_MIS X_5_SEC 8. X_5_DMY DATETIME20.
					X_5_DMYW 2.
				;
				ID = VAR1;
				X_1 = VAR2 + 0;
				X_2 = VAR3 + 0;
				X_1V = X_1/20037508.34*180;
				X_2V = 180/CONSTANT("PI")*(2*ATAN(EXP(X_2/20037508.34*180*CONSTANT("PI")/180))-CONSTANT("PI")/2);
				X_3 = VAR4 + 0;
				X_3_1 = X_3 > 0;
				X_3_2 = X_3/100;
				X_4 = VAR5 + 0;
				X_4_1 = X_4 > 0;
				X_4_2 = X_4/360;
				X_3_4 = X_3 > 0 OR X_4 > 0;
				X_5_MON = SUBSTR(COMPRESS(VAR6,"0123456789","K"),1,2)+0;
				X_5_DAY = SUBSTR(COMPRESS(VAR6,"0123456789","K"),3,2)+0;
				X_5_HOR = SUBSTR(COMPRESS(VAR6,"0123456789","K"),5,2)+0;
				X_5_MIS = SUBSTR(COMPRESS(VAR6,"0123456789","K"),7,2)+0;
				X_5_SEC = SUBSTR(COMPRESS(VAR6,"0123456789","K"),9,2)+0;
				X_5_DMY = DHMS(MDY(X_5_MON,X_5_DAY,2019),X_5_HOR,X_5_MIS,X_5_SEC);
				X_5_DMYW = WEEKDAY(DATEPART(X_5_DMY));
				KEEP ID X_:;
				RUN;

				PROC APPEND DATA = DEMO BASE = ANA.TEST;RUN;
				PROC DELETE DATA = DEMO;RUN;
			%END;
		%MEND;

        %R();
		%T();

	PROC PRINTTO LOG = LOG;RUN;
```
变量如下：
> X_1 X_2 X_1V X_2V 24.12
>
> X_3 X_3_1 8.2 X_3_2 8.4
>
> X_4 X_4_1 8.2 X_4_2 8.4 X_3_4 8.2
>
> X_5_MON X_5_DAY X_5_HOR X_5_MIS X_5_SEC 8. 
>
> X_5_DMY DATETIME20.
>
> X_5_DMYW 2.
>
其中，
``` JAVA
/*	//墨卡托转经纬度*/
/*        public Vector2D Mercator2lonLat(Vector2D mercator)*/
/*        {*/
/*            Vector2D lonLat = new Vector2D();*/
/*            double x = mercator.X / 20037508.34 * 180;*/
/*            double y = mercator.Y / 20037508.34 * 180;*/
/*            y = 180 / Math.PI * (2 * Math.Atan(Math.Exp(y * Math.PI / 180)) - Math.PI / 2);*/
/*            lonLat.X = x;*/
/*            lonLat.Y = y;*/
/*            return lonLat;*/
/*        }*/
```
转化为SAS代码
``` SAS
/*	DATA _T;*/
/*	SET ANA.TEST(KEEP = X_1 X_2);*/
/*	X_1V = X_1/20037508.34*180;*/
/*	X_2V = 180/CONSTANT("PI")*(2*ATAN(EXP(X_2/20037508.34*180*CONSTANT("PI")/180))-CONSTANT("PI")/2);*/
/*	RUN;*/
```

同样的，我们对测试集也处理相同变量
``` SAS
	DATA ANA.TEST;
	SET ANA.TEST;
	FORMAT 
		X_1V X_2V 24.12
		X_3_1 8.2 X_3_2 8.4
		X_4_1 8.2 X_4_2 8.4 X_3_4 8.2
		X_5_DMYW 2.
	;
	X_1V = X_1/20037508.34*180;
	X_2V = 180/CONSTANT("PI")*(2*ATAN(EXP(X_2/20037508.34*180*CONSTANT("PI")/180))-CONSTANT("PI")/2);
	X_3_1 = X_3 > 0;
	X_3_2 = X_3/100;
	X_4_1 = X_4 > 0;
	X_4_2 = X_4/360;
	X_3_4 = X_3 > 0 OR X_4 > 0;
	X_5_DMYW = WEEKDAY(DATEPART(X_5_DMY));
	RUN;
```
刺网、围网、拖网，拆分为三个数据集
``` SAS
/*	02.训练数据集 */
	PROC FREQ DATA = ANA.RESULT;
	TABLE VAR7/OUT = DEMO;
	RUN;

	DATA ANA.DATA1;
	SET ANA.RESULT;
	FORMAT 
		ID $4.
		TARGET 1. X_1 X_2 X_1V X_2V 24.12 
		X_3 X_3_1 8.2 X_3_2 8.4
		X_4 X_4_1 8.2 X_4_2 8.4 X_3_4 8.2 
		X_5_MON X_5_DAY X_5_HOR X_5_MIS X_5_SEC 8. X_5_DMY DATETIME20.
		X_5_DMYW 2.
	;
	IF VAR7 = "刺网" THEN TARGET = 1;
	IF VAR7 = "围网" THEN TARGET = 0;
	IF VAR7 = "拖网" THEN TARGET = 0;
	ID = VAR1;
	X_1 = VAR2 + 0;
	X_2 = VAR3 + 0;
	X_1V = X_1/20037508.34*180;
	X_2V = 180/CONSTANT("PI")*(2*ATAN(EXP(X_2/20037508.34*180*CONSTANT("PI")/180))-CONSTANT("PI")/2);
	X_3 = VAR4 + 0;
	X_3_1 = X_3 > 0;
	X_3_2 = X_3/100;
	X_4 = VAR5 + 0;
	X_4_1 = X_4 > 0;
	X_4_2 = X_4/360;
	X_3_4 = X_3 > 0 OR X_4 > 0;
	X_5_MON = SUBSTR(COMPRESS(VAR6,"0123456789","K"),1,2)+0;
	X_5_DAY = SUBSTR(COMPRESS(VAR6,"0123456789","K"),3,2)+0;
	X_5_HOR = SUBSTR(COMPRESS(VAR6,"0123456789","K"),5,2)+0;
	X_5_MIS = SUBSTR(COMPRESS(VAR6,"0123456789","K"),7,2)+0;
	X_5_SEC = SUBSTR(COMPRESS(VAR6,"0123456789","K"),9,2)+0;
	X_5_DMY = DHMS(MDY(X_5_MON,X_5_DAY,2019),X_5_HOR,X_5_MIS,X_5_SEC);
	X_5_DMYW = WEEKDAY(DATEPART(X_5_DMY));
	KEEP ID X_: TARGET;
	RUN;
	PROC PRINT DATA = ANA.DATA1(OBS=10);
	RUN;


	DATA ANA.DATA2;
	SET ANA.RESULT;
	FORMAT 
		ID $4.
		TARGET 1. X_1 X_2 X_1V X_2V 24.12 
		X_3 X_3_1 8.2 X_3_2 8.4
		X_4 X_4_1 8.2 X_4_2 8.4 X_3_4 8.2 
		X_5_MON X_5_DAY X_5_HOR X_5_MIS X_5_SEC 8. X_5_DMY DATETIME20.
		X_5_DMYW 2.
	;
	IF VAR7 = "刺网" THEN TARGET = 0;
	IF VAR7 = "围网" THEN TARGET = 1;
	IF VAR7 = "拖网" THEN TARGET = 0;
	ID = VAR1;
	X_1 = VAR2 + 0;
	X_2 = VAR3 + 0;
	X_1V = X_1/20037508.34*180;
	X_2V = 180/CONSTANT("PI")*(2*ATAN(EXP(X_2/20037508.34*180*CONSTANT("PI")/180))-CONSTANT("PI")/2);
	X_3 = VAR4 + 0;
	X_3_1 = X_3 > 0;
	X_3_2 = X_3/100;
	X_4 = VAR5 + 0;
	X_4_1 = X_4 > 0;
	X_4_2 = X_4/360;
	X_3_4 = X_3 > 0 OR X_4 > 0;
	X_5_MON = SUBSTR(COMPRESS(VAR6,"0123456789","K"),1,2)+0;
	X_5_DAY = SUBSTR(COMPRESS(VAR6,"0123456789","K"),3,2)+0;
	X_5_HOR = SUBSTR(COMPRESS(VAR6,"0123456789","K"),5,2)+0;
	X_5_MIS = SUBSTR(COMPRESS(VAR6,"0123456789","K"),7,2)+0;
	X_5_SEC = SUBSTR(COMPRESS(VAR6,"0123456789","K"),9,2)+0;
	X_5_DMY = DHMS(MDY(X_5_MON,X_5_DAY,2019),X_5_HOR,X_5_MIS,X_5_SEC);
	X_5_DMYW = WEEKDAY(DATEPART(X_5_DMY));
	KEEP ID X_: TARGET;
	RUN;


	DATA ANA.DATA3;
	SET ANA.RESULT;
	FORMAT 
		ID $4.
		TARGET 1. X_1 X_2 X_1V X_2V 24.12 
		X_3 X_3_1 8.2 X_3_2 8.4
		X_4 X_4_1 8.2 X_4_2 8.4 X_3_4 8.2 
		X_5_MON X_5_DAY X_5_HOR X_5_MIS X_5_SEC 8. X_5_DMY DATETIME20.
		X_5_DMYW 2.
	;
	IF VAR7 = "刺网" THEN TARGET = 0;
	IF VAR7 = "围网" THEN TARGET = 0;
	IF VAR7 = "拖网" THEN TARGET = 1;
	ID = VAR1;
	X_1 = VAR2 + 0;
	X_2 = VAR3 + 0;
	X_1V = X_1/20037508.34*180;
	X_2V = 180/CONSTANT("PI")*(2*ATAN(EXP(X_2/20037508.34*180*CONSTANT("PI")/180))-CONSTANT("PI")/2);
	X_3 = VAR4 + 0;
	X_3_1 = X_3 > 0;
	X_3_2 = X_3/100;
	X_4 = VAR5 + 0;
	X_4_1 = X_4 > 0;
	X_4_2 = X_4/360;
	X_3_4 = X_3 > 0 OR X_4 > 0;
	X_5_MON = SUBSTR(COMPRESS(VAR6,"0123456789","K"),1,2)+0;
	X_5_DAY = SUBSTR(COMPRESS(VAR6,"0123456789","K"),3,2)+0;
	X_5_HOR = SUBSTR(COMPRESS(VAR6,"0123456789","K"),5,2)+0;
	X_5_MIS = SUBSTR(COMPRESS(VAR6,"0123456789","K"),7,2)+0;
	X_5_SEC = SUBSTR(COMPRESS(VAR6,"0123456789","K"),9,2)+0;
	X_5_DMY = DHMS(MDY(X_5_MON,X_5_DAY,2019),X_5_HOR,X_5_MIS,X_5_SEC);
	X_5_DMYW = WEEKDAY(DATEPART(X_5_DMY));
	KEEP ID X_: TARGET;
	RUN;
```
二分类分别建模，导入模型宏

详见附录
``` SAS
/*	03.导入模型宏 */
	%INCLUDE "&CP.\02.MacroS1.sas";
	%INCLUDE "&CP.\02.MacroS2.sas";
	%INCLUDE "&CP.\02.MacroS3.sas";
```
计算模型概率，转计F1分数
``` SAS
/*	04.F1分数计算 */
	DATA DATA1;
	SET ANA.DATA1;
	%S1();
	KEEP P_TARGET1 TARGET;
	RUN;

	DATA DATA2;
	SET ANA.DATA2;
	%S2();
	KEEP P_TARGET1 TARGET;
	RUN;

	DATA DATA3;
	SET ANA.DATA3;
	%S3();
	KEEP P_TARGET1 TARGET;
	RUN;


	%MACRO F1(DT=,);
		PROC DELETE DATA = &DT._F1;
		RUN;

		%DO I = 1 %TO 100;
			%LET T = %SYSEVALF(&I./100);
			%PUT &T.;

			PROC SQL;
			CREATE TABLE _T1 AS 
			SELECT
				&T. AS T,
				SUM(CASE WHEN P_TARGET1 > &T. AND TARGET = 1 THEN 1 ELSE 0 END)/
					SUM(CASE WHEN P_TARGET1 > &T. THEN 1 ELSE 0 END) AS P,
				SUM(CASE WHEN P_TARGET1 > &T. AND TARGET = 1 THEN 1 ELSE 0 END)/
					SUM(CASE WHEN TARGET = 1 THEN 1 ELSE 0 END) AS R,
				2 * (CALCULATED P) * (CALCULATED R)/((CALCULATED P) + (CALCULATED R)) AS F1
			FROM &DT.;
			QUIT;

			PROC APPEND DATA = _T1 BASE = &DT._F1;RUN;
			PROC DELETE DATA = _T1;RUN;
		%END;
		
		PROC SQL;
		CREATE TABLE &DT._F1 AS 
		SELECT
			*,
			MAX(F1) AS MAXF1,
			CASE WHEN F1 = CALCULATED MAXF1 THEN 1 ELSE 0 END AS IS_MAXF1
		FROM &DT._F1;
		QUIT;

		PROC UNIVARIATE DATA = &DT._F1;
		VAR F1;
		RUN;

	%MEND;
	%F1(DT=DATA1);
	%F1(DT=DATA2);
	%F1(DT=DATA3);
```
记录最大F1分数至宏变量
``` SAS
	PROC SQL NOPRINT;
		SELECT 
			MAX(CASE WHEN IS_MAXF1 = 1 THEN T ELSE . END) INTO: TDATA1 
		FROM DATA1_F1;
		SELECT 
			MAX(CASE WHEN IS_MAXF1 = 1 THEN T ELSE . END) INTO: TDATA2 
		FROM DATA2_F1;
		SELECT 
			MAX(CASE WHEN IS_MAXF1 = 1 THEN T ELSE . END) INTO: TDATA3
		FROM DATA3_F1;
		
		CREATE TABLE ANA.R&PN. AS 
		SELECT * 
		FROM (
			SELECT * FROM DATA1_F1 UNION ALL 
			SELECT * FROM DATA2_F1 UNION ALL
			SELECT * FROM DATA3_F1
		)
		WHERE IS_MAXF1 = 1;
	QUIT;
	%PUT &TDATA1. &TDATA2. &TDATA3.;

	PROC PRINT DATA = ANA.R&PN.;
	RUN;
```
根据最大F1的cut off概率，预测为0/1
``` SAS
/*	05.模型结果 */
	DATA TEST1;
	SET ANA.TEST;
	%S1();
	FORMAT MTYPE 1. FID $32.;
	MTYPE = 1;
	FID = PUT(MD5(CAT(ID, CATX("|", OF X_:))), HEX32.);
	RUN;
	PROC SORT DATA = TEST1 NODUPKEY;
	BY FID;
	RUN;

	DATA TEST2;
	SET ANA.TEST;
	%S2();
	FORMAT MTYPE 1. FID $32.;
	MTYPE = 1;
	FID = PUT(MD5(CAT(ID, CATX("|", OF X_:))), HEX32.);
	RUN;
	PROC SORT DATA = TEST2 NODUPKEY;
	BY FID;
	RUN;

	DATA TEST3;
	SET ANA.TEST;
	%S3();
	FORMAT MTYPE 1. FID $32.;
	MTYPE = 1;
	FID = PUT(MD5(CAT(ID, CATX("|", OF X_:))), HEX32.);
	RUN;
	PROC SORT DATA = TEST3 NODUPKEY;
	BY FID;
	RUN;

	DATA TEST123;
	MERGE 
		TEST1(IN=T1 RENAME = (P_TARGET1 = P1) KEEP = FID ID P_TARGET1)
		TEST2(IN=T2 RENAME = (P_TARGET1 = P2) KEEP = FID P_TARGET1)
		TEST3(IN=T3 RENAME = (P_TARGET1 = P3) KEEP = FID P_TARGET1)
	;
	BY FID;
	IF T1 OR T2 OR T3;
	TP1 = P1 > &TDATA1.;
	TP2 = P2 > &TDATA2.;
	TP3 = P3 > &TDATA3.;
	RUN;
```
根据模型结果，整合3个模型
``` SAS
	PROC SQL;
	CREATE TABLE RESULT AS 
	SELECT
		ID,
		SUM(1) AS CNT,
		MEAN(TP1) AS TP1,
		MEAN(TP2) AS TP2,
		MEAN(TP3) AS TP3
	FROM TEST123
	GROUP BY ID;
	QUIT;

	DATA RESULT;
	SET RESULT;
	FORMAT T $6.;
	/* 1 刺网 2 围网 3 拖网 */
	IF TP1 = MAX(TP1, TP2, TP3)
		THEN T = "刺网";
	ELSE IF TP2 = MAX(TP1, TP2, TP3)
		THEN T = "围网";
	ELSE IF TP3 = MAX(TP1, TP2, TP3)
		THEN T = "拖网";
	ELSE T = "拖网";
	RUN;
	
	FILENAME EXPT ".\data\hy_round1\result&PN..csv" ENCODING="UTF-8";
	PROC EXPORT 
	    DATA = RESULT(KEEP = ID T)
	    OUTFILE = EXPT DBMS = CSV REPLACE;
	    PUTNAMES = NO;
	RUN;
```
相关链接，https://github.com/IvanaXu/TianChiProj/tree/master/HOcean_Round1

#### 二、附录
%INCLUDE "&CP.\02.MacroS1.sas";
``` SAS
%MACRO S1();

LENGTH _UFormat $ 32;
_UFormat = ' ';
LENGTH _WARN_ $4;
LABEL _WARN_ = 'Warnings';
_WARN_ = '    ';
_nInputMiss = 0;
_nInputOutRange = 0;
 
/******************************************************************************/
/*                    Calculate Standardized Inputs                           */
/******************************************************************************/
 
_I1 = X_1;
IF MISSING(X_1) THEN DO;
    _I1 = 6276993.21620483; /* impute average value */
    _nInputMiss = 1;
END;
_I1 = 2 * ( _I1 - 5125337.87599689 ) / 2008336.87763215 - 1;
 
_I2 = X_2;
IF MISSING(X_2) THEN DO;
    _I2 = 5271062.52994067; /* impute average value */
    _nInputMiss = 1;
END;
_I2 = 2 * ( _I2 - 4129342.08552849 ) / 2524064.6945741 - 1;
 
_I3 = X_3;
IF MISSING(X_3) THEN DO;
    _I3 = 1.78228339801757; /* impute average value */
    _nInputMiss = 1;
END;
_I3 = 2 * ( _I3 ) / 80.73 - 1;
 
_I4 = X_4;
IF MISSING(X_4) THEN DO;
    _I4 = 115.136029295035; /* impute average value */
    _nInputMiss = 1;
END;
_I4 = 2 * ( _I4 ) / 360 - 1;
 
_I5 = X_5_DAY;
IF MISSING(X_5_DAY) THEN DO;
    _I5 = 14.5991103840437; /* impute average value */
    _nInputMiss = 1;
END;
_I5 = 2 * ( _I5 - 1 ) / 30 - 1;
 
_I6 = X_5_HOR;
IF MISSING(X_5_HOR) THEN DO;
    _I6 = 11.6113902290413; /* impute average value */
    _nInputMiss = 1;
END;
_I6 = 2 * ( _I6 ) / 23 - 1;
 
_I7 = X_5_MIS;
IF MISSING(X_5_MIS) THEN DO;
    _I7 = 29.5610779411831; /* impute average value */
    _nInputMiss = 1;
END;
_I7 = 2 * ( _I7 ) / 59 - 1;
 
_I8 = X_5_MON;
IF MISSING(X_5_MON) THEN DO;
    _I8 = 10.851861707355; /* impute average value */
    _nInputMiss = 1;
END;
_I8 = 2 * ( _I8 - 10 ) / 1 - 1;
 
_I9 = X_5_SEC;
IF MISSING(X_5_SEC) THEN DO;
    _I9 = 29.8596586983243; /* impute average value */
    _nInputMiss = 1;
END;
_I9 = 2 * ( _I9 ) / 59 - 1;
 
/******************************************************************************/
/*                              Set _WARN_ Value                              */
/******************************************************************************/
IF ( _nInputMiss GT 0) THEN DO;
    SUBSTR( _WARN_, 1, 1) = 'M';
END;
IF ( _nInputOutRange GT 0) THEN DO;
    SUBSTR( _WARN_, 2, 1) = 'U';
END;
 
/******************************************************************************/
/*                   Calculate Output of Hidden Layer 1                       */
/******************************************************************************/
 
_H1 = 10.8984949330612
    - 11.9523979644358 * _I1
    + 40.0845104372483 * _I2
    + 2.45790417320227 * _I3
    - 0.02312046298801 * _I4
    + 0.01487211618644 * _I5
    + 0.00307091613153 * _I6
    - 0.00344221873451 * _I7
    + 0.01354268368639 * _I8
    - 0.00249949086698 * _I9;
IF ( _H1 GE 0 ) THEN DO;
    _H1 = EXP( -2 * _H1 );
    _H1 = ( 1 - _H1 ) / ( 1 + _H1 );
END;
ELSE DO;
    _H1 = EXP( 2 * _H1 );
    _H1 = ( _H1 - 1 ) / ( _H1 + 1 );
END;
 
_H2 = 0.98003573867117
    + 43.3659955546155 * _I1
    + 42.1259477407316 * _I2
    - 1.52401218767423 * _I3
    - 0.01349911694966 * _I4
    + 0.03465100870368 * _I5
    + 0.04308140706846 * _I6
    + 0.02158147023848 * _I7
    + 0.06202423595799 * _I8
    + 0.0066904711393 * _I9;
IF ( _H2 GE 0 ) THEN DO;
    _H2 = EXP( -2 * _H2 );
    _H2 = ( 1 - _H2 ) / ( 1 + _H2 );
END;
ELSE DO;
    _H2 = EXP( 2 * _H2 );
    _H2 = ( _H2 - 1 ) / ( _H2 + 1 );
END;
 
/******************************************************************************/
/*                   Calculate Output of Hidden Layer 2                       */
/******************************************************************************/
 
_H3 = 18.8871319184006
    - 36.2173916271691 * _H1
    + 16.5104439557658 * _H2;
IF ( _H3 GE 0 ) THEN DO;
    _H3 = EXP( -2 * _H3 );
    _H3 = ( 1 - _H3 ) / ( 1 + _H3 );
END;
ELSE DO;
    _H3 = EXP( 2 * _H3 );
    _H3 = ( _H3 - 1 ) / ( _H3 + 1 );
END;
 
/******************************************************************************/
/*                   Calculate Output of Target Layer                         */
/******************************************************************************/
 
_T1_0 = -9.40941416028989
    - 2.47778457254763 * _I1
    + 1.80888827363531 * _I2
    - 7.57267344209617 * _I3
    + 0.08999732016242 * _I4
    + 0.10892869253909 * _I5
    - 0.04270986592275 * _I6
    - 0.01436052898125 * _I7
    - 0.00528981758441 * _I8
    + 0.00775498202586 * _I9
    - 1.93259143672535 * _H3;
LABEL P_TARGET1 = "预测: TARGET=1";
IF ABS(_T1_0) < 36 THEN
    P_TARGET1 = 1.0 / (1.0 + EXP(-_T1_0));
ELSE
    IF _T1_0 < 0 THEN
        P_TARGET1 = 0;
    ELSE
        P_TARGET1 = 1;
LABEL P_TARGET0 = "预测: TARGET=0";
P_TARGET0 = 1.0 - P_TARGET1;
 
/******************************************************************************/
/*                        Drop temporary variables                            */
/******************************************************************************/
DROP _UFormat _nInputMiss _nInputOutRange;
DROP _I1 _I2 _I3 _I4 _I5 _I6 _I7 _I8 _I9;
DROP _H1 _H2 _H3;
DROP _T1_0;
*------------------------------------------------------------*;
*Computing Classification Vars: TARGET;
*------------------------------------------------------------*;
length I_TARGET $32;
label  I_TARGET = '到: TARGET';
length _format200 $200;
drop _format200;
_format200= ' ' ;
_p_= 0 ;
drop _p_ ;
if P_TARGET1 - _p_ > 1e-8 then do ;
   _p_= P_TARGET1 ;
   _format200='1';
end;
if P_TARGET0 - _p_ > 1e-8 then do ;
   _p_= P_TARGET0 ;
   _format200='0';
end;
I_TARGET=dmnorm(_format200,32); ;
label U_TARGET = 'Unnormalized Into: TARGET';
format U_TARGET 1.;
if I_TARGET='1' then
U_TARGET=1;
if I_TARGET='0' then
U_TARGET=0;


%MEND S1;

```

%INCLUDE "&CP.\02.MacroS2.sas";
``` SAS
%MACRO S2();

***********************************;
*** Begin Scoring Code for Neural;
***********************************;
DROP _DM_BAD _EPS _NOCL_ _MAX_ _MAXP_ _SUM_ _NTRIALS;
 _DM_BAD = 0;
 _NOCL_ = .;
 _MAX_ = .;
 _MAXP_ = .;
 _SUM_ = .;
 _NTRIALS = .;
 _EPS =                1E-10;
LENGTH _WARN_ $4
;
      label S_X_1 = '标准: X_1' ;

      label S_X_2 = '标准: X_2' ;

      label S_X_3 = '标准: X_3' ;

      label S_X_4 = '标准: X_4' ;

      label S_X_5_DAY = '标准: X_5_DAY' ;

      label S_X_5_HOR = '标准: X_5_HOR' ;

      label S_X_5_MIS = '标准: X_5_MIS' ;

      label S_X_5_MON = '标准: X_5_MON' ;

      label S_X_5_SEC = '标准: X_5_SEC' ;

      label H1x1_1 = '隐藏: H1x1_=1' ;

      label H1x1_2 = '隐藏: H1x1_=2' ;

      label H1x2_1 = '隐藏: H1x2_=1' ;

      label H1x2_2 = '隐藏: H1x2_=2' ;

      label H1x3_1 = '隐藏: H1x3_=1' ;

      label H1x3_2 = '隐藏: H1x3_=2' ;

      label I_TARGET = '到: TARGET' ;

      label U_TARGET = '非正规化至: TARGET' ;

      label P_TARGET1 = '预测: TARGET=1' ;

      label P_TARGET0 = '预测: TARGET=0' ;

      label  _WARN_ = "Warnings";

*** *************************;
*** Checking missing input Interval
*** *************************;

IF NMISS(
   X_1 ,
   X_2 ,
   X_3 ,
   X_4 ,
   X_5_DAY ,
   X_5_HOR ,
   X_5_MIS ,
   X_5_MON ,
   X_5_SEC   ) THEN DO;
   SUBSTR(_WARN_, 1, 1) = 'M';

   _DM_BAD = 1;
END;
*** *************************;
*** Writing the Node interval ;
*** *************************;
IF _DM_BAD EQ 0 THEN DO;
   S_X_1  =    -23.2425819539563 +   3.7030159146828E-6 * X_1 ;
   S_X_2  =    -20.6878307411216 +    3.924845619383E-6 * X_2 ;
   S_X_3  =    -0.72152390230399 +     0.40525568485421 * X_3 ;
   S_X_4  =    -0.98375180988624 +     0.00855321727148 * X_4 ;
   S_X_5_DAY  =    -1.68252096661115 +     0.11526772904371 * X_5_DAY ;
   S_X_5_HOR  =    -1.66629155404202 +     0.14351108507234 * X_5_HOR ;
   S_X_5_MIS  =    -1.71008910796224 +     0.05788829287721 * X_5_MIS ;
   S_X_5_MON  =    -30.5539110384655 +     2.81552869150283 * X_5_MON ;
   S_X_5_SEC  =    -1.73660976678013 +     0.05821454802157 * X_5_SEC ;
END;
ELSE DO;
   IF MISSING( X_1 ) THEN S_X_1  = . ;
   ELSE S_X_1  =    -23.2425819539563 +   3.7030159146828E-6 * X_1 ;
   IF MISSING( X_2 ) THEN S_X_2  = . ;
   ELSE S_X_2  =    -20.6878307411216 +    3.924845619383E-6 * X_2 ;
   IF MISSING( X_3 ) THEN S_X_3  = . ;
   ELSE S_X_3  =    -0.72152390230399 +     0.40525568485421 * X_3 ;
   IF MISSING( X_4 ) THEN S_X_4  = . ;
   ELSE S_X_4  =    -0.98375180988624 +     0.00855321727148 * X_4 ;
   IF MISSING( X_5_DAY ) THEN S_X_5_DAY  = . ;
   ELSE S_X_5_DAY  =    -1.68252096661115 +     0.11526772904371 * X_5_DAY ;
   IF MISSING( X_5_HOR ) THEN S_X_5_HOR  = . ;
   ELSE S_X_5_HOR  =    -1.66629155404202 +     0.14351108507234 * X_5_HOR ;
   IF MISSING( X_5_MIS ) THEN S_X_5_MIS  = . ;
   ELSE S_X_5_MIS  =    -1.71008910796224 +     0.05788829287721 * X_5_MIS ;
   IF MISSING( X_5_MON ) THEN S_X_5_MON  = . ;
   ELSE S_X_5_MON  =    -30.5539110384655 +     2.81552869150283 * X_5_MON ;
   IF MISSING( X_5_SEC ) THEN S_X_5_SEC  = . ;
   ELSE S_X_5_SEC  =    -1.73660976678013 +     0.05821454802157 * X_5_SEC ;
END;
*** *************************;
*** Writing the Node H1x1_ ;
*** *************************;
IF _DM_BAD EQ 0 THEN DO;
   H1x1_1  =    -4.67487561479997 * S_X_1  +     7.63479704173979 * S_X_2
          +    -0.22601914116034 * S_X_3  +     0.16914723836277 * S_X_4
          +    -0.19759629782742 * S_X_5_DAY  +     0.06605439028426 *
        S_X_5_HOR  +     0.01531434497012 * S_X_5_MIS
          +    -0.14757627029962 * S_X_5_MON  +     0.01049322888926 *
        S_X_5_SEC ;
   H1x1_2  =    -1.78314266921554 * S_X_1  +     0.72930171492106 * S_X_2
          +     2.43215843413582 * S_X_3  +     0.20037094466577 * S_X_4
          +    -0.19684531309349 * S_X_5_DAY  +     0.00622601553355 *
        S_X_5_HOR  +    -0.03468800677165 * S_X_5_MIS
          +    -0.21844667995369 * S_X_5_MON  +     0.01282879812057 *
        S_X_5_SEC ;
   H1x1_1  =     3.16976459811599 + H1x1_1 ;
   H1x1_2  =     4.62545258528217 + H1x1_2 ;
   DROP _EXP_BAR;
   _EXP_BAR=50;
   H1x1_1  = EXP(MIN(-0.5 * H1x1_1 **2, _EXP_BAR));
   H1x1_2  = EXP(MIN(-0.5 * H1x1_2 **2, _EXP_BAR));
END;
ELSE DO;
   H1x1_1  = .;
   H1x1_2  = .;
END;
*** *************************;
*** Writing the Node H1x2_ ;
*** *************************;
IF _DM_BAD EQ 0 THEN DO;
   H1x2_1  =    -0.91949184984534 * S_X_1  +      2.9158791209794 * S_X_2
          +     0.11892476703962 * S_X_3  +    -0.31907612004876 * S_X_4
          +     -0.0902328934483 * S_X_5_DAY  +    -0.04095544431737 *
        S_X_5_HOR  +    -0.01026668764717 * S_X_5_MIS
          +    -0.06495689670943 * S_X_5_MON  +    -0.00157785974371 *
        S_X_5_SEC ;
   H1x2_2  =    -10.5864590347477 * S_X_1  +     5.03334847324649 * S_X_2
          +     0.10427358625857 * S_X_3  +     0.06284331051971 * S_X_4
          +     0.15464314918942 * S_X_5_DAY  +    -0.00795336935373 *
        S_X_5_HOR  +     -0.0111685051473 * S_X_5_MIS
          +     0.02328838137323 * S_X_5_MON  +     0.00108990652583 *
        S_X_5_SEC ;
   H1x2_1  =    -1.12475482062052 + H1x2_1 ;
   H1x2_2  =    -1.90100582119541 + H1x2_2 ;
   DROP _EXP_BAR;
   _EXP_BAR=50;
   H1x2_1  = EXP(MIN(-0.5 * H1x2_1 **2, _EXP_BAR));
   H1x2_2  = EXP(MIN(-0.5 * H1x2_2 **2, _EXP_BAR));
END;
ELSE DO;
   H1x2_1  = .;
   H1x2_2  = .;
END;
*** *************************;
*** Writing the Node H1x3_ ;
*** *************************;
IF _DM_BAD EQ 0 THEN DO;
   H1x3_1  =     29.2173029986981 * S_X_1  +    -6.45998623823227 * S_X_2
          +     -12.339739290723 * S_X_3  +     0.34286336786173 * S_X_4
          +     3.14765656041372 * S_X_5_DAY  +    -0.00349978257838 *
        S_X_5_HOR  +    -0.04041049463998 * S_X_5_MIS
          +     0.34767666475627 * S_X_5_MON  +    -0.04597040713612 *
        S_X_5_SEC ;
   H1x3_2  =    -0.17617985290225 * S_X_1  +    -0.34713014333351 * S_X_2
          +    -0.49910205760803 * S_X_3  +     0.08793451460352 * S_X_4
          +     0.01308089371452 * S_X_5_DAY  +     0.03099180153591 *
        S_X_5_HOR  +    -0.00138947269526 * S_X_5_MIS
          +    -0.01352242710089 * S_X_5_MON  +    -0.01297419735338 *
        S_X_5_SEC ;
   H1x3_1  =    -3.78085884376367 + H1x3_1 ;
   H1x3_2  =     1.53448397962223 + H1x3_2 ;
   H1x3_1  = TANH(H1x3_1 );
   H1x3_2  = TANH(H1x3_2 );
END;
ELSE DO;
   H1x3_1  = .;
   H1x3_2  = .;
END;
*** *************************;
*** Writing the Node TARGET ;
*** *************************;
IF _DM_BAD EQ 0 THEN DO;
   P_TARGET1  =    -2.57318466830944 * H1x1_1  +     4.05657572470422 * H1x1_2
         ;
   P_TARGET1  = P_TARGET1  +     1.35556154470612 * H1x2_1
          +    -3.28511410316945 * H1x2_2 ;
   P_TARGET1  = P_TARGET1  +     0.50258311282615 * H1x3_1
          +    -1.54072992668773 * H1x3_2 ;
   P_TARGET1  =    -0.01908906905699 + P_TARGET1 ;
   P_TARGET0  = 0;
   _MAX_ = MAX (P_TARGET1 , P_TARGET0 );
   _SUM_ = 0.;
   P_TARGET1  = EXP(P_TARGET1  - _MAX_);
   _SUM_ = _SUM_ + P_TARGET1 ;
   P_TARGET0  = EXP(P_TARGET0  - _MAX_);
   _SUM_ = _SUM_ + P_TARGET0 ;
   P_TARGET1  = P_TARGET1  / _SUM_;
   P_TARGET0  = P_TARGET0  / _SUM_;
END;
ELSE DO;
   P_TARGET1  = .;
   P_TARGET0  = .;
END;
IF _DM_BAD EQ 1 THEN DO;
   P_TARGET1  =     0.23210600400055;
   P_TARGET0  =     0.76789399599944;
END;
*** *************************;
*** Writing the I_TARGET  AND U_TARGET ;
*** *************************;
_MAXP_ = P_TARGET1 ;
I_TARGET  = "1" ;
U_TARGET  =                    1;
IF( _MAXP_ LT P_TARGET0  ) THEN DO;
   _MAXP_ = P_TARGET0 ;
   I_TARGET  = "0" ;
   U_TARGET  =                    0;
END;
********************************;
*** End Scoring Code for Neural;
********************************;
drop S_:;


%MEND S2;
```

%INCLUDE "&CP.\02.MacroS3.sas";
``` SAS
%MACRO S3();

*************************************;
*** begin scoring code for regression;
*************************************;

length _WARN_ $4;
label _WARN_ = '警告' ;

length I_TARGET $ 1;
label I_TARGET = '到: TARGET' ;
*** Target Values;
array REG3DRF [2] $1 _temporary_ ('1' '0' );
label U_TARGET = '非正规化至: TARGET' ;
format U_TARGET 1.;
*** Unnormalized target values;
ARRAY REG3DRU[2]  _TEMPORARY_ (1 0);

drop _DM_BAD;
_DM_BAD=0;

*** Check X_1 for missing values ;
if missing( X_1 ) then do;
   substr(_warn_,1,1) = 'M';
   _DM_BAD = 1;
end;

*** Check X_2 for missing values ;
if missing( X_2 ) then do;
   substr(_warn_,1,1) = 'M';
   _DM_BAD = 1;
end;

*** Check X_3 for missing values ;
if missing( X_3 ) then do;
   substr(_warn_,1,1) = 'M';
   _DM_BAD = 1;
end;

*** Check X_4 for missing values ;
if missing( X_4 ) then do;
   substr(_warn_,1,1) = 'M';
   _DM_BAD = 1;
end;

*** Check X_5_DAY for missing values ;
if missing( X_5_DAY ) then do;
   substr(_warn_,1,1) = 'M';
   _DM_BAD = 1;
end;

*** Check X_5_HOR for missing values ;
if missing( X_5_HOR ) then do;
   substr(_warn_,1,1) = 'M';
   _DM_BAD = 1;
end;

*** Check X_5_MIS for missing values ;
if missing( X_5_MIS ) then do;
   substr(_warn_,1,1) = 'M';
   _DM_BAD = 1;
end;

*** Check X_5_MON for missing values ;
if missing( X_5_MON ) then do;
   substr(_warn_,1,1) = 'M';
   _DM_BAD = 1;
end;

*** Check X_5_SEC for missing values ;
if missing( X_5_SEC ) then do;
   substr(_warn_,1,1) = 'M';
   _DM_BAD = 1;
end;
*** If missing inputs, use averages;
if _DM_BAD > 0 then do;
   _P0 = 0.622761272;
   _P1 = 0.377238728;
   goto REG3DR1;
end;

*** Compute Linear Predictor;
drop _TEMP;
drop _LP0;
_LP0 = 0;

***  Effect: X_1 ;
_TEMP = X_1 ;
_LP0 = _LP0 + (  -2.060065236162E-7 * _TEMP);

***  Effect: X_2 ;
_TEMP = X_2 ;
_LP0 = _LP0 + ( -5.7221995827393E-6 * _TEMP);

***  Effect: X_3 ;
_TEMP = X_3 ;
_LP0 = _LP0 + (    0.15364577339732 * _TEMP);

***  Effect: X_4 ;
_TEMP = X_4 ;
_LP0 = _LP0 + (    0.00004828317148 * _TEMP);

***  Effect: X_5_DAY ;
_TEMP = X_5_DAY ;
_LP0 = _LP0 + (   -0.00531672301753 * _TEMP);

***  Effect: X_5_HOR ;
_TEMP = X_5_HOR ;
_LP0 = _LP0 + (    0.00330258855139 * _TEMP);

***  Effect: X_5_MIS ;
_TEMP = X_5_MIS ;
_LP0 = _LP0 + (   -0.00040430730158 * _TEMP);

***  Effect: X_5_MON ;
_TEMP = X_5_MON ;
_LP0 = _LP0 + (   -0.04183986489373 * _TEMP);

***  Effect: X_5_SEC ;
_TEMP = X_5_SEC ;
_LP0 = _LP0 + (   -0.00021255644793 * _TEMP);

*** Naive Posterior Probabilities;
drop _MAXP _IY _P0 _P1;
_TEMP =     32.2689961075288 + _LP0;
if (_TEMP < 0) then do;
   _TEMP = exp(_TEMP);
   _P0 = _TEMP / (1 + _TEMP);
end;
else _P0 = 1 / (1 + exp(-_TEMP));
_P1 = 1.0 - _P0;

REG3DR1:


*** Posterior Probabilities and Predicted Level;
label P_TARGET1 = '预测: TARGET=1' ;
label P_TARGET0 = '预测: TARGET=0' ;
P_TARGET1 = _P0;
_MAXP = _P0;
_IY = 1;
P_TARGET0 = _P1;
if (_P1 >  _MAXP + 1E-8) then do;
   _MAXP = _P1;
   _IY = 2;
end;
I_TARGET = REG3DRF[_IY];
U_TARGET = REG3DRU[_IY];

*************************************;
***** end scoring code for regression;
*************************************;


%MEND S3;
```


