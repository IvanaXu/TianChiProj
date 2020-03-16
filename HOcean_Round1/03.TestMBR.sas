	
	OPTIONS COMPRESS = YES;
/*	00.逻辑库 */
	LIBNAME ANA "C:\Users\IvanXu\Desktop\data\hy_round1";
	%LET CP = C:\Users\IvanXu\Desktop\code\Hy_round1;
	%LET DP = C:\Users\IvanXu\Desktop\data\hy_round1\hy_round1_train_20200102\hy_round1_train_20200102;
	%LET TP = C:\Users\IvanXu\Desktop\data\hy_round1\hy_round1_testA_20200102\hy_round1_testA_20200102;

	%LET PN = 000009;


%LET P = 1;
	*------------------------------------------------------------*;
	* MBR: CREATE DECISION MATRIX;
	*------------------------------------------------------------*;
	DATA WORK.TARGET;
	  LENGTH   TARGET                           $  32
	           COUNT                                8
	           DATAPRIOR                            8
	           TRAINPRIOR                           8
	           DECPRIOR                             8
	           DECISION1                            8
	           DECISION2                            8
	           ;

	  LABEL    COUNT="LEVEL COUNTS"
	           DATAPRIOR="DATA PROPORTIONS"
	           TRAINPRIOR="TRAINING PROPORTIONS"
	           DECPRIOR="DECISION PRIORS"
	           DECISION1="1"
	           DECISION2="0"
	           ;
	  FORMAT   COUNT 10.
	           ;
TARGET="1"; COUNT=391806; DATAPRIOR=0.14513279187802; TRAINPRIOR=0.14513279187802; DECPRIOR=.; DECISION1=1; DECISION2=0;
output;
TARGET="0"; COUNT=2307832; DATAPRIOR=0.85486720812197; TRAINPRIOR=0.85486720812197; DECPRIOR=.; DECISION1=0; DECISION2=1;
output;
	;
	RUN;
	PROC DATASETS LIB=WORK NOLIST;
	MODIFY TARGET(TYPE=PROFIT LABEL=TARGET);
	LABEL DECISION1= '1';
	LABEL DECISION2= '0';
	RUN;
	QUIT;

	*------------------------------------------------------------* ;
	* MBR: DMDBCLASS MACRO ;
	*------------------------------------------------------------* ;
	%MACRO DMDBCLASS;
	    TARGET(DESC)
	%MEND DMDBCLASS;
	*------------------------------------------------------------* ;
	* MBR: DMDBVAR MACRO ;
	*------------------------------------------------------------* ;
	%MACRO DMDBVAR;
	    X_1 X_1V X_2 X_2V X_3 X_3_1 X_3_2 X_3_4 X_4 X_4_1 X_4_2 X_5_DAY X_5_HOR X_5_MIS X_5_MON X_5_SEC
	%MEND DMDBVAR;
	*------------------------------------------------------------*;
	* MBR: CREATE DMDB;
	*------------------------------------------------------------*;

	DATA ADATA&P.;
	SET ANA.DATA&P. NOBS = N;
	_RX1 = RAND("UNIFORM");
	_RX2 = RAND("UNIFORM");
	_DATAOBS_ = _N_;
	CALL SYMPUT("NS", N);
	RUN;
	%PUT &NS.;
	PROC SORT DATA = ADATA&P.;
	BY _RX1 _RX2;
	RUN;
	DATA 
		TRANS&P._TRAIN
		TRANS&P._VALIDATE
		TRANS&P._TEST
	;
	SET ADATA&P.(DROP = _RX1 _RX2);
	IF _N_ < INT(&NS.*0.4)
		THEN OUTPUT TRANS&P._TRAIN;
	ELSE IF _N_ < INT(&NS.*0.7)
		THEN OUTPUT TRANS&P._VALIDATE;
	ELSE OUTPUT TRANS&P._TEST;
	RUN;


	PROC DMDB BATCH DATA=TRANS&P._TRAIN
	DMDBCAT=WORK.MBR_DMDB
	MAXLEVEL = 513
	;
	ID
	_DATAOBS_
	;
	CLASS %DMDBCLASS;
	VAR %DMDBVAR;
	TARGET
	TARGET
	;
	RUN;
	QUIT;


/* TRAIN */
	*------------------------------------------------------------* ;
	* MBR: INTERVAL VARIABLES MACRO ;
	*------------------------------------------------------------* ;
	%MACRO PMBRVAR;
	    X_1 X_1V X_2 X_2V X_3 X_3_1 X_3_2 X_3_4 X_4 X_4_1 X_4_2 X_5_DAY X_5_HOR X_5_MIS X_5_MON X_5_SEC
	%MEND PMBRVAR;
	PROC PMBR DATA=TRANS&P._TRAIN DMDBCAT=WORK.MBR_DMDB
	VALIDATA = TRANS&P._VALIDATE
	TESTDATA = TRANS&P._TEST
	OUTEST = MBR_ESTIMATE
	K = 16
	EPSILON = 0
	BUCKETS = 8
	METHOD = RDTREE
	WEIGHTED
	NEIGHBORS
	;
	VAR %PMBRVAR;
	TARGET TARGET;
	SCORE DATA=TRANS&P._TRAIN
	OUT=MBR&P._TRAIN
	ROLE = TRAIN
	;
	ID _DATAOBS_;
	RUN;
	QUIT;
	*------------------------------------------------------------* ;
	* MBR: INTERVAL VARIABLES MACRO ;
	*------------------------------------------------------------* ;
	%MACRO PMBRVAR;
	    X_1 X_1V X_2 X_2V X_3 X_3_1 X_3_2 X_3_4 X_4 X_4_1 X_4_2 X_5_DAY X_5_HOR X_5_MIS X_5_MON X_5_SEC
	%MEND PMBRVAR;
	PROC PMBR DATA=TRANS&P._TRAIN DMDBCAT=WORK.MBR_DMDB
	OUTEST = WORK.MBR_OUTEST
	K = 16
	EPSILON = 0
	BUCKETS = 8
	METHOD = RDTREE
	WEIGHTED
	NEIGHBORS
	;
	VAR %PMBRVAR;
	TARGET TARGET;
	SCORE DATA=TRANS&P._VALIDATE
	OUT=MBR&P._VALIDATE
	ROLE = VALID
	;
	ID _DATAOBS_;
	RUN;
	QUIT;
	*------------------------------------------------------------* ;
	* MBR: INTERVAL VARIABLES MACRO ;
	*------------------------------------------------------------* ;
	%MACRO PMBRVAR;
	    X_1 X_1V X_2 X_2V X_3 X_3_1 X_3_2 X_3_4 X_4 X_4_1 X_4_2 X_5_DAY X_5_HOR X_5_MIS X_5_MON X_5_SEC
	%MEND PMBRVAR;
	PROC PMBR DATA=TRANS&P._TRAIN DMDBCAT=WORK.MBR_DMDB
	OUTEST = WORK.MBR_OUTEST
	K = 16
	EPSILON = 0
	BUCKETS = 8
	METHOD = RDTREE
	WEIGHTED
	NEIGHBORS
	;
	VAR %PMBRVAR;
	TARGET TARGET;
	SCORE DATA=TRANS&P._TEST
	OUT=MBR&P._TEST
	ROLE = TEST
	;
	ID _DATAOBS_;
	RUN;
	QUIT;


/* TEST */
	%MACRO PMBRVAR;
	    X_1 X_1V X_2 X_2V X_3 X_3_1 X_3_2 X_3_4 X_4 X_4_1 X_4_2 X_5_DAY X_5_HOR X_5_MIS X_5_MON X_5_SEC
	%MEND PMBRVAR;
	PROC PMBR DATA=TRANS&P._TRAIN DMDBCAT=WORK.MBR_DMDB
	OUTEST = WORK.MBR_OUTEST
	K = 16
	EPSILON = 0
	BUCKETS = 8
	METHOD = RDTREE
	WEIGHTED
	NEIGHBORS
	;
	VAR %PMBRVAR;
	TARGET TARGET;
	SCORE DATA=ADATA&P.
	OUT=ADATA&P._SCORE
	ROLE = SCORE
	;
	ID _DATAOBS_;
	RUN;
	QUIT;

	
	%MACRO F1_(DT=,);
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
	%F1_(DT=ADATA&P._SCORE);

	
	DATA TEST&P.;
	SET ANA.TEST;
	_DATAOBS_ = _N_ + 1000000;
	RUN;

	%MACRO PMBRVAR;
	    X_1 X_1V X_2 X_2V X_3 X_3_1 X_3_2 X_3_4 X_4 X_4_1 X_4_2 X_5_DAY X_5_HOR X_5_MIS X_5_MON X_5_SEC
	%MEND PMBRVAR;
	PROC PMBR DATA=TRANS&P._TRAIN DMDBCAT=WORK.MBR_DMDB
	OUTEST = WORK.MBR_OUTEST
	K = 16
	EPSILON = 0
	BUCKETS = 8
	METHOD = RDTREE
	WEIGHTED
	NEIGHBORS
	;
	VAR %PMBRVAR;
	TARGET TARGET;
	SCORE DATA=TEST&P.
	OUT=TEST&P._SCORE
	ROLE = SCORE
	;
	ID _DATAOBS_;
	RUN;
	QUIT;


%LET P = 2;
	*------------------------------------------------------------*;
	* MBR: CREATE DECISION MATRIX;
	*------------------------------------------------------------*;
	DATA WORK.TARGET;
	  LENGTH   TARGET                           $  32
	           COUNT                                8
	           DATAPRIOR                            8
	           TRAINPRIOR                           8
	           DECPRIOR                             8
	           DECISION1                            8
	           DECISION2                            8
	           ;

	  LABEL    COUNT="LEVEL COUNTS"
	           DATAPRIOR="DATA PROPORTIONS"
	           TRAINPRIOR="TRAINING PROPORTIONS"
	           DECPRIOR="DECISION PRIORS"
	           DECISION1="1"
	           DECISION2="0"
	           ;
	  FORMAT   COUNT 10.
	           ;
TARGET="1"; COUNT=626607; DATAPRIOR=0.2321077863032; TRAINPRIOR=0.2321077863032; DECPRIOR=.; DECISION1=1; DECISION2=0;
output;
TARGET="0"; COUNT=2073031; DATAPRIOR=0.76789221369679; TRAINPRIOR=0.76789221369679; DECPRIOR=.; DECISION1=0; DECISION2=1;
output;
	;
	RUN;
	PROC DATASETS LIB=WORK NOLIST;
	MODIFY TARGET(TYPE=PROFIT LABEL=TARGET);
	LABEL DECISION1= '1';
	LABEL DECISION2= '0';
	RUN;
	QUIT;

	*------------------------------------------------------------* ;
	* MBR: DMDBCLASS MACRO ;
	*------------------------------------------------------------* ;
	%MACRO DMDBCLASS;
	    TARGET(DESC)
	%MEND DMDBCLASS;
	*------------------------------------------------------------* ;
	* MBR: DMDBVAR MACRO ;
	*------------------------------------------------------------* ;
	%MACRO DMDBVAR;
	    X_1 X_1V X_2 X_2V X_3 X_3_1 X_3_2 X_3_4 X_4 X_4_1 X_4_2 X_5_DAY X_5_HOR X_5_MIS X_5_MON X_5_SEC
	%MEND DMDBVAR;
	*------------------------------------------------------------*;
	* MBR: CREATE DMDB;
	*------------------------------------------------------------*;

	DATA ADATA&P.;
	SET ANA.DATA&P. NOBS = N;
	_RX1 = RAND("UNIFORM");
	_RX2 = RAND("UNIFORM");
	_DATAOBS_ = _N_;
	CALL SYMPUT("NS", N);
	RUN;
	%PUT &NS.;
	PROC SORT DATA = ADATA&P.;
	BY _RX1 _RX2;
	RUN;
	DATA 
		TRANS&P._TRAIN
		TRANS&P._VALIDATE
		TRANS&P._TEST
	;
	SET ADATA&P.(DROP = _RX1 _RX2);
	IF _N_ < INT(&NS.*0.4)
		THEN OUTPUT TRANS&P._TRAIN;
	ELSE IF _N_ < INT(&NS.*0.7)
		THEN OUTPUT TRANS&P._VALIDATE;
	ELSE OUTPUT TRANS&P._TEST;
	RUN;


	PROC DMDB BATCH DATA=TRANS&P._TRAIN
	DMDBCAT=WORK.MBR_DMDB
	MAXLEVEL = 513
	;
	ID
	_DATAOBS_
	;
	CLASS %DMDBCLASS;
	VAR %DMDBVAR;
	TARGET
	TARGET
	;
	RUN;
	QUIT;


/* TRAIN */
	*------------------------------------------------------------* ;
	* MBR: INTERVAL VARIABLES MACRO ;
	*------------------------------------------------------------* ;
	%MACRO PMBRVAR;
	    X_1 X_1V X_2 X_2V X_3 X_3_1 X_3_2 X_3_4 X_4 X_4_1 X_4_2 X_5_DAY X_5_HOR X_5_MIS X_5_MON X_5_SEC
	%MEND PMBRVAR;
	PROC PMBR DATA=TRANS&P._TRAIN DMDBCAT=WORK.MBR_DMDB
	VALIDATA = TRANS&P._VALIDATE
	TESTDATA = TRANS&P._TEST
	OUTEST = MBR_ESTIMATE
	K = 16
	EPSILON = 0
	BUCKETS = 8
	METHOD = RDTREE
	WEIGHTED
	NEIGHBORS
	;
	VAR %PMBRVAR;
	TARGET TARGET;
	SCORE DATA=TRANS&P._TRAIN
	OUT=MBR&P._TRAIN
	ROLE = TRAIN
	;
	ID _DATAOBS_;
	RUN;
	QUIT;
	*------------------------------------------------------------* ;
	* MBR: INTERVAL VARIABLES MACRO ;
	*------------------------------------------------------------* ;
	%MACRO PMBRVAR;
	    X_1 X_1V X_2 X_2V X_3 X_3_1 X_3_2 X_3_4 X_4 X_4_1 X_4_2 X_5_DAY X_5_HOR X_5_MIS X_5_MON X_5_SEC
	%MEND PMBRVAR;
	PROC PMBR DATA=TRANS&P._TRAIN DMDBCAT=WORK.MBR_DMDB
	OUTEST = WORK.MBR_OUTEST
	K = 16
	EPSILON = 0
	BUCKETS = 8
	METHOD = RDTREE
	WEIGHTED
	NEIGHBORS
	;
	VAR %PMBRVAR;
	TARGET TARGET;
	SCORE DATA=TRANS&P._VALIDATE
	OUT=MBR&P._VALIDATE
	ROLE = VALID
	;
	ID _DATAOBS_;
	RUN;
	QUIT;
	*------------------------------------------------------------* ;
	* MBR: INTERVAL VARIABLES MACRO ;
	*------------------------------------------------------------* ;
	%MACRO PMBRVAR;
	    X_1 X_1V X_2 X_2V X_3 X_3_1 X_3_2 X_3_4 X_4 X_4_1 X_4_2 X_5_DAY X_5_HOR X_5_MIS X_5_MON X_5_SEC
	%MEND PMBRVAR;
	PROC PMBR DATA=TRANS&P._TRAIN DMDBCAT=WORK.MBR_DMDB
	OUTEST = WORK.MBR_OUTEST
	K = 16
	EPSILON = 0
	BUCKETS = 8
	METHOD = RDTREE
	WEIGHTED
	NEIGHBORS
	;
	VAR %PMBRVAR;
	TARGET TARGET;
	SCORE DATA=TRANS&P._TEST
	OUT=MBR&P._TEST
	ROLE = TEST
	;
	ID _DATAOBS_;
	RUN;
	QUIT;


/* TEST */
	%MACRO PMBRVAR;
	    X_1 X_1V X_2 X_2V X_3 X_3_1 X_3_2 X_3_4 X_4 X_4_1 X_4_2 X_5_DAY X_5_HOR X_5_MIS X_5_MON X_5_SEC
	%MEND PMBRVAR;
	PROC PMBR DATA=TRANS&P._TRAIN DMDBCAT=WORK.MBR_DMDB
	OUTEST = WORK.MBR_OUTEST
	K = 16
	EPSILON = 0
	BUCKETS = 8
	METHOD = RDTREE
	WEIGHTED
	NEIGHBORS
	;
	VAR %PMBRVAR;
	TARGET TARGET;
	SCORE DATA=ADATA&P.
	OUT=ADATA&P._SCORE
	ROLE = SCORE
	;
	ID _DATAOBS_;
	RUN;
	QUIT;

	
	%MACRO F1_(DT=,);
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
	%F1_(DT=ADATA&P._SCORE);

	
	DATA TEST&P.;
	SET ANA.TEST;
	_DATAOBS_ = _N_ + 1000000;
	RUN;

	%MACRO PMBRVAR;
	    X_1 X_1V X_2 X_2V X_3 X_3_1 X_3_2 X_3_4 X_4 X_4_1 X_4_2 X_5_DAY X_5_HOR X_5_MIS X_5_MON X_5_SEC
	%MEND PMBRVAR;
	PROC PMBR DATA=TRANS&P._TRAIN DMDBCAT=WORK.MBR_DMDB
	OUTEST = WORK.MBR_OUTEST
	K = 16
	EPSILON = 0
	BUCKETS = 8
	METHOD = RDTREE
	WEIGHTED
	NEIGHBORS
	;
	VAR %PMBRVAR;
	TARGET TARGET;
	SCORE DATA=TEST&P.
	OUT=TEST&P._SCORE
	ROLE = SCORE
	;
	ID _DATAOBS_;
	RUN;
	QUIT;


%LET P = 3;
	*------------------------------------------------------------*;
	* MBR: CREATE DECISION MATRIX;
	*------------------------------------------------------------*;
	DATA WORK.TARGET;
	  LENGTH   TARGET                           $  32
	           COUNT                                8
	           DATAPRIOR                            8
	           TRAINPRIOR                           8
	           DECPRIOR                             8
	           DECISION1                            8
	           DECISION2                            8
	           ;

	  LABEL    COUNT="LEVEL COUNTS"
	           DATAPRIOR="DATA PROPORTIONS"
	           TRAINPRIOR="TRAINING PROPORTIONS"
	           DECPRIOR="DECISION PRIORS"
	           DECISION1="1"
	           DECISION2="0"
	           ;
	  FORMAT   COUNT 10.
	           ;
TARGET="1"; COUNT=1681225; DATAPRIOR=0.62275942181877; TRAINPRIOR=0.62275942181877; DECPRIOR=.; DECISION1=1; DECISION2=0;
output;
TARGET="0"; COUNT=1018413; DATAPRIOR=0.37724057818122; TRAINPRIOR=0.37724057818122; DECPRIOR=.; DECISION1=0; DECISION2=1;
output;
	;
	RUN;
	PROC DATASETS LIB=WORK NOLIST;
	MODIFY TARGET(TYPE=PROFIT LABEL=TARGET);
	LABEL DECISION1= '1';
	LABEL DECISION2= '0';
	RUN;
	QUIT;

	*------------------------------------------------------------* ;
	* MBR: DMDBCLASS MACRO ;
	*------------------------------------------------------------* ;
	%MACRO DMDBCLASS;
	    TARGET(DESC)
	%MEND DMDBCLASS;
	*------------------------------------------------------------* ;
	* MBR: DMDBVAR MACRO ;
	*------------------------------------------------------------* ;
	%MACRO DMDBVAR;
	    X_1 X_1V X_2 X_2V X_3 X_3_1 X_3_2 X_3_4 X_4 X_4_1 X_4_2 X_5_DAY X_5_HOR X_5_MIS X_5_MON X_5_SEC
	%MEND DMDBVAR;
	*------------------------------------------------------------*;
	* MBR: CREATE DMDB;
	*------------------------------------------------------------*;

	DATA ADATA&P.;
	SET ANA.DATA&P. NOBS = N;
	_RX1 = RAND("UNIFORM");
	_RX2 = RAND("UNIFORM");
	_DATAOBS_ = _N_;
	CALL SYMPUT("NS", N);
	RUN;
	%PUT &NS.;
	PROC SORT DATA = ADATA&P.;
	BY _RX1 _RX2;
	RUN;
	DATA 
		TRANS&P._TRAIN
		TRANS&P._VALIDATE
		TRANS&P._TEST
	;
	SET ADATA&P.(DROP = _RX1 _RX2);
	IF _N_ < INT(&NS.*0.4)
		THEN OUTPUT TRANS&P._TRAIN;
	ELSE IF _N_ < INT(&NS.*0.7)
		THEN OUTPUT TRANS&P._VALIDATE;
	ELSE OUTPUT TRANS&P._TEST;
	RUN;


	PROC DMDB BATCH DATA=TRANS&P._TRAIN
	DMDBCAT=WORK.MBR_DMDB
	MAXLEVEL = 513
	;
	ID
	_DATAOBS_
	;
	CLASS %DMDBCLASS;
	VAR %DMDBVAR;
	TARGET
	TARGET
	;
	RUN;
	QUIT;


/* TRAIN */
	*------------------------------------------------------------* ;
	* MBR: INTERVAL VARIABLES MACRO ;
	*------------------------------------------------------------* ;
	%MACRO PMBRVAR;
	    X_1 X_1V X_2 X_2V X_3 X_3_1 X_3_2 X_3_4 X_4 X_4_1 X_4_2 X_5_DAY X_5_HOR X_5_MIS X_5_MON X_5_SEC
	%MEND PMBRVAR;
	PROC PMBR DATA=TRANS&P._TRAIN DMDBCAT=WORK.MBR_DMDB
	VALIDATA = TRANS&P._VALIDATE
	TESTDATA = TRANS&P._TEST
	OUTEST = MBR_ESTIMATE
	K = 16
	EPSILON = 0
	BUCKETS = 8
	METHOD = RDTREE
	WEIGHTED
	NEIGHBORS
	;
	VAR %PMBRVAR;
	TARGET TARGET;
	SCORE DATA=TRANS&P._TRAIN
	OUT=MBR&P._TRAIN
	ROLE = TRAIN
	;
	ID _DATAOBS_;
	RUN;
	QUIT;
	*------------------------------------------------------------* ;
	* MBR: INTERVAL VARIABLES MACRO ;
	*------------------------------------------------------------* ;
	%MACRO PMBRVAR;
	    X_1 X_1V X_2 X_2V X_3 X_3_1 X_3_2 X_3_4 X_4 X_4_1 X_4_2 X_5_DAY X_5_HOR X_5_MIS X_5_MON X_5_SEC
	%MEND PMBRVAR;
	PROC PMBR DATA=TRANS&P._TRAIN DMDBCAT=WORK.MBR_DMDB
	OUTEST = WORK.MBR_OUTEST
	K = 16
	EPSILON = 0
	BUCKETS = 8
	METHOD = RDTREE
	WEIGHTED
	NEIGHBORS
	;
	VAR %PMBRVAR;
	TARGET TARGET;
	SCORE DATA=TRANS&P._VALIDATE
	OUT=MBR&P._VALIDATE
	ROLE = VALID
	;
	ID _DATAOBS_;
	RUN;
	QUIT;
	*------------------------------------------------------------* ;
	* MBR: INTERVAL VARIABLES MACRO ;
	*------------------------------------------------------------* ;
	%MACRO PMBRVAR;
	    X_1 X_1V X_2 X_2V X_3 X_3_1 X_3_2 X_3_4 X_4 X_4_1 X_4_2 X_5_DAY X_5_HOR X_5_MIS X_5_MON X_5_SEC
	%MEND PMBRVAR;
	PROC PMBR DATA=TRANS&P._TRAIN DMDBCAT=WORK.MBR_DMDB
	OUTEST = WORK.MBR_OUTEST
	K = 16
	EPSILON = 0
	BUCKETS = 8
	METHOD = RDTREE
	WEIGHTED
	NEIGHBORS
	;
	VAR %PMBRVAR;
	TARGET TARGET;
	SCORE DATA=TRANS&P._TEST
	OUT=MBR&P._TEST
	ROLE = TEST
	;
	ID _DATAOBS_;
	RUN;
	QUIT;


/* TEST */
	%MACRO PMBRVAR;
	    X_1 X_1V X_2 X_2V X_3 X_3_1 X_3_2 X_3_4 X_4 X_4_1 X_4_2 X_5_DAY X_5_HOR X_5_MIS X_5_MON X_5_SEC
	%MEND PMBRVAR;
	PROC PMBR DATA=TRANS&P._TRAIN DMDBCAT=WORK.MBR_DMDB
	OUTEST = WORK.MBR_OUTEST
	K = 16
	EPSILON = 0
	BUCKETS = 8
	METHOD = RDTREE
	WEIGHTED
	NEIGHBORS
	;
	VAR %PMBRVAR;
	TARGET TARGET;
	SCORE DATA=ADATA&P.
	OUT=ADATA&P._SCORE
	ROLE = SCORE
	;
	ID _DATAOBS_;
	RUN;
	QUIT;

	
	%MACRO F1_(DT=,);
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
	%F1_(DT=ADATA&P._SCORE);

	
	DATA TEST&P.;
	SET ANA.TEST;
	_DATAOBS_ = _N_ + 1000000;
	RUN;

	%MACRO PMBRVAR;
	    X_1 X_1V X_2 X_2V X_3 X_3_1 X_3_2 X_3_4 X_4 X_4_1 X_4_2 X_5_DAY X_5_HOR X_5_MIS X_5_MON X_5_SEC
	%MEND PMBRVAR;
	PROC PMBR DATA=TRANS&P._TRAIN DMDBCAT=WORK.MBR_DMDB
	OUTEST = WORK.MBR_OUTEST
	K = 16
	EPSILON = 0
	BUCKETS = 8
	METHOD = RDTREE
	WEIGHTED
	NEIGHBORS
	;
	VAR %PMBRVAR;
	TARGET TARGET;
	SCORE DATA=TEST&P.
	OUT=TEST&P._SCORE
	ROLE = SCORE
	;
	ID _DATAOBS_;
	RUN;
	QUIT;



	
	DATA TEST1;
	SET Test1_score;
	FORMAT MTYPE 1. FID $32.;
	MTYPE = 1;
	FID = PUT(MD5(CAT(ID, CATX("|", OF X_:))), HEX32.);
	RUN;
	PROC SORT DATA = TEST1 NODUPKEY;
	BY FID;
	RUN;

	DATA TEST2;
	SET Test2_score;
	FORMAT MTYPE 1. FID $32.;
	MTYPE = 1;
	FID = PUT(MD5(CAT(ID, CATX("|", OF X_:))), HEX32.);
	RUN;
	PROC SORT DATA = TEST2 NODUPKEY;
	BY FID;
	RUN;

	DATA TEST3;
	SET Test3_score;
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
	TP1 = P1 > 0.30;
	TP2 = P2 > 0.40;
	TP3 = P3 > 0.42;
	RUN;
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

	FILENAME EXPT "C:\Users\IvanXu\Desktop\data\hy_round1\result&PN..csv" ENCODING="UTF-8";
	PROC EXPORT 
	    DATA = RESULT(KEEP = ID T)
	    OUTFILE = EXPT DBMS = CSV REPLACE;
	    PUTNAMES = NO;
	RUN;




/*	PROC UNIVARIATE DATA = ANA.DATA1;*/
/*	VAR X_3 X_4;*/
/*	RUN;*/
	

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


/*	DATA _T;*/
/*	SET ANA.TEST(KEEP = X_1 X_2);*/
/*	X_1V = X_1/20037508.34*180;*/
/*	X_2V = 180/CONSTANT("PI")*(2*ATAN(EXP(X_2/20037508.34*180*CONSTANT("PI")/180))-CONSTANT("PI")/2);*/
/*	RUN;*/
	


