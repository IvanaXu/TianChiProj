	
	DATA Adata1_score_T;
	SET Adata1_score;
	FORMAT MTYPE 1. FID $32.;
	MTYPE = 1;
	FID = PUT(MD5(CAT(ID, CATX("|", OF X_:))), HEX32.);
	RUN;
	PROC SORT DATA = Adata1_score_T NODUPKEY;
	BY FID;
	RUN;

	DATA Adata2_score_T;
	SET Adata2_score;
	FORMAT MTYPE 1. FID $32.;
	MTYPE = 1;
	FID = PUT(MD5(CAT(ID, CATX("|", OF X_:))), HEX32.);
	RUN;
	PROC SORT DATA = Adata2_score_T NODUPKEY;
	BY FID;
	RUN;

	DATA Adata3_score_T;
	SET Adata3_score;
	FORMAT MTYPE 1. FID $32.;
	MTYPE = 1;
	FID = PUT(MD5(CAT(ID, CATX("|", OF X_:))), HEX32.);
	RUN;
	PROC SORT DATA = Adata3_score_T NODUPKEY;
	BY FID;
	RUN;

	DATA Adata123_score_T;
	MERGE 
		Adata1_score_T(IN=TB1 RENAME = (P_TARGET1 = P1 TARGET = T1 ) KEEP = FID ID P_TARGET1 TARGET)
		Adata2_score_T(IN=TB2 RENAME = (P_TARGET1 = P2 TARGET = T2 ) KEEP = FID P_TARGET1 TARGET)
		Adata3_score_T(IN=TB3 RENAME = (P_TARGET1 = P3 TARGET = T3 ) KEEP = FID P_TARGET1 TARGET)
	;
	BY FID;
	IF TB1 OR TB2 OR TB3;
	TP1 = P1 > 0.30;
	TP2 = P2 > 0.40;
	TP3 = P3 > 0.42;
	RUN;
	PROC SQL;
	CREATE TABLE RAdata123_score_T AS 
	SELECT
		PUT(INPUT(ID,8.),Z4.) AS ID,
		SUM(1) AS CNT,
		MEAN(TP1) AS TP1,
		MEAN(TP2) AS TP2,
		MEAN(TP3) AS TP3,
		MEAN(T1) AS T1,
		MEAN(T2) AS T2,
		MEAN(T3) AS T3
	FROM Adata123_score_T
	GROUP BY ID
	ORDER BY ID;
	QUIT;

	DATA RAdata123_score_T;
	SET RAdata123_score_T;
	FORMAT T $6.;
	/* 1 刺网 2 围网 3 拖网 */
	IF TP1 = MAX(TP1, TP2, TP3)
		THEN T = "刺网";
	ELSE IF TP2 = MAX(TP1, TP2, TP3)
		THEN T = "围网";
	ELSE IF TP3 = MAX(TP1, TP2, TP3)
		THEN T = "拖网";
	ELSE T = "拖网";

	FORMAT TT $6.;
	/* 1 刺网 2 围网 3 拖网 */
	IF T1 = MAX(T1, T2, T3)
		THEN TT = "刺网";
	ELSE IF T2 = MAX(T1, T2, T3)
		THEN TT = "围网";
	ELSE IF T3 = MAX(T1, T2, T3)
		THEN TT = "拖网";
	ELSE TT = "拖网";

	IS_TTT = T = TT;
	RUN;
	PROC SORT DATA = RAdata123_score_T;
	BY IS_TTT;
	RUN;
	PROC FREQ DATA = RAdata123_score_T;
	TABLE IS_TTT;
	RUN;



