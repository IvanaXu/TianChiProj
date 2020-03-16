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



