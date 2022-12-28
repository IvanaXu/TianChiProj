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



