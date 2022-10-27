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



