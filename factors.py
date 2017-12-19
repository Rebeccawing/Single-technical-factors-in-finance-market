#!/usr/bin/env python
# coding=utf-8
# python 2.7.12
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from numpy import NaN
#这个程序按第二种理解，即因子的factor period来变

# 读取所有文件
#AdjF = pd.read_csv("C:\DELL\quantitative research\Firtest\Origidata\AdjFactor_20070131_20130301.csv", sep=',', index_col=0)
DCloP = pd.read_csv("C:\DELL\quantitative research\Firtest\Origidata\ClosePrice_20070131_20130301.csv", sep=',', index_col=0)
DOpenP =  pd.read_csv("C:\DELL\quantitative research\Firtest\Origidata\OpenPrice_20070131_20130301.csv", sep=',', index_col=0)
DHP =  pd.read_csv("C:\DELL\quantitative research\Firtest\Origidata\HighestPrice_20070131_20130301.csv", sep=',', index_col=0)
DLP =  pd.read_csv("C:\DELL\quantitative research\Firtest\Origidata\LowestPrice_20070131_20130301.csv", sep=',', index_col=0)
DVol =  pd.read_csv("C:\DELL\quantitative research\Firtest\Origidata\Volume_20070131_20130301.csv", sep=',', index_col=0)
DVWap =  pd.read_csv("C:\DELL\quantitative research\Firtest\Origidata\Vwap_20070131_20130301.csv", sep=',', index_col=0)
HS =  pd.read_csv("C:\DELL\quantitative research\Firtest\Origidata\hs_zz_zz_20070131_20130301.csv", sep=',', index_col=0)

#对所有价格进行复权
#AdjF = AdjF.fillna(1.0)
#DCloP = CloP/AdjF
#DOpenP = OpenP/AdjF
#DHP = HP/AdjF
#DLP = LP/AdjF
#DVol = Vol*AdjF
developarr = [DCloP,DOpenP,DHP,DLP,DVol,HS]

#定义一个函数，把所有的df index转化为datetime格式
def timeconvert(x):
    x.index = x.index.astype('str')
    x.index = pd.to_datetime(pd.DatetimeIndex(x.index).date)
for i in range(0,6):
    timeconvert(developarr[i])
# 求出待处理的表格nor与前一天的表格ref
def ref(x):
    y=x.iloc[1:,:]
    z=x.iloc[:-1,:]
    z.index = y.index
    return z
def nor(x):
    return x.iloc[1:,:]

#求开盘价
def first(x):
    return x[0]
DOpenP_5 = (DOpenP.resample('W',label='left').first()).dropna(how='all')
DOpenP_10 = (DOpenP.resample('2W',label='left').first()).dropna(how='all')
DOpenP_20 = (DOpenP.resample('4W',label='left').first()).dropna(how='all')
DOpenP_60 = (DOpenP.resample('12W',label='left').first()).dropna(how='all')
[REFDOpenP,REFDOpenP_5,REFDOpenP_10,REFDOpenP_20,REFDOpenP_60] = [ref(DOpenP),ref(DOpenP_5),ref(DOpenP_10),ref(DOpenP_20),ref(DOpenP_60)]
[DOpenP,DOpenP_5,DOpenP_10,DOpenP_20,DOpenP_60] = [nor(DOpenP),nor(DOpenP_5),nor(DOpenP_10),nor(DOpenP_20),nor(DOpenP_60)]
#求收盘价
DCloP_5  = (DCloP.resample('W',label='left').last()).dropna(how='all')
DCloP_10  = (DCloP.resample('2W',label='left').last()).dropna(how='all')
DCloP_20 = (DCloP.resample('4W',label='left').last()).dropna(how='all')
DCloP_60  = (DCloP.resample('12W',label='left').last()).dropna(how='all')
[REFDCloP,REFDCloP_5,REFDCloP_10,REFDCloP_20,REFDCloP_60] = [ref(DCloP),ref(DCloP_5),ref(DCloP_10),ref(DCloP_20),ref(DCloP_60)]
[DCloP,DCloP_5,DCloP_10,DCloP_20,DCloP_60] = [nor(DCloP),nor(DCloP_5),nor(DCloP_10),nor(DCloP_20),nor(DCloP_60)]
#求最高价
DHP_5  = (DHP.resample('W',label='left').max()).dropna(how='all')
DHP_10  = (DHP.resample('2W',label='left').max()).dropna(how='all')
DHP_20  = (DHP.resample('4W',label='left').max()).dropna(how='all')
DHP_60  = (DHP.resample('12W',label='left').max()).dropna(how='all')
[REFDHP,REFDHP_5,REFDHP_10,REFDHP_20,REFDHP_60] = [ref(DHP),ref(DHP_5),ref(DHP_10),ref(DHP_20),ref(DHP_60)]
[DHP,DHP_5,DHP_10,DHP_20,DHP_60] = [nor(DHP),nor(DHP_5),nor(DHP_10),nor(DHP_20),nor(DHP_60)]
#求最低价
DLP_5 = (DLP.resample('W',label='left').min()).dropna(how='all')
DLP_10 = (DLP.resample('2W',label='left').min()).dropna(how='all')
DLP_20 = (DLP.resample('4W',label='left').min()).dropna(how='all')
DLP_60 = (DLP.resample('12W',label='left').min()).dropna(how='all')
[REFDLP,REFDLP_5,REFDLP_10,REFDLP_20,REFDLP_60] = [ref(DLP),ref(DLP_5),ref(DLP_10),ref(DLP_20),ref(DLP_60)]
[DLP,DLP_5,DLP_10,DLP_20,DLP_60] = [nor(DLP),nor(DLP_5),nor(DLP_10),nor(DLP_20),nor(DLP_60)]
#求成交量
DVol_5 = (DVol.resample('W',label='left').sum()).dropna(how='all')
DVol_10 = (DVol.resample('2W',label='left').sum()).dropna(how='all')
DVol_20 = (DVol.resample('4W',label='left').sum()).dropna(how='all')
DVol_60 = (DVol.resample('12W',label='left').sum()).dropna(how='all')
[REFDVol,REFDVol_5,REFDVol_10,REFDVol_20,REFDVol_60] = [ref(DVol),ref(DVol_5),ref(DVol_10),ref(DVol_20),ref(DVol_60)]
[DVol,DVol_5,DVol_10,DVol_20,DVol_60] = [nor(DVol),nor(DVol_5),nor(DVol_10),nor(DVol_20),nor(DVol_60)]
#求大盘指数
HS5 = (HS.resample('W',label='left').last()).dropna(how='all')
HS10 = (HS.resample('2W',label='left').last()).dropna(how='all')
HS20 = (HS.resample('4W',label='left').last()).dropna(how='all')
HS60 = (HS.resample('12W',label='left').last()).dropna(how='all')
#收益率函数
def Return(x):
    x1=x.iloc[:-1,:]
    x2=x.iloc[1:,:]
    x1.index=x2.index
    df = (x2/x1) -1
    df1=df.shift(-1)
    return df1
#算每个交易周期的benchmark
[HS_5,HS_10,HS_20,HS_60] = [Return(HS5),Return(HS10),Return(HS20),Return(HS60)]


#算每个交易周期的股票收益
[FR_5,FR_10,FR_20,FR_60] = [Return(DCloP_5),Return(DCloP_10),Return(DCloP_20),Return(DCloP_60)]

#定义批量存文件函数
def dfSave(x,y):
    x.to_csv('C:\\DELL\\quantitative research\\Firtest\\secondfactor\\'+y+'.csv')
#构造因子的函数
#AR
def AR(x,M):
     x=str(x)
     vn = globals()
     x1 = (vn['DHP_' + x] - vn['DOpenP_' + x])
     x2 = (vn['DOpenP_' + x] - vn['DLP_' + x])
     y1 = pd.rolling_mean(x1,M,min_periods=1)
     y2 = pd.rolling_mean(x2,M,min_periods=1)
     y = y1/y2
#    x = x1/x2
     return y*100

#def AR(M):
#    x1 = DHP-DOpenP
#    x2 = DOpenP-DLP
#    y1 = pd.rolling_sum(x1,M,min_periods=1)
#    y2 = pd.rolling_sum(x2,M,min_periods=1)
#    y = y1/y2
#    return y*100
[AR_5,AR_10,AR_20,AR_60]=[AR(5,4),AR(10,4),AR(20,4),AR(60,4)]
ARfact=[AR_5,AR_10,AR_20,AR_60]
ARname=['AR_5','AR_10','AR_20','AR_60']
for i in range(0,4):
    dfSave(ARfact[i],ARname[i])

#BR
def BR(x,M):
    x=str(x)
    vn = globals()
    df1=(vn['DHP_' + x]-vn['REFDCloP_' + x])
    df1[df1<0]=0
    df2=(vn['REFDCloP_' + x]-vn['DLP_' + x])
    df2[df2<0]=0
    x1 = pd.rolling_sum(df1,M,min_periods=1)
    x2 = pd.rolling_sum(df2,M,min_periods=1)
    y = x1/x2
    return y*100

[BR_5,BR_10,BR_20,BR_60]=[BR(5,4),BR(10,4),BR(20,4),BR(60,4)]
BRfact=[BR_5,BR_10,BR_20,BR_60]
BRname=['BR_5','BR_10','BR_20','BR_60']
for i in range(0,4):
    dfSave(BRfact[i],BRname[i])

#ASI
def ASI(x):
    x = str(x)
    vn = globals()
    aa=abs(vn['DHP_' + x]-vn['REFDCloP_' + x])
    bb=abs(vn['DLP_' + x]-vn['REFDCloP_' + x])
    cc=abs(vn['DHP_' + x]-vn['REFDLP_' + x])
    dd=abs(vn['REFDCloP_' + x]-vn['REFDOpenP_' + x])
    r = cc+dd*0.25
    r[(aa>bb)&(aa>cc)]=aa+bb*0.5+dd*0.25
    r[(bb>cc)&(bb>aa)]=bb+aa*0.5+dd*0.25
    X=vn['DCloP_' + x]*1.5-vn['DOpenP_' + x]*0.5-vn['REFDOpenP_' + x]
    aa[aa<bb]=bb
    si=16*X*aa/r
    asi=si.cumsum(axis=0)
    return asi

[ASI_5,ASI_10,ASI_20,ASI_60]=[ASI(5),ASI(10),ASI(20),ASI(60)]
ASIfact=[ASI_5,ASI_10,ASI_20,ASI_60]
ASIname=['ASI_5','ASI_10','ASI_20','ASI_60']
for i in range(0,4):
    dfSave(ASIfact[i],ASIname[i])

#DDI
def DDI(x,M):
    x = str(x)
    vn = globals()
    df1=abs(vn['DHP_' + x]-vn['REFDHP_' + x])
    df2=abs(vn['DLP_' + x]-vn['REFDLP_' + x])
    df1[df1<df2]=df2
    DMZ=df1.copy()
    DMF=df1.copy()
    DMZ[(vn['DHP_' + x]+vn['DLP_' + x])<(vn['REFDHP_' + x]+vn['REFDLP_' + x])] = 0
    DMF[(vn['DHP_' + x] + vn['DLP_' + x]) >= (vn['REFDHP_' + x] + vn['REFDLP_' + x])] = 0
    x1=pd.rolling_mean(DMZ,M,min_periods=1)
    x2=pd.rolling_mean(DMF,M,min_periods=1)
    DIZ=x1/(x1+x2)
    DIF=x2/(x1+x2)
    ddi=DIZ-DIF
    return ddi

[DDI_5, DDI_10, DDI_20, DDI_60] = [DDI(5,20), DDI(10,20), DDI(20,20), DDI(60,20)]
DDIfact = [DDI_5, DDI_10, DDI_20, DDI_60]
DDIname = ['DDI_5', 'DDI_10', 'DDI_20', 'DDI_60']
for i in range(0,4):
    dfSave(DDIfact[i],DDIname[i])

#Ease of Movement
def EMV(x,M):
    x=str(x)
    vn=globals()
    df1 = (vn['DHP_' + x]+vn['DLP_' + x]-vn['REFDHP_' + x]-vn['REFDLP_' + x])*0.5
    df2 = (vn['DHP_' + x]-vn['DLP_' + x])/vn['DVol_' + x]
    emv=pd.ewma((df1*df2),span=M,min_periods=1)
    return emv

[EMV_5, EMV_10, EMV_20, EMV_60] = [EMV(5,4), EMV(10,4), EMV(20,4), EMV(60,4)]
EMVfact = [EMV_5, EMV_10, EMV_20, EMV_60]
EMVname = ['EMV_5', 'EMV_10', 'EMV_20', 'EMV_60']
for i in range(0,4):
    dfSave(EMVfact[i],EMVname[i])

#Elder
def Elder(x,M):
    x=str(x)
    vn=globals()
    df1=pd.ewma(vn['DCloP_' + x],span=M,min_periods=1)
    long_pow = vn['DHP_' + x]-df1
    short_pow = vn['DLP_' + x]-df1
    elder=(long_pow-short_pow)/vn['DCloP_' + x]
    return elder

[Elder_5, Elder_10, Elder_20, Elder_60] = [Elder(5,4), Elder(10,4), Elder(20,4), Elder(60,4)]
Elderfact = [Elder_5, Elder_10, Elder_20, Elder_60]
Eldername = ['Elder_5', 'Elder_10', 'Elder_20', 'Elder_60']
for i in range(0,4):
    dfSave(Elderfact[i],Eldername[i])

#Hurst
def Hurst(x,M):
    x=str(x)
    vn=globals()
    m = pd.rolling_mean((vn['DCloP_' + x]),M,min_periods=1)
    y = vn['DCloP_' + x]-m
    z = y.cumsum(axis=0)
    r = pd.rolling_max((vn['DCloP_' + x]),M,min_periods=1)-pd.rolling_min((vn['DCloP_' + x]),M,min_periods=1)
    s = (vn['DCloP_' + x]).std()
    hurst = r/s
    return hurst

[Hurst_5, Hurst_10, Hurst_20, Hurst_60] = [Hurst(5,4), Hurst(10,4), Hurst(20,4), Hurst(60,4)]
Hurstfact = [Hurst_5, Hurst_10, Hurst_20, Hurst_60]
Hurstname = ['Hurst_5', 'Hurst_10', 'Hurst_20', 'Hurst_60']
for i in range(0,4):
    dfSave(Hurstfact[i],Hurstname[i])

#KDJ
def KDJ(x,M,P1,P2):
    x=str(x)
    vn=globals()
    lv = pd.rolling_min((vn['DLP_' + x]),M,min_periods=1)
    hv = pd.rolling_max((vn['DHP_' + x]),M, min_periods=1)
    rsv = 100*(vn['DCloP_' + x]-lv)/(hv-lv)
    ind = rsv.index
    k = rsv.copy()
    for col in rsv.columns:
        for i in range(0,len(ind)):
            if i==0:
                pass
            else:
                k.loc[ind[i],col] = (rsv.loc[ind[i-1],col]*(P1-1) + rsv.loc[ind[i],col])/P1
    d = k.copy()
    for col in k.columns:
        for i in range(0,len(ind)):
            if i==0:
                pass
            else:
                d.loc[ind[i], col] = (k.loc[ind[i-1], col] * (P2 - 1) + k.loc[ind[i], col]) / P2
    j = k*3-d*2
    return j

[KDJ_5, KDJ_10, KDJ_20, KDJ_60] = [KDJ(5,4,3,3), KDJ(10,4,3,3), KDJ(20,4,3,3), KDJ(60,4,3,3)]
KDJfact = [KDJ_5, KDJ_10, KDJ_20, KDJ_60]
KDJname = ['KDJ_5', 'KDJ_10', 'KDJ_20', 'KDJ_60']
for i in range(0,4):
    dfSave(KDJfact[i],KDJname[i])

#MFI
def MFI(x,M):
    x=str(x)
    vn=globals()
    typ = (vn['DHP_' + x]+vn['DLP_' + x]+vn['DCloP_' + x])/3
    reftyp = (vn['REFDHP_' + x]+vn['REFDLP_' + x]+vn['REFDCloP_' + x])/3
    typ1=typ*vn['DVol_' + x]
    typ2=typ1.copy()
    typ1[typ<=reftyp]=0
    typ2[typ>=reftyp]=0
    x1=pd.rolling_mean(typ1,M,min_periods=1)
    x2=pd.rolling_mean(typ2,M,min_periods=1)
    v1=x1/x2
    mfi = 100-(100/(1+v1))
    return mfi

[MFI_5, MFI_10, MFI_20, MFI_60] = [MFI(5,7), MFI(10,7), MFI(20,7), MFI(60,7)]
MFIfact = [MFI_5, MFI_10, MFI_20, MFI_60]
MFIname = ['MFI_5', 'MFI_10', 'MFI_20', 'MFI_60']
for i in range(0,4):
    dfSave(MFIfact[i],MFIname[i])

#PVI
def PVI(x):
    x=str(x)
    vn=globals()
    df = vn['DCloP_' + x] - vn['REFDCloP_' + x]
    for col in df.columns:
        ind = df.index
        pi = pd.Series(range(len(df)+1))
        pi[0] = 1
        pvi = df.copy()
        for i in range(len(df)):
            if (df.loc[ind[i],col]>0):
                pvi.loc[ind[i],col] = df.loc[ind[i],col]*pi[i]/(vn['REFDCloP_' + x]).loc[ind[i],col]
            else:
                pvi.loc[ind[i],col] = pi[i]
            pi[i+1]=pvi.loc[ind[i],col]
    return pvi

[PVI_5, PVI_10, PVI_20, PVI_60] = [PVI(5), PVI(10), PVI(20), PVI(60)]
PVIfact = [PVI_5, PVI_10, PVI_20, PVI_60]
PVIname = ['PVI_5', 'PVI_10', 'PVI_20', 'PVI_60']
for i in range(0,4):
    dfSave(PVIfact[i],PVIname[i])

#Ulcer
def likevar(ser):
    ser = ser.astype('float')
    ser2 = ser.copy()
    ind = ser.index
    for i in range(len(ind)):
        ser1 = ser[ind[0]:ind[i]]
        ser2[ind[i]] = ((ser1**2).sum()/(i+1))**0.5
    return ser2
def Ulcer(x,M):
    x=str(x)
    vn=globals()
    mp = pd.rolling_max((vn['DCloP_' + x]),M,min_periods=1)
    ri = 100*(vn['DCloP_' + x]-mp)/mp
    ulcer = ri.apply(likevar,axis=1)
    return ri

[Ulcer_5, Ulcer_10, Ulcer_20, Ulcer_60] = [Ulcer(5,4), Ulcer(10,4), Ulcer(20,4), Ulcer(60,4)]
Ulcerfact = [Ulcer_5, Ulcer_10, Ulcer_20, Ulcer_60]
Ulcername = ['Ulcer_5', 'Ulcer_10', 'Ulcer_20', 'Ulcer_60']
for i in range(0,4):
    dfSave(Ulcerfact[i],Ulcername[i])
