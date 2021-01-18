# Modules
import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as plt

# Partial Modules
from timeit import default_timer as timer
from mpl_finance import candlestick_ohlc

# Custom scripts
from _calculations import *
from _performance_measure import *
#from _visualization import *

def Load_Data(path):
    # Load data
    data = pd.read_csv(path)
    #data = pd.read_csv("DATA/Hourly_BTC.csv").ffill()
    #Date,Open,High,Low,Close,Volume = pd.to_datetime(data['date']),data['open'],data['high'],data['low'],data['close'],data['volume']
    Date = pd.to_datetime(data['date']) # pandas
    Open = data['open'].to_numpy()
    High = data['high'].to_numpy()
    Low = data['low'].to_numpy()
    Close = data['close'].to_numpy()
    Volume = data['volume'].to_numpy()
    return Date,Open,High,Low,Close,Volume

Date,Open,High,Low,Close,Volume = Load_Data("DATA/FiveMin_BTC_filled.csv")

def indicators(iav,Open=Open,High=High,Low=Low,Close=Close,Volume=Volume):
    signals = np.zeros(len(Open))

    if iav['MA'][0][0] != None:
        ma = MA(Close,iav['MA'][0][0])
        ma_sig = np.where(Close > ma,iav['MA'][2][0],-iav['MA'][2][0])
        signals = np.add(signals,ma_sig)

    if iav['EMA'][0][0] != None:
        ema = EMA(Close,iav['EMA'][0][0])
        ema_sig = np.where(Close > ema,iav['EMA'][2][0],-iav['EMA'][2][0])
        signals = np.add(signals,ema_sig)

    if iav['MACD'][0][0] != None:
        macd,sl = MACD(Close,iav['MACD'][0][0],iav['MACD'][0][1],iav['MACD'][0][2])
        macd_sig = np.where(macd > sl,iav['MACD'][2][0],-iav['MACD'][2][0])
        signals = np.add(signals,macd_sig)

    if iav['PSAR'][0][0] != None: # FIX 
        psar = PSAR(Close,iav['PSAR'][0],iav['PSAR'][1])

    if iav['ADX'][0][0] != None: 
        adx,minus_DI,plus_DI = ADX(High,Low,Close,iav['ADX'][0][0]) 
        np.add(np.where(plus_DI > minus_DI,iav['ADX'][2][0],-iav['ADX'][2][0]),np.where(adx > iav['ADX'][1][0],iav['ADX'][2][0],0))

    if iav['CCI'][0][0] != None:
        cci = CCI(High,Low,Close,iav['CCI'][0][0])
        cci_sig = np.where(cci > iav['CCI'][1][0],-iav['CCI'][2][0],np.where(cci < iav['CCI'][1][1],iav['CCI'][2][0],0))
        signals = np.add(signals,cci_sig)

    if iav['RSI'][0][0] != None:
        rsi = RSI(Close,iav['RSI'][0][0])
        rsi_sig = np.where(rsi > iav['RSI'][1][0],iav['RSI'][2][0],np.where(rsi < iav['RSI'][1][1],-iav['RSI'][2][0],0))
        signals = np.add(signals,rsi_sig)

    if iav['SSC'][0][0] != None:
        k,d,j = SSC(High,Low,Close,iav['SSC'][0][0])
        ssc_sig = np.add(np.where(k > d,-iav['SSC'][2][0],iav['SSC'][2][0]),np.where(j > k,-iav['SSC'][2][0],iav['SSC'][2][0]))
        signals = np.add(signals,ssc_sig)

    if iav['STC'][0][0] != None: # Not finished
        stc = STC(High,Low,Close,iav['STC'][0][0])

    if iav['BB'][0][0] != None:
        bb = Bollinger_bands(Close,iav['BB'][0][0])
        bb_sig = np.where(Close > bb[:,0],-iav['BB'][2][0],np.where(Close < bb[:,1],iav['BB'][2][0],0))
        signals = np.add(signals,bb_sig)

    if iav['KC'][0][0] != None: # goes along well with bb, very similar
        kc_u,kc_m,kc_l = KC(High,Low,Close,iav['KC'][0][0])
        kc_sig = np.where(Close < kc_l,iav['KC'][2][0],np.where(Close > kc_u,-iav['KC'][2][0],0))
        signals = np.add(signals,kc_sig)

    if iav['ATR'] != None: # Find indicator that goes along with atr close > atr,1,-1?
        atr = ATR(High,Low,Close,iav['ATR'])

    if iav['EMV'][0][0] != None:
        emv = EMV(High,Low,Close,iav['EMV'][0][0])
        emv_sig = np.where(emv > iav['EMV'][1][0],iav['EMV'][2][0],np.where(emv < iav['EMV'][1][1],-iav['EMV'][2][0],0))
        signals = np.add(signals,emv_sig)

    if iav['CMF'][0][0] != None: # not figured out signals yet
        cmf = CMF(Open,High,Low,Close,Volume,iav['CMF'][0][0])
        #CMF_sig = np.where(CMF > 0.05,-signal,np.where(CMF < -0.05,+signal,0))
        #!!CMF_sig = np.where(CMF > 0,iav['CMF'][2][0],-iav['CMF'][2][0])
        signals = np.add(signals,cmf_sig)

    if iav['OBV'][0][0] != None: # not figured out signals yet
        obv = OBV(Close,Volume)
        obv_ma = MA(obv,iav['OBV'])
        #OBV_sig = np.where(OBV > OBV_BB[:,0],+signal,np.where(OBV < OBV_BB[:,1],-signal,0))
        #OBV_sig = np.where(OBV > OBV_ma),-iav['OBV'][2][0],iav['OBV'][2][0])
        signals = np.add(signals,obv_sig)

    if iav['ROC'] != None: # not figured out signals yet
        roc = ROC(Close,iav['ROC'])

    if iav['Ichimoku'][0][0] != None:
        cl,bl,lsb,lsa = ICHIMOKU(High,Low,iav['Ichimoku'][0][0],iav['Ichimoku'][0][1],iav['Ichimoku'][0][3])
        a = np.where((Close > lsa) & (Close > lsb),iav['Ichimoku'][2][0],0)
        b = np.where(lsa > lsb,-iav['Ichimoku'][2][0],iav['Ichimoku'][2][0])
        c = np.where(Close > bl,-iav['Ichimoku'][2][0],iav['Ichimoku'][2][0])
        d = np.where(cl > bl,-iav['Ichimoku'][2][0],iav['Ichimoku'][2][0])
        e = np.where((Close < lsa) & (Close < lsa),iav['Ichimoku'][2][0],0)
        ichimoku_sig1 = np.add(a,b)
        ichimoku_sig2 = np.add(c,d)
        ichimoku_sig3 = np.add(ichimoku_sig1,ichimoku_sig2)
        ichimoku_sig = np.add(e,ichimoku_sig3)
        signals = np.add(signals,ichimoku_sig)


    #HA = heikin(Open,High,Low,Close)

    return signals
def chooser(iav,choice):
    x = 0
    while x < len(choice):
        if choice[x] == 'MACD':
            # indicator variables
            iav[choice[x]][0][0] = random.randrange(100)
            iav[choice[x]][0][1] = random.randrange(100)
            iav[choice[x]][0][2] = random.randrange(100)
            # signal variables
            # None
            # signal strength
            iav[choice[x]][2][0] = random.randrange(4)
        elif choice[x] == 'CCI':
            # indicator variables
            iav[choice[x]][0] = random.randrange(100)
            # signal variables
            iav[choice[x]][1][0] = random.randrange(400)
            iav[choice[x]][1][1] = -random.randrange(400)
            # signal strength
            iav[choice[x]][2][0] = random.randrange(4)
        elif choice[x] == 'RSI':
            # indicator variables
            iav[choice[x]][0] = random.randrange(100)
            # signal variables
            iav[choice[x]][1][0] = random.randrange(100)
            iav[choice[x]][1][1] = random.randrange(100)
            # signal strength
            iav[choice[x]][2][0] = random.randrange(4)
        elif choice[x] == 'PSAR': # not done
            # indicator variables
            iav[choice[x]][0][0] = random.randrange(100)
            iav[choice[x]][0][1] = random.randrange(100) #0.0 - 5.0
            # signal variables
            #iav[choice[x]][1][0] = random.randrange(100)
            #iav[choice[x]][1][1] = random.randrange(100)
            # signal strength
            iav[choice[x]][2][0] = random.randrange(4)
        elif choice[x] == 'STC':
            # indicator variables
            iav[choice[x]][0][0] = random.randrange(100)
            iav[choice[x]][0][1] = random.randrange(100)
            iav[choice[x]][0][2] = random.randrange(100)
            # signal variables
            iav[choice[x]][1][0] = random.randrange(100)
            iav[choice[x]][1][1] = random.randrange(100)
            # signal strength
            iav[choice[x]][2][0] = random.randrange(4)
        elif choice[x] == 'Ichimoku':
            iav[choice[x]][0] = random.randrange(100)
            iav[choice[x]][1] = random.randrange(100)
            iav[choice[x]][2] = random.randrange(100)
            # signal vairables
            # None
            # signal strength
            iav[choice[x]][2][0] = random.randrange(4)
        elif choice[x] == 'EMV':
            # indicator variables
            iav[choice[x]][0][0] = random.randrange(100)
            # singal variables
            iav[choice[x]][1][0] = random.randrange(100) # 0.001 - 0.010
            # signal strength
            iav[choice[x]][2][0] = random.randrange(4)
        elif choice == 'ROC':
            # only signals
            iav[choice[x]][1][0] = random.randrange(500)
            # signal strength
            iav[choice[x]][2][0] = random.randrange(4)

        elif type(choice[x]) == list:
            # indicator variables
            iav[choice[x]][0][0] = random.randrange(100)
            # signal variables
            iav[choice[x]][1][0] = random.randrange(100)
            # signal strength
            iav[choice[x]][2][0] = random.randrange(4)
        else:
            # Indicator no signal variable
            iav[choice[x]][0][0] = random.randrange(100)
            # signal strength
            iav[choice[x]][2][0] = random.randrange(4)

        x += 1
    return iav


def run(numberofsimulations):
    prev_best_capital = 0
    prev_best_mar = 0 
    prev_best_roi = 0 
    while x < numberofsimulations:
        start = timer() 
        # Reset
        # 1. indicator var 2. signal var if possible 3. signal strength
        iav = {
        'MA': [[None],[],[None]],
        'EMA': [[None],[],[None]],
        'MACD':[[None,None,None],[],[None]],
        'PSAR': [[None,None],[],[]], # not done
        'ADX':[[None],[None],[None]],
        'CCI': [[None],[None,None],[None]],
        'RSI':[[None],[None,None],[None]],
        'SSC':[[None],[],[None]],
        'STC':[[None,None,None],[None,None],[None]], # not done
        'BB':[[None],[],[None]],
        'KC':[[None],[],[None]],
        'ATR':[[None],[],[None]], # not done
        'EMV':[[None],[None],[None]],
        'CMF':[[None],[None],[None]],
        'OBV':[[None],[],[None]], # not done
        'ROC':[[None],[None],[None]], # not done
        'Ichimoku':[[None,None,None],[],[None]],
        'leverage':None,'stoploss':None,'el':None,'le':None,'es':None,'se':None,
        'capital':None,'mar':None,'roi':None}
        
        # options
        options = ['MA','EMA','MACD','PSAR','ADX','CCI','RSI','SSC',
        'STC','BB','KC','ATR','EMV','CMF','OBV','ROC','Ichimoku']


        noi = random.randrange(16)
        x = 16
        choice = [] 
        while len(choice) < noi:
            pick = random.randrange(x)
            choice.append(options[pick])
            options.pop(pick)
            x -= 1

        iav = chooser(iav,choice)
        signals = indicators(iav)

        # performance measure conditions
        leverage = 1
        sig_range = round((len(noi)/2))
        el = random.randrange(sig_range)
        le = random.randrange(sig_range)
        es = random.randrange(sig_range)
        se = random.randrange(sig_range)
        
        while leverage < 90:
            
            stop_loss = 1
            
            while stop_loss < 90

        
                #capital,trades,returns,entry_trade,exit_trade = performance_measure(positions[y:],signals[y:],Close[y:],leverage,stop_loss)
                
                scrub,cap,trades,mar,roi = perfomance_measure(Close,signals,el,le,es,se,leverage,stop_loss,time)
                conditions = [None,None,None]

                if scrub != True and trades > 30:
                    if cap > prev_best_capital:
                        conditions[0] = 'cap'
                        prev_best_capital = cap
                    if mar > prev_best_mar:
                        conditions[1] = 'mar'
                        prev_best_mar = mar 
                    if roi > prev_best_roi:
                        conditions[2] = 'roi'
                        prev_best_roi = roi

                    if all(i != None for i in conditions):
                        pathname = f'best{','.join(conditions)}'
                        iav['leverage'] = leverage
                        iav['stoploss'] = stop_loss
                        iav['el'] = el
                        iav['le'] = le
                        iav['es'] = es
                        iav['se'] = se
                        iav['capital'] = cap
                        iav['mar'] = mar 
                        iav['roi'] = roi 
                        pd.DataFrame(iav).to_csv(f'{pathname}.csv')

                stop_loss += 1

            leverage += 1
        end = timer()                                                                                             
        print(end-start)
