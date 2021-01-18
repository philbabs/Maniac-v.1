import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# Tools to calculate indicators
def MA(data,dp):
    ma = data.rolling(dp).mean()
    return ma
def ema(data,dp):
    a = (2/(dp+1))
    b = np.multiply(data,a)
    c = (1 - a)
    d_p = np.append([0],data[:-1])
    EMA = np.add(np.multiply(data,a),np.multiply(d_p,np.subtract(1,a)))
    return EMA
    '''
    a = (2/(dp+1))
    b = np.multiply(data,a)
    c = (1 - a)
    EMAs = [4]
    x = 1
    
    while x < len(data):
        EMA = (b[x] + (EMAs[x-1] * c))
        EMAs.append(EMA)
        x += 1  
    '''
    '''

    l = tp
    EMA,EMAs,m,x = 0,[0],(2 / (l+1)),1
    while x < len(data):
        cur_price = data[x]
        prev_EMA = EMAs[x-1]
        EMA = ((cur_price * m) + (prev_EMA * (1 - m)))
        EMAs.append(EMA)
        x += 1
    '''
    #return EMAs
def EMA(data,dp):
    
    a = (2/(dp+1))
    b = np.multiply(data,a)
    c = (1 - a)
    d_p = np.append([0],data[:-1])
    EMA = np.add(np.multiply(data,a),np.multiply(d_p,np.subtract(1,a)))
    return EMA
    '''
    
    a = (2/(dp+1))
    b = np.multiply(data,a)
    c = (1 - a)
    EMAs = [4]
    x = 1
    
    while x < len(data):
        EMA = (b[x] + (EMAs[x-1] * c))
        EMAs.append(EMA)
        x += 1  
    '''
    
    '''
    l = dp
    EMA,EMAs,m,x = 0,[0],(2 / (l+1)),1
    while x < len(data):
        cur_price = data[x]
        prev_EMA = EMAs[x-1]
        EMA = ((cur_price * m) + (prev_EMA * (1 - m)))
        EMAs.append(EMA)
        x += 1
    '''
    return EMAs
def smooth(data,dp):
    return pd.DataFrame(data).rolling(dp).mean().to_numpy().flatten()
    '''
    if type(data) == list:
        arr = np.array(data)

        rolling_mean = (np.convolve(arr,np.ones(dp+1),'valid') / dp)

        dp_arr = np.zeros(dp)
    
        result = np.append(dp_arr,rolling_mean,axis=0)
    else:
        rolling_mean = (np.convolve(data,np.ones(dp+1),'valid') / dp)

        dp_arr = np.ones(dp)
    
        result = np.append(dp_arr,rolling_mean,axis=0)
    #return result.reshape((len(result)),1)
    return result
    '''
# Trend indicators 
def MACD(data,dp1,dp2,dp3=9):
    MACD,MACDs = 0,[]
    a = np.array(EMA(data,dp1))
    b = np.array(EMA(data,dp2))
    macd = np.subtract(a,b)
    sl = np.array(EMA(macd,dp3))
    return macd,sl
def PSAR(data,sens,dp): # not finished somethings wrong

    idx = np.arange(dp)+np.arange(len(data))[:,None] # indexes of subRanges
    idx = np.minimum(len(data)-1,idx)
    arr = np.array(data[idx])
    sub_arr = np.where(arr[:,0] < arr[:,dp-1],1,-1)
    EPs = np.where(sub_arr == 1,np.max(arr,axis=1),np.min(arr,axis=1))
    #np.linspace(0,sens,2)
    #AFs_l = np.where(sub_arr == 1,,0)
    #AFs_s = np.where(sub_arr == -1,,0)
    # NO FOR LOOPS
    a = np.array(np.zeros(len(data)))
    x = 0
    y = 0
    for i in sub_arr:
        if sub_arr[x-1] != i:
            y = 0
        else:
            a[x] = y
        if round(y) < 0.2:
            y += sens
        x += 1

    PSAR = []    
    if EPs[0] < EPs[5]:
        start = max(EPs[:5])
        PSAR.append(start)
    else:
        start = min(EPs[:5])
        PSAR.append(start)
    x = 1
    while x < len(data):
        if sub_arr[x] == 1:

            PSAR.append(PSAR[x-1] + (a[x-1] * (EPs[x-1] - PSAR[x-1])))
        else:
            
            PSAR.append(PSAR[x-1] - (a[x-1] * (PSAR[x-1]- EPs[x-1])))
        x += 1 
    
    return PSAR
    '''
    EP = []
    ​AF = sens
    
    RPSAR = Prior PSAR + [Prior AF(Prior EP-Prior PSAR)]
    
    FPSAR = Prior PSAR − [Prior AF(Prior PSAR-Prior EP)]
    
    where:
    RPSAR = Rising PSAR
    AF = Acceleration Factor, it starts at 0.02 and
    increases by 0.02, up to a maximum of 0.2, each
    time the extreme point makes a new low (falling
    SAR) or high(rising SAR)
    FPSAR = Falling PSAR
    EP = Extreme Point, the lowest low in the current
    downtrend(falling SAR)or the highest high in the
    current uptrend(rising SAR)

    The indicator can also be used used to set stop loss orders.
     This can be achieved by moving the stop loss to match the 
     level of the SAR indicator.
    
    # Uptrend PSAR = Prior PSAR + Prior AF (Prior EP - Prior PSAR)
    # Downtrend PSAR = Prior PSAR - Prior AF (Prior PSAR - Prior EP)
    PSAR,PSARs,AF,AFs,EPs,step = 0,[0],0,[0],[0],0
    Uptrend,Downtrend = False,False

    x = 0
    y = 1
    while len(EPs) < 5:
        if data[x] > EPs[y-1]:
            EPs.append(data[x])
            y += 1
        x += 1
    x = 1
    PSAR = EPs[-1]
    PSARs.append(PSAR)
    AF = (EPs[y-1] - PSARs[x-1])
    AFs.append(AF)
    if EPs[y-1] > EPs[y-2]:

        Uptrend = True
    else:

        Downtrend = True

    while True:
        while Uptrend:
            if x < len(data-10):
                if step >= 0.2:
                    step = 0
                    Uptrend,Downtrend = False,True
                    break

                if data[x] > EPs[y-1]:
                    EPs.append(data[x])
                    y += 1
                    step += sens
                AF = (EPs[y-1] - PSARs[x-1])
                AFs.append(AF)
                
                PSAR = (PSARs[x-1] + AFs[x-1])
                PSARs.append(PSAR)
                
                x += 1
                
            else:

                PSARs.pop()
                return PSARs    
                
            
            
        while Downtrend:
            if x < len(data-10):
                if step >= 0.2:
                    step = 0
                    Uptrend,Downtrend = True,False
                    break
                if data[x] > EPs[y-1]:
                    EPs.append(data[x])
                    y += 1
                    step += sens
                AF = (EPs[y-1] - PSARs[x-1])
                AFs.append(AF)
                PSAR = (AFs[x-1] - PSARs[x-1])

                PSARs.append(PSAR)
                
             
                x += 1
                
            else:
                print('out')
                PSARs.pop()
                return PSARs
        '''         
def ADX(high,low,close,dp):
    ''' 
    h = np.append([0],high)
    h_p = np.append(high,[0])
    l = np.append([0],low)
    l_p = np.append(low,[0])
    '''
    h = high
    h_p = np.append([0],high[:-1])
    l = low
    l_p = np.append([0],low[:-1])
    a = np.subtract(h,h_p)
    a = np.where(a == 0,1,a)
    b = np.subtract(l_p,l)
    b = np.where(b == 0,1,b)
    #atr = np.append([0],ATR(high,low,close,dp))
    atr = ATR(high,low,close,dp)
    atr = np.where(atr == 0,None,atr)
    atr = np.array(pd.DataFrame(atr).bfill()).flatten()
    plus_DM = np.where(a > b,a,0.001)
    minus_DM = np.where(b > a,b,0.001)
    plus_DM14 = smooth(plus_DM,dp)
    minus_DM14 = smooth(minus_DM,dp)
    plus_DI = np.multiply(np.divide(plus_DM14,atr),100) #This +DI14 is the green Plus Directional Indicator line (+DI) that is plotted along with the ADX line.
    minus_DI = np.multiply(np.divide(minus_DM14,atr),100) #This -DI14 is the red Minus Directional Indicator line (-DI) that is plotted along with the ADX line.
    ADX = smooth(np.multiply(np.divide(np.absolute(np.subtract(plus_DI,minus_DI)),np.add(plus_DI,minus_DI)),100),dp)
    return ADX,minus_DI,plus_DI
# Momentum indicators
def CCI(h,l,c,dp):
    typPrices = np.divide(np.add(np.add(h,l),c),3)
    smatp = smooth(typPrices,dp)
    CCI = ((typPrices - smatp) / (0.015*(pd.DataFrame(typPrices).mad().to_numpy().flatten()/20)))
    #CCI = (Typical Price  -  20-period SMA of TP) / (.015 x Mean Deviation)
    return CCI
def RSI(close,dp):
    close_previous = np.append([0],close[:-1])
    diff = np.array((close - close_previous))
    Up_prices = np.where(diff > 0,diff,0.000001) # Up prices
    Down_prices = np.where(diff < 0,diff,0.00001) # Down prices
    avg_gain = np.array(smooth(Up_prices,dp)) 
    avg_loss = np.array(abs(smooth(Down_prices,dp)))
    RSI = np.array((100 - (100/(1 + avg_gain / avg_loss))))
    return RSI
    '''indicators_df['RSI'] = RSI
    df_signals['RSI_SHORT'] = np.where(indicators_df['RSI'] > 70,-1,0)
    df_signals['RSI_LONG'] = np.where(indicators_df['RSI'] < 30,1,0)
    return indicators_df,df_signals
    
    An asset is usually considered overbought
    when the RSI is above 70% and oversold when it is below 30%.
    '''
def SSC(h,l,c,dp): # Stochastic often used with RSI
    '''
    def calc(data,p,hol):
        a = np.zeros(len(data))
        x = 0
        if hol == True:
            while x < len(a):
                b = max(data[x:x+p])
                a[x] = b
                x += 1
        elif hol == False:
            while x < len(a):
                b = min(data[x:x+p])
                a[x] = b
                x += 1
        else:
            pass
        return a
    '''
    def calc(data,dp,HoL):
        a = np.zeros(len(data))
        idx = np.arange(dp)+np.arange(len(data))[:,None] # indexes of subRanges
        idx = np.minimum(len(data)-1,idx)               # don't overflow indexes
        if HoL == True:
            rollingMax = np.max(data[idx],axis=1) # apply maximums on every subrange
        elif HoL == False:
            rollingMax = np.min(data[idx],axis=1) # apply maximums on every subrange
        return rollingMax
    hh = calc(h,dp,True)
    ll = calc(l,dp,False)
    k = np.multiply(np.divide(np.where(np.subtract(c,ll) == 0,1,np.subtract(c,ll)),np.where(np.subtract(hh,ll)== 0,1,np.subtract(hh,ll))),100)
    d = smooth(k,3)
    j = np.subtract(np.multiply(3,k),np.multiply(2,d))
    return k,d,j
def STC(high,low,close,s,l,dp):
    c,nan = MACD(close,s,l)
    k,d,j = SSC(c,c,c,dp)
    D = np.array(EMA(k,dp))
    kd,d,j = SSC(D,D,D,3) 
    STC = np.array(EMA(kd,3))
    return STC
def heikin(o,h,l,c):
    o_p = np.append([0],o[:-1])
    c_p = np.append([0],c[:-1])
    Close = np.divide(np.add(np.add(o,c),np.add(l,c)),4).reshape(len(o),1) 
    Open = np.divide(np.add(o_p,c_p),2).reshape(len(o),1) 
    #High = np.max(Open,h,l,Close,axis=1)
    #Low = np.min(Open,h,l,Close,axis=1)
    High = np.array([Open,h.reshape(len(o),1),l.reshape(len(o),1),Close]).max(0)#.reshape(len(o),1) 
    pd.DataFrame(High).to_csv('High.csv') 
    Low = np.array([Open,h.reshape(len(o),1),l.reshape(len(o),1),Close]).min(0).reshape(len(o),1)
  
    #heikin = np.array([Open,High,Low,Close])
    heikin = np.append(np.append(Open,High,axis=1),np.append(Low,Close,axis=1),axis=1)
    return heikin
# Volatility indicators 
def Bollinger_bands(data,dp):

    ma = pd.DataFrame(data).rolling(dp).mean().to_numpy()
    std = pd.DataFrame(data).rolling(dp).std().to_numpy()
    upper_band = np.array([ma + (std*2)]).reshape(len(data),1)
    lower_band = np.array([ma - (std*2)]).reshape(len(data),1)
    bollinger_bands = np.append(upper_band,lower_band,axis=1)
    return bollinger_bands,ma.flatten()
def KC(h,l,c,dp):
    #although the multiplier can also be adjusted based on personal preference. A larger multiplier will result in a wider channel.
    b = np.array(EMA(c,dp))
    #print(len(b))
    d = ATR(h,l,c,dp)
    #print(len(d))
    a = np.add(b,np.multiply(2,d))
    c = np.subtract(b,np.multiply(2,d))
    return a,b,c
def ATR(high,low,close,dp):
    '''
    If you're long, then minus X ATR from the highs and that's your trailing stop loss.
    If you're short, then add X ATR from the lows and that's your trailing stop loss.
    ''' 

    c_p = np.append([0],close[:-1])
    a = np.subtract(high,low)
    b = np.abs(np.subtract(high,c_p))
    c = np.abs(np.subtract(low,c_p))
    ATRs = smooth(np.maximum(a,b,c),dp)
    #ATRs = MA(pd.DataFrame(np.maximum(a,b,c)),dp).to_numpy().flatten()
    return ATRs
    '''
    TR,TRs,ATRs = 0,[0],[]
    x = 1
    while x < len(close):
        h,l,p_c = high[x],low[x],close[x-1]
        TR = max((h-l),abs(h-p_c),abs(l-p_c))
        TRs.append(TR)
        x += 1
    #TRs = pd.DataFrame(TRs)
    #ATRs = TRs.rolling(tp).mean()
    ATRs = smooth(TRs,tp)
    return ATRs
    '''
# Volume indicators 
def EMV(h,l,c,v,dp):
    phigh = np.append([1],h[:-1])
    high = h
    plow = np.append([1],l[:-1])
    low = l
    vol = v
    sub = np.subtract(high,low)
    sub = np.where(sub == 0,1,sub)
    dm = np.subtract(np.divide(np.add(high,low),2),np.divide(np.add(phigh,plow),2))
    boxr = np.divide(vol,sub)
    boxr = np.where(boxr == 0,1,boxr)
    dm = np.where(dm == 0,1,dm)
    EMV = smooth(np.divide(dm,boxr),dp)
    
    '''
    high_plus_low = np.add(high,low)
    #print(high_plus_low)
    phigh_plus_plow = np.add(phigh,plow)
    #print(phigh_plus_plow)

    movement = np.where(np.subtract(np.divide(high_plus_low,2),np.divide(phigh_plus_plow,2)) == 0,1,np.subtract(np.divide(high_plus_low,2),np.divide(phigh_plus_plow,2)))
    #print(movement)
    boxr = np.where(np.divide(np.divide(vol,1000000),np.where(np.subtract(high,low) == 0,1,np.subtract(high,low))) == 0,1,np.divide(np.divide(vol,1000000),np.where(np.subtract(high,low) == 0,1,np.subtract(high,low))))
    EMV = np.divide(movement,boxr)
    '''
    return EMV
def CMF(o,h,l,c,v,dp):# AD stands for Accumulation Distribution
    cl = np.where(np.subtract(c,l) == 0,1,np.subtract(c,l))
    hc = np.where(np.subtract(h,c) == 0,1,np.subtract(h,c))
    hl = np.where(np.subtract(h,l) == 0,1,np.subtract(h,l))
    multiplier = np.divide(np.subtract(cl,hc),hl)
    multiplier = np.where(multiplier == 0,1,multiplier)
    volume = np.multiply(multiplier,v)
    CMF = np.divide(smooth(multiplier,dp),smooth(volume,dp))
    return CMF
def OBV(close,volume): # Bolls fast numpy code.
    '''
    close = close.to_numpy()
    volume = volume.to_numpy()
    volume[0] = 0.0
    
    obv = np.where(close > np.roll(close, 1), volume,
          np.where(close < np.roll(close, 1), -volume,0)).cumsum()

    obv_sma = pd.DataFrame(obv).rolling(14).mean()
    return obv,obv_sma
    
    #CUSTOM INDICATOR? - NOT OBV
    c_p = np.append([0],close[:-1])
    obv = np.where(close > c_p,+volume,np.where(close < c_p,-volume,0))
    pd.DataFrame(obv).to_csv('obv.csv')
    return obv
    '''
    #CUSTOM INDICATOR? - NOT OBV
    c_p = np.append([0],close[:-1])
    obv = np.where(close > c_p,+volume,np.where(close < c_p,-volume,0)).cumsum()

    return obv
    '''
    obv_arr = np.array([])
    obv = 0
    i = 1
    while i < len(close):
        c_p = close[i-1]
        c_c = close[i]
        c_v = volume[i]
        
        if c_c > c_p:
            obv += c_v
            #indicators_df['OBV'][i] = obv
            obv_arr = np.append(obv_arr,obv)
        elif c_c < c_p:
            obv -= c_v
            #indicators_df['OBV'][i] = obv
            obv_arr = np.append(obv_arr,obv)
        else:
            #indicators_df['OBV'][i] = obv
            obv_arr = np.append(obv_arr,obv)
        #print(i)
        i += 1
    return obv_arr
    '''    
def ROC(data,dp): # fast.
    cp = np.append(np.ones(dp),data[0:-dp])
    ROC = np.multiply(np.divide(np.subtract(data,cp),cp),100)
    return ROC
    '''
    n_close = np.append(data,np.zeros(200))
    c_close = np.append(np.zeros(200),data)
    close = data#.to_numpy()
    ROCs = np.array([])
    #n = np.array(data[range(1,len(data),200)]) Useful to know that you can do this with arrays
    n = 200
    x = 200
    print(len(close))
    while x < len(close):
        ROC = ((close[x] - close[x-n]) / (close[x-n]) * 100)
        ROCs = np.append(ROCs,ROC)
        print(x)
        x += 1
    return ROCs
    '''
def WHAT_CMF(o,h,l,c,v,dp):# AD stands for Accumulation Distribution
    clop = np.subtract(c,o)
    clop = np.where(clop == 0,0.005,clop)
    hilo = np.subtract(h,l)
    hilo = np.where(hilo == 0,4,hilo)
    division = np.divide(clop,hilo)
    AD = np.multiply(v,division)
    #CMF = np.divide(np.sum([AD,period]),np.sum([v,period]))
    CMF = np.divide(np.add(AD,dp),np.add(v,dp))
    return CMF
# Ichimoku indicator 
def ICHIMOKU(h,l,dp1,dp2,dp3):
    def high_low(h,l,p):
        '''
        a = np.zeros(len(h))
        x = 0
        while x < (len(h)):
            b = max(h[x:x+p])
            c = min(l[x:x+p])
            a[x] = ((b+c)/2)
            x += 1
        return a
        '''
        
        ''' 
        DO NOT DELETE - EVEN FASTER
        def rollingMax2(data,window=9):
    		result = data.copy()
    		for offset in range(1,window):
        		result[:-1] = np.maximum(result[:-1],result[1:])
    		return result
    	'''
        a = np.zeros(len(h))
        idx = np.arange(p)+np.arange(len(h))[:,None] # indexes of subRanges
        idx = np.minimum(len(h)-1,idx)
        hh = np.max(h[idx],axis=1) # apply maximums on every subrange  
        ll = np.min(l[idx],axis=1) # apply maximums on every subrange
        a = np.divide(np.add(hh,ll),2)
        return a
    CL = high_low(h,l,dp1)
    BL = high_low(h,l,dp2)
    LSB = high_low(h,l,dp3)
    LSA = np.divide(np.add(CL,BL),2)
    print('done')
    return CL,BL,LSB,LSA
# Custom indicators
def WBA():

    pass
def CUSTOM_OBV():
    #CUSTOM INDICATOR? - NOT OBV
    c_p = np.append([0],close[:-1])
    obv = np.where(close > c_p,+volume,np.where(close < c_p,-volume,0))
    pd.DataFrame(obv).to_csv('obv.csv')
    return obv
# Other Tools
def stop_loss(): # never risk more than 2% of your account
    pass
