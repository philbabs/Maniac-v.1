def perfomance_measure(c,signals,el,le,es,se,size,leverage,stop_loss,time):
    # initial variables
    x = 0
    init_cap = 1
    cap = init_cap
    trades = 0
    scrub = False
    capital = np.zeros(len(signals)) 

    # run
    while x < (len(signals)-5):

        if signals[x] == el:
            # enter long

            # trade vars 
            trades += 1
            #entry price
            p = c[x]
            # amount in usd
            usd = ((cap * size)*leverage)
            startFee = (usd * 0.1)
            cap -= (usd + startFee)
            # amount in btc
            btc = (usd / p)
            # stop loss
            sl = (p - (p - (p * stop_loss)))
            
            # info
            print(f'type: Long - entry price: {p} - stop loss: {sl} - usd: {usd} - btc: {btc}')

            while x < (len(signals)-5):

                diff = (((c[x] - p )/ p)*leverage)
                
                # exit long

                if diff <= -1: # liquidation
                    print(f'!LIQUIDATED! - pos diff: {diff}')
                    capital[x] = cap
                    x += 1
                    break

                if c[x] <= sl:  # Stop loss
                    print(f'stop loss reached - {sl} - return: {btc * c[x]}')
                    usd = (btc * c[x])
                    cap += (usd - (usd * 0.1))
                    capital[x] = cap
                    x += 1
                    break

                if signals[x] == le: # Exit
                    print(f'Exited trade: Long - exit price: {c[x]} - return: {btc * c[x]}')
                    usd = (btc * c[x])
                    cap += (usd - (usd * 0.1))
                    capital[x] = cap
                    x += 1  
                    break

                else:
                    capital[x] = cap
                    x += 1

        if init_cap < 0.003:
            scrub = True
            break

        if signals[x] == es:
            # enter short

            # trade vars 
            trades += 1
            #entry price
            p = c[x]
            # amount in usd
            usd = ((cap * size)*leverage)
            startFee = (usd * 0.1)
            cap -= (usd + startFee)
            # amount in btc
            btc = (usd / p)
            # stop loss
            sl = (p + (p - (p * stop_loss)))
            
            # info
            print(f'type: Short - entry price: {p} - stop loss: {sl} - usd: {usd} - btc: {btc}')

            while x < (len(signals)-5):

                diff = (((p - c[x])/ p)*leverage)
                
                # exit long

                if diff <= -1: # liquidation
                    print(f'!LIQUIDATED!')
                    capital[x] = cap
                    x += 1
                    break

                if c[x] >= sl:  # Stop loss
                    print(f'stop loss reached - {sl} - return: {btc * c[x]}')
                    usd = (btc * c[x])
                    cap += (usd - (usd * 0.1))
                    capital[x] = cap
                    x += 1
                    break

                if signals[x] == se: # Exit
                    print(f'Exited trade: Short - exit price: {c[x]} - return: {btc * c[x]}')
                    usd = (btc * c[x])
                    cap += (usd - (usd * 0.1))
                    capital[x] = cap
                    x += 1  
                    break

                else:
                    capital[x] = cap
                    x += 1
                
        else:
            capital[x] = cap
            x += 1

    # Cagr
    capital_pd = pd.DataFrame(capital)
    capital_pd.to_csv('capital.csv')
    cagr = (((cap / init_cap) ** (1/time)) - 1)
    # maxdd
    window = len(signals)
    Roll_Max = capital_pd.rolling(window, min_periods=1).max()
    drawdown = np.subtract(np.divide(capital_pd,Roll_Max),1.0)
    maxdd = drawdown.rolling(window, min_periods=1).min()
    # mar
    mar = (cagr/maxdd)
    # roi
    roi = (cap/init_cap)
    #print(f'capital: {cap} - mar: {mar} - roi: {roi}')
    print(trades)

    return scrub,capital,trades,mar,roi
