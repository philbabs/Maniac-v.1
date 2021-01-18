def perfomance_measure(c,signals,el,le,es,se,leverage,stop_loss,time):
    # initial variables
    x = 0
    init_cap = 1
    cap = init_cap
    trades = 0
    scrub = False
    capital = np.zeros(len(signals)) 

    # run
    while x < len(signals):

        if singals[x] == el:
            # enter long
            trades += 1
            p = c[x]
            pa = (((cap * 0.3) / p)*leverage)
            stop_loss = (p / stop_loss)
            while x < len(signals):
                pos_diff = (((c[x] - p )/ p)*leverage)
                # exit long
                if pos_diff <= -1:
                    cap -= (pa * c[x])
                    capital[x] = cap
                    x += 1
                    break
                if c[x] <= stop_loss:
                    cap += (pa * c[x])
                    capital[x] = cap
                    x += 1
                    break    
                if signals[x] == le:
                    cap += (pa * c[x])
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
            trades += 1
            # enter short
            p = c[x]
            pa = (((cap * 0.3) / p)*leverage)
            stop_loss = (p / stop_loss)
            while x < len(signals):
                pos_diff = (((p - c[x])/ p)*leverage)
                # exit short
                if pos_diff <= -1:
                    cap -= (pa * c[x])
                    capital[x] = cap
                    x += 1
                    break
                if c[x] >= stop_loss:
                    cap += (pa * c[x])
                    capital[x] = cap
                    x += 1
                    break    
                if signals[x] == se:
                    cap += (pa * c[x])
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

    return scrub,cap,trades,mar,roi
