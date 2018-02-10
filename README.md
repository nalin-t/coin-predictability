# Predict future returns of cryptocurrencies
---
**Function:**
Predict future return of cryptocurrencies based on past returns using regressors.


## coin_predictability.py
---
**Procedures:**
* Access cryptocurrency data stored on Amazon RDS MySQL database
* Calculate the average hourly logarithmic returns to be used as features to predict future returns
* Predict future returns using the most optimal regressor and hyperparameters, and the performance is evaluated based on the coefficient of determination


## Results
---
 Consistent with the efficient-market hypothesis and the modern portfolio theory, future returns cannot predicted. Rather, stock market and cryptocurrency prices evolve according to a random walk.

 At any given time, prices fully reflect all information available. Therefore, since 1) all market participants have access to the same information, and 2) market prices are only affected by new information, it is not possible to have an advantage and therefore predict a return.

 This plot is a comparison the predicted returns of some of the cryptocurrencies (BTC, QTUM, XRP, BTG, and OMG) against their actual returns, i.e. labels of the development set. The 1-hour future predictions for each currency (n = 28) were made based on the average logarithmic returns of the last 24 hours of 29 cryptocurrencies, whose minimum market capitalization exceeds USD 1 billion. Each line is the linear regression, and the shaded region is the 68% confidence interval.

 ![](https://github.com/nalin-t/coin-predictability/blob/master/ActualPredictedReturns.png)


This kernel density estimate is a summary of how the predicted returns of all 29 cryptocurrencies compare against their actual returns.  

![](https://github.com/nalin-t/coin-predictability/blob/master/ActualPredictedReturnsDist.png)


In efficient cryptocurrency markets, prices are unpredictable, and investment pattern is not easily discerned, as illustrated in these plots.


### References:
---
* Schulmerich, M. *Applied Asset and Risk Management*.
Springer-Verlag Berlin Heidelberg, 2015
