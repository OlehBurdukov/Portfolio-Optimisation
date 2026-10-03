# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sbn
from pypfopt.efficient_frontier import EfficientFrontier
from pypfopt import risk_models
from pypfopt import expected_returns
import statsmodels.formula.api as smf

# Import of the initial investments data
GBPUSD = pd.read_csv('Client/Proxies/GBP_USD Historical Data.csv', parse_dates = ['Date'], index_col = 'Date', thousands = ',', na_values = 'n/a').sort_index(ascending = True)
sp500 = pd.read_csv('Client/Proxies/S&P 500 Historical Data.csv', parse_dates = ['Date'], index_col = 'Date', thousands = ',', na_values = 'n/a').sort_index(ascending = True)
ftse_allworld = pd.read_csv('Client/Proxies/FTSE All World Historical Data.csv', parse_dates = ['Date'], index_col = 'Date', na_values = 'n/a').sort_index(ascending = True)

# Assignment of returns data
GBPUSD['Return, daily'] = GBPUSD['Price'].pct_change(periods = 1)

sp500['Return, daily'] = (sp500['Price'].pct_change(periods = 1).add(1)).div(GBPUSD['Return, daily'].add(1)).sub(1)
sp500['Log return, daily'] = np.log(sp500['Return, daily'].add(1))
sp500['Cumulative log return, daily'] = np.cumsum(sp500['Log return, daily'])

ftse_allworld['Return, daily'] = (ftse_allworld['Price'].pct_change(periods = 1).add(1)).div(GBPUSD['Return, daily'].add(1)).sub(1)
ftse_allworld['Log return, daily'] = np.log(ftse_allworld['Return, daily'].add(1))
ftse_allworld['Cumulative log return, daily'] = np.cumsum(ftse_allworld['Log return, daily'])

sp500 = sp500.loc['10/14/2013':'09/15/2026'] # FTSE All-world index starts a bit later than 15 years ago. We assume it doesn't corrupt analysis and allign length of both time series

# Assignment of the log returns dataframe for further modelling manipulations
LogReturnsData = pd.DataFrame({'sp500' : sp500['Log return, daily'],
                               'sp500, cumulative' : sp500['Cumulative log return, daily'],
                               'ftse_allworld' : ftse_allworld['Log return, daily'],
                               'ftse_allworld, cumulative' : ftse_allworld['Cumulative log return, daily']})

# Plots of the log returns data
LogReturnsData[['sp500', 'ftse_allworld']].plot(subplots = True)
plt.show()
LogReturnsData[['sp500, cumulative', 'ftse_allworld, cumulative']].plot(kind = 'line')
plt.show()

# Analysis of descriptive statistics
statistics1 = LogReturnsData[['sp500', 'ftse_allworld']].describe()
print(statistics1)
statistics2 = LogReturnsData[['sp500', 'ftse_allworld']].agg(['median', 'var', 'skew', 'kurtosis'])
print(statistics2)

annual_mean = np.exp(statistics1.loc['mean'].mul(252)).sub(1)
print('annual mean values are:\n', annual_mean)

annual_std = statistics1.loc['std'].mul(np.sqrt(252))
print('annual standard deviation values are:\n', annual_std)

# Plot of distribution
plt.hist(x = LogReturnsData[['sp500', 'ftse_allworld']], 
         bins = 50, 
         density = True, 
         alpha = 0.5, 
         label = ['Distribution of Vanguard S&P500', 'Distribution of Vanguard FTSE World'])
plt.xlabel('Daily log returns')
plt.ylabel('Density')
plt.title('Distribution of both ETFs')
plt.grid(True)
plt.legend()
plt.show()

# Correlation
correlation = LogReturnsData[['sp500', 'ftse_allworld']].corr()
sbn.heatmap(correlation,
            annot = True,
            cmap = 'YlGnBu',
            vmin = -1, vmax = 1)
plt.show()

# Construction of portfolio for further comparison purposes
risk_free = 0.0375
PriceData = pd.DataFrame({'sp500' : sp500['Price'].div(GBPUSD['Price']),
                          'ftse_allworld' : ftse_allworld['Price'].div(GBPUSD['Price'])})

mu = expected_returns.mean_historical_return(PriceData)
Sigma = risk_models.sample_cov(PriceData)

EF_undiv = EfficientFrontier(mu, Sigma)
EF_undiv.set_weights({'sp500' : 0.1354,
                      'ftse_allworld' : 0.8646})

EF_undiv.portfolio_performance(verbose = True, risk_free_rate = risk_free)
cleaned_weights = EF_undiv.clean_weights()
print(cleaned_weights)

# We now observe the substitute asset, which is gold ETC.

# Import of the initial investments data
sp500 = pd.read_csv('Client/Proxies/S&P 500 Historical Data.csv', parse_dates = ['Date'], index_col = 'Date', thousands = ',', na_values = 'n/a').sort_index(ascending = True) # reimport of this dataset due to it being cut during previous analysis part
gold = pd.read_csv('Client/Proxies/XAU_USD Historical Data.csv', parse_dates = ['Date'], index_col = 'Date', thousands = ',', na_values = 'n/a').sort_index(ascending = True)

# Assignment of returns data
sp500['Return, daily'] = (sp500['Price'].pct_change(periods = 1).add(1)).div(GBPUSD['Return, daily'].add(1)).sub(1)
sp500['Log return, daily'] = np.log(sp500['Return, daily'].add(1))
sp500['Cumulative log return, daily'] = np.cumsum(sp500['Log return, daily'])

gold['Return, daily'] = (gold['Price'].pct_change(periods = 1).add(1)).div(GBPUSD['Return, daily'].add(1)).sub(1) # Here we address the mentioned issue about Gold GBP Spot. Considering we have indirect quote in the data, we need to divide the log transformed quote by log returns.
gold['Log return, daily'] = np.log(gold['Return, daily'].add(1)) 
gold['Cumulative log return, daily'] = np.cumsum(gold['Log return, daily'])

# Assignment of the log returns dataframe for further modelling manipulations
LogReturnsData = pd.DataFrame({'sp500' : sp500['Log return, daily'],
                               'sp500, cumulative' : sp500['Cumulative log return, daily'],
                               'gold' : gold['Log return, daily'],
                               'gold, cumulative' : gold['Cumulative log return, daily']})

# Plots of the log returns data
LogReturnsData[['sp500', 'gold']].plot(subplots = True)
plt.show()
LogReturnsData[['sp500, cumulative', 'gold, cumulative']].plot(kind = 'line')
plt.show()

# # Analysis of descriptive statistics
statistics1 = LogReturnsData[['sp500', 'gold']].describe()
print(statistics1)
statistics2 = LogReturnsData[['sp500', 'gold']].agg(['median', 'var', 'skew', 'kurtosis'])
print(statistics2)

annual_mean = np.exp(statistics1.loc['mean'].mul(252)).sub(1)
print('annualised mean return values are:\n', annual_mean)

annual_std = statistics1.loc['std'].mul(np.sqrt(252))
print('annualised standard deviation values of returns are:\n', annual_std)

# Plot of distributions
plt.hist(x = LogReturnsData[['sp500', 'gold']],
         bins = 50,
         density = True,
         alpha = 0.5,
         label = ['Distribution of S&P500', 'Distribution of XAU/GBP Gold Spot'])
plt.xlabel('Daily log returns')
plt.ylabel('Density')
plt.title('Distribution of both assets')
plt.grid(True)
plt.legend()
plt.show()

# Correlation
correlation = LogReturnsData[['sp500', 'gold']].corr()
sbn.heatmap(correlation,
            annot = True,
            cmap = 'YlGnBu',
            vmin = -1, vmax = 1)
plt.show()

# Construction of portfolio
PriceData = pd.DataFrame({'sp500' : sp500['Price'].div(GBPUSD['Price']),
                          'gold' : gold['Price'].div(GBPUSD['Price'])})

mu = expected_returns.mean_historical_return(PriceData)
Sigma = risk_models.sample_cov(PriceData)

EF_div = EfficientFrontier(mu, Sigma)
EF_div.max_sharpe(risk_free)

EF_div.portfolio_performance(verbose = True, risk_free_rate = risk_free)
cleaned_weights = EF_div.clean_weights()
print(cleaned_weights)

# Market risk Analysis
LogReturnsData['Benchmark log return, daily'] = LogReturnsData['sp500']
# Given sufficiently small difference between approaches, we can immediately estimate the excess return from log returns
LogReturnsData['Excess Benchmark log return, daily'] = LogReturnsData['Benchmark log return, daily'].sub(np.log(1 + risk_free)/252)

LogReturnsData['Portfolio log return, daily'] = np.log(sp500['Return, daily'].mul(cleaned_weights['sp500']).add(gold['Return, daily'].mul(cleaned_weights['gold'])).add(1))
LogReturnsData['Excess Portfolio log return, daily'] = LogReturnsData['Portfolio log return, daily'].sub(np.log(1 + risk_free)/252)

model = smf.ols(formula = 'Q("Excess Portfolio log return, daily") ~ Q("Excess Benchmark log return, daily")', data = LogReturnsData)
fit = model.fit()

print(fit.summary())
pf_beta = fit.params['Q("Excess Benchmark log return, daily")']
print(pf_beta)

# Our linear model tells the following: beta factor of the portfolio is 0.8245; This is a good result, considering the fact our benchmark is actually the proxy analysed on one of our assets stead. In other words, diversification strategy has proven beneficial, as our beta has dropped to this value from value 1 
# The model also indicates adjusted R-squared on level of 97.1% which is surprisingly high

# Plot of linear regression
sbn.regplot(data = LogReturnsData,
            x = 'Excess Benchmark log return, daily',
            y = 'Excess Portfolio log return, daily',
            line_kws = {'color' : 'red'},
            scatter_kws = {'alpha' : 0.6})
plt.show()

# Value at Risk analytics
# Let's define VaR and CVaR metric for our portfolio
weights = np.array(list(cleaned_weights.values()))

pf_mean = np.dot(weights.T, mu)
pf_std = np.sqrt(np.dot(weights.T, np.dot(Sigma, weights)))
T = 252

simulated_returns = []
for i in range(5000):
    random_returns = np.random.normal(pf_mean, pf_std, T)
    simulated_returns.append(random_returns)

simulated_returns = np.array(simulated_returns).T

VaR_90 = np.percentile(simulated_returns, 100 - 90)
VaR_95 = np.percentile(simulated_returns, 100 - 95)
VaR_99 = np.percentile(simulated_returns, 100 - 99)

CVaR_90 = simulated_returns[simulated_returns < VaR_90].mean()
CVaR_95 = simulated_returns[simulated_returns < VaR_95].mean()
CVaR_99 = simulated_returns[simulated_returns < VaR_99].mean()

# Plot portfolio distribution with VaR and CVaR sets
np.exp(LogReturnsData['Portfolio log return, daily']).sub(1).plot(kind = 'density',
                                                                  rot = 45,
                                                                  title = 'VaR-CVaR metrics of the portfolio')
plt.axvline(x = VaR_90, color = 'yellow', linestyle = '-', label = 'VaR 90%')
plt.axvline(x = VaR_95, color = 'pink', linestyle = '-', label = 'VaR 95%')
plt.axvline(x = VaR_99, color = 'red', linestyle = '-', label = 'VaR 99%')
plt.axvline(x = CVaR_90, color = 'yellow', linestyle = '--', label = 'CVaR 90%')
plt.axvline(x = CVaR_95, color = 'pink', linestyle = '--', label = 'CVaR 95%')
plt.axvline(x = CVaR_99, color = 'red', linestyle = '--', label = 'CVaR 99%')

plt.xlabel('Daily portfolio returns')
plt.ylabel('Density')
plt.title('VaR-CVaR metrics of the portfolio')
plt.grid(True)
plt.legend()
plt.show()

# Plot the whole Monte Carlo process
# plt.plot(range(T), simulated_returns)
# plt.show()