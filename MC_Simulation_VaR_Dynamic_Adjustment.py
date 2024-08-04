import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import datetime as dt
from pandas_datareader import data as pdr
import yfinance as yf
from scipy.optimize import minimize
import time

# Importing data from Yahoo Finance
def get_data(tickers, start_date, end_date):
    yf.pdr_override()
    stock_data = pdr.get_data_yahoo(tickers, start_date, end_date)
    stock_data = stock_data['Close']
    returns = stock_data.pct_change()
    mean_returns = returns.mean()
    cov_matrix = returns.cov()
    return stock_data, mean_returns, cov_matrix

# Define stock list and initial time period
stock_list = ['RELIANCE.NS', 'HDFCBANK.NS', 'INFY.NS', 'TCS.NS']
end_date = dt.datetime.now()
start_date = end_date - dt.timedelta(days=300)

stock_data, mean_returns, cov_matrix = get_data(stock_list, start_date, end_date)

# Initial random weights and portfolio settings
weights = np.random.random(len(stock_list))
weights /= np.sum(weights)

# Monte Carlo Simulation settings
mc_sims = 10000
T = 100
initial_portfolio_value = 10000
VaR_threshold = 0.05  # 5% of initial portfolio value

# Helper functions for VaR and CVaR
def mcVaR(returns, alpha=5):
    if isinstance(returns, pd.Series):
        return np.percentile(returns, alpha)
    else:
        raise ValueError('Expected a pandas Series')

def mcCVaR(returns, alpha=5):
    if isinstance(returns, pd.Series):
        belowVaR = returns <= mcVaR(returns, alpha=alpha)
        return returns[belowVaR].mean()
    else:
        raise ValueError('Expected a pandas Series')

# Function to calculate portfolio VaR given weights
def portfolio_var(weights, mean_returns, cov_matrix, initial_value, mc_sims=10000, alpha=5):
    meanM = np.full(shape=(T, len(weights)), fill_value=mean_returns).T
    portfolio_sims = np.full(shape=(T, mc_sims), fill_value=0.0)

    for m in range(mc_sims):
        Z = np.random.normal(size=(T, len(weights)))
        L = np.linalg.cholesky(cov_matrix)
        daily_returns = meanM + np.inner(L, Z)
        portfolio_sims[:, m] = np.cumprod(np.inner(weights, daily_returns.T) + 1) * initial_value

    port_results = pd.Series(portfolio_sims[-1, :])
    return initial_value - mcVaR(port_results, alpha=alpha)

# Optimize portfolio weights to minimize VaR
def optimize_weights(mean_returns, cov_matrix, initial_value, mc_sims=10000, alpha=5):
    constraints = ({'type': 'eq', 'fun': lambda weights: np.sum(weights) - 1})
    bounds = tuple((0, 1) for _ in range(len(stock_list)))
    
    result = minimize(portfolio_var, weights, args=(mean_returns, cov_matrix, initial_value, mc_sims, alpha),
                      method='SLSQP', bounds=bounds, constraints=constraints)
    return result.x

# Function to update data and re-optimize portfolio
def update_portfolio():
    global stock_data, mean_returns, cov_matrix, weights, initial_portfolio_value
    end_date = dt.datetime.now()
    start_date = end_date - dt.timedelta(days=300)
    
    stock_data, mean_returns, cov_matrix = get_data(stock_list, start_date, end_date)
    weights = optimize_weights(mean_returns, cov_matrix, initial_portfolio_value)

# Live data simulation
portfolio_values = []

for day in range(T):
    update_portfolio()
    
    meanM = np.full(shape=(T, len(weights)), fill_value=mean_returns).T
    portfolio_sims = np.full(shape=(T, mc_sims), fill_value=0.0)

    for m in range(mc_sims):
        Z = np.random.normal(size=(T, len(weights)))
        L = np.linalg.cholesky(cov_matrix)
        daily_returns = meanM + np.inner(L, Z)
        portfolio_sims[:, m] = np.cumprod(np.inner(weights, daily_returns.T) + 1) * initial_portfolio_value

    port_results = pd.Series(portfolio_sims[-1, :])
    VaR = initial_portfolio_value - mcVaR(port_results, alpha=5)
    CVaR = initial_portfolio_value - mcCVaR(port_results, alpha=5)

    portfolio_values.append(initial_portfolio_value)
    print(f'Day {day+1}: VaR INR {round(VaR, 2)}, CVaR INR {round(CVaR, 2)}')

    # Pause to simulate real-time updates (e.g., one day delay)
    time.sleep(1)

# Plot the simulation results
plt.plot(portfolio_values)
plt.ylabel('Portfolio Value (INR)')
plt.xlabel('Trading Days')
plt.title('Live Monte Carlo Simulation of Stock Portfolio')
plt.show()

# Final VaR and CVaR
VaR = initial_portfolio_value - mcVaR(port_results, alpha=5)
CVaR = initial_portfolio_value - mcCVaR(port_results, alpha=5)

print('Final VaR INR {}'.format(round(VaR, 2)))
print('Final CVaR INR {}'.format(round(CVaR, 2)))
