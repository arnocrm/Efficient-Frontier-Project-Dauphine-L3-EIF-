#Question 1

#import data structures 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


#Indicate the path on your computer to find the right location to extract the document
from pathlib import Path 
import os 

file_location=input("enter your computer path to the file without the quotation mark, eg:  /Users/nitou1/Downloads   ")
directory_path = Path(file_location)
os.chdir(directory_path)

#Extract data from the computer 
stockdata = pd.read_excel('Data.xlsx')


#Cancel all the rows with Nan
df = stockdata.dropna()
df_2=df.drop('Date', axis=1)

#Put date as the index to annualize the returns 
df['Date'] = pd.to_datetime(df['Date'])
df.info()
df.set_index('Date', inplace=True)

#Question 2

#VL Graph of each asset
df.plot(figsize=(6,4))

#Graph properties
plt.legend()
plt.ylabel('Value')
plt.xlabel('Time')
plt.title('VL')

#Question 3

#find the daily returns of each asset with pct.change 
returns_df = df.pct_change()

#find the volatily of each asset with std
vol=df.std()

#MDD:the maximum observed loss from a peak to a trough of a portfolio, before a new peak is attained

#DD
cumul_returns = (1 + returns_df).cumprod()
max_cumul_returns = cumul_returns.cummax()
DD_df = (max_cumul_returns - cumul_returns)/max_cumul_returns

#MDD
MDD_df = DD_df.max() * 100


#Question 4 

#Covariance matrix
cov_matrix = df.pct_change().apply(lambda x: np.log(1+x)).cov()
cov_matrix

#Correlation matrix (useless but allows one to see the different link relation the assets)
corr_matrix = df.pct_change().apply(lambda x: np.log(1+x)).corr()
corr_matrix


#Calculation of assets daily return and then annualization 
ind_er = df.resample('Y').last().pct_change().mean()
ind_er


#Volatility calculation (assuming there are 250 business days in a year)
ann_sd = df.pct_change().apply(lambda x: np.log(1+x)).std().apply(lambda x: x*np.sqrt(250))
ann_sd

#New dataframe to sum up the Returns and Volatility per assets
assets = pd.concat([ind_er, ann_sd], axis=1) # Creating a table for visualising returns and volatility of assets
assets.columns = ['Returns', 'Volatility']
assets

#Creation of 3 lists to append the Vol, Return according to the random weights
p_ret = [] # Define an empty array for portfolio returns
p_vol = [] # Define an empty array for portfolio volatility
p_weights = [] # Define an empty array for asset weights

#We define the number of portfolios we want
num_assets = len(df.columns)
num_portfolios = 25000

for portfolio in range(num_portfolios):
    weights = np.random.random(num_assets) #random weights for each asset
    weights = weights/np.sum(weights)
    if weights[0]< 0.9:    #constraint to avoid S&P 500 to exceed 0.9   
        p_weights.append(weights) #we add the weights to our weight array
        returns = np.dot(weights, ind_er) #we calcultate the returns in function of the weights
        p_ret.append(returns) #we add the returns to our return array
        var = cov_matrix.mul(weights, axis=0).mul(weights, axis=1).sum().sum()# Portfolio Variance
        sd = np.sqrt(var) # Daily standard deviation
        ann_sd = sd*np.sqrt(250) # Annual standard deviation = volatility
        p_vol.append(ann_sd) #we add the std  to our std array
        
data = {'Returns':p_ret, 'Volatility':p_vol}

for counter, symbol in enumerate(df.columns.tolist()):
    #print(counter, symbol)
    data[symbol+' weight'] = [w[counter] for w in p_weights]
    
    
portfolios  = pd.DataFrame(data)
portfolios.head() # Dataframe of the 25,000 portfolios created

#Plot efficient frontier
portfolios.plot.scatter(x='Volatility', y='Returns', marker='o', s=10, alpha=0.3, grid=True, figsize=[10,10])

#idxmin() gives us the minimum value in the column specified = find the minimum variance portfolio
min_vol_port = portfolios.iloc[portfolios['Volatility'].idxmin()]                            
min_vol_port

#Find the portfolio with the best Sharpe ratio, with a risk free rate defined, here 0,01
rf = 0.01 
optimal_risky_port = portfolios.iloc[((portfolios['Returns']-rf)/portfolios['Volatility']).idxmax()]
optimal_risky_port

#idxmax() gives us the maximum value in the column specified = find the maximum return portfolio
max_return_port= portfolios.iloc[portfolios['Returns'].idxmax()]
max_return_port

plt.subplots(figsize=[10,10])
plt.scatter(portfolios['Volatility'], portfolios['Returns'],marker='o', s=10, alpha=0.3)
plt.scatter(max_return_port[1], max_return_port[0], color='c', marker='*', s=500) # turquoise star
plt.scatter(min_vol_port[1], min_vol_port[0], color='r', marker='*', s=500) #red star
plt.scatter(optimal_risky_port[1], optimal_risky_port[0], color='g', marker='*', s=500) #green star
plt.title('Efficient Frontier')
plt.xlabel('Risk')
plt.ylabel('Return')

# Question 5 

#MM, VOL, Bollinger condition and VL's strategy 
df_inv=df.iloc[::-1]
#MM, VOL, Bollinger condition and VL's strategy 
for n in [5, 10, 15, 20, 30]:
    df_inv[f'MM_Boll_{n}']=df_inv['S&P 500'].rolling(window=n).mean().iloc[n-1:]
    df_inv[f'VOL_Boll_{n}']=df_inv['S&P 500'].rolling(window=n).std().iloc[n-1:]
    df_inv[f'Boll_band_{n}']=(df_inv['S&P 500'].iloc[n-1:] < df_inv[f'MM_Boll_{n}'].iloc[n-1:] - df_inv[f'VOL_Boll_{n}'].iloc[n-1:]).astype(int)
    df_inv[f'Invested_or_not_{n}']=df_inv[f'Boll_band_{n}'].shift(1).iloc[1:]
    df_inv[f'VL_{n}'] = (1 + df_inv[f'Invested_or_not_{n}'] * (df_inv['S&P 500'].pct_change())).cumprod()

#VL charts 
df_inv[['VL_5', 'VL_10', 'VL_15', 'VL_20', 'VL_30']].plot(figsize=(6,4))
plt.title('VL strategies')
plt.xlabel('Time')
plt.ylabel('VL')
