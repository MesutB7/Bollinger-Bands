import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Download EUR/CHF data
data = yf.download('EURCHF=X', start='2022-01-01', end='2024-08-31')

# Calculate Bollinger Bands
data['SMA'] = data['Close'].rolling(window=20).mean()
data['STD'] = data['Close'].rolling(window=20).std()
data['Upper Band'] = data['SMA'] + (2 * data['STD'])
data['Lower Band'] = data['SMA'] - (2 * data['STD'])

# Define trading signals
data['Buy Signal'] = np.where(data['Close'] < data['Lower Band'], 1, 0)
data['Sell Signal'] = np.where(data['Close'] > data['Upper Band'], -1, 0)
data['Position'] = data['Buy Signal'] + data['Sell Signal']

# Calculate strategy returns
data['Market Return'] = data['Close'].pct_change()
data['Strategy Return'] = data['Market Return'] * data['Position'].shift(1)
data['Cumulative Market Return'] = (1 + data['Market Return']).cumprod()
data['Cumulative Strategy Return'] = (1 + data['Strategy Return']).cumprod()

# Plot Bollinger Bands with signals
plt.figure(figsize=(14,8))
plt.plot(data.index, data['Close'], label='Close Price', alpha=0.5)
plt.plot(data.index, data['Upper Band'], label='Upper Bollinger Band', linestyle='--', alpha=0.5)
plt.plot(data.index, data['Lower Band'], label='Lower Bollinger Band', linestyle='--', alpha=0.5)
plt.fill_between(data.index, data['Lower Band'], data['Upper Band'], color='lightgray', alpha=0.3)

# Filter the index for Buy and Sell signals to match the size
buy_signals = data[data['Buy Signal'] == 1]
sell_signals = data[data['Sell Signal'] == -1]

plt.scatter(buy_signals.index, buy_signals['Close'], label='Buy Signal', marker='^', color='green', alpha=1)
plt.scatter(sell_signals.index, sell_signals['Close'], label='Sell Signal', marker='v', color='red', alpha=1)

plt.title('Bollinger Bands Strategy')
plt.legend()
plt.show()

# Plot cumulative returns
plt.figure(figsize=(14,8))
plt.plot(data.index, data['Cumulative Market Return'], label='Cumulative Market Return', color='blue')
plt.plot(data.index, data['Cumulative Strategy Return'], label='Cumulative Strategy Return', color='green')
plt.title('Cumulative Returns: Bollinger Bands Strategy vs Market')
plt.legend()
plt.show()

# Backtest performance (separate plot)
plt.figure(figsize=(14,8))
plt.plot(data.index, data['Strategy Return'].cumsum(), label='Backtest: Strategy Return', color='orange')
plt.title('Backtest: Cumulative Strategy Returns')
plt.legend()
plt.show()
