import pandas as pd
import numpy as np
import yfinance as yf

data = pd.read_csv('RELIANCE_1d.csv')

# Function Definitions for Each Alpha Factor
def ts_argmax(series, window):
    return series.rolling(window).apply(lambda x: np.argmax(x), raw=True)

def signed_power(x, power):
    return np.sign(x) * np.abs(x) ** power

def delta(series, periods):
    return series.diff(periods)

def scale(series):
    return (series - series.mean()) / series.std()

def vwap(data):
    return (data['close'] * data['volume']).cumsum() / data['volume'].cumsum()

# Prepare Data
data['returns'] = data['close'].pct_change()
data['vwap'] = vwap(data)
data['adv20'] = data['volume'].rolling(window=20).mean()

# Function to Calculate Alpha Factors for Given Range
def calculate_alpha_factors(data, start_index, end_index):
    # Slice the DataFrame for the given range
    sliced_data = data.iloc[start_index:end_index].copy()

    # Feature Calculations
# Adjust Alpha_2 to handle log application properly
    sliced_data['Alpha_1'] = (-1 * sliced_data['volume'].rolling(2).apply(lambda x: np.log(x).sum(), raw=False).diff().rolling(6).corr((sliced_data['close'] - sliced_data['open']) / sliced_data['open'])).rank()
    sliced_data['Alpha_2'] = (-1 * sliced_data['open'].rolling(10).corr(sliced_data['volume'])).rank()
    sliced_data['Alpha_3'] = (-1 * sliced_data['low'].rolling(9).apply(lambda x: x.rank().iloc[-1])).rank()
    sliced_data['Alpha_4'] = (sliced_data['open'] - (sliced_data['vwap'].rolling(10).sum() / 10)) * (-1 * np.abs(sliced_data['close'] - sliced_data['vwap']).rank()).rank()
    sliced_data['Alpha_5'] = (-1 * sliced_data['open'].rolling(10).corr(sliced_data['volume'])).rank()
    sliced_data['Alpha_6'] = (-1 * (sliced_data['open'].rolling(5).sum() * sliced_data['returns'].rolling(5).sum() - sliced_data['open'].rolling(5).sum().shift(10))).rank()
    sliced_data['Alpha_7'] = sliced_data['close'].diff(1).rolling(4).apply(lambda x: x.rank().iloc[-1] if 0 < x.min() else (-1 * delta(sliced_data['close'], 1)).rank().iloc[-1])
    sliced_data['Alpha_8'] = ((sliced_data['vwap'] - sliced_data['close']).rolling(3).max().rank() + (sliced_data['vwap'] - sliced_data['close']).rolling(3).min().rank()) * delta(sliced_data['volume'], 3).rank()
    sliced_data['Alpha_9'] = np.sign(delta(sliced_data['volume'], 1)) * -1 * delta(sliced_data['close'], 1)
    sliced_data['Alpha_10'] = (-1 * sliced_data['close'].rolling(5).corr(sliced_data['volume'])).rank()
    sliced_data['Alpha_11'] = (-1 * delta(sliced_data['returns'], 3).rank() * sliced_data['open'].rolling(10).corr(sliced_data['volume'])).rank()
    sliced_data['Alpha_1'] = (-1 * sliced_data['high'].rolling(3).corr(sliced_data['volume']).rolling(3).sum()).rank()
    sliced_data['Alpha_12'] = (-1 * sliced_data['high'].rolling(5).corr(sliced_data['volume'])).rank()
    sliced_data['Alpha_13'] = (-1 * sliced_data['close'].rolling(10).rank() * delta(delta(sliced_data['close'], 1), 1).rank() * (sliced_data['volume'] / sliced_data['adv20']).rolling(5).rank()).rank()
    sliced_data['Alpha_14'] = (-1 * (sliced_data['close'].rolling(5).std() + (sliced_data['close'] - sliced_data['open'])) + sliced_data['close'].rolling(10).corr(sliced_data['open'])).rank()
    sliced_data['Alpha_15'] = ((-1 * (sliced_data['open'] - sliced_data['high'].shift(1)).rank()) * (sliced_data['open'] - sliced_data['close'].shift(1)).rank() * (sliced_data['open'] - sliced_data['low'].shift(1)).rank()).rank()
    sliced_data['Alpha_16'] = -1 * (delta(sliced_data['high'].rolling(5).corr(sliced_data['volume']), 5) * sliced_data['close'].rolling(20).std()).rank()
    sliced_data['Alpha_17'] = (-1 * sliced_data['returns'] * sliced_data['adv20'] * (sliced_data['high'] - sliced_data['close'])).rank()
    sliced_data['Alpha_18'] = -1 * sliced_data['volume'].rolling(5).corr(sliced_data['high']).rolling(3).max()
    sliced_data['Alpha_19'] = scale((sliced_data['adv20'].rolling(5).corr(sliced_data['low']) + ((sliced_data['high'] + sliced_data['low']) / 2) - sliced_data['close']))
# Update for Alpha_29
    sliced_data['Alpha_20'] = (np.log(sliced_data['close'].rolling(5).min()).rolling(2).apply(np.prod) + delta(-1 * sliced_data['returns'], 6).rank()).rank()
    sliced_data['Alpha_21'] = ((1.0 - (pd.Series(np.sign(sliced_data['close'] - sliced_data['close'].shift(1)) + np.sign(sliced_data['close'].shift(1) - sliced_data['close'].shift(2)) + np.sign(sliced_data['close'].shift(2) - sliced_data['close'].shift(3)))).rank()) * sliced_data['volume'].rolling(5).sum()) / sliced_data['volume'].rolling(20).sum()
# Update for Alpha_31
    rolling_corr = sliced_data['adv20'].rolling(12).corr(sliced_data['low'])
    scaled_corr = (rolling_corr - rolling_corr.mean()) / rolling_corr.std()

    sliced_data['Alpha_22'] = ((delta(sliced_data['close'], 10).rolling(20).apply(lambda x: -1 * x.rank().iloc[-1])).rank().rank().rank() 
                            + (-1 * delta(sliced_data['close'], 3)).rank()) + scaled_corr.rank()
    sliced_data['Alpha_23'] = (-1 * ((1 - (sliced_data['open'] / sliced_data['close']))**1)).rank()
    sliced_data['Alpha_24'] = ((1 - ((sliced_data['returns'].rolling(2).std() / sliced_data['returns'].rolling(5).std()).rank())) + (1 - delta(sliced_data['close'], 1).rank())).rank()
    sliced_data['Alpha_25'] = (((2.21 * (sliced_data['close'].rolling(15).corr(sliced_data['volume'].shift(1))).rank()) + (0.7 * ((sliced_data['open'] - sliced_data['close']).rank()))) + (0.73 * (ts_argmax(delta(-1 * sliced_data['returns'], 6), 5).rank()))) + (abs(sliced_data['vwap'].rolling(6).corr(sliced_data['adv20']))).rank()
    sliced_data['Alpha_26'] = (-1 * sliced_data['close'].rolling(10).apply(lambda x: x.rank().iloc[-1]).rank()) * (sliced_data['close'] / sliced_data['open']).rank()
    sliced_data['Alpha_27'] = (-1 * sliced_data['high'].rolling(10).std().rank()) * sliced_data['high'].rolling(10).corr(sliced_data['volume'])
    sliced_data['Alpha_28'] = np.where(sliced_data['close'].diff() > 0, 1, 0)
    sliced_data['Alpha_29'] = sliced_data['close']
    return {f'Alpha_{i}': sliced_data[f'Alpha_{i}'].iloc[-1] for i in range(1, 30)} 

# Example: Calculate alpha factors from index 0 to 10
# Example: Calculate alpha factors for all possible sliding windows
alpha_factors_list = []
max_possible = len(data) - 100  # Adjust based on the window size (100 rows)

for i in range(max_possible):
    start_index = len(data) - 100 - i
    end_index = len(data) - i
    
    # Calculate alpha factors
    alpha_factors = calculate_alpha_factors(data, start_index, end_index)
    
    # Append the results to the list
    alpha_factors_list.append(alpha_factors)

# Convert the list of alpha factors into a DataFrame
alpha_factors_df = pd.DataFrame(alpha_factors_list)

# Reverse the DataFrame to ensure the alpha factors are in the correct serial order
alpha_factors_df = alpha_factors_df.iloc[::-1].reset_index(drop=True)

# Save the DataFrame to a CSV file
alpha_factors_df.to_csv('features.csv', index=False)
print("Features saved to 'features.csv'.")
