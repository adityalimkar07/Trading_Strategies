import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, SimpleRNN
import matplotlib.pyplot as plt
import csv

def create_results_csv(predictions, strategy_returns, close_prices, sequence_length):
    filename = 'trading_results.csv'
    
    # Calculate cumulative returns
    cumulative_returns = [sum(strategy_returns[:i+1]) for i in range(len(strategy_returns))]
    
    # Determine buy/sell prices
    buy_sell_prices = []
    for i, pred in enumerate(predictions):
        if pred > 0.5:  # Buy signal
            buy_sell_prices.append((close_prices[i+sequence_length], -1))
        else:  # Sell signal
            buy_sell_prices.append((-1, close_prices[i+sequence_length]))
    
    with open(filename, 'w', newline='') as csvfile:
        csvwriter = csv.writer(csvfile)
        
        # Write header
        csvwriter.writerow(['Trade_Number', 'Prediction', 'Buy_Price', 'Sell_Price', 
                            'Close_Price', 'Returns_Current_Trade', 'Cumulative_Returns'])
        # Write data
        for i in range(len(predictions)):
            csvwriter.writerow([
                i+1,
                predictions[i],
                buy_sell_prices[i][0],
                buy_sell_prices[i][1],
                close_prices[i+sequence_length],
                strategy_returns[i],
                cumulative_returns[i]
            ])
    
    print(f"Results saved to {filename}")

def load_data(file_path, target_column):
    data = pd.read_csv(file_path)
    
    # Convert all columns to numeric, replacing non-numeric values with NaN
    for column in data.columns:
        data[column] = pd.to_numeric(data[column], errors='coerce')
    
    # Drop rows with NaN values
    data = data.dropna()
    
    features = data.drop(columns=[target_column, 'Alpha_29'])
    target = data[target_column]
    close_prices = data['Alpha_29']
    
    return features.values, target.values, close_prices.values, features.shape[1]

# Standardize data using mean and std from training data only
def standardize_data(train_data, test_data):
    scaler = StandardScaler()
    train_data_scaled = scaler.fit_transform(train_data)
    test_data_scaled = scaler.transform(test_data)
    return train_data_scaled, test_data_scaled, scaler

def build_model(seq_length, n_features):
    model = Sequential([
        SimpleRNN(50, return_sequences=True, input_shape=(seq_length, n_features)),
        Dropout(0.2),
        SimpleRNN(50, return_sequences=False),
        Dropout(0.2),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

def train_and_predict(model, X_train, y_train, X_test, epochs=40, batch_size=32):
    model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, verbose=0)
    prediction = model.predict(X_test, verbose=0)
    return prediction[0][0]

def trading_strategy(prediction, close_price, prev_close_price):
    if prediction > 0.5:
        position = 1  # Buy/Long
    else:
        position = -1  # Sell/Short
    
    return_ = (close_price - prev_close_price) / prev_close_price
    strategy_return = position * return_
    
    return strategy_return

def calculate_metrics(returns):
    total_return = np.sum(returns)
    sharpe_ratio = np.mean(returns) / np.std(returns) * np.sqrt(252)
    max_dd = np.min(np.cumsum(returns) - np.maximum.accumulate(np.cumsum(returns)))
    win_rate = np.sum([1 for r in returns if r > 0]) / len(returns)
    
    return total_return, sharpe_ratio, max_dd, win_rate

def initialize_plot():
    plt.ion()
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True)
    line1, = ax1.plot([], [], label='Cumulative Returns (PnL)')
    line2, = ax2.plot([], [], label='Close Price', color='orange')
    
    ax1.set_title('Trading Strategy Performance')
    ax1.set_ylabel('Cumulative Returns')
    ax1.grid(True)
    ax1.legend()
    
    ax2.set_xlabel('Trading Days')
    ax2.set_ylabel('Close Price')
    ax2.grid(True)
    ax2.legend()
    
    return fig, (ax1, ax2), (line1, line2)

def update_plot(axes, lines, cumulative_returns, close_prices, i, sequence_length):
    ax1, ax2 = axes
    line1, line2 = lines
    
    x = np.arange(len(cumulative_returns))
    
    line1.set_xdata(x)
    line1.set_ydata(cumulative_returns)
    
    adjusted_close = close_prices[sequence_length:i+sequence_length+1]
    line2.set_xdata(x[:len(adjusted_close)])
    line2.set_ydata(adjusted_close)
    
    for ax in axes:
        ax.relim()
        ax.autoscale_view()
    
    plt.draw()
    plt.pause(0.01)

if __name__ == "__main__":
    # Parameters
    file_path = 'alpha_factors.csv'
    target_column = 'Alpha_28'
    sequence_length = 100
    training_window = 252  # Use one year of data for training
    
    # Load data
    features, target, close_prices, n_features = load_data(file_path, target_column)
    
    # Initialize plot
    fig, axes, lines = initialize_plot()
    
    # Initialize lists for storing results
    predictions = []
    strategy_returns = []
    cumulative_returns = []
    
    # Sliding window prediction
    for i in range(len(features) - sequence_length - training_window):
        # Define training data window
        train_start = i
        train_end = i + training_window
        
        # Get training data (past data only)
        X_train = features[train_start:train_end]
        y_train = target[train_start:train_end]
        
        # Get test data (next point after training window)
        X_test = features[train_end:train_end+sequence_length]
        
        # Standardize data using only training data statistics
        X_train_scaled, X_test_scaled, _ = standardize_data(X_train, X_test)
        
        # Reshape data for RNN
        X_train_reshaped = X_train_scaled[-sequence_length:].reshape(1, sequence_length, n_features)
        X_test_reshaped = X_test_scaled.reshape(1, sequence_length, n_features)
        y_train_point = y_train[-1]
        
        # Build and train model
        model = build_model(sequence_length, n_features)
        prediction = train_and_predict(model, X_train_reshaped, np.array([y_train_point]), X_test_reshaped)
        
        predictions.append(prediction)
        
        # Implement trading strategy
        current_idx = train_end + sequence_length - 1
        strategy_return = trading_strategy(prediction, close_prices[current_idx], close_prices[current_idx-1])
        strategy_returns.append(strategy_return)
        
        if cumulative_returns:
            cumulative_returns.append(cumulative_returns[-1] + strategy_return)
        else:
            cumulative_returns.append(strategy_return)
        
        # Update plot
        update_plot(axes, lines, cumulative_returns, close_prices, i, sequence_length)
        
        # Print progress
        if (i + 1) % 10 == 0:
            print(f"Processed {i + 1}/{len(features) - sequence_length - training_window} predictions")
    
    # Calculate final performance metrics
    total_return, sharpe_ratio, max_dd, win_rate = calculate_metrics(strategy_returns)
    
    # Print results
    print(f"\nFinal Results:")
    print(f"Total Return: {total_return:.2f}")
    print(f"Sharpe Ratio: {sharpe_ratio:.2f}")
    print(f"Maximum Drawdown: {max_dd:.2f}")
    print(f"Win Rate: {win_rate:.2f}")
    
    create_results_csv(predictions, strategy_returns, close_prices, sequence_length)
    
    plt.ioff()
    plt.show()