#!/usr/bin/env python
# coding: utf-8

# # I. Algorithm with raw data

# ## a) Initial configuration

# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Input, ReLU
from tensorflow.keras.optimizers import Adam
import os


# In[ ]:


# Getting the directory of the current script and defining the relative paths to the data files
dir = os.path.dirname(__file__)
mbg_path = os.path.join(dir, 'Data', 'D1. Raw starting data', 'MBG.DE')
msft_path = os.path.join(dir, 'Data', 'D1. Raw starting data', 'MSFT')

df_mbg = pd.read_csv(mbg_path, delimiter=",")
df_msft = pd.read_csv(msft_path, delimiter=",")

# Computing daily returns and rolling volatility (standard deviation of returns) for our two datasets
df_mbg['Return'] = df_mbg['Close'].pct_change()
df_msft['Return'] = df_msft['Close'].pct_change()

# We use a window size of 30 days for calculating the volatility
window_size_vol = 30  

# Annualizing the volatilities
df_mbg['Volatility'] = df_mbg['Return'].rolling(window=window_size_vol).std() * np.sqrt(252)  
df_msft['Volatility'] = df_msft['Return'].rolling(window=window_size_vol).std() * np.sqrt(252)

# Dropping rows with NaN and extracting the last 30 rows
df_mbg = df_mbg.dropna().reset_index(drop=True)
df_msft = df_msft.dropna().reset_index(drop=True)
df_mbg_last_30 = df_mbg.tail(30).reset_index(drop=True)
df_msft_last_30 = df_msft.tail(30).reset_index(drop=True)

print("Initial MBG data:")
print(df_mbg.head())
print("Initial MSFT data:")
print(df_msft.head())

# Scaling the data with MinMaxScaler()
scaler_mbg = MinMaxScaler()
scaler_msft = MinMaxScaler()
df_mbg['Close'] = scaler_mbg.fit_transform(df_mbg[['Close']])
df_msft['Close'] = scaler_msft.fit_transform(df_msft[['Close']])

print("Preprocessed MBG data:")
print(df_mbg.head())
print("Preprocessed MSFT data:")
print(df_msft.head())


# In[ ]:


# Preparation of the training data
# The function prepare_training_data transforms our preprocessed data (stock prices) into input-output pairs for model training
# For each position in the series, it extracts a subsequence of length n_steps as the input (X) and the following value as the output (y)
# We generate multiple input-output pairs, which are then converted into numpy arrays and returned to train the stock prices
def prepare_training_data(series, n_steps):
    X, y = [], []
    for i in range(len(series) - n_steps):
        seq_X = series[i:i+n_steps]
        seq_y = series[i+n_steps]
        X.append(seq_X)
        y.append(seq_y)
    return np.array(X), np.array(y)

n_steps = 30
X_mbg, y_mbg = prepare_training_data(df_mbg['Close'].values, n_steps)
X_msft, y_msft = prepare_training_data(df_msft['Close'].values, n_steps)

# Reshaping data for our LSTM model and splitting the data into training and testing
# We train the data on the whole dataset minus the last 30 days of the dataset and test it on the last 30 days of the dataset
X_mbg = X_mbg.reshape((X_mbg.shape[0], X_mbg.shape[1], 1))
X_msft = X_msft.reshape((X_msft.shape[0], X_msft.shape[1], 1))

split_idx_mbg = len(X_mbg) - 30  
split_idx_msft = len(X_msft) - 30

X_train_mbg, X_test_mbg = X_mbg[:split_idx_mbg], X_mbg[split_idx_mbg:]
y_train_mbg, y_test_mbg = y_mbg[:split_idx_mbg], y_mbg[split_idx_mbg:]

X_train_msft, X_test_msft = X_msft[:split_idx_msft], X_msft[split_idx_msft:]
y_train_msft, y_test_msft = y_msft[:split_idx_msft], y_msft[split_idx_msft:]



# Definition of the LSTM model
# The function create_lstm_model constructs and compiles our LSTM-based neural network model
# We initializes a Sequential model and adds an input layer with a specified shape, followed by two LSTM layers with a specified number of neurons and ReLU activation
# The compiled model is returned using the Adam optimizer and a specified loss function (which is specified right after in our code, depending on the dataset)
def create_lstm_model(input_shape, num_neurons, loss_function):
    model = Sequential([
        Input(shape=input_shape),
        LSTM(num_neurons, activation='relu', return_sequences=True),
        LSTM(num_neurons, activation='relu'),
        Dense(1, activation='sigmoid')  
    ])
    model.compile(optimizer='adam', loss=loss_function)
    return model


# Training of the model on both dataset, with MSE as a loss function for MSFT and MAE for MBG
model_msft = create_lstm_model((n_steps, 1), 200, 'mse')
history_msft = model_msft.fit(X_train_msft, y_train_msft, epochs=30, batch_size=32, verbose=0)

model_mbg = create_lstm_model((n_steps, 1), 20, 'mae')
history_mbg = model_mbg.fit(X_train_mbg, y_train_mbg, epochs=30, batch_size=16, verbose=0)

# Plotting the training loss for checking
plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
plt.plot(history_mbg.history['loss'])
plt.title('MBG Model Training Loss')

plt.subplot(1, 2, 2)
plt.plot(history_msft.history['loss'])
plt.title('MSFT Model Training Loss')

plt.tight_layout()
plt.show()

# Predicting prices using our pre-trained models and printing them
predictions_msft = model_msft.predict(X_test_msft)
predictions_mbg = model_mbg.predict(X_test_mbg)

print("Raw MSFT Predictions:")
print(predictions_msft)

print("Raw MBG Predictions:")
print(predictions_mbg)

# Computing deltas with the function compute_deltas 
# The function calculates the differences between our consecutive prices predictions values : it initializes an array deltas with the same shape as predictions, sets the first element to zero, and computes the difference between each pair of consecutive prices predictions
# The results are stored in the subsequent elements of deltas, and the function returns the resulting array of deltas
def compute_deltas(predictions):
    deltas = np.zeros_like(predictions)
    deltas[1:] = predictions[1:] - predictions[:-1]
    return deltas

deltas_mbg = compute_deltas(predictions_mbg)
deltas_msft = compute_deltas(predictions_msft)

# Normalization and adjustment of deltas
deltas_mbg = np.abs(deltas_mbg)  
deltas_msft = np.abs(deltas_msft)  

deltas_mbg = MinMaxScaler().fit_transform(deltas_mbg)
deltas_msft = MinMaxScaler().fit_transform(deltas_msft)

# Initialization of NN_Delta with NaNs and filling of NN_Delta with the computed deltas
df_mbg_last_30['NN_Delta'] = np.nan
df_msft_last_30['NN_Delta'] = np.nan

df_mbg_last_30.loc[:, 'NN_Delta'] = deltas_mbg.flatten()
df_msft_last_30.loc[:, 'NN_Delta'] = deltas_msft.flatten()

# Printing the results
print(df_mbg_last_30)
print(df_msft_last_30)


# In[ ]:


# Defining the Black-Scholes function with the given formulas from the BS model
def black_scholes(S, K, T, r, sigma, option_type='call'):
    from scipy.stats import norm
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    if option_type == 'call':
        price = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
        delta = norm.cdf(d1)
    elif option_type == 'put':
        price = K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
        delta = -norm.cdf(-d1)
    return price, delta

# The calculate_bs function computes Black-Scholes prices and deltas using a rolling ATM strike (cf report part 3.3 for more informations on it)
# We initialize the lists bs_prices and bs_deltas with NaNs for the specified window_size (in our case it will be 30 days) 
# For each data point beyond the initial window_size, we calculate the current price (S), the strike price from window_size days ago (K), one day to maturity (T), a risk-free rate (r), and current volatility (sigma)
# If sigma is not NaN, we use the black_scholes function to compute the price and delta, and append them to the lists; otherwise, we append NaNs to the list
# The function finally returns the lists bs_prices and bs_deltas.
def calculate_bs(df, window_size):
    bs_prices = [np.nan] * window_size  
    bs_deltas = [np.nan] * window_size  

    for i in range(window_size, len(df)):
        S = df['Close'].iloc[i]  # Current price
        K = df['Close'].iloc[i - window_size]  # Price window_size days ago
        T = 1 / 252  # One day to maturity
        r = 0.0  # Risk-free rate
        sigma = df['Volatility'].iloc[i]  # Current volatility

        if not np.isnan(sigma):
            price, delta = black_scholes(S, K, T, r, sigma)
            bs_prices.append(price)
            bs_deltas.append(delta)
        else:
            bs_prices.append(np.nan)
            bs_deltas.append(np.nan)

    return bs_prices, bs_deltas

# Calculation of Black-Scholes prices and deltas with rolling ATM strike prices (we choose a window size of 30 days, more on this on part 3.3 of our report) for each dataset
window_size = 30  
bs_prices_mbg, bs_deltas_mbg = calculate_bs(df_mbg, window_size)
bs_prices_msft, bs_deltas_msft = calculate_bs(df_msft, window_size)


# In[ ]:


# Adding the calculated BS prices and deltas to the dataframes and printing the last few rows to verify
# Since our model aims to predict deltas for the last 30 days of our dataset with the training on the whole dataset minus these last 30 days
df_mbg_last_30['BS_Price'] = bs_prices_mbg[-30:]  
df_mbg_last_30['BS_Delta'] = bs_deltas_mbg[-30:]  

df_msft_last_30['BS_Price'] = bs_prices_msft[-30:] 
df_msft_last_30['BS_Delta'] = bs_deltas_msft[-30:] 

print("Final MBG Data with BS calculations:")
print(df_mbg_last_30[['Date', 'Close', 'BS_Price', 'BS_Delta', 'NN_Delta']])
print("Final MSFT Data with BS calculations:")
print(df_msft_last_30[['Date', 'Close', 'BS_Price', 'BS_Delta', 'NN_Delta']])


# In[ ]:


# Visualization of the hedge ratios for the last 30 days
plt.figure(figsize=(14, 7))

plt.subplot(2, 1, 1)
plt.plot(df_mbg_last_30['NN_Delta'], label='NN delta', color='blue')
plt.plot(df_mbg_last_30['BS_Delta'], label='BS delta', color='red')
plt.fill_between(range(len(df_mbg_last_30['NN_Delta'])), df_mbg_last_30['NN_Delta'], alpha=0.3, color='blue')
plt.fill_between(range(len(df_mbg_last_30['BS_Delta'])), df_mbg_last_30['BS_Delta'], alpha=0.3, color='red')
plt.title('MBG hedge ratios')
plt.xlabel('Date')
plt.ylabel('Hedge ratio')
plt.legend()

plt.subplot(2, 1, 2)
plt.plot(df_msft_last_30['NN_Delta'], label='NN delta', color='blue')
plt.plot(df_msft_last_30['BS_Delta'], label='BS delta', color='red')
plt.fill_between(range(len(df_msft_last_30['NN_Delta'])), df_msft_last_30['NN_Delta'], alpha=0.3, color='blue')
plt.fill_between(range(len(df_msft_last_30['BS_Delta'])), df_msft_last_30['BS_Delta'], alpha=0.3, color='red')
plt.title('MSFT hedge ratios')
plt.xlabel('Date')
plt.ylabel('Hedge ratio')
plt.legend()

plt.tight_layout()
plt.show()

# Denormalization of predicted prices for visualization
denorm_predictions_mbg = scaler_mbg.inverse_transform(predictions_mbg)
denorm_predictions_msft = scaler_msft.inverse_transform(predictions_msft)

# Visualize predicted prices vs actual prices for last 30 days
plt.figure(figsize=(14, 7))

plt.subplot(2, 1, 1)
plt.plot(df_mbg_last_30['Close'], label='Actual price', color='black')
plt.plot(denorm_predictions_mbg, label='Predicted price', color='green')
plt.title('MBG predicted vs actual prices')
plt.xlabel('Date')
plt.ylabel('Price')
plt.legend()

plt.subplot(2, 1, 2)
plt.plot(df_msft_last_30['Close'], label='Actual price', color='black')
plt.plot(denorm_predictions_msft, label='Predicted price', color='green')
plt.title('MSFT predicted vs actual prices')
plt.xlabel('Date')
plt.ylabel('Price')
plt.legend()

plt.tight_layout()
plt.show()


# In[1]:


# Definition of a PnL test with the function calculate_pnl
# We base the computation on a specified hedge column, and initialize a list pnl with zero for the first row
# Then, for each subsequent row, we calculates the PnL as the difference in closing prices adjusted by the previous hedge value, and append the result to PnL 
# The function returns the cumulative sum of the PnL values
def calculate_pnl(df, hedge_column):
    pnl = [0]  
    for i in range(1, len(df)):
        pnl.append(df['Close'].iloc[i] - df['Close'].iloc[i-1] * df[hedge_column].iloc[i-1])
    return np.cumsum(pnl)

# Calculation of the PnL for each dataset and adding them to the results dataframe and printing of the last few rows to verify
df_mbg_last_30['NN_PnL'] = calculate_pnl(df_mbg_last_30, 'NN_Delta')
df_mbg_last_30['BS_PnL'] = calculate_pnl(df_mbg_last_30, 'BS_Delta')

df_msft_last_30['NN_PnL'] = calculate_pnl(df_msft_last_30, 'NN_Delta')
df_msft_last_30['BS_PnL'] = calculate_pnl(df_msft_last_30, 'BS_Delta')

print("Final MBG Data with BS calculations and PnL:")
print(df_mbg_last_30[['Date', 'Close', 'BS_Price', 'BS_Delta', 'NN_Delta', 'NN_PnL', 'BS_PnL']])
print("Final MSFT Data with BS calculations and PnL:")
print(df_msft_last_30[['Date', 'Close', 'BS_Price', 'BS_Delta', 'NN_Delta', 'NN_PnL', 'BS_PnL']])


# Visualization of the PnL for the last 30 days
plt.figure(figsize=(14, 7))

plt.subplot(2, 1, 1)
plt.plot(df_mbg_last_30['NN_PnL'], label='NN PnL', color='blue')
plt.plot(df_mbg_last_30['BS_PnL'], label='BS PnL', color='red')
plt.title('MBG PnL comparison')
plt.xlabel('Date')
plt.ylabel('PnL')
plt.legend()

plt.subplot(2, 1, 2)
plt.plot(df_msft_last_30['NN_PnL'], label='NN PnL', color='blue')
plt.plot(df_msft_last_30['BS_PnL'], label='BS PnL', color='red')
plt.title('MSFT PnL comparison')
plt.xlabel('Date')
plt.ylabel('PnL')
plt.legend()

plt.tight_layout()
plt.show()


# In[ ]:


# Calculate daily returns for the NN PnL and replacing infinities and NaNs resulting from pct_change calculation
df_mbg_last_30['NN_Return'] = df_mbg_last_30['NN_PnL'].pct_change()
df_msft_last_30['NN_Return'] = df_msft_last_30['NN_PnL'].pct_change()

df_mbg_last_30['NN_Return'].replace([np.inf, -np.inf], np.nan, inplace=True)
df_mbg_last_30['NN_Return'].fillna(0, inplace=True)

df_msft_last_30['NN_Return'].replace([np.inf, -np.inf], np.nan, inplace=True)
df_msft_last_30['NN_Return'].fillna(0, inplace=True)

# Calculate Sharpe ratio with a risk-free rate of 1%
# We compute the mean and standard deviation of the 'NN_Return' column for the last 30 days of both df_mbg_last_30 and df_msft_last_30
# Then, the Sharpe ratio is calculated by subtracting the daily risk-free rate (annual rate divided by 252 trading days) from the mean return and dividing by the standard deviation of returns for both datasets
# The results are stored in sharpe_ratio_mbg and sharpe_ratio_msft
risk_free_rate = 0.01  

mean_return_mbg = df_mbg_last_30['NN_Return'].mean()
std_return_mbg = df_mbg_last_30['NN_Return'].std()
sharpe_ratio_mbg = (mean_return_mbg - risk_free_rate/252) / std_return_mbg

mean_return_msft = df_msft_last_30['NN_Return'].mean()
std_return_msft = df_msft_last_30['NN_Return'].std()
sharpe_ratio_msft = (mean_return_msft - risk_free_rate/252) / std_return_msft

print(f'Sharpe ratio for MBG: {sharpe_ratio_mbg}')
print(f'Sharpe ratio for MSFT: {sharpe_ratio_msft}')

# The function calculate_max_drawdown computes the maximum drawdown of our PnL series then finds the cumulative maximum of these returns (peak)
# We compute the drawdown as the difference between cumulative returns and the peak, divided by the peak
# The function returns this maximum drawdown value
def calculate_max_drawdown(pnl_series):
    cumulative_returns = (1 + pnl_series).cumprod()
    peak = cumulative_returns.cummax()
    drawdown = (cumulative_returns - peak) / peak
    max_drawdown = drawdown.min()
    return max_drawdown

# Appliying the maximum drawdown for both datasets and printing them
max_drawdown_mbg = calculate_max_drawdown(df_mbg_last_30['NN_Return'])
max_drawdown_msft = calculate_max_drawdown(df_msft_last_30['NN_Return'])

print(f'Maximum drawdown for MBG: {max_drawdown_mbg}')
print(f'Maximum drawdown for MSFT: {max_drawdown_msft}')


# ## b) Additional loss functions and number of neurons

# In[ ]:


from scipy.optimize import minimize
from tensorflow.keras.losses import Huber, LogCosh

df_mbg_v2 = pd.read_csv(mbg_path, delimiter=",")
df_msft_v2 = pd.read_csv(msft_path, delimiter=",")

# Computing daily returns and rolling volatility
df_mbg_v2['Return'] = df_mbg_v2['Close'].pct_change()
df_msft_v2['Return'] = df_msft_v2['Close'].pct_change()

window_size_vol = 30  
df_mbg_v2['Volatility'] = df_mbg_v2['Return'].rolling(window=window_size_vol).std() * np.sqrt(252)  
df_msft_v2['Volatility'] = df_msft_v2['Return'].rolling(window=window_size_vol).std() * np.sqrt(252)  

# Dropping rows with NaN values and extracting the last 30 rows
df_mbg_v2 = df_mbg_v2.dropna().reset_index(drop=True)
df_msft_v2 = df_msft_v2.dropna().reset_index(drop=True)

df_mbg_last_30_v2 = df_mbg_v2.tail(30).reset_index(drop=True)
df_msft_last_30_v2 = df_msft_v2.tail(30).reset_index(drop=True)

print("Initial MBG data:")
print(df_mbg_v2.head())
print("Initial MSFT data:")
print(df_msft_v2.head())

# Scaling the data using the scaler functions (scaler_mbg and scaler_msft) defined previously
df_mbg_v2['Close'] = scaler_mbg.fit_transform(df_mbg[['Close']])
df_msft_v2['Close'] = scaler_msft.fit_transform(df_msft[['Close']])

print("Preprocessed MBG data:")
print(df_mbg_v2.head())
print("Preprocessed MSFT data:")
print(df_msft_v2.head())

# Preparing the training of the data using the previously defined function prepare_training_data
X_mbg_v2, y_mbg_v2 = prepare_training_data(df_mbg_v2['Close'].values, n_steps)
X_msft_v2, y_msft_v2 = prepare_training_data(df_msft_v2['Close'].values, n_steps)


# Changing the shape of X_mbg and X_msft to have three dimensions: the number of samples, the number of time steps, and one feature per time step
# Necessary for compatibility with the LSTM layer, which expects input in the form (samples, time steps, features)
X_mbg_v2 = X_mbg_v2.reshape((X_mbg_v2.shape[0], X_mbg_v2.shape[1], 1))
X_msft_v2 = X_msft_v2.reshape((X_msft_v2.shape[0], X_msft_v2.shape[1], 1))

# Splitting data into training and testing as in the initial run
split_idx_mbg_v2 = len(X_mbg_v2) - 30 
split_idx_msft_v2 = len(X_msft_v2) - 30

X_train_mbg_v2, X_test_mbg_v2 = X_mbg_v2[:split_idx_mbg_v2], X_mbg_v2[split_idx_mbg_v2:]
y_train_mbg_v2, y_test_mbg_v2 = y_mbg_v2[:split_idx_mbg_v2], y_mbg_v2[split_idx_mbg_v2:]

X_train_msft_v2, X_test_msft_v2 = X_msft_v2[:split_idx_msft_v2], X_msft_v2[split_idx_msft_v2:]
y_train_msft_v2, y_test_msft_v2 = y_msft_v2[:split_idx_msft_v2], y_msft_v2[split_idx_msft_v2:]


# Training of the model on both datasets with different loss functions and neuron counts (we choose number of neurons that are around the initial parameters upward and downward)
loss_functions_msft = [MeanSquaredError(), MeanAbsoluteError(), Huber(), LogCosh()]
neuron_counts_msft = [200, 180, 150, 220, 250]
results_msft = {}

for neurons in neuron_counts_msft:
    for loss_function in loss_functions_msft:
        model_msft = create_lstm_model((n_steps, 1), neurons, loss_function)
        history_msft = model_msft.fit(X_train_msft_v2, y_train_msft_v2, epochs=30, batch_size=32, verbose=0)
        predictions_msft = model_msft.predict(X_test_msft_v2)
        results_msft[f'{loss_function.name}_{neurons}'] = {
            'history': history_msft.history['loss'],
            'predictions': scaler_msft.inverse_transform(predictions_msft)
        }

loss_functions_mbg = [MeanSquaredError(), MeanAbsoluteError(), Huber(), LogCosh()]
neuron_counts_mbg = [20, 10, 15, 30, 40]
results_mbg = {}

for neurons in neuron_counts_mbg:
    for loss_function in loss_functions_mbg:
        model_mbg = create_lstm_model((n_steps, 1), neurons, loss_function)
        history_mbg = model_mbg.fit(X_train_mbg_v2, y_train_mbg_v2, epochs=30, batch_size=16, verbose=0)
        predictions_mbg = model_mbg.predict(X_test_mbg_v2)
        results_mbg[f'{loss_function.name}_{neurons}'] = {
            'history': history_mbg.history['loss'],
            'predictions': scaler_mbg.inverse_transform(predictions_mbg)
        }

# Plotting training loss for each loss function and neuron count
plt.figure(figsize=(16, 16))

plt.subplot(2, 2, 1)
for key, result in results_msft.items():
    plt.plot(result['history'], label=key)
plt.title('MSFT model training loss')
plt.legend()

plt.subplot(2, 2, 2)
for key, result in results_mbg.items():
    plt.plot(result['history'], label=key)
plt.title('MBG model training loss')
plt.legend()

plt.tight_layout()
plt.show()


# Initializing dataFrame for last 30 days of data with NaNs
df_mbg_last_30_v2.drop(columns=['NN_Delta'], inplace=True, errors='ignore')
df_msft_last_30_v2.drop(columns=['NN_Delta'], inplace=True, errors='ignore')

# Computing and filling NN_Delta for each loss function and neuron count (we use the compute_deltas function that was already initialized in the initial configuration of our NN)
for key, result in results_mbg.items():
    deltas_mbg_v2 = compute_deltas(result['predictions'])
    deltas_mbg_v2 = np.abs(deltas_mbg_v2)  
    deltas_mbg_v2 = MinMaxScaler().fit_transform(deltas_mbg_v2)
    df_mbg_last_30_v2[f'NN_Delta_{key}'] = deltas_mbg_v2.flatten()

for key, result in results_msft.items():
    deltas_msft_v2 = compute_deltas(result['predictions'])
    deltas_msft_v2 = np.abs(deltas_msft_v2)  
    deltas_msft_v2 = MinMaxScaler().fit_transform(deltas_msft_v2)
    df_msft_last_30_v2[f'NN_Delta_{key}'] = deltas_msft_v2.flatten()

# Printing results
print(df_mbg_last_30_v2)
print(df_msft_last_30_v2)


# # II. Robustness testing

# In[ ]:


df_mbg_v3 = pd.read_csv(mbg_path, delimiter=",")
df_msft_v3 = pd.read_csv(msft_path, delimiter=",")

# Displaying the first few rows of the datasets
print("MBG.DE data head:")
print(df_mbg_v3.head())
print("MSFT data head:")
print(df_msft_v3.head())

# Calculating log returns and historical volatility (annualized)
df_mbg_v3['Log_Returns'] = np.log(df_mbg_v3['Close'] / df_mbg_v3['Close'].shift(1))
df_msft_v3['Log_Returns'] = np.log(df_msft_v3['Close'] / df_msft_v3['Close'].shift(1))
df_mbg_v3.dropna(inplace=True)
df_msft_v3.dropna(inplace=True)

historical_volatility_mbg = df_mbg_v3['Log_Returns'].std() * np.sqrt(252)
historical_volatility_msft = df_msft_v3['Log_Returns'].std() * np.sqrt(252)

print(f"Historical volatility for MBG.DE: {historical_volatility_mbg}")
print(f"Historical volatility for MSFT: {historical_volatility_msft}")

# Normalizing the data
scaler_mbg = MinMaxScaler(feature_range=(0, 1))
scaler_msft = MinMaxScaler(feature_range=(0, 1))
scaled_mbg_v3 = scaler_mbg.fit_transform(df_mbg_v3[['Close', 'Log_Returns']])
scaled_msft_v3 = scaler_msft.fit_transform(df_msft_v3[['Close', 'Log_Returns']])

# Defining the heston_model_params function that estimates the parameters of the Heston model
# It takes two arguments: log_returns and historical_volatility, and an inner function named objective calculates the sum of squared differences between the log_returns and their mean for given parameters (kappa, theta, sigma, rho, and v0). This objective function serves as the target for the optimization process.
# The minimize function from the SciPy library is used to perform the optimization, employing the L-BFGS-B method, and function finds the set of parameters that minimize the objective function
# The estimated Heston model parameters are returned
def heston_model_params(log_returns, historical_volatility):
    def objective(params):
        kappa, theta, sigma, rho, v0 = params
        return np.sum(np.square(log_returns - np.mean(log_returns)))
    
    initial_guess = [historical_volatility, historical_volatility, 1.0, -0.6, historical_volatility]
    bounds = [(0.001, None), (0.001, None), (0.001, None), (-1, 1), (0.001, None)]
    result = minimize(objective, initial_guess, bounds=bounds, method='L-BFGS-B')
    return result.x

mbg_params = heston_model_params(df_mbg_v3['Log_Returns'].values, historical_volatility_mbg)
print(f"Estimated Heston parameters for MBG.DE: {mbg_params}")

msft_params = heston_model_params(df_msft_v3['Log_Returns'].values, historical_volatility_msft)
print(f"Estimated Heston parameters for MSFT: {msft_params}")

# Defining initial guesses for Bates, VG-CIR, and rBergomi models in the same way as we did with Heston model
# Bates model parameters estimation function
def bates_model_params(log_returns, historical_volatility):
    def objective(params):
        kappa, theta, sigma, rho, v0, lambda_, mu_j, sigma_j = params
        return np.sum(np.square(log_returns - np.mean(log_returns)))
    
    initial_guess = [historical_volatility, historical_volatility, 1.0, -0.6, historical_volatility, 0.2, -0.075, 0.15]
    bounds = [(0.001, None), (0.001, None), (0.001, None), (-1, 1), (0.001, None), (0.001, None), (-0.2, 0.2), (0.001, None)]
    result = minimize(objective, initial_guess, bounds=bounds, method='L-BFGS-B')
    return result.x

mbg_bates_params = bates_model_params(df_mbg_v3['Log_Returns'].values, historical_volatility_mbg)
print(f"Estimated Bates parameters for MBG.DE: {mbg_bates_params}")

msft_bates_params = bates_model_params(df_msft_v3['Log_Returns'].values, historical_volatility_msft)
print(f"Estimated Bates parameters for MSFT: {msft_bates_params}")

# VG-CIR model parameters estimation function
def vg_cir_model_params(log_returns, historical_volatility):
    def objective(params):
        kappa, theta, sigma, v0, sigma_vg, theta_vg, kappa_vg = params
        return np.sum(np.square(log_returns - np.mean(log_returns)))
    
    initial_guess = [historical_volatility, historical_volatility, 1.0, historical_volatility, 0.25, -0.15, 0.2]
    bounds = [(0.001, None), (0.001, None), (0.001, None), (0.001, None), (0.1, 0.3), (-0.2, -0.1), (0.1, 0.3)]
    result = minimize(objective, initial_guess, bounds=bounds, method='L-BFGS-B')
    return result.x

mbg_vg_cir_params = vg_cir_model_params(df_mbg_v3['Log_Returns'].values, historical_volatility_mbg)
print(f"Estimated VG-CIR parameters for MBG.DE: {mbg_vg_cir_params}")

msft_vg_cir_params = vg_cir_model_params(df_msft_v3['Log_Returns'].values, historical_volatility_msft)
print(f"Estimated VG-CIR parameters for MSFT: {msft_vg_cir_params}")

# rBergomi model parameters estimation function
def rbergomi_model_params(log_returns):
    def objective(params):
        eta, H, xi = params
        return np.sum(np.square(log_returns - np.mean(log_returns)))
    
    initial_guess = [1.0, 0.2, 0.04]
    bounds = [(0.001, None), (0.1, 0.3), (0.001, None)]
    result = minimize(objective, initial_guess, bounds=bounds, method='L-BFGS-B')
    return result.x

mbg_rbergomi_params = rbergomi_model_params(df_mbg_v3['Log_Returns'].values)
print(f"Estimated rBergomi parameters for MBG.DE: {mbg_rbergomi_params}")

msft_rbergomi_params = rbergomi_model_params(df_msft_v3['Log_Returns'].values)
print(f"Estimated rBergomi parameters for MSFT: {msft_rbergomi_params}")

# simulate_heston_paths function generates synthetic price paths for a given asset using the Heston model
# It takes as inputs the initial price (S0), model parameters (kappa, theta, sigma, rho, v0), the total time period (T), the number of time steps (N), and the number of simulated paths (M)
# It initializes arrays to store the simulated prices and volatilities by setting the initial values to S0 and v0
# Then it iterates over each time step, generating random variables z1 and z2 that are correlated via rho, and the volatility at each time step is updated using these random variables and the Heston model's stochastic differential equation
# The price is updated using an exponential function that incorporates the new volatility value, and the loop continues for all time steps, simulating M paths
# The function returns the array of simulated prices
def simulate_heston_paths(S0, params, T=1, N=252, M=10):
    kappa, theta, sigma, rho, v0 = params
    dt = T / N
    prices = np.zeros((N + 1, M))
    prices[0] = S0
    v = np.zeros((N + 1, M))
    v[0] = v0
    for t in range(1, N + 1):
        z1 = np.random.normal(size=M)
        z2 = rho * z1 + np.sqrt(1 - rho**2) * np.random.normal(size=M)
        v[t] = np.abs(v[t-1] + kappa * (theta - v[t-1]) * dt + sigma * np.sqrt(v[t-1] * dt) * z1)
        prices[t] = prices[t-1] * np.exp((0 - 0.5 * v[t]) * dt + np.sqrt(v[t] * dt) * z2)
    return prices

# Function to simulate Bates model paths
def simulate_bates_paths(S0, params, T=1, N=252, M=10):
    kappa, theta, sigma, rho, v0, lambda_, mu_j, sigma_j = params
    dt = T / N
    prices = np.zeros((N + 1, M))
    prices[0] = S0
    v = np.zeros((N + 1, M))
    v[0] = v0
    for t in range(1, N + 1):
        z1 = np.random.normal(size=M)
        z2 = rho * z1 + np.sqrt(1 - rho**2) * np.random.normal(size=M)
        jump = (np.random.poisson(lambda_ * dt, M) * (np.exp(mu_j + sigma_j * np.random.normal(size=M)) - 1))
        v[t] = np.abs(v[t-1] + kappa * (theta - v[t-1]) * dt + sigma * np.sqrt(v[t-1] * dt) * z1)
        prices[t] = prices[t-1] * np.exp((0 - 0.5 * v[t]) * dt + np.sqrt(v[t] * dt) * z2) * (1 + jump)
    return prices

# Function to simulate VG-CIR model paths
def simulate_vg_cir_paths(S0, params, T=1, N=252, M=10):
    kappa, theta, sigma, v0, sigma_vg, theta_vg, kappa_vg = params
    dt = T / N
    prices = np.zeros((N + 1, M))
    prices[0] = S0
    v = np.zeros((N + 1, M))
    v[0] = v0
    for t in range(1, N + 1):
        z1 = np.random.normal(size=M)
        z2 = np.random.normal(size=M)
        v[t] = np.abs(v[t-1] + kappa * (theta - v[t-1]) * dt + sigma * np.sqrt(v[t-1] * dt) * z1)
        prices[t] = prices[t-1] * np.exp((0 - 0.5 * v[t]) * dt + np.sqrt(v[t] * dt) * z2)
    return prices

# Function to simulate rBergomi model paths
def simulate_rbergomi_paths(S0, params, T=1, N=252, M=10):
    eta, H, xi = params
    dt = T / N
    prices = np.zeros((N + 1, M))
    prices[0] = S0
    v = np.zeros((N + 1, M))
    v[0] = xi
    for t in range(1, N + 1):
        z1 = np.random.normal(size=M)
        z2 = np.random.normal(size=M)
        fBM = np.random.normal(size=M) * dt ** H
        v[t] = np.abs(v[t-1] + eta * np.sqrt(v[t-1] * dt) * fBM)
        prices[t] = prices[t-1] * np.exp((0 - 0.5 * v[t]) * dt + np.sqrt(v[t] * dt) * z2)
    return prices

# Simulating paths using estimated parameters and combining all synthetic paths for each stock
S0_mbg = df_mbg_v3['Close'].iloc[-1]
S0_msft = df_msft_v3['Close'].iloc[-1]

synthetic_paths_mbg_heston = simulate_heston_paths(S0_mbg, mbg_params)
synthetic_paths_msft_heston = simulate_heston_paths(S0_msft, msft_params)

synthetic_paths_mbg_bates = simulate_bates_paths(S0_mbg, mbg_bates_params)
synthetic_paths_msft_bates = simulate_bates_paths(S0_msft, msft_bates_params)

synthetic_paths_mbg_vg_cir = simulate_vg_cir_paths(S0_mbg, mbg_vg_cir_params)
synthetic_paths_msft_vg_cir = simulate_vg_cir_paths(S0_msft, msft_vg_cir_params)

synthetic_paths_mbg_rbergomi = simulate_rbergomi_paths(S0_mbg, mbg_rbergomi_params)
synthetic_paths_msft_rbergomi = simulate_rbergomi_paths(S0_msft, msft_rbergomi_params)

synthetic_paths_combined_mbg = np.hstack((synthetic_paths_mbg_heston, synthetic_paths_mbg_bates, synthetic_paths_mbg_vg_cir, synthetic_paths_mbg_rbergomi))
synthetic_paths_combined_msft = np.hstack((synthetic_paths_msft_heston, synthetic_paths_msft_bates, synthetic_paths_msft_vg_cir, synthetic_paths_msft_rbergomi))

# Create training and testing sets
def create_train_test(paths, look_back=30):
    X_train, y_train, X_test, y_test = [], [], [], []
    for path in paths.T:
        for i in range(look_back, len(path) - 30):
            X_train.append(path[i-look_back:i])
            y_train.append(path[i])
        for i in range(len(path) - 30, len(path) - 1):
            X_test.append(path[i-look_back:i])
            y_test.append(path[i])
    return np.array(X_train), np.array(y_train), np.array(X_test), np.array(y_test)

look_back = 30
X_train_mbg_v3, y_train_mbg_v3, X_test_mbg_v3, y_test_mbg_v3 = create_train_test(synthetic_paths_combined_mbg, look_back)
X_train_msft_v3, y_train_msft_v3, X_test_msft_v3, y_test_msft_v3 = create_train_test(synthetic_paths_combined_msft, look_back)

# Reshaping input data to 3D for LSTM [samples, time steps, features]
X_train_mbg_v3 = X_train_mbg_v3.reshape(X_train_mbg_v3.shape[0], look_back, 1)
X_test_mbg_v3 = X_test_mbg_v3.reshape(X_test_mbg_v3.shape[0], look_back, 1)
X_train_msft_v3 = X_train_msft_v3.reshape(X_train_msft_v3.shape[0], look_back, 1)
X_test_msft_v3 = X_test_msft_v3.reshape(X_test_msft_v3.shape[0], look_back, 1)

# Define LSTM model
def create_lstm_model(neurons, input_shape, loss):
    model = Sequential()
    model.add(LSTM(neurons, return_sequences=True, input_shape=input_shape))
    model.add(LSTM(neurons))
    model.add(Dense(1))
    model.compile(optimizer=Adam(), loss=loss)
    return model

# Model parameters
epochs = 30
batch_size_mbg = 16
batch_size_msft = 32

# Training of LSTM models for both datasets
model_mbg_v3 = create_lstm_model(20, (look_back, 1), 'mean_absolute_error')
model_mbg_v3.fit(X_train_mbg_v3, y_train_mbg_v3, epochs=epochs, batch_size=batch_size_mbg, validation_data=(X_test_mbg_v3, y_test_mbg_v3))

model_msft_v3 = create_lstm_model(200, (look_back, 1), 'mean_squared_error')
model_msft_v3.fit(X_train_msft_v3, y_train_msft_v3, epochs=epochs, batch_size=batch_size_msft, validation_data=(X_test_msft_v3, y_test_msft_v3))

# evaluate_model function computes the test loss and generates predictions
# It takes a model and test data (X_test and y_test) as inputs and computes the test loss using model.evaluate(X_test, y_test), which returns the loss value
# It then prints the test loss, and generates predictions using model.predict(X_test), and returns these predictions.
def evaluate_model(model, X_test, y_test):
    test_loss = model.evaluate(X_test, y_test)
    print(f'test loss: {test_loss}')
    predictions = model.predict(X_test)
    return predictions

predictions_mbg_v3 = evaluate_model(model_mbg_v3, X_test_mbg_v3, y_test_mbg_v3)
predictions_msft_v3 = evaluate_model(model_msft_v3, X_test_msft_v3, y_test_msft_v3)

# Approximation of hedge ratios as the difference between consecutive predictions
def compute_hedge_ratios(predictions):
    hedge_ratios = np.diff(predictions, axis=0)
    return hedge_ratios

hedge_ratios_mbg_v3 = compute_hedge_ratios(predictions_mbg_v3)
hedge_ratios_msft_v3 = compute_hedge_ratios(predictions_msft_v3)

# Results for the last 30 days
print("predicted prices for MBG.DE (last 30 days):")
print(predictions_mbg_v3[-30:])
print("hedge ratios for MBG.DE (last 30 days):")
print(hedge_ratios_mbg_v3[-30:])

print("predicted prices for MSFT (last 30 days):")
print(predictions_msft_v3[-30:])
print("hedge ratios for MSFT (last 30 days):")
print(hedge_ratios_msft_v3[-30:])


# Plotting of predicted prices and actual prices for the last 30 days for MBG.DE
plt.figure(figsize=(14, 7))
plt.plot(predictions_mbg_v3[-30:], label='predicted prices')
plt.plot(y_test_mbg_v3[-30:], label='actual prices')
plt.title('Predicted vs Actual prices for MBG.DE (last 30 days)')
plt.xlabel('Time steps')
plt.ylabel('Price')
plt.legend()
plt.show()

# Plotting hedge ratios for the last 30 days for MBG.DE
plt.figure(figsize=(14, 7))
plt.plot(hedge_ratios_mbg_v3[-30:], label='hedge ratios')
plt.title('Hedge ratios for MBG.DE (last 30 days)')
plt.xlabel('Time steps')
plt.ylabel('Hedge ratio')
plt.legend()
plt.show()

# Plotting predicted prices and actual prices for the last 30 days for MSFT
plt.figure(figsize=(14, 7))
plt.plot(predictions_msft_v3[-30:], label='predicted prices')
plt.plot(y_test_msft_v3[-30:], label='actual prices')
plt.title('Predicted vs Actual prices for MSFT (last 30 days)')
plt.xlabel('Time steps')
plt.ylabel('Price')
plt.legend()
plt.show()

# Plotting hedge ratios for the last 30 days for MSFT
plt.figure(figsize=(14, 7))
plt.plot(hedge_ratios_msft_v3[-30:], label='hedge ratios')
plt.title('Hedge ratios for MSFT (last 30 days)')
plt.xlabel('Time steps')
plt.ylabel('Hedge ratio')
plt.legend()
plt.show()


# # III. Additional inputs

# In[ ]:


# Additional file paths (historical data for interest rates and performance index)
fedfunds_path = os.path.join(dir, 'Data', 'D2. Additional inputs', 'Macroeconomic indicators (interest rates)', 'Fed rates')
ecb_rates_path = os.path.join(dir, 'Data', 'D2. Additional inputs', 'Macroeconomic indicators (interest rates)', 'ECB rates')
spx_path = os.path.join(dir, 'Data', 'D2. Additional inputs', 'Market indicators (market indexes)', '^SPX')
daxi_path = os.path.join(dir, 'Data', 'D2. Additional inputs', 'Market indicators (market indexes)', '^GDAXI')

# Loading data
df_mbg_v4 = pd.read_csv(mbg_path)
df_msft_v4 = pd.read_csv(msft_path)
df_fedfunds = pd.read_csv(fedfunds_path)
df_ecb_rates = pd.read_csv(ecb_rates_path)
df_spx = pd.read_csv(spx_path)
df_daxi = pd.read_csv(daxi_path)

# Verifying columns in each DataFrame and renaming columns for consistency
print("Columns in df_fedfunds:", df_fedfunds.columns)
print("Columns in df_ecb_rates:", df_ecb_rates.columns)

if 'DATE' in df_fedfunds.columns:
    df_fedfunds.rename(columns={'DATE': 'Date'}, inplace=True)
if 'FEDFUNDS' in df_fedfunds.columns:
    df_fedfunds.rename(columns={'FEDFUNDS': 'Fed_Rate'}, inplace=True)

if 'DATE' in df_ecb_rates.columns:
    df_ecb_rates.rename(columns={'DATE': 'Date'}, inplace=True)
if 'Main refinancing operations - Minimum bid rate/fixed rate (date of changes) - Level (FM.D.U2.EUR.4F.KR.MRR_RT.LEV)' in df_ecb_rates.columns:
    df_ecb_rates.rename(columns={'Main refinancing operations - Minimum bid rate/fixed rate (date of changes) - Level (FM.D.U2.EUR.4F.KR.MRR_RT.LEV)': 'ECB_Rate'}, inplace=True)

# Converting date columns to datetime format
df_mbg_v4['Date'] = pd.to_datetime(df_mbg_v4['Date'])
df_msft_v4['Date'] = pd.to_datetime(df_msft_v4['Date'])
df_fedfunds['Date'] = pd.to_datetime(df_fedfunds['Date'])
df_ecb_rates['Date'] = pd.to_datetime(df_ecb_rates['Date'])
df_spx['Date'] = pd.to_datetime(df_spx['Date'])
df_daxi['Date'] = pd.to_datetime(df_daxi['Date'])

# Handling missing values in GDAXI
df_daxi.fillna(method='ffill', inplace=True)

# Normalizing the relevant columns in each DataFrame and merging DataFrames on 'Date'
scaler = MinMaxScaler()
df_mbg_v4[['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']] = scaler.fit_transform(df_mbg_v4[['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']])
df_msft_v4[['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']] = scaler.fit_transform(df_msft_v4[['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']])
df_fedfunds[['Fed_Rate']] = scaler.fit_transform(df_fedfunds[['Fed_Rate']])
df_ecb_rates[['ECB_Rate']] = scaler.fit_transform(df_ecb_rates[['ECB_Rate']])
df_spx[['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']] = scaler.fit_transform(df_spx[['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']])
df_daxi[['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']] = scaler.fit_transform(df_daxi[['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']])

df_mbg_merged = pd.merge(df_mbg_v4, df_daxi[['Date', 'Close']], on='Date', how='left', suffixes=('', '_DAX_Close'))
df_mbg_merged = pd.merge(df_mbg_merged, df_ecb_rates[['Date', 'ECB_Rate']], on='Date', how='left')
df_msft_merged = pd.merge(df_msft_v4, df_spx[['Date', 'Close']], on='Date', how='left', suffixes=('', '_SPX_Close'))
df_msft_merged = pd.merge(df_msft_merged, df_fedfunds[['Date', 'Fed_Rate']], on='Date', how='left')

# calculate_rsi function computes the Relative Strength Index (RSI) for a given dataframe
# It takes a dataframe (df), a column name (column), and a period (period) as inputs, and then calculates the difference (delta) between consecutive values in the specified column
# It then identifies gains (gain) as the mean of positive differences over the specified period, and losses (loss) as the mean of negative differences (converted to positive) over the same period 
# The relative strength (rs) is calculated as the ratio of average gain to average loss
# The RSI is then computed using the formula 100 - (100 / (1 + rs))
def calculate_rsi(df, column='Close', period=14):
    delta = df[column].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

df_mbg_merged['RSI'] = calculate_rsi(df_mbg_merged, column='Close')
df_msft_merged['RSI'] = calculate_rsi(df_msft_merged, column='Close')

# Defining columns to normalize and normalizing DataFrames
columns_to_normalize_mbg = ['Close', 'RSI', 'ECB_Rate', 'Close_DAX_Close']
columns_to_normalize_msft = ['Close', 'RSI', 'Fed_Rate', 'Close_SPX_Close']

scaler = MinMaxScaler()
df_mbg_merged[columns_to_normalize_mbg] = scaler.fit_transform(df_mbg_merged[columns_to_normalize_mbg])
df_msft_merged[columns_to_normalize_msft] = scaler.fit_transform(df_msft_merged[columns_to_normalize_msft])

# Filling NaN values with column means
df_mbg_merged[columns_to_normalize_mbg] = df_mbg_merged[columns_to_normalize_mbg].fillna(df_mbg_merged[columns_to_normalize_mbg].mean())
df_msft_merged[columns_to_normalize_msft] = df_msft_merged[columns_to_normalize_msft].fillna(df_msft_merged[columns_to_normalize_msft].mean())

# create_sequences function generates sequences and corresponding targets from time series data
# It takes as inputs the data (data) and the number of time steps for each sequence 
# The function iterates over the data to extract sequences of the specified length and their corresponding targets
# For each position i, it creates a sequence from data[i:i + time_steps] and a target as the value at data[i + time_steps]. 
# These sequences and targets are then collected into lists and converted to numpy arrays before being returned
def create_sequences(data, time_steps=30):
    sequences = []
    targets = []
    for i in range(len(data) - time_steps):
        sequence = data[i:i + time_steps]
        target = data[i + time_steps]
        sequences.append(sequence)
        targets.append(target)
    return np.array(sequences), np.array(targets)

# prepare_data function formats the data for an LSTM model
# It takes as inputs a dataframe (df), the names of the feature columns (feature_columns), the name of the target column (target_column), and the number of time steps for each sequence as inputs 
# The function extracts the feature data and target data from the dataframe as numpy arrays, and then calls the create_sequences function to generate sequences (X) and their corresponding targets (y) from the feature data
# It returns these sequences and targets
def prepare_data(df, feature_columns, target_column, time_steps=30):
    data = df[feature_columns].values
    target = df[target_column].values
    X, y = create_sequences(data, time_steps)
    return X, y

# List of configurations for inputs (cf report part 5.3)
configurations_mbg = [
    ['Close'],
    ['Close', 'RSI'],
    ['Close', 'ECB_Rate'],
    ['Close', 'Close_DAX_Close'],
    ['Close', 'RSI', 'ECB_Rate'],
    ['Close', 'RSI', 'Close_DAX_Close'],
    ['Close', 'Close_DAX_Close', 'ECB_Rate'],
    ['Close', 'RSI', 'ECB_Rate', 'Close_DAX_Close']
]

configurations_msft = [
    ['Close'],
    ['Close', 'RSI'],
    ['Close', 'Fed_Rate'],
    ['Close', 'Close_SPX_Close'],
    ['Close', 'RSI', 'Fed_Rate'],
    ['Close', 'RSI', 'Close_SPX_Close'],
    ['Close', 'Close_SPX_Close', 'Fed_Rate'],
    ['Close', 'RSI', 'Fed_Rate', 'Close_SPX_Close']
]

# build_model function constructs an LSTM neural network model by taking as inputs the shape of the input data (input_shape) and the number of neurons in each LSTM layer (neurons)
# It initializes a Sequential model and adds two LSTM layers, each with the specified number of neurons and ReLU activation
# The first LSTM layer is configured to return sequences, while the second one is not, and a dense layer with a single neuron is added as the output layer
# The model is then compiled using the Adam optimizer and mean squared error loss function, and the compiled model is returned
def build_model(input_shape, neurons):
    model = Sequential()
    model.add(LSTM(neurons, activation='relu', return_sequences=True, input_shape=input_shape))
    model.add(LSTM(neurons, activation='relu'))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

# train_and_evaluate function trains and evaluates an LSTM model using different configurations of feature columns from the input dataframe
# It takes as inputs the dataframe (df), a list of feature configurations (configurations), the name of the target column (target_column), the number of neurons in the LSTM layers (neurons), the number of training epochs (epochs), and the batch size (batch_size)
# For each configuration in configurations, it prints the configuration being used, and then prepares the data by calling prepare_data, which formats the feature and target data into sequences and their corresponding targets
# It also prints the shapes of the input features (X) and targets (y)
# Next, the function builds an LSTM model with the specified input shape and number of neurons by calling build_model, and trains the model using model.fit with the prepared input features and the first column of the target data, for the specified number of epochs and batch size
# After training, it generates predictions using model.predict and prints the shape of the predictions, and then calculates the deltas by comparing the predictions to the first column of the target data
# The deltas for the last 30 days are stored in the results dictionary, with the configuration tuple as the key
# The function returns the results dictionary containing the deltas for the last 30 days for each configuration
def train_and_evaluate(df, configurations, target_column, neurons, epochs, batch_size):
    results = {}
    time_steps = 30

    for config in configurations:
        print(f"\nTraining with configuration: {config}")
        X, y = prepare_data(df, config, target_column, time_steps)
        print(f"Shape of X: {X.shape}")
        print(f"Shape of y: {y.shape}")

        model = build_model((X.shape[1], X.shape[2]), neurons)
        model.fit(X, y[:, 0], epochs=epochs, batch_size=batch_size, verbose=1)  # Fit the model with the first column of y
        
        predictions = model.predict(X)
        print(f"Shape of predictions: {predictions.shape}")

        deltas = predictions - y[:, 0].reshape(-1, 1)  # Compare predictions to the first column of y
        results[tuple(config)] = deltas[-30:]  # Last 30 days of deltas
    return results

mbg_deltas_v4 = train_and_evaluate(df_mbg_merged, configurations_mbg, 'Close', 20, 30, 16)
msft_deltas_v4 = train_and_evaluate(df_msft_merged, configurations_msft, 'Close', 200, 30, 32)

# Plotting results
def plot_results(deltas, title):
    plt.figure(figsize=(14, 7))
    for config, delta in deltas.items():
        plt.plot(delta, label=str(config))
    plt.title(title)
    plt.xlabel("Days")
    plt.ylabel("Hedge ratios")
    plt.legend()
    plt.show()

plot_results(mbg_deltas_v4, "MBG hedge ratios comparison (last 30 days)")
plot_results(msft_deltas_v4, "MSFT hedge ratios comparison (last 30 days)")


# # IV. Market frictions

# In[ ]:


df_mbg_v5 = pd.read_csv(mbg_path, delimiter=",")
df_msft_v5 = pd.read_csv(msft_path, delimiter=",")

# Preprocessing data
def preprocess_data(df):
    df['Return'] = df['Adj Close'].pct_change()
    df['Volatility'] = df['Return'].rolling(window=30).std() * np.sqrt(252)
    df.dropna(inplace=True)
    return df

df_mbg_tc = preprocess_data(df_mbg_v5)
df_msft_tc = preprocess_data(df_msft_v5)

# Preparing data for use in the LSTM model by creating sequences of a specified length (look_back) and their corresponding target values
# Initializing empty lists to store the sequences (dataX) and targets (dataY), and looping through the dataframe to extract sequences and targets, appended to the lists
# Converting the lists to numpy arrays and returning them
# The look_back period is set to 30, and the datasets for MBG and MSFT are created using the 'Adj Close' column from the respective dataframes
def create_dataset(df, look_back=30):
    dataX, dataY = [], []
    for i in range(len(df) - look_back):
        dataX.append(df.iloc[i:(i + look_back), :].values)
        dataY.append(df['Adj Close'].iloc[i + look_back])
    return np.array(dataX), np.array(dataY)

look_back = 30
X_mbg_v5, y_mbg_v5 = create_dataset(df_mbg_tc[['Adj Close']], look_back)
X_msft_v5, y_msft_v5 = create_dataset(df_msft_tc[['Adj Close']], look_back)

# Defining LSTM model
def create_model():
    model = Sequential()
    model.add(LSTM(50, input_shape=(look_back, 1)))
    model.add(Dense(1))
    model.compile(loss='mae', optimizer='adam')
    return model

# Training and prediction for the last 30 days
model_mbg_v5 = create_model()
model_mbg_v5.fit(X_mbg_v5, y_mbg_v5, epochs=5, batch_size=32, verbose=1)

model_msft_v5 = create_model()
model_msft_v5.fit(X_msft_v5, y_msft_v5, epochs=5, batch_size=32, verbose=1)

pred_mbg_v5 = model_mbg_v5.predict(X_mbg[-30:])
pred_msft_v5 = model_msft_v5.predict(X_msft[-30:])

# Normalizing NN predictions
scaler_mbg = MinMaxScaler(feature_range=(0, 1))
scaler_msft = MinMaxScaler(feature_range=(0, 1))
scaled_pred_mbg = scaler_mbg.fit_transform(pred_mbg_v5)
scaled_pred_msft = scaler_msft.fit_transform(pred_msft_v5)

# Calculating BS deltas
def black_scholes_delta(S, K, T, r, sigma):
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    return norm.cdf(d1)

K = df_mbg_v5['Adj Close'].iloc[-30:].values[0] 
T = 30 / 252 
r = 0.01  

df_mbg_v5.loc[df_mbg_tc.index[-30:], 'BS_Delta'] = black_scholes_delta(df_mbg_tc['Adj Close'].iloc[-30:].values, K, T, r, df_mbg_tc['Volatility'].iloc[-30:].values)
df_msft_v5.loc[df_msft_tc.index[-30:], 'BS_Delta'] = black_scholes_delta(df_msft_tc['Adj Close'].iloc[-30:].values, K, T, r, df_msft_tc['Volatility'].iloc[-30:].values)

# Defining transaction costs rate (cf report part 5.4)
k = 0.001 

# Calculating the transaction costs for both MBG and MSFT based on the absolute changes in NN_Delta and the adjusted close price (Adj Close)
# Initializing the portfolio value (Portfolio_Value) for the last 30 days using the adjusted close price 30 days before the end
# For each of the last 30 days, the portfolio value is updated by calculating the new portfolio value based on the previous value, the current and previous NN_Delta, the adjusted close price, and the transaction costs 
# The exponential term np.exp(r * (1 / 252)) accounts for the risk-free rate over a single trading day, with the loop iterating to update the portfolio value for each day, considering transaction costs
df_mbg_tc['NN_Transaction_Cost'] = k * np.abs(df_mbg_tc['NN_Delta'].diff()) * df_mbg_v5['Adj Close']
df_msft_tc['NN_Transaction_Cost'] = k * np.abs(df_msft_tc['NN_Delta'].diff()) * df_msft_v5['Adj Close']

V0_mbg = df_mbg_tc['Adj Close'].iloc[-30]  
V0_msft = df_msft_tc['Adj Close'].iloc[-30] 

df_mbg_tc['Portfolio_Value'] = V0_mbg
df_msft_tc['Portfolio_Value'] = V0_msft

for t in range(1, 30):
    df_mbg.loc[df_mbg.index[-30 + t], 'Portfolio_Value'] = df_mbg_tc['NN_Delta'].iloc[-30 + t] * df_mbg_tc['Adj Close'].iloc[-30 + t] + \
                                               (df_mbg_tc['Portfolio_Value'].iloc[-30 + t - 1] - df_mbg_tc['NN_Delta'].iloc[-30 + t - 1] * df_mbg_tc['Adj Close'].iloc[-30 + t - 1]) * np.exp(r * (1 / 252)) - \
                                               df_mbg_tc['NN_Transaction_Cost'].iloc[-30 + t]

    df_msft.loc[df_msft.index[-30 + t], 'Portfolio_Value'] = df_msft_tc['NN_Delta'].iloc[-30 + t] * df_msft_tc['Adj Close'].iloc[-30 + t] + \
                                               (df_msft_tc['Portfolio_Value'].iloc[-30 + t - 1] - df_msft_tc['NN_Delta'].iloc[-30 + t - 1] * df_msft_tc['Adj Close'].iloc[-30 + t - 1]) * np.exp(r * (1 / 252)) - \
                                               df_msft_tc['NN_Transaction_Cost'].iloc[-30 + t]

import matplotlib.ticker as mtick

# Plotting results
fig, axs = plt.subplots(2, 2, figsize=(15, 10))

# Plotting hedge ratios with shaded areas
axs[0, 0].plot(range(1, 31), df_mbg_tc['NN_Delta'].iloc[-30:], label='NN hedge ratio')
axs[0, 0].plot(range(1, 31), df_mbg_tc['BS_Delta'].iloc[-30:], label='BS hedge ratio')
axs[0, 0].fill_between(range(1, 31), df_mbg_tc['NN_Delta'].iloc[-30:], alpha=0.2)
axs[0, 0].fill_between(range(1, 31), df_mbg_tc['BS_Delta'].iloc[-30:], alpha=0.2)
axs[0, 0].set_title('Hedge ratios (NN vs BS) - MBG')
axs[0, 0].set_xlabel('Time')
axs[0, 0].set_ylabel('Hedge ratio')
axs[0, 0].legend()

axs[0, 1].plot(range(1, 31), df_msft_tc['NN_Delta'].iloc[-30:], label='NN hedge ratio')
axs[0, 1].plot(range(1, 31), df_msft_tc['BS_Delta'].iloc[-30:], label='BS hedge ratio')
axs[0, 1].fill_between(range(1, 31), df_msft_tc['NN_Delta'].iloc[-30:], alpha=0.2)
axs[0, 1].fill_between(range(1, 31), df_msft_tc['BS_Delta'].iloc[-30:], alpha=0.2)
axs[0, 1].set_title('Hedge ratios (NN vs BS) - MSFT')
axs[0, 1].set_xlabel('Time')
axs[0, 1].set_ylabel('Hedge ratio')
axs[0, 1].legend()

plt.tight_layout()
plt.show()


# calculate_pnl function computes the profit and loss (PnL) for a given dataframe based on the specified hedge column. 
# It initializes a list pnl with zero for the first row, and for each subsequent row, it calculates the PnL as the difference between the current and previous close prices, adjusted by the previous value of the specified hedge column
# It returns the cumulative sum of the PnL values
def calculate_pnl(df, hedge_column):
    pnl = [0]  # Initialize with zero for the first row
    for i in range(1, len(df)):
        pnl.append(df['Close'].iloc[i] - df['Close'].iloc[i-1] * df[hedge_column].iloc[i-1])
    return np.cumsum(pnl)

# Selecting the last 30 days data for MBG and MSFT, calculating PnL and add it to the dataframe
df_mbg_last_30_tc = df_mbg_tc.iloc[-30:].copy()
df_msft_last_30_tc = df_msft_tc.iloc[-30:].copy()

df_mbg_last_30_tc['NN_PnL'] = calculate_pnl(df_mbg_last_30_tc, 'NN_Delta')
df_mbg_last_30_tc['BS_PnL'] = calculate_pnl(df_mbg_last_30_tc, 'BS_Delta')

df_msft_last_30_tc['NN_PnL'] = calculate_pnl(df_msft_last_30_tc, 'NN_Delta')
df_msft_last_30_tc['BS_PnL'] = calculate_pnl(df_msft_last_30_tc, 'BS_Delta')

# Print the last few rows to check
print("Final MBG data with BS calculations and PnL:")
print(df_mbg_last_30_tc[['Date', 'Close', 'BS_Delta', 'NN_Delta', 'NN_PnL', 'BS_PnL']])
print("Final MSFT data with BS calculations and PnL:")
print(df_msft_last_30_tc[['Date', 'Close', 'BS_Delta', 'NN_Delta', 'NN_PnL', 'BS_PnL']])

# Visualizing PnL
plt.figure(figsize=(14, 7))

plt.subplot(2, 1, 1)
plt.plot(range(1, 31), df_mbg_last_30_tc['NN_PnL'], label='NN PnL', color='blue')
plt.fill_between(range(1, 31), df_mbg_last_30_tc['NN_PnL'], color='blue', alpha=0.1)
plt.plot(range(1, 31), df_mbg_last_30_tc['BS_PnL'], label='BS PnL', color='red')
plt.fill_between(range(1, 31), df_mbg_last_30_tc['BS_PnL'], color='red', alpha=0.1)
plt.title('MBG PnL comparison')
plt.xlabel('Time')
plt.ylabel('PnL')
plt.legend()

plt.subplot(2, 1, 2)
plt.plot(range(1, 31), df_msft_last_30_tc['NN_PnL'], label='NN PnL', color='blue')
plt.fill_between(range(1, 31), df_msft_last_30_tc['NN_PnL'], color='blue', alpha=0.1)
plt.plot(range(1, 31), df_msft_last_30_tc['BS_PnL'], label='BS PnL', color='red')
plt.fill_between(range(1, 31), df_msft_last_30_tc['BS_PnL'], color='red', alpha=0.1)
plt.title('MSFT PnL comparison')
plt.xlabel('Time')
plt.ylabel('PnL')
plt.legend()

plt.tight_layout()
plt.show()


# In[ ]:


# Calculating the stats
def calculate_stats(df, column_name):
    return {
        'Mean': np.mean(df[column_name]),
        'Median': np.median(df[column_name]),
        'Std Dev': np.std(df[column_name]),
        'Min': np.min(df[column_name]),
        'Max': np.max(df[column_name])
    }

# Collecting stats in tables
mbg_stats = pd.DataFrame({
    'Metric': ['Mean', 'Median', 'Std Dev', 'Min', 'Max'],
    'Without TC': [calculate_stats(df_mbg_last_30, 'NN_Delta')[key] for key in ['Mean', 'Median', 'Std Dev', 'Min', 'Max']],
    'With TC': [calculate_stats(df_mbg_last_30_tc, 'NN_Delta')[key] for key in ['Mean', 'Median', 'Std Dev', 'Min', 'Max']]
})

msft_stats = pd.DataFrame({
    'Metric': ['Mean', 'Median', 'Std Dev', 'Min', 'Max'],
    'Without TC': [calculate_stats(df_msft_last_30, 'NN_Delta')[key] for key in ['Mean', 'Median', 'Std Dev', 'Min', 'Max']],
    'With TC': [calculate_stats(df_msft_last_30_tc, 'NN_Delta')[key] for key in ['Mean', 'Median', 'Std Dev', 'Min', 'Max']]
})

# Plotting the comparison of hedge ratios for the last 30 days
plt.figure(figsize=(14, 7))

plt.subplot(2, 1, 1)
plt.plot(range(1, 31), df_mbg_last_30['NN_Delta'], label='Without TC', color='blue')
plt.plot(range(1, 31), df_mbg_last_30_tc['NN_Delta'], label='With TC', color='red')
plt.fill_between(range(1, 31), df_mbg_last_30['NN_Delta'], color='blue', alpha=0.1)
plt.fill_between(range(1, 31), df_mbg_last_30_tc['NN_Delta'], color='red', alpha=0.1)
plt.title('MBG hedge ratio comparison (last 30 days)')
plt.xlabel('Time')
plt.ylabel('Hedge ratio')
plt.legend()

plt.subplot(2, 1, 2)
plt.plot(range(1, 31), df_msft_last_30['NN_Delta'], label='Without TC', color='blue')
plt.plot(range(1, 31), df_msft_last_30_tc['NN_Delta'], label='With TC', color='red')
plt.fill_between(range(1, 31), df_msft_last_30['NN_Delta'], color='blue', alpha=0.1)
plt.fill_between(range(1, 31), df_msft_last_30_tc['NN_Delta'], color='red', alpha=0.1)
plt.title('MSFT hedge ratio comparison (last 30 days)')
plt.xlabel('Time')
plt.ylabel('Hedge ratio')
plt.legend()

plt.tight_layout()
plt.show()

# Printing the stats for verification
print("MBG hedge ratio stats:")
print(mbg_stats)
print("\nMSFT hedge ratio stats:")
print(msft_stats)


# # V. Computational justifications for the choices in the neural network architecture and features (based on our datasets features)

# # a) Number of neurons in the LSTM NN

# In[ ]:


# We define a range of neurons to test for the LSTM models and initialize a dictionary to store the results
# For each number of neurons in the range, we create and train an LSTM model for both MBG and MSFT using the specified number of neurons
# The training is done for 20 epochs with a batch size of 32, and the verbose output is turned off
# After training, we generate predictions for the training data and evaluates the model using mean squared error (MSE) and mean absolute error (MAE) metrics. 
# The results, including the number of neurons, MSE, and MAE for both MBG and MSFT, are stored in the dictionary
neuron_range = [10, 20, 50, 100, 200]
results_v6 = {'neurons': [], 'mse_mbg': [], 'mae_mbg': [], 'mse_msft': [], 'mae_msft': []}

for num_neurons in neuron_range:
    # Creating and training the model
    model_mbg_v6 = create_lstm_model((n_steps, 1), num_neurons)
    model_mbg_v6.fit(X_mbg_v6, y_mbg_v6, epochs=20, batch_size=32, verbose=0)
    
    model_msft_v6 = create_lstm_model((n_steps, 1), num_neurons)
    model_msft_v6.fit(X_msft_v6, y_msft_v6, epochs=20, batch_size=32, verbose=0)
    
    # Predicting and evaluating the model
    predictions_mbg_v6 = model_mbg_v6.predict(X_mbg)
    mse_mbg = mean_squared_error(y_mbg_v6, predictions_mbg_v6)
    mae_mbg = mean_absolute_error(y_mbg_v6, predictions_mbg_v6)

    predictions_msft_v6 = model_msft_v6.predict(X_msft_v6)
    mse_msft = mean_squared_error(y_msft_v6, predictions_msft_v6)
    mae_msft = mean_absolute_error(y_msft_v6, predictions_msft_v6)
    
    # Store the results
    results_v6['neurons'].append(num_neurons)
    results_v6['mse_mbg'].append(mse_mbg)
    results_v6['mae_mbg'].append(mae_mbg)
    results_v6['mse_msft'].append(mse_msft)
    results_v6['mae_msft'].append(mae_msft)

# Converting results to DataFrame
results_df_v6 = pd.DataFrame(results)

# Plot the results
plt.figure(figsize=(14, 7))

plt.subplot(2, 1, 1)
plt.plot(results_df_v6['neurons'], results_df_v6['mse_mbg'], marker='o', label='MSE MBG')
plt.plot(results_df_v6['neurons'], results_df_v6['mae_mbg'], marker='o', label='MAE MBG')
plt.xlabel('Number of neurons')
plt.ylabel('Error')
plt.title('Error vs number of neurons (MBG)')
plt.legend()

plt.subplot(2, 1, 2)
plt.plot(results_df_v6['neurons'], results_df_v6['mse_msft'], marker='o', label='MSE MSFT')
plt.plot(results_df_v6['neurons'], results_df_v6['mae_msft'], marker='o', label='MAE MSFT')
plt.xlabel('Number of neurons')
plt.ylabel('Error')
plt.title('Error vs number of neurons (MSFT)')
plt.legend()

plt.tight_layout()
plt.show()

# Displaying the results
print(results_df_v6)


# # b) Loss function

# In[ ]:


# We train and evaluate the LSTM models for MBG and MSFT using two different loss functions: Mean Squared Error (MSE) and Mean Absolute Error (MAE)
# We initialize a dictionary to store the results and for each loss function, we create and compile LSTM models for both MBG and MSFT, as well as train the models using the specified loss function, and generate predictions for the training data
# The evaluation metrics (Root Mean Squared Error (RMSE) and Mean Absolute Error (MAE)) are calculated and stored in the results dictionary 
results = {}
for loss_function in ['mse', 'mae']:
    model_mbg_v7 = create_lstm_model((n_steps, 1), loss_function)
    model_msft_v7 = create_lstm_model((n_steps, 1), loss_function)

    model_mbg_v7.fit(X_mbg_v7, y_mbg_v7, epochs=20, batch_size=32, verbose=0)
    model_msft_v7.fit(X_msft_v7, y_msft_v7, epochs=20, batch_size=32, verbose=0)

    y_mbg_pred_v7 = model_mbg_v7.predict(X_mbg_v7)
    y_msft_pred_v7 = model_msft_v7.predict(X_msft_v7)

    results_v7[loss_function] = {
        'MBG RMSE': np.sqrt(mean_squared_error(y_mbg_v7, y_mbg_pred_v7)),
        'MBG MAE': mean_absolute_error(y_mbg_v7, y_mbg_pred_v7),
        'MSFT RMSE': np.sqrt(mean_squared_error(y_msft_v7, y_msft_pred_v7)),
        'MSFT MAE': mean_absolute_error(y_msft_v7, y_msft_pred_v7)
    }

# Displaying the results
results_df_v7 = pd.DataFrame(results_v7)
print(results_df_v7)


# # c) Batch size and number of epochs

# In[ ]:


# We define lists of batch sizes and epochs to test and we initialize an empty list to store the results
# We iterate over each combination of batch size and number of epochs, creating and compiling LSTM models for MBG and MSFT 
# The start time is recorded before training the models with the current batch size and number of epochs
# The training time is calculated after training, and predictions for the training data are generated for both MBG and MSFT models
# The results for the current combination, including the batch size, number of epochs, RMSE for MBG, RMSE for MSFT, and training time, are stored in the results list
# The results list is then converted to a DataFrame and printed for an easier analysis

batch_sizes = [16, 32, 64]
epochs_list = [20, 30, 50]
initial_results = []

for batch_size in batch_sizes:
    for epochs in epochs_list:
        model_mbg_v8 = create_lstm_model((n_steps, 1))
        model_msft_v8 = create_lstm_model((n_steps, 1))
        
        start_time = time.time()
        model_mbg_v8.fit(X_mbg_v8, y_mbg_v8, epochs=epochs, batch_size=batch_size, verbose=0)
        model_msft_v8.fit(X_msft_v8, y_msft_v8, epochs=epochs, batch_size=batch_size, verbose=0)
        training_time = time.time() - start_time
        
        y_mbg_pred_v8 = model_mbg_v8.predict(X_mbg_v8)
        y_msft_pred_v8 = model_msft_v8.predict(X_msft_v8)
        
        initial_results_v8.append({
            'Batch Size': batch_size,
            'Epochs': epochs,
            'MBG RMSE': np.sqrt(mean_squared_error(y_mbg_v8, y_mbg_pred_v8)),
            'MSFT RMSE': np.sqrt(mean_squared_error(y_msft_v8, y_msft_pred_v8)),
            'Training Time (s)': training_time
        })

initial_results_df_v8 = pd.DataFrame(initial_results_v8)
print("Initial manual exploration results:")
print(initial_results_df_v8)

import seaborn as sns

# Plotting RMSE for different batch sizes and epochs
plt.figure(figsize=(12, 6))

for batch_size in batch_sizes:
    subset = initial_results_df_v8[initial_results_df_v8['Batch size'] == batch_size]
    plt.plot(subset['Epochs'], subset['MBG RMSE'], label=f'MBG RMSE (Batch size={batch_size})')
    plt.plot(subset['Epochs'], subset['MSFT RMSE'], label=f'MSFT RMSE (Batch size={batch_size})')

plt.xlabel('Epochs')
plt.ylabel('RMSE')
plt.title('RMSE for different batch sizes and epochs')
plt.legend()
plt.show()

