

# Importing necessary libraries
import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout
import matplotlib.pyplot as plt
import seaborn as sns
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score



# Download historical data for Infosys (INFY)
ticker = yf.Ticker("INFY.NS")
df = ticker.history(period="2y", interval="1d")

# Display the first few rows of the dataframe
print("Dataframe head:")
df.head()





# Data visualization
plt.figure(figsize=(12, 6))
plt.plot(df['Close'], label='Closing Price')
plt.title('Infosys Stock Price Over Time')
plt.xlabel('Date')
plt.ylabel('Closing Price')
plt.legend()
plt.show()




# Moving averages
df['MA20'] = df['Close'].rolling(window=20).mean()
df['STD20'] = df['Close'].rolling(window=20).std()

plt.figure(figsize=(12, 6))
plt.plot(df['Close'], label='Closing Price')
plt.plot(df['MA20'], label='20-Day Moving Average')
plt.fill_between(df.index, df['MA20'] - df['STD20'], df['MA20'] + df['STD20'], alpha=0.2)
plt.legend()
plt.show()





# Statistical analysis
correlation_matrix = df.corr()
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Correlation Matrix')
plt.show()





# Data preparation for LSTM
X = df["Close"].values.reshape(-1, 1)
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

sequence_length = 5
sequences = []
target = []

for i in range(len(X_scaled) - sequence_length):
    seq = X_scaled[i:i + sequence_length, 0]
    label = X_scaled[i + sequence_length, 0]
    sequences.append(seq)
    target.append(label)
    
X_seq, y_seq = np.array(sequences), np.array(target)





# Creating a new dataframe with sequences and target
df_sequences = pd.DataFrame(X_seq, columns=[f'Day_{i+1}' for i in range(sequence_length)])
df_sequences['Target'] = y_seq

# Displaying the new dataframe with sequences and target
print("\nDataframe with Sequences and Target:")
df_sequences.head()




# Displaying the differences in the dataframe after creating sequences and target
print("\nDifferences in the Dataframe:")
df_sequences.diff().dropna().head()





# Splitting the data into training, validation, and testing sets
split_ratio_train = 0.8
split_ratio_val = 0.9

split_index_train = int(split_ratio_train * len(X_seq))
split_index_val = int(split_ratio_val * len(X_seq))

X_train, X_val, X_test = X_seq[:split_index_train], X_seq[split_index_train:split_index_val], X_seq[split_index_val:]
y_train, y_val, y_test = y_seq[:split_index_train], y_seq[split_index_train:split_index_val], y_seq[split_index_val:]





# Creating LSTM model
model = Sequential([
    LSTM(units=64, return_sequences=True, input_shape=(sequence_length, 1)),
    Dropout(0.3),
    LSTM(units=32),
    Dropout(0.4),
    Dense(1)
])





# Compile the model with learning rate and MAE metric
model.compile(loss='mean_squared_error', optimizer=Adam(learning_rate=0.001), metrics=['mae'])





# Train the model with validation data and early stopping
history = model.fit(X_train, y_train, epochs=100, batch_size=64, verbose=2,
                    validation_data=(X_val, y_val),
                    callbacks=[EarlyStopping(patience=10, restore_best_weights=True)])





# Visualize the training and validation loss over epochs
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Training and Validation Loss over Epochs')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()





# Evaluate on the training set
train_predictions = model.predict(X_train)
train_predictions = scaler.inverse_transform(train_predictions)
y_train_actual = scaler.inverse_transform(y_train.reshape(-1, 1))





# Visualize the predictions on the training set
plt.plot(y_train_actual, label='Actual Stock Price (Training)')
plt.plot(train_predictions, label='Predicted Stock Price (Training)')
plt.title('Infosys Stock Price Prediction on Training Set using LSTM')
plt.xlabel('Time')
plt.ylabel('Stock Price')
plt.legend()
plt.show()





# Calculate metrics for the training set
train_mae = mean_absolute_error(y_train_actual, train_predictions)
train_mse = mean_squared_error(y_train_actual, train_predictions)
train_r2 = r2_score(y_train_actual, train_predictions)*100

print('\nTraining Metrics:')
print(f'Mean Absolute Error (MAE): {train_mae:.2f}')
print(f'Mean Squared Error (MSE): {train_mse:.2f}')
print(f'R-squared (R2): {train_r2:.2f}%')





# Evaluate on the validation set
val_predictions = model.predict(X_val)
val_predictions = scaler.inverse_transform(val_predictions)
y_val_actual = scaler.inverse_transform(y_val.reshape(-1, 1))

# Visualize the predictions on the validation set
plt.plot(y_val_actual, label='Actual Stock Price (Validation)')
plt.plot(val_predictions, label='Predicted Stock Price (Validation)')
plt.title('Infosys Stock Price Prediction on Validation Set using LSTM')
plt.xlabel('Time')
plt.ylabel('Stock Price')
plt.legend()
plt.show()





# Calculate metrics for the validation set
val_mae = mean_absolute_error(y_val_actual, val_predictions)
val_mse = mean_squared_error(y_val_actual, val_predictions)
val_r2 = r2_score(y_val_actual, val_predictions)*100

print('\nValidation Metrics:')
print(f'Mean Absolute Error (MAE): {val_mae:.2f}')
print(f'Mean Squared Error (MSE): {val_mse:.2f}')
print(f'R-squared (R2): {val_r2:.2f}%')





# Evaluate on the test set
test_predictions = model.predict(X_test)
test_predictions = scaler.inverse_transform(test_predictions)
y_test_actual = scaler.inverse_transform(y_test.reshape(-1, 1))

# Visualize the predictions on the test set
plt.plot(y_test_actual, label='Actual Stock Price')
plt.plot(test_predictions, label='Predicted Stock Price')
plt.title('Infosys Stock Price Prediction using LSTM')
plt.xlabel('Time')
plt.ylabel('Stock Price')
plt.legend()
plt.show()





# Evaluate different metrics on the test set
mae = mean_absolute_error(y_test_actual, test_predictions)
mse = mean_squared_error(y_test_actual, test_predictions)
rmse = np.sqrt(mse)
r2 = r2_score(y_test_actual, test_predictions) * 100

print('\nTest Set Metrics:')
print(f'Mean Absolute Error (MAE): {mae:.2f}')
print(f'Mean Squared Error (MSE): {mse:.2f}')
print(f'Root Mean Squared Error (RMSE): {rmse:.2f}')
print(f'R-squared (R2): {r2:.2f}%')

