import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import date
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler

# Fetch stock data
def get_stock_data(ticker, start, end):
    stock = yf.download(ticker, start=start, end=end)
    stock['Date'] = stock.index
    return stock

# Prepare the data
def prepare_data(stock):
    stock['SMA_10'] = stock['Close'].rolling(window=10).mean()  # 10-day moving average
    stock['SMA_50'] = stock['Close'].rolling(window=50).mean()  # 50-day moving average
    stock['RSI'] = 100 - (100 / (1 + stock['Close'].pct_change().rolling(14).mean()))  # Relative Strength Index
    stock['Prediction'] = stock['Close'].shift(-1)  # Predicting next day's close price
    stock.dropna(inplace=True)
    
    X = stock[['Open', 'High', 'Low', 'Volume', 'SMA_10', 'SMA_50', 'RSI']]
    y = stock['Prediction']
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    return train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Train the model
def train_model(X_train, y_train):
    model = RandomForestRegressor(n_estimators=200, random_state=42)
    model.fit(X_train, y_train)
    return model

# Predict and evaluate
def predict(model, X_test, y_test):
    predictions = model.predict(X_test)
    error = mean_absolute_error(y_test, predictions)
    accuracy = r2_score(y_test, predictions) * 100
    return predictions, error, accuracy

# Menu-driven system
def main():
    companies = {
        "1": "AAPL",
        "2": "GOOGL",
        "3": "MSFT",
        "4": "TSLA"
    }
    
    while True:
        print("\nStock Market Prediction")
        print("1. Apple (AAPL)")
        print("2. Google (GOOGL)")
        print("3. Microsoft (MSFT)")
        print("4. Tesla (TSLA)")
        print("5. Exit")
        choice = input("Enter your choice: ")
        
        if choice == "5":
            print("Exiting...")
            break
        elif choice in companies:
            ticker = companies[choice]
            start_date = "2023-01-01"
            end_date = date.today()
            
            stock_data = get_stock_data(ticker, start_date, end_date)
            X_train, X_test, y_train, y_test = prepare_data(stock_data)
            model = train_model(X_train, y_train)
            predictions, error, accuracy = predict(model, X_test, y_test)
            
            # Plot results
            plt.figure(figsize=(10, 5))
            plt.plot(y_test.values, label="Actual Price")
            plt.plot(predictions, label="Predicted Price", linestyle='dashed')
            plt.xlabel("Days")
            plt.ylabel("Stock Price")
            plt.legend()
            plt.title(f"Stock Price Prediction for {ticker}\nMean Absolute Error: {error:.2f}, Accuracy: {accuracy:.2f}%")
            plt.show()
        else:
            print("Invalid choice, please try again.")

if __name__ == "__main__":
    main()
