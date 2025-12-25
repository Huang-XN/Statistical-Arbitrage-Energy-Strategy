import yfinance as yf
import matplotlib.pyplot as plt

### Fetch annualhistorical stock data for Apple Inc. (AAPL)

data = yf.download("AAPL", start="2024-01-01", end="2025-01-01", interval="1d")

### Display the close price data
data['Close'].plot(title='Apple Stock Price - The Start of My Revenge')
plt.show()

### Print the first five rows of the data
print(data.head())