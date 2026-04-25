import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# --- PHASE 1: DATA LOADING (External CSV) ---
print("Loading real-world sales data from Sales_Data.csv...")

# Loading the CSV you uploaded
df = pd.read_csv('c:/Users/adity/Desktop/Thiranex/Task 4/Sales_Data.csv')

# Preprocessing: Convert Order_Date to datetime
# Using dayfirst=False as the sample shows MM/DD/YYYY or MM-DD-YYYY
df['Order_Date'] = pd.to_datetime(df['Order_Date'], errors='coerce')

# Drop rows where dates couldn't be parsed
df = df.dropna(subset=['Order_Date'])

# Feature Engineering: Calculate Revenue and Profit
df['Total_Revenue'] = df['Units_Sold'] * df['Unit_SellingPrice']
df['Total_Cost'] = df['Units_Sold'] * df['Unit_MakingCost']
df['Total_Profit'] = df['Total_Revenue'] - df['Total_Cost']

# Time Features
df['Month'] = df['Order_Date'].dt.month
df['DayOfWeek'] = df['Order_Date'].dt.day_name()
df = df.sort_values('Order_Date')

print(f"Data Loaded. Shape: {df.shape}")

# --- PHASE 2: END-TO-END ANALYSIS ---

# 1. Revenue Trends Over Time
print("\nAnalyzing Revenue Trends...")
plt.figure(figsize=(12, 6))
# Resampling to Month to make the trend line smoother for the report
monthly_trend = df.set_index('Order_Date')['Total_Revenue'].resample('ME').sum()
plt.plot(monthly_trend.index, monthly_trend.values, marker='o', color='royalblue', linewidth=2) # type: ignore
plt.title('Monthly Revenue Trend (Real-world Data)')
plt.xlabel('Date')
plt.ylabel('Total Revenue ($)')
plt.grid(True, alpha=0.3)
plt.show()

# 2. Performance by Item Type (Domain Specific Insight)
print("Analyzing Item Type performance...")
plt.figure(figsize=(12, 6))
item_perf = df.groupby('Item_Type')['Total_Profit'].sum().sort_values(ascending=False)
sns.barplot(x=item_perf.values, y=item_perf.index, palette='viridis')
plt.title('Total Profit by Item Type')
plt.xlabel('Total Profit ($)')
plt.show()

# --- PHASE 3: PREDICTION (Forecasting) ---
print("\nBuilding Profit Prediction Model...")

# We'll predict Profit trends based on days from the first order
df['DayCount'] = (df['Order_Date'] - df['Order_Date'].min()).dt.days
X = df[['DayCount']]
y = df['Total_Profit']

model = LinearRegression()
model.fit(X, y)

# Predict for the next 60 days
last_day = df['DayCount'].max()
future_days = np.arange(last_day, last_day + 60).reshape(-1, 1)
future_preds = model.predict(future_days)

# Visualize the Forecast
plt.figure(figsize=(12, 6))
plt.scatter(df['DayCount'], df['Total_Profit'], alpha=0.3, label='Actual Daily Profit')
plt.plot(df['DayCount'], model.predict(X), color='red', label='Trend Line')
plt.plot(future_days, future_preds, color='green', linestyle='--', linewidth=3, label='60-Day Forecast')
plt.title('Retail Profit Forecast')
plt.xlabel('Days since start of records')
plt.ylabel('Profit ($)')
plt.legend()
plt.show()

# --- FINAL CONCLUSIONS ---
print("\n--- Project Conclusions ---")
top_item = item_perf.index[0]
print(f"1. Top Performer: '{top_item}' is the most profitable category.")
print(f"2. Profitability: Average profit per order is ${df['Total_Profit'].mean():.2f}.")
print(f"3. Forecast: The trend suggests a {'steady increase' if model.coef_[0] > 0 else 'slight decline'} in profits for the next two months.")