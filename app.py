import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from scipy.stats import linregress
import plotly.graph_objects as go
from datetime import datetime, timedelta

# Genesis block date
genesis = datetime(2009, 1, 3)

st.title('Bitcoin Power Law Dashboard in Terms of Gold')

# Fetch historical data
@st.cache_data(ttl=3600)  # Cache for 1 hour
def fetch_data():
    # BTC data (starts ~2014 in yfinance)
    btc = yf.download('BTC-USD', start='2014-01-01', end=datetime.now().strftime('%Y-%m-%d'), progress=False)
    btc.columns = btc.columns.get_level_values(0)  # Flatten MultiIndex if present
    # Gold futures (per oz in USD)
    gold = yf.download('GC=F', start='2014-01-01', end=datetime.now().strftime('%Y-%m-%d'), progress=False)
    gold.columns = gold.columns.get_level_values(0)  # Flatten MultiIndex if present
    # Align dates and compute BTC in gold oz
    df = pd.DataFrame({'BTC_USD': btc['Close'], 'Gold_USD': gold['Close']}).dropna()
    df['BTC_in_Gold'] = df['BTC_USD'] / df['Gold_USD']
    # Days since genesis
    df['Days'] = (df.index - genesis).days
    return df

df = fetch_data()

# Fit power law: log(price) = log(A) + B * log(days)
df_fit = df[(df['Days'] > 0) & (df['BTC_in_Gold'] > 0)].copy()
log_days = np.log(df_fit['Days'])
log_price = np.log(df_fit['BTC_in_Gold'])
slope, intercept, r_value, p_value, std_err = linregress(log_days, log_price)

# Fair value function
def fair_value(days):
    return np.exp(intercept) * days ** slope

# Current values
current_date = datetime.now()
current_days = (current_date - genesis).days
current_btc_usd = df['BTC_USD'].iloc[-1]
current_gold_usd = df['Gold_USD'].iloc[-1]
current_btc_gold = df['BTC_in_Gold'].iloc[-1]
current_fair = fair_value(current_days)
valuation = (current_btc_gold - current_fair) / current_fair * 100

# Display key metrics
col1, col2, col3, col4 = st.columns(4)
col1.metric("Current BTC (USD)", f"${current_btc_usd:,.0f}")
col2.metric("Current Gold/oz (USD)", f"${current_gold_usd:,.0f}")
col3.metric("Current BTC in Gold oz", f"{current_btc_gold:.2f}")
col4.metric("Fair Value (Power Law)", f"{current_fair:.2f}")

col5, col6, col7 = st.columns(3)
col5.metric("Valuation (% over/under fair)", f"{valuation:.1f}%")
col6.metric("Power Law Exponent (B)", f"{slope:.2f}")
col7.metric("Fit R²", f"{r_value**2:.2f}")

# Plot log-log chart with fit and projection
st.subheader('Log-Log Plot: BTC in Gold oz vs. Days Since Genesis')
fig = go.Figure()

# Actual data
fig.add_trace(go.Scatter(x=df['Days'], y=df['BTC_in_Gold'], mode='lines', name='Actual BTC in Gold', line=dict(color='blue')))

# Power law fit line (extend 5 years into future)
fit_days = np.arange(df['Days'].min(), current_days + 365 * 5)
fit_price = fair_value(fit_days)
fig.add_trace(go.Scatter(x=fit_days, y=fit_price, mode='lines', name='Power Law Fit', line=dict(color='red', dash='dash')))

# Current point
fig.add_trace(go.Scatter(x=[current_days], y=[current_btc_gold], mode='markers', name='Current', marker=dict(color='green', size=10)))

# Layout
fig.update_layout(
    xaxis_type='log',
    yaxis_type='log',
    xaxis_title='Days Since Genesis (Log Scale)',
    yaxis_title='BTC in Gold oz (Log Scale)',
    height=600,
    hovermode='x unified'
)

st.plotly_chart(fig, use_container_width=True)

# Data table (last 10 days)
st.subheader('Recent Data')
st.dataframe(df.tail(10)[['BTC_USD', 'Gold_USD', 'BTC_in_Gold', 'Days']])

# Explanation
st.markdown("""
### How This Works
- **Bitcoin Power Law in Gold Terms**: Models BTC's value in ounces of gold as a power law function of time: BTC_in_Gold ≈ A * days^B.
- **Data Sources**: Yahoo Finance for BTC-USD and gold futures (GC=F).
- **Fit Details**: Linear regression on log-log data. The exponent (B) typically hovers around 5-6 based on historical fits.
- **Fair Value**: Projected value today based on the fit. Positive valuation means BTC is over fair value in gold terms.
""")
