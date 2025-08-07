import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from scipy.stats import linregress
import plotly.graph_objects as go
from datetime import datetime, timedelta
import time

# Configure page for auto-refresh
st.set_page_config(page_title="Bitcoin Power Law Dashboard", layout="wide")

# Genesis block date
genesis = datetime(2009, 1, 3)

st.title('Bitcoin Power Law Dashboard in Terms of Gold (Real-time)')

# Add auto-refresh functionality
auto_refresh = st.sidebar.checkbox("Auto-refresh (every 5 minutes)", value=True)
refresh_interval = st.sidebar.selectbox("Refresh interval", [1, 5, 10, 30], index=1)

if auto_refresh:
    # Auto-refresh the page
    time.sleep(refresh_interval * 60)
    st.rerun()

# Display last update time
st.sidebar.write(f"Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

# Fetch historical data with much shorter cache for real-time updates
@st.cache_data(ttl=300)  # Cache for 5 minutes only
def fetch_data():
    try:
        # BTC data (starts ~2014 in yfinance)
        btc = yf.download('BTC-USD', start='2014-01-01', end=datetime.now().strftime('%Y-%m-%d'), progress=False)
        if btc.columns.nlevels > 1:
            btc.columns = btc.columns.get_level_values(0)  # Flatten MultiIndex if present
        
        # Gold futures (per oz in USD)
        gold = yf.download('GC=F', start='2014-01-01', end=datetime.now().strftime('%Y-%m-%d'), progress=False)
        if gold.columns.nlevels > 1:
            gold.columns = gold.columns.get_level_values(0)  # Flatten MultiIndex if present
        
        # Align dates and compute BTC in gold oz
        df = pd.DataFrame({'BTC_USD': btc['Close'], 'Gold_USD': gold['Close']}).dropna()
        df['BTC_in_Gold'] = df['BTC_USD'] / df['Gold_USD']
        
        # Days since genesis
        df['Days'] = (df.index - genesis).days
        
        return df
    except Exception as e:
        st.error(f"Error fetching data: {e}")
        return None

# Add manual refresh button
if st.sidebar.button("🔄 Refresh Now"):
    st.cache_data.clear()
    st.rerun()

df = fetch_data()

if df is not None and not df.empty:
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

    # Display key metrics with color coding for valuation
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Current BTC (USD)", f"${current_btc_usd:,.0f}")
    col2.metric("Current Gold/oz (USD)", f"${current_gold_usd:,.0f}")
    col3.metric("Current BTC in Gold oz", f"{current_btc_gold:.2f}")
    col4.metric("Fair Value (Power Law)", f"{current_fair:.2f}")

    col5, col6, col7 = st.columns(3)
    
    # Color code the valuation metric
    if valuation > 0:
        col5.metric("Valuation (% over fair)", f"+{valuation:.1f}%", delta=None)
    else:
        col5.metric("Valuation (% under fair)", f"{valuation:.1f}%", delta=None)
    
    col6.metric("Power Law Exponent (B)", f"{slope:.2f}")
    col7.metric("Fit R²", f"{r_value**2:.2f}")

    # Add status indicator
    if valuation > 20:
        st.warning("⚠️ BTC appears significantly overvalued relative to gold")
    elif valuation < -20:
        st.success("💰 BTC appears undervalued relative to gold")
    else:
        st.info("📊 BTC is trading near fair value relative to gold")

    # Plot log-log chart with fit and projection
    st.subheader('Log-Log Plot: BTC in Gold oz vs. Days Since Genesis')
    
    fig = go.Figure()

    # Actual data
    fig.add_trace(go.Scatter(
        x=df['Days'], 
        y=df['BTC_in_Gold'], 
        mode='lines', 
        name='Actual BTC in Gold', 
        line=dict(color='blue', width=2)
    ))

    # Power law fit line (extend 5 years into future)
    fit_days = np.arange(df['Days'].min(), current_days + 365 * 5)
    fit_price = fair_value(fit_days)
    fig.add_trace(go.Scatter(
        x=fit_days, 
        y=fit_price, 
        mode='lines', 
        name='Power Law Fit', 
        line=dict(color='red', dash='dash', width=2)
    ))

    # Current point
    fig.add_trace(go.Scatter(
        x=[current_days], 
        y=[current_btc_gold], 
        mode='markers', 
        name='Current Position', 
        marker=dict(color='green', size=12, symbol='diamond')
    ))

    # Layout
    fig.update_layout(
        xaxis_type='log',
        yaxis_type='log',
        xaxis_title='Days Since Genesis (Log Scale)',
        yaxis_title='BTC in Gold oz (Log Scale)',
        height=600,
        hovermode='x unified',
        showlegend=True
    )
    
    st.plotly_chart(fig, use_container_width=True)

    # Recent data table
    st.subheader('Recent Data (Last 10 Trading Days)')
    recent_df = df.tail(10)[['BTC_USD', 'Gold_USD', 'BTC_in_Gold', 'Days']].copy()
    recent_df['BTC_USD'] = recent_df['BTC_USD'].apply(lambda x: f"${x:,.0f}")
    recent_df['Gold_USD'] = recent_df['Gold_USD'].apply(lambda x: f"${x:,.0f}")
    recent_df['BTC_in_Gold'] = recent_df['BTC_in_Gold'].apply(lambda x: f"{x:.2f}")
    
    st.dataframe(recent_df, use_container_width=True)

    # Statistical summary
    st.subheader('Statistical Summary')
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.write("**Power Law Statistics:**")
        st.write(f"- Exponent (B): {slope:.3f}")
        st.write(f"- Intercept (log A): {intercept:.3f}")
        st.write(f"- R-squared: {r_value**2:.3f}")
        st.write(f"- P-value: {p_value:.2e}")
    
    with col2:
        st.write("**Current Metrics:**")
        st.write(f"- Days since Genesis: {current_days:,}")
        st.write(f"- BTC/Gold ratio: {current_btc_gold:.2f}")
        st.write(f"- Fair value ratio: {current_fair:.2f}")
        st.write(f"- Deviation: {valuation:.1f}%")
    
    with col3:
        st.write("**Data Quality:**")
        st.write(f"- Total data points: {len(df):,}")
        st.write(f"- Data range: {len(df)} days")
        st.write(f"- Last update: {df.index[-1].strftime('%Y-%m-%d')}")

    # Explanation
    st.markdown("""
    ### How This Dashboard Works
    
    **Real-time Features:**
    - Data refreshes every 5 minutes automatically (configurable in sidebar)
    - Manual refresh button available in sidebar
    - Live status indicators for over/under valuation
    - Current timestamp showing last update
    
    **Bitcoin Power Law in Gold Terms:**
    - Models BTC's value in ounces of gold as: `BTC_in_Gold ≈ A × days^B`
    - Uses linear regression on log-transformed data for power law fitting
    - Fair value represents the expected BTC/Gold ratio based on historical trends
    
    **Data Sources:**
    - Bitcoin: Yahoo Finance (BTC-USD)
    - Gold: Yahoo Finance Gold Futures (GC=F)
    - Genesis date: January 3, 2009 (Bitcoin's first block)
    
    **Interpretation:**
    - **Positive valuation**: BTC is trading above fair value relative to gold
    - **Negative valuation**: BTC is trading below fair value relative to gold
    - **Power law exponent**: Typically ranges 5-6, indicating exponential growth trend
    """)

else:
    st.error("Unable to fetch data. Please check your internet connection and try refreshing the page.")
    st.info("The dashboard will attempt to refresh automatically in a few minutes.")
