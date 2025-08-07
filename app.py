import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from scipy.stats import linregress
import plotly.graph_objects as go
from datetime import datetime, timedelta
import time

# Configure page
st.set_page_config(page_title="Bitcoin Power Law Dashboard", layout="wide")

# Genesis block date
genesis = datetime(2009, 1, 3)

st.title('Bitcoin Power Law Dashboard in Terms of Gold (Real-time)')

# Sidebar controls
st.sidebar.header("Dashboard Controls")
manual_refresh = st.sidebar.button("🔄 Refresh Data Now")
st.sidebar.write(f"Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

# Much shorter cache for real-time data
@st.cache_data(ttl=300, show_spinner=True)  # 5 minute cache
def fetch_data():
    try:
        with st.spinner('Fetching latest market data...'):
            # Get current date for data range
            end_date = datetime.now().strftime('%Y-%m-%d')
            
            # Fetch BTC data
            btc = yf.download('BTC-USD', start='2014-01-01', end=end_date, progress=False, timeout=10)
            if btc.empty:
                raise Exception("No BTC data received")
            
            # Handle MultiIndex columns
            if hasattr(btc.columns, 'nlevels') and btc.columns.nlevels > 1:
                btc.columns = btc.columns.get_level_values(0)
            
            # Fetch Gold data  
            gold = yf.download('GC=F', start='2014-01-01', end=end_date, progress=False, timeout=10)
            if gold.empty:
                raise Exception("No Gold data received")
            
            # Handle MultiIndex columns
            if hasattr(gold.columns, 'nlevels') and gold.columns.nlevels > 1:
                gold.columns = gold.columns.get_level_values(0)
            
            # Combine data
            df = pd.DataFrame({
                'BTC_USD': btc['Close'], 
                'Gold_USD': gold['Close']
            }).dropna()
            
            if df.empty:
                raise Exception("No overlapping data between BTC and Gold")
            
            # Calculate BTC in Gold terms
            df['BTC_in_Gold'] = df['BTC_USD'] / df['Gold_USD']
            
            # Days since genesis
            df['Days'] = (df.index - genesis).days
            
            return df, None
            
    except Exception as e:
        return None, str(e)

# Clear cache if manual refresh is clicked
if manual_refresh:
    st.cache_data.clear()

# Fetch data
df, error = fetch_data()

if error:
    st.error(f"❌ Data fetch error: {error}")
    st.info("💡 Try clicking 'Refresh Data Now' or check your internet connection.")
    st.stop()

if df is None or df.empty:
    st.error("❌ No data available")
    st.stop()

try:
    # Filter valid data for power law fitting
    df_fit = df[(df['Days'] > 0) & (df['BTC_in_Gold'] > 0)].copy()
    
    if len(df_fit) < 10:
        st.error("❌ Insufficient data for power law fitting")
        st.stop()
    
    # Fit power law: log(price) = log(A) + B * log(days)
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

    # Success indicator
    st.success(f"✅ Data updated successfully - {len(df)} data points loaded")

    # Display key metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            "Current BTC (USD)", 
            f"${current_btc_usd:,.0f}",
            help="Latest Bitcoin price in USD"
        )
    
    with col2:
        st.metric(
            "Gold Price/oz (USD)", 
            f"${current_gold_usd:,.0f}",
            help="Latest gold futures price per ounce"
        )
    
    with col3:
        st.metric(
            "BTC in Gold oz", 
            f"{current_btc_gold:.2f}",
            help="How many ounces of gold 1 BTC is worth"
        )
    
    with col4:
        st.metric(
            "Power Law Fair Value", 
            f"{current_fair:.2f}",
            help="Expected BTC/Gold ratio based on power law"
        )

    # Second row of metrics
    col5, col6, col7 = st.columns(3)
    
    with col5:
        delta_color = "normal"
        if abs(valuation) > 50:
            delta_color = "inverse"
        
        st.metric(
            "Valuation vs Fair Value", 
            f"{valuation:+.1f}%",
            delta=f"{valuation:+.1f}%",
            help="Percentage over/under power law fair value"
        )
    
    with col6:
        st.metric(
            "Power Law Exponent", 
            f"{slope:.3f}",
            help="The 'B' in the power law equation: BTC = A × days^B"
        )
    
    with col7:
        st.metric(
            "Model Fit (R²)", 
            f"{r_value**2:.3f}",
            help="How well the power law fits the data (1.0 = perfect fit)"
        )

    # Status indicators
    if valuation > 50:
        st.error("🔴 **EXTREMELY OVERVALUED** - BTC is significantly above fair value relative to gold")
    elif valuation > 20:
        st.warning("🟡 **OVERVALUED** - BTC is above fair value relative to gold")
    elif valuation < -50:
        st.success("🟢 **EXTREMELY UNDERVALUED** - BTC is significantly below fair value relative to gold")
    elif valuation < -20:
        st.info("🔵 **UNDERVALUED** - BTC is below fair value relative to gold")
    else:
        st.info("⚪ **FAIRLY VALUED** - BTC is trading near power law fair value")

    # Main chart
    st.subheader('Power Law Chart: BTC in Gold oz vs. Days Since Genesis')
    
    fig = go.Figure()

    # Actual price data
    fig.add_trace(go.Scatter(
        x=df['Days'], 
        y=df['BTC_in_Gold'], 
        mode='lines', 
        name='Actual BTC/Gold Ratio', 
        line=dict(color='#1f77b4', width=2),
        hovertemplate='<b>Day %{x}</b><br>BTC/Gold: %{y:.3f}<extra></extra>'
    ))

    # Power law fit line
    fit_days = np.linspace(df['Days'].min(), current_days + 365 * 2, 1000)
    fit_price = fair_value(fit_days)
    
    fig.add_trace(go.Scatter(
        x=fit_days, 
        y=fit_price, 
        mode='lines', 
        name=f'Power Law Fit (y = {np.exp(intercept):.2e} × x^{slope:.3f})', 
        line=dict(color='#ff7f0e', dash='dash', width=2),
        hovertemplate='<b>Day %{x}</b><br>Fair Value: %{y:.3f}<extra></extra>'
    ))

    # Current point
    fig.add_trace(go.Scatter(
        x=[current_days], 
        y=[current_btc_gold], 
        mode='markers', 
        name=f'Current Position ({datetime.now().strftime("%Y-%m-%d")})', 
        marker=dict(color='#2ca02c', size=15, symbol='diamond'),
        hovertemplate='<b>Today</b><br>Current: %{y:.3f}<br>Fair Value: {:.3f}<br>Valuation: {:+.1f}%<extra></extra>'.format(current_fair, valuation)
    ))

    # Chart layout
    fig.update_layout(
        xaxis_type='log',
        yaxis_type='log',
        xaxis_title='Days Since Genesis Block (Log Scale)',
        yaxis_title='BTC Price in Gold Ounces (Log Scale)',
        height=650,
        hovermode='x unified',
        showlegend=True,
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=0.01
        )
    )
    
    fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='rgba(128,128,128,0.2)')
    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='rgba(128,128,128,0.2)')
    
    st.plotly_chart(fig, use_container_width=True)

    # Data table
    st.subheader('Recent Market Data')
    recent_data = df.tail(10).copy()
    recent_data.index = recent_data.index.strftime('%Y-%m-%d')
    recent_data_display = recent_data[['BTC_USD', 'Gold_USD', 'BTC_in_Gold']].copy()
    recent_data_display.columns = ['BTC Price (USD)', 'Gold Price (USD)', 'BTC/Gold Ratio']
    
    # Format for display
    recent_data_display['BTC Price (USD)'] = recent_data_display['BTC Price (USD)'].apply(lambda x: f"${x:,.0f}")
    recent_data_display['Gold Price (USD)'] = recent_data_display['Gold Price (USD)'].apply(lambda x: f"${x:,.0f}")
    recent_data_display['BTC/Gold Ratio'] = recent_data_display['BTC/Gold Ratio'].apply(lambda x: f"{x:.3f}")
    
    st.dataframe(recent_data_display, use_container_width=True)

    # Statistics
    st.subheader('Model Statistics')
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Power Law Parameters:**")
        st.code(f"""
Equation: BTC_in_Gold = A × days^B
A (coefficient): {np.exp(intercept):.2e}
B (exponent): {slope:.3f}
R-squared: {r_value**2:.3f}
P-value: {p_value:.2e}
Standard Error: {std_err:.3f}
        """)
    
    with col2:
        st.write("**Current Analysis:**")
        st.code(f"""
Days since Genesis: {current_days:,}
Current BTC/Gold: {current_btc_gold:.3f}
Fair Value: {current_fair:.3f}
Deviation: {valuation:+.1f}%
Data Points: {len(df):,}
Date Range: {df.index[0].strftime('%Y-%m-%d')} to {df.index[-1].strftime('%Y-%m-%d')}
        """)

except Exception as e:
    st.error(f"❌ Analysis error: {str(e)}")
    st.info("This might be due to insufficient or invalid data. Try refreshing.")

# Information section
with st.expander("ℹ️ How This Dashboard Works"):
    st.markdown("""
    ### Real-time Bitcoin Power Law Analysis
    
    **What it shows:**
    - Bitcoin's price expressed in ounces of gold over time
    - A power law trend line showing the expected "fair value"
    - Current valuation relative to this fair value
    
    **The Power Law Model:**
    - Follows the equation: `BTC_in_Gold = A × days^B`
    - Where `days` is the number of days since Bitcoin's Genesis block (Jan 3, 2009)
    - Uses linear regression on log-transformed data for fitting
    
    **Data Sources:**
    - Bitcoin price: Yahoo Finance (BTC-USD)
    - Gold price: Yahoo Finance Gold Futures (GC=F)
    - Updates every 5 minutes automatically
    
    **Interpretation:**
    - **Positive %**: BTC is overvalued relative to the long-term trend
    - **Negative %**: BTC is undervalued relative to the long-term trend
    - **Near 0%**: BTC is trading close to fair value
    
    **Note:** This is a mathematical model based on historical data and should not be considered investment advice.
    """)

# Auto-refresh instructions
st.info("🔄 **Real-time Updates:** This dashboard caches data for 5 minutes. Click 'Refresh Data Now' for immediate updates, or wait for automatic refresh.")
