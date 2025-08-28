import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from scipy.stats import linregress
import plotly.graph_objects as go
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Genesis block date
genesis = datetime(2009, 1, 3)

st.title('Bitcoin Power Law Dashboard: Gold & ZROZ Comparison')

# Load BTC historical data (CSV preferred)
@st.cache_data(ttl=14400)
def load_btc_data():
    try:
        df_btc = pd.read_csv("btc_usd_full.csv", parse_dates=["Date"], index_col="Date")
        st.success("Loaded BTC data from CSV")
    except Exception:
        st.warning("BTC CSV not found, falling back to Yahoo Finance (since 2014)")
        df_btc = yf.download('BTC-USD', start='2009-01-03', end=datetime.now().strftime('%Y-%m-%d'),
                             progress=False, auto_adjust=True)
    return df_btc

# Load Gold and ZROZ from Yahoo Finance
@st.cache_data(ttl=14400)
def load_macro_data():
    gold = yf.download('GC=F', start='2009-01-03', end=datetime.now().strftime('%Y-%m-%d'),
                       progress=False, auto_adjust=True)
    zroz = yf.download('ZROZ', start='2009-01-03', end=datetime.now().strftime('%Y-%m-%d'),
                       progress=False, auto_adjust=True)
    return gold, zroz

btc_data = load_btc_data()
gold_data, zroz_data = load_macro_data()

# Align datasets
df = pd.DataFrame({
    'BTC_USD': btc_data['Close'],
    'Gold_USD': gold_data['Close'],
    'ZROZ_USD': zroz_data['Close']
}).dropna()

df['BTC_in_Gold'] = df['BTC_USD'] / df['Gold_USD']
df['BTC_in_ZROZ'] = df['BTC_USD'] / df['ZROZ_USD']
df['Days'] = (df.index - genesis).days

# --- Power Law Fit Function ---
def fit_power_law(x_days, y_ratio):
    mask = (x_days > 0) & (y_ratio > 0)
    log_days = np.log(x_days[mask])
    log_y = np.log(y_ratio[mask])
    slope, intercept, r_value, p_value, std_err = linregress(log_days, log_y)
    def fair_value(days):
        return np.exp(intercept) * days ** slope
    return slope, intercept, r_value, fair_value

# Fit BTC/Gold
slope_gold, intercept_gold, r_gold, fair_gold = fit_power_law(df['Days'], df['BTC_in_Gold'])

# Fit BTC/ZROZ
slope_zroz, intercept_zroz, r_zroz, fair_zroz = fit_power_law(df['Days'], df['BTC_in_ZROZ'])

# --- Chart Function ---
def make_powerlaw_chart(df, y_col, fair_func, slope, r_value, label, unit):
    current_date = datetime.now()
    current_days = (current_date - genesis).days
    current_val = df[y_col].iloc[-1]
    current_fair = fair_func(current_days)

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df['Days'], y=df[y_col], mode='lines', name=f'Actual {label}'))
    fit_days = np.arange(1, current_days + 365*5)
    fig.add_trace(go.Scatter(x=fit_days, y=fair_func(fit_days), mode='lines', name='Power Law Fit', line=dict(dash='dash')))
    fig.add_trace(go.Scatter(x=[current_days], y=[current_val], mode='markers', name='Current Value', marker=dict(size=12, color='green', symbol='star')))
    fig.add_trace(go.Scatter(x=[current_days], y=[current_fair], mode='markers', name='Fair Value', marker=dict(size=10, color='orange', symbol='diamond')))
    fig.update_layout(
        xaxis_type='log', yaxis_type='log',
        xaxis_title='Days Since Genesis (log)',
        yaxis_title=f'{label} ({unit}) (log)',
        title=f'Bitcoin Power Law in {label} Terms (R²={r_value**2:.3f}, B={slope:.3f})'
    )
    return fig

# --- R² Convergence Function ---
@st.cache_data(ttl=14400)
def calculate_r_squared_convergence(df, y_col):
    df_sorted = df[(df['Days'] > 0) & (df[y_col] > 0)].sort_values('Days')
    min_points = 30
    if len(df_sorted) < min_points:
        return None
    r_squared_values, dates, sample_sizes = [], [], []
    for i in range(min_points, len(df_sorted), max(1, len(df_sorted)//200)):
        subset = df_sorted.iloc[:i]
        try:
            log_days = np.log(subset['Days'])
            log_price = np.log(subset[y_col])
            slope, intercept, r_value, p_value, std_err = linregress(log_days, log_price)
            r_squared_values.append(r_value**2)
            dates.append(subset.index[-1])
            sample_sizes.append(i)
        except Exception:
            continue
    if not r_squared_values:
        return None
    return pd.DataFrame({
        'Date': dates,
        'R_squared': r_squared_values,
        'Sample_Size': sample_sizes
    })

# --- Display ---
st.subheader("BTC vs. Gold Power Law")
fig_gold = make_powerlaw_chart(df, 'BTC_in_Gold', fair_gold, slope_gold, r_gold, 'BTC/Gold', 'oz')
st.plotly_chart(fig_gold, use_container_width=True)

st.subheader("BTC vs. ZROZ Power Law")
fig_zroz = make_powerlaw_chart(df, 'BTC_in_ZROZ', fair_zroz, slope_zroz, r_zroz, 'BTC/ZROZ', 'ETF shares')
st.plotly_chart(fig_zroz, use_container_width=True)

# R² Convergence Charts + Summaries
def display_r2_convergence(df, y_col, label):
    r2_df = calculate_r_squared_convergence(df, y_col)
    if r2_df is not None:
        fig_r2 = go.Figure()
        fig_r2.add_trace(go.Scatter(x=r2_df['Date'], y=r2_df['R_squared'], mode='lines', name='R² Convergence'))
        fig_r2.update_layout(title=f"{label} Power Law R² Convergence", yaxis=dict(range=[0,1]))
        st.plotly_chart(fig_r2, use_container_width=True)

        # Summary metrics
        current_r2 = r2_df['R_squared'].iloc[-1]
        peak_r2 = r2_df['R_squared'].max()
        min_r2 = r2_df['R_squared'].min()
        start_r2 = r2_df['R_squared'].iloc[0]
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Current R²", f"{current_r2:.4f}")
        col2.metric("Peak R²", f"{peak_r2:.4f}")
        col3.metric("Lowest R²", f"{min_r2:.4f}")
        col4.metric("Initial R²", f"{start_r2:.4f}")

st.subheader("R² Convergence: BTC vs Gold")
display_r2_convergence(df, 'BTC_in_Gold', 'BTC/Gold')

st.subheader("R² Convergence: BTC vs ZROZ")
display_r2_convergence(df, 'BTC_in_ZROZ', 'BTC/ZROZ')

st.markdown("""
### Notes
- BTC data is loaded from CSV if available (must be named **btc_usd_full.csv** in the app directory, with columns `Date`, `Close`).
- Gold (`GC=F`) and ZROZ (`ZROZ`) pulled from Yahoo Finance.
- Both ratios are modeled as power laws in time since Bitcoin's genesis (2009-01-03).
- R² Convergence charts show how model fit improves or weakens as more data is included.
- Summary metrics show current, peak, and historical ranges of R².
""")
