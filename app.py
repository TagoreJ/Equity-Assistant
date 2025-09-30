import pandas as pd
import numpy as np
import yfinance as yf
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta

# --- Load your stock list files once here ---
us_stocks = pd.read_excel("USStocks.xlsx")
uk_stocks = pd.read_excel("UKStocks.xlsx")
ind_stocks = pd.read_excel("INDStocks.xlsx")

# --- Utility: Load selected region ---
def set_region(country):
    if country == 'India':
        return ind_stocks
    elif country == 'US':
        return us_stocks
    elif country == 'UK':
        return uk_stocks
    else:
        return pd.DataFrame()

# --- Calculate portfolio value for selected stocks and shares ---
def get_portfolio_data(df, stock_num_dict):
    portfolio_df = df[df['Company Name'].isin(stock_num_dict.keys())].copy()
    portfolio_df['Number of Shares'] = portfolio_df['Company Name'].map(stock_num_dict).fillna(0).astype(int)
    
    # Fetch current prices
    prices = []
    for ticker in portfolio_df['Ticker']:
        try:
            price = yf.Ticker(ticker).history(period="1d")['Close'].iloc[-1]
        except Exception:
            price = np.nan
        prices.append(price)
    portfolio_df['Current Price'] = prices
    
    portfolio_df['Total Value'] = portfolio_df['Current Price'] * portfolio_df['Number of Shares']
    return portfolio_df.dropna(subset=['Current Price'])

# --- Fetch historical price data for tickers ---
@st.cache_data(show_spinner=False)
def fetch_price_data(tickers, period='5y'):
    data = {}
    for tk in tickers:
        try:
            df = yf.Ticker(tk).history(period=period)[['Close']]
            df.rename(columns={'Close': tk}, inplace=True)
            data[tk] = df[tk]
        except Exception:
            data[tk] = pd.Series(dtype=float)
    price_df = pd.concat(data.values(), axis=1)
    price_df.columns = data.keys()
    price_df.dropna(how='all', inplace=True)
    return price_df

# --- Compute portfolio log returns ---
def calc_log_returns(price_df):
    return np.log(price_df / price_df.shift(1)).dropna()

# --- Plot historical price charts for each stock ---
def plot_historical_prices(price_df):
    st.subheader("Historical Stock Price Charts")
    for ticker in price_df.columns:
        st.line_chart(price_df[ticker], height=200, use_container_width=True, key=f"price_{ticker}")

# --- Plot portfolio cumulative returns ---
def plot_cumulative_returns(log_returns, weights):
    st.subheader("Portfolio Cumulative Returns")
    weighted_returns = log_returns.dot(weights)
    cumulative = np.exp(weighted_returns.cumsum())
    st.line_chart(cumulative, height=300)

# --- Plot correlation heatmap ---
def plot_correlation_heatmap(log_returns):
    st.subheader("Correlation Heatmap")
    fig, ax = plt.subplots(figsize=(10,8))
    sns.heatmap(log_returns.corr(), annot=True, cmap='coolwarm', ax=ax)
    st.pyplot(fig)

# --- Calculate and plot drawdowns ---
def plot_drawdowns(price_df):
    st.subheader("Max Drawdown Periods")
    drawdowns = {}
    for ticker in price_df.columns:
        cum_max = price_df[ticker].cummax()
        drawdown = (price_df[ticker] - cum_max) / cum_max
        drawdowns[ticker] = drawdown.min()
    drawdown_df = pd.Series(drawdowns).sort_values()
    st.bar_chart(drawdown_df)

# --- Dividends calendar ---
@st.cache_data
def fetch_dividends(tickers):
    dividends = {}
    now = datetime.today()
    future = now + timedelta(days=90)  # next 3 months
    for ticker in tickers:
        try:
            d = yf.Ticker(ticker).dividends
            upcoming = d[d.index > now]
            dividends[ticker] = upcoming
        except Exception:
            dividends[ticker] = pd.Series(dtype='float64')
    return dividends

def show_dividend_calendar(dividends):
    st.subheader("Upcoming Dividends (next 3 months)")
    for ticker, divs in dividends.items():
        if len(divs) > 0:
            st.write(f"**{ticker}:**")
            for date, amount in divs.items():
                st.write(f"- {date.date()}: {amount:.2f}")

# --- Portfolio metrics: PE, EPS, etc ---
@st.cache_data
def fetch_fundamentals(tickers):
    fundamentals = {}
    for ticker in tickers:
        try:
            info = yf.Ticker(ticker).info
            fundamentals[ticker] = {
                'PE Ratio': info.get('trailingPE', np.nan),
                'Dividend Yield': info.get('dividendYield', np.nan),
                'EPS': info.get('trailingEps', np.nan),
                'ROE': info.get('returnOnEquity', np.nan),
                'Debt to Equity': info.get('debtToEquity', np.nan),
            }
        except Exception:
            fundamentals[ticker] = None
    return fundamentals

def display_fundamentals(fundamentals):
    st.subheader("Fundamental Metrics")
    df = pd.DataFrame(fundamentals).T
    st.dataframe(df.style.format("{:.2f}"))

# --- Portfolio rebalancing suggestion ---
def rebalance_portfolio(portfolio_df, weights, total_value):
    st.subheader("Rebalancing Suggestions")
    portfolio_df = portfolio_df.copy()
    portfolio_df['Optimized Weight'] = weights
    portfolio_df['Current Value'] = portfolio_df['Total Value']
    portfolio_df['Target Value'] = portfolio_df['Optimized Weight'] * total_value
    portfolio_df['Difference ($)'] = portfolio_df['Target Value'] - portfolio_df['Current Value']
    portfolio_df['Action'] = portfolio_df['Difference ($)'].apply(lambda x: 'Buy' if x > 0 else 'Sell')
    portfolio_df['Shares to Trade'] = (portfolio_df['Difference ($)'] / portfolio_df['Current Price']).abs().round(0).astype(int)
    st.table(portfolio_df[['Company Name', 'Current Value', 'Optimized Weight', 'Target Value', 'Difference ($)', 'Action', 'Shares to Trade']])

# --- Alerts system ---
def user_alerts(portfolio_df):
    st.subheader("Set Price Alerts")
    alerts = {}
    for idx, row in portfolio_df.iterrows():
        price_alert = st.number_input(f"Alert Price for {row['Company Name']} ({row['Ticker']})", value=float(row['Current Price']))
        alerts[row['Ticker']] = price_alert
    
    st.write("### Price Alert Status")
    for ticker, alert_price in alerts.items():
        current_price = portfolio_df.loc[portfolio_df['Ticker'] == ticker, 'Current Price'].values[0]
        if current_price > alert_price:
            st.markdown(f"**{ticker}** price is above alert: {current_price:.2f} > {alert_price:.2f} 🔔")
        else:
            st.markdown(f"{ticker}: price below alert.")

# --- Benchmark comparison ---
@st.cache_data
def fetch_benchmark_data(benchmark_symbol, period='5y'):
    return yf.Ticker(benchmark_symbol).history(period=period)['Close']

def plot_benchmark_comparison(portfolio_log_returns, benchmark_log_return):
    st.subheader("Portfolio vs Benchmark Returns")
    portfolio_cum = np.exp(portfolio_log_returns.cumsum())
    benchmark_cum = np.exp(benchmark_log_return.cumsum())
    combined_df = pd.DataFrame({'Portfolio': portfolio_cum, 'Benchmark': benchmark_cum})
    st.line_chart(combined_df)

# --- Risk-adjusted ratios: Sortino (simplified) ---
def sortino_ratio(log_returns, risk_free_rate=0.0):
    downside_returns = log_returns[log_returns < 0]
    expected_return = log_returns.mean() * 252
    downside_std = downside_returns.std() * np.sqrt(252)
    if downside_std == 0:
        return np.nan
    return (expected_return - risk_free_rate) / downside_std

# --- Monte Carlo Simulation of portfolio growth ---
def monte_carlo_simulation(log_returns, weights, start_value=10000, years=5, sims=1000):
    st.subheader("Monte Carlo Simulation of Portfolio Value")
    mean_returns = log_returns.mean() * 252
    cov = log_returns.cov() * 252
    port_mean = np.dot(weights, mean_returns)
    port_std = np.sqrt(np.dot(weights.T, np.dot(cov, weights)))

    dt = 1/252
    iterations = int(years * 252)
    results = np.zeros((iterations, sims))
    for sim in range(sims):
        shock = np.random.normal(loc=(port_mean - 0.5 * port_std ** 2) * dt,
                                 scale=port_std * np.sqrt(dt),
                                 size=iterations)
        price_paths = start_value * np.exp(np.cumsum(shock))
        results[:, sim] = price_paths
    
    # Plot percentile bands
    percentiles = [5, 25, 50, 75, 95]
    df = pd.DataFrame(results)
    fig, ax = plt.subplots(figsize=(12, 6))
    for p in percentiles:
        ax.plot(df.quantile(p / 100, axis=1), label=f'{p}th Percentile')
    ax.set_title('Monte Carlo Simulation - Portfolio Value Over Time')
    ax.set_xlabel('Trading Days')
    ax.set_ylabel('Portfolio Value')
    ax.legend()
    st.pyplot(fig)

# ------------- Streamlit UI starts here ----------------

st.title("Enhanced Portfolio Dashboard")

region = st.selectbox('Choose your Exchange', ['India', 'US', 'UK'])
df = set_region(region)

options = st.multiselect('Select Companies', df['Company Name'])

if options:
    initial_shares = {name: 1 for name in options}  # default one share per stock

    with st.form("shares_form"):
        st.write("Modify number of shares per stock:")
        for stock in options:
            initial_shares[stock] = st.number_input(stock, min_value=0, value=initial_shares[stock], step=1)
        submitted = st.form_submit_button("Update Portfolio")

    portfolio_df = get_portfolio_data(df, initial_shares)

    if portfolio_df.empty:
        st.warning("No valid price data for selected stocks.")
    else:
        tickers = portfolio_df['Ticker'].tolist()

        # Display portfolio dataframe summary
        st.subheader("Portfolio Summary")
        st.dataframe(portfolio_df[['Company Name', 'Ticker', 'Number of Shares', 'Current Price', 'Total Value']])

        # Fetch historical data
        price_df = fetch_price_data(tickers)
        log_returns = calc_log_returns(price_df)

        # Calculate current allocation weights by value
        total_value = portfolio_df['Total Value'].sum()
        weights = portfolio_df['Total Value'] / total_value

        # Show key analytics and charts
        plot_historical_prices(price_df)
        plot_cumulative_returns(log_returns, weights)
        plot_correlation_heatmap(log_returns)
        plot_drawdowns(price_df)

        # Fetch and display fundamentals
        fundamentals = fetch_fundamentals(tickers)
        display_fundamentals(fundamentals)

        # Optimization example placeholder (weights optimization code can be plugged here)
        # For demo, we'll just simulate with current weights
        rebalance_portfolio(portfolio_df, weights.values, total_value)

        # Alerts system
        user_alerts(portfolio_df)

        # Dividends Calendar
        dividends = fetch_dividends(tickers)
        show_dividend_calendar(dividends)

        # Benchmark comparison
        benchmark_dict = {
            'India': '^NSEI',  # Nifty 50
            'US': '^GSPC',     # S&P 500
            'UK': '^FTSE',
        }
        benchmark_symbol = benchmark_dict.get(region, None)
        if benchmark_symbol:
            benchmark_price = fetch_benchmark_data(benchmark_symbol)
            benchmark_log = np.log(benchmark_price / benchmark_price.shift(1)).dropna()
            portfolio_returns = log_returns.dot(weights)
            plot_benchmark_comparison(portfolio_returns, benchmark_log)

        # Risk Adjusted Ratios
        st.subheader("Risk Adjusted Performance")
        st.write(f"Sortino Ratio: {sortino_ratio(log_returns.dot(weights)):.2f}")

        # Monte Carlo Simulation
        monte_carlo_simulation(log_returns, weights.values)

else:
    st.info("Select one or more companies above to begin analysis.")

