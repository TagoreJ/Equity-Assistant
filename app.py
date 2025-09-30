import pandas as pd
import numpy as np
import yfinance as yf
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta

# ==== Load Your Stock Lists ====

us_stocks = pd.read_excel("USStocks.xlsx")
uk_stocks = pd.read_excel("UKStocks.xlsx")
ind_stocks = pd.read_excel("INDStocks.xlsx")

# ==== Utility Functions ====

def set_region(country):
    if country == 'India':
        return ind_stocks
    elif country == 'US':
        return us_stocks
    elif country == 'UK':
        return uk_stocks
    else:
        return pd.DataFrame()

def get_portfolio_data(df, stock_num_dict):
    portfolio_df = df[df['Company Name'].isin(stock_num_dict.keys())].copy()
    portfolio_df['Number of Shares'] = portfolio_df['Company Name'].map(stock_num_dict).fillna(0).astype(int)
    
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

def calc_log_returns(price_df):
    return np.log(price_df / price_df.shift(1)).dropna()

def plot_historical_prices(price_df):
    st.subheader("Historical Stock Price Charts")
    for ticker in price_df.columns:
        series = price_df[ticker].dropna()
        if series.empty:
            st.warning(f"No price data to plot for {ticker}")
            continue
        st.line_chart(series, height=200, use_container_width=True, key=f"price_{ticker}")

def plot_cumulative_returns(log_returns, weights):
    st.subheader("Portfolio Cumulative Returns")
    weighted_returns = log_returns.dot(weights)
    cumulative = np.exp(weighted_returns.cumsum())
    st.line_chart(cumulative, height=300)

def plot_correlation_heatmap(log_returns):
    st.subheader("Correlation Heatmap")
    fig, ax = plt.subplots(figsize=(10,8))
    sns.heatmap(log_returns.corr(), annot=True, cmap='coolwarm', ax=ax)
    st.pyplot(fig)

def plot_drawdowns(price_df):
    st.subheader("Max Drawdown Periods")
    drawdowns = {}
    for ticker in price_df.columns:
        cum_max = price_df[ticker].cummax()
        drawdown = (price_df[ticker] - cum_max) / cum_max
        drawdowns[ticker] = drawdown.min()
    drawdown_df = pd.Series(drawdowns).sort_values()
    st.bar_chart(drawdown_df)

@st.cache_data
def fetch_dividends(tickers):
    dividends = {}
    now = datetime.today()
    future = now + timedelta(days=90)
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

@st.cache_data
def fetch_benchmark_data(benchmark_symbol, period='5y'):
    return yf.Ticker(benchmark_symbol).history(period=period)['Close']

def plot_benchmark_comparison(portfolio_log_returns, benchmark_log_return):
    st.subheader("Portfolio vs Benchmark Returns")
    portfolio_cum = np.exp(portfolio_log_returns.cumsum())
    benchmark_cum = np.exp(benchmark_log_return.cumsum())
    combined_df = pd.DataFrame({'Portfolio': portfolio_cum, 'Benchmark': benchmark_cum})
    st.line_chart(combined_df)

def sortino_ratio(log_returns, risk_free_rate=0.0):
    downside_returns = log_returns[log_returns < 0]
    expected_return = log_returns.mean() * 252
    downside_std = downside_returns.std() * np.sqrt(252)
    if downside_std == 0:
        return np.nan
    return (expected_return - risk_free_rate) / downside_std

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


# ==== Main App ====

st.set_page_config(
    page_title='Ekalavya Portfolio Assistant',
    page_icon="💬",
    layout='wide',
    initial_sidebar_state='expanded'
)

st.header('Ekalavya Portfolio Assistant')

region = st.selectbox('Choose your Exchange', ['India', 'US', 'UK'])
df = set_region(region)

options = st.multiselect('Choose your companies', df['Company Name'])

if options:
    # Default initial shares
    initial_shares = {name: 1 for name in options}

    with st.form("modify_portfolio"):
        st.write("Modify Number of Shares:")
        for stock in options:
            initial_shares[stock] = st.number_input(stock, 1, 9999999, value=initial_shares[stock], step=1)
        submitted = st.form_submit_button("Update Portfolio")

    portfolio_df = get_portfolio_data(df, initial_shares)

    if portfolio_df.empty:
        st.warning("No valid price data for selected stocks.")
    else:
        st.subheader("Portfolio Summary")
        st.dataframe(portfolio_df[['Company Name', 'Ticker', 'Number of Shares', 'Current Price', 'Total Value']])

        tickers = portfolio_df['Ticker'].tolist()
        price_df = fetch_price_data(tickers)
        log_returns = calc_log_returns(price_df)

        # Compute allocation weights based on current total value
        total_value = portfolio_df['Total Value'].sum()
        weights = portfolio_df['Total Value'] / total_value

        # Your original old features could also be here (like daily returns etc.) — 
        # but to keep completeness within this merged example,
        # you could add them similarly as needed.

        # New features below:
        plot_historical_prices(price_df)
        plot_cumulative_returns(log_returns, weights)
        plot_correlation_heatmap(log_returns)
        plot_drawdowns(price_df)

        fundamentals = fetch_fundamentals(tickers)
        display_fundamentals(fundamentals)

        rebalance_portfolio(portfolio_df, weights.values, total_value)
        user_alerts(portfolio_df)

        dividends = fetch_dividends(tickers)
        show_dividend_calendar(dividends)

        benchmark_dict = {
            'India': '^NSEI',  # NSE Nifty 50
            'US': '^GSPC',     # S&P 500
            'UK': '^FTSE',
        }
        benchmark_symbol = benchmark_dict.get(region, None)
        if benchmark_symbol:
            benchmark_price = fetch_benchmark_data(benchmark_symbol)
            benchmark_log = np.log(benchmark_price / benchmark_price.shift(1)).dropna()
            portfolio_returns = log_returns.dot(weights)
            plot_benchmark_comparison(portfolio_returns, benchmark_log)

        st.subheader("Risk Adjusted Performance")
        st.write(f"Sortino Ratio: {sortino_ratio(log_returns.dot(weights)):.2f}")

        monte_carlo_simulation(log_returns, weights.values)


else:
    st.info("Please select one or more companies above to begin analysis.")

