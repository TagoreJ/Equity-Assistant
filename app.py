from duckduckgo_search import DDGS
import pandas as pd
from PIL import Image, ImageDraw, ImageOps
import requests
from io import BytesIO
import streamlit as st
import base64
import datetime
import yfinance as yf
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from datetime import datetime, timedelta
import numpy as np
from scipy.stats import norm
import time

us_stocks = r"C:\Users\herop\Downloads\python\USStocks.xlsx"
uk_stocks = r"C:\Users\herop\Downloads\python\UKStocks.xlsx"
ind_stocks = r"C:\Users\herop\Downloads\python\INDStocks.xlsx"
def set_region(country):
    if country == 'India':
        df = ind_stocks
    elif country == 'US':
        df = us_stocks
    elif country == 'UK':
        df = uk_stocks
    return df


# Function to create circular logos
def make_logo(company):
    try:
        results = DDGS().images(
            keywords=company + ' minimal Favicon',
            region="wt-wt",
            safesearch="off",
            max_results=1,
        )
        image_url = results[0]['image']
        response = requests.get(image_url) 
        img = Image.open(BytesIO(response.content))
    except:
        img = Image.new("RGB", (200, 200), (255, 255, 255))
    width, height = img.size
    min_dim = min(width, height)
    left = (width - min_dim) // 2
    top = (height - min_dim) // 2
    right = left + min_dim
    bottom = top + min_dim
    img = img.crop((left, top, right, bottom))

    # Create a circular mask
    mask = Image.new("L", img.size, 0)
    draw = ImageDraw.Draw(mask)
    draw.ellipse((0, 0, img.size[0], img.size[1]), fill=255)

    # Apply the mask to make the image round
    img = ImageOps.fit(img, mask.size, centering=(0.5, 0.5))
    img.putalpha(mask)
    img = img.convert("RGBA")
    return img

# Convert PIL image to base64 string
def pil_to_base64(img):
    buffer = BytesIO()
    img.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode()

def classify_mcap(mcap):
    if region == 'India':
        if mcap < 240:
            return "Small"
        elif mcap > 1200:
            return "Large"
        else:
            return "Mid"
    else:
        if mcap < 2:
            return "Small"
        elif mcap > 10:
            return "Large"
        else:
            return "Mid"




def get_daily_returns(ticker):
    stock = yf.Ticker(ticker)
    stock_info = stock.history(period="5d") 
    latest_price = stock_info['Close'].iloc[-1]
    prev_price = stock_info['Close'].iloc[-2]
    daily_returns = (latest_price - prev_price)
    return f'{daily_returns:.2f}'

def get_daily_return_percentage(ticker):
    stock = yf.Ticker(ticker)
    stock_info = stock.history(period="5d") 
    latest_price = stock_info['Close'].iloc[-1]
    prev_price = stock_info['Close'].iloc[-2]
    daily_percent = (latest_price - prev_price)/prev_price*100
    return f'{daily_percent:.2f}'

# Function to generate a portfolio dataframe with logos
def generate_portfolio(df):
    industry_list = []
    logo_list = []
    mcap_list = []
    class_mcap_list = []
    price_list = []
    for name in df['Company Name']:
        logo = make_logo(name)
        logo_base64 = f'data:image/png;base64,{pil_to_base64(logo)}'
        logo_list.append(logo_base64)

    df.insert(0, 'Logo', logo_list)

    for ticker in df['Ticker']:
        stock = yf.Ticker(ticker)
        industry = stock.info.get("industry")
        industry_list.append(industry)
    df.insert(3, 'Industry', industry_list)

    for ticker in df['Ticker']:
        stock = yf.Ticker(ticker)
        mcap = stock.info.get("marketCap")
        if mcap is not None:
            mcap_list.append(f'{mcap/1000000000:.2f}')
        else:
            mcap_list.append('Error')
    df.insert(4, 'Market Cap', mcap_list)

    for ticker in df['Ticker']:
        stock = yf.Ticker(ticker)
        mcap = stock.info.get("marketCap")
        if mcap is not None:
            mcap_class = classify_mcap(stock.info.get("marketCap")/1000000000)
        else:
            mcap_class = 'Error'
        class_mcap_list.append(mcap_class)
    df.insert(5, 'Market Cap Size', class_mcap_list)

    for price in df['Ticker']:
        stock = yf.Ticker(price)
        stock_info = stock.history(period="1d") 
        latest_price = stock_info['Close'].iloc[-1]  
        price_list.append(f'{latest_price:.2f}')
    df.insert(6, 'Current Price', price_list)

    beta_list = []
    for ticker in df['Ticker']:
        stock = yf.Ticker(ticker)
        beta = stock.info.get("beta")
        beta_list.append(beta)
    df.insert(7, 'Beta', beta_list)
    return df

st.set_page_config(
    page_title='Chat Playground',
    page_icon="💬",
    layout='wide',
    initial_sidebar_state='expanded'
)
st.header('Ekalavya Portfolio Assistant')


region = st.selectbox('Choose your Exchange', ['India', 'US', 'UK'])
if region == 'India':
    regdf = ind_stocks
elif region == 'US':
    regdf = us_stocks
elif region == 'UK':
    regdf = uk_stocks

df = pd.read_excel(regdf)
options = st.multiselect('Choose your companies', df['Company Name'])

portfolio_df = generate_portfolio(df[df['Company Name'].isin(options)])

stock_list = portfolio_df['Company Name'].tolist()   
ticker_list = portfolio_df['Ticker'].tolist()

with st.sidebar:
    buy_date = st.select_slider("Enter Time Frame", options=['1d', '5d', '1mo', '3mo', '6mo', '1y', '2y', '5y', '10y', 'ytd', 'max'], value='1y')
    def get_total_returns(ticker):
        try:
            def_stock = yf.Ticker(ticker)
            stock_info = def_stock.history(period=buy_date)
            latest_price = stock_info['Close'].iloc[-1]
            oldest_price = stock_info['Close'].iloc[0]
        except (IndexError, ValueError):
            stock_info = yf.Ticker(ticker).history(period='max')
            latest_price = stock_info['Close'].iloc[-1]
            oldest_price = stock_info['Close'].iloc[0]
        total_returns = latest_price - oldest_price
        return float(total_returns) if pd.notnull(total_returns) else 0 
    with st.form("Modify Portfolio"):
        st.write("Modify Number of Stocks")
        stock_num_dict = {}
        for stock in stock_list:
            my_number = int(st.number_input(stock, 1, 9999999, value=1 , step=1))
            print(my_number)
            stock_num_dict[stock] = my_number
        st.form_submit_button('Modify Portfolio')
    
        portfolio_df['Number of Shares'] = portfolio_df['Company Name'].map(stock_num_dict)
        portfolio_df['Current Price'] = pd.to_numeric(portfolio_df['Current Price'], errors='coerce')

        portfolio_df['Total Value'] = portfolio_df['Current Price'] * portfolio_df['Number of Shares']
        portfolio_df['Allocation (%)'] = (portfolio_df['Total Value'] / portfolio_df['Total Value'].sum()) * 100
        portfolio_df['Allocation (%)'] = portfolio_df['Allocation (%)'].apply(lambda x: f'{x:.2f}')

        portfolio_df['Daily Return'] = (pd.to_numeric(portfolio_df['Ticker'].apply(get_daily_returns)))*pd.to_numeric(portfolio_df['Number of Shares'])
        portfolio_df['Daily Return (in %)'] = portfolio_df['Ticker'].apply(get_daily_return_percentage)
        portfolio_df['Total Returns'] = portfolio_df['Ticker'].apply(get_total_returns) * pd.to_numeric(portfolio_df['Number of Shares'], errors='coerce')

    var_days = st.number_input('Number of Days for VaR Calculation', min_value=1, max_value=9999999, value=5, step=1)
    var_conf_interval = st.number_input('Confidence Interval for VaR Calculation', min_value=float(0), max_value=1.00, value=0.95, step=0.01)
    rf_rate = st.number_input('Risk Free Rate in (%)', min_value=float(0), max_value=float(100), value=7.365, step=0.01)
portfolio_df['Logo'] = portfolio_df['Logo'].apply(
    lambda x: f'<img src="{x}" width="50" style="border-radius: 50%;" />')

st.markdown(
    f"""
    <div style="overflow-x: auto; text-align: left;">
        {portfolio_df.to_html(
            escape=False,
            index=False,
            classes='dataframe').replace('<th>', '<th style="text-align: center;">')}
    </div>
    """,
    unsafe_allow_html=True
)

st.divider()

st.subheader('Portfolio Summary:')

# Create a Pie Chart

# Show the chart
row1, row2, row3 = st.columns(3)

with row1:

    market_cap_summary = portfolio_df['Market Cap Size'].value_counts()
    values = market_cap_summary.tolist()
    labels = market_cap_summary.index.tolist()

    fig, ax = plt.subplots(figsize=(10, 10))
    fig.patch.set_alpha(0)  
    ax.pie(
        values,
        labels=labels,
        autopct='%1.1f%%', 
        textprops={'color': 'white'},  
        startangle=90
    )
    st.pyplot(fig)
    st.text('Market Cap Distribution')

with row2:
    
    industry_summary = portfolio_df['Industry'].value_counts()
    ind_values = industry_summary.tolist()
    ind_labels = industry_summary.index.tolist()

    fig, ax = plt.subplots(figsize=(10, 10))
    fig.patch.set_alpha(0) 
    ax.pie(
        ind_values,
        labels=ind_labels,
        textprops={'color': 'white'}, 
        startangle=90,
        labeldistance=0.3  

    )
    st.pyplot(fig)
    st.text('Industry Distribution')

with row3:
    company_summary = portfolio_df[['Company Name', 'Allocation (%)']] 
    comp_values = company_summary['Allocation (%)'].values
    comp_labels = company_summary['Company Name'].tolist()

    fig, ax = plt.subplots(figsize=(10, 10))
    fig.patch.set_alpha(0) 
    ax.pie(
        comp_values,
        labels=comp_labels,
        textprops={'color': 'white'}, 
        startangle=90,
        labeldistance=0.3  

    )
    st.pyplot(fig)
    st.text('Company Distribution')

st.divider()
st.subheader('Portfolio Performance:')

st.markdown(
    """
    <style>
    .center-text {
        text-align: center;
    }
    .green {
        color: green;
        font-weight: bold;
    }
    .red {
        color: red;
        font-weight: bold;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

col1, col2, col3, col4 = st.columns(4)

with col1:
    with st.container():
        st.write('<div class="center-text">Total Portfolio Value</div>', unsafe_allow_html=True)
        total_value = portfolio_df['Total Value'].sum()
        st.markdown(
            f'<div class="center-text">{total_value:,.2f}</div>',
            unsafe_allow_html=True,
        )

with col2:    
    with st.container():
        st.write('<div class="center-text">Total Portfolio Return</div>', unsafe_allow_html=True)
        total_returns = portfolio_df['Total Returns'].sum()
        color = "green" if total_returns >= 0 else "red"
        st.markdown(
            f'<div class="center-text {color}">{total_returns:,.2f}</div>',
            unsafe_allow_html=True,
        )

with col3:
    with st.container():
        st.write('<div class="center-text">Daily Gain/Loss</div>', unsafe_allow_html=True)
        daily_gain_loss = pd.to_numeric(portfolio_df["Daily Return"], errors="coerce").sum()
        color = "green" if daily_gain_loss >= 0 else "red"
        st.markdown(
            f'<div class="center-text {color}">{daily_gain_loss:,.2f}</div>',
            unsafe_allow_html=True,
        )

with col4:
    with st.container():
        portfolio_df["Allocation (%)"] = pd.to_numeric(portfolio_df["Allocation (%)"], errors="coerce")
        portfolio_df["Daily Return (in %)"] = pd.to_numeric(portfolio_df["Daily Return (in %)"], errors="coerce")

        st.write('<div class="center-text">Daily Gain/Loss %</div>', unsafe_allow_html=True)
        daily_gain_loss_percent = pd.to_numeric(portfolio_df["Daily Return (in %)"]*portfolio_df["Allocation (%)"], errors="coerce").sum()
        color = "green" if daily_gain_loss_percent >= 0 else "red"
        st.markdown(
            f'<div class="center-text {color}">{daily_gain_loss_percent/100:,.2f}%</div>',
            unsafe_allow_html=True,
        )

st.divider()

st.subheader('Risk Metrics:')

tickers = portfolio_df['Ticker'].tolist()
end_date = datetime.today()
start_date = end_date - timedelta(days=365 * 5)

adj_close_df = pd.DataFrame()
for ticker in tickers:
    data = yf.download(ticker, start=start_date, end=end_date)
    adj_close_df[ticker] = data['Close']

# Calculate log returns
log_returns = np.log(adj_close_df / adj_close_df.shift(1))
log_returns = log_returns.dropna()

# Covariance matrix
cov_matrix = log_returns.cov() * 252

# Define functions
def standard_deviation(weights, cov_matrix):
    variance = weights.T @ cov_matrix @ weights
    return np.sqrt(variance)

def expected_returns(weights, log_returns):
    mean_returns = log_returns.mean()*252
    return np.dot(weights, mean_returns)

def sharpe_ratio(weights, log_returns, risk_free_rate, cov_matrix):
    return (expected_returns(weights, log_returns) - risk_free_rate) / standard_deviation(weights, cov_matrix)

def neg_sharpe_ratio(weights, log_returns, cov_matrix, risk_free_rate):
    return -sharpe_ratio(weights, log_returns, risk_free_rate, cov_matrix)

# Optimization
risk_free_rate = rf_rate / 100
constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})  
bounds = [(0.05, 0.5) for _ in range(len(tickers))]  
initial_weights = portfolio_df['Allocation (%)'].values / 100

optimized_results = minimize(
    neg_sharpe_ratio, 
    initial_weights, 
    args=(log_returns, cov_matrix, risk_free_rate), 
    method='SLSQP', 
    bounds=bounds, 
    constraints=constraints
)

optimal_weights = optimized_results.x

#VaR and CVar Calculation. 
#Get random z-score based on the number of assets in the portfolio.
def random_z_score():
    return np.random.normal(0,1)

days = var_days
portfolio_value = portfolio_df['Total Value'].sum()
portfolio_expected_return = expected_returns(initial_weights, log_returns)*252
portfolio_std_dev = standard_deviation(initial_weights, cov_matrix)
def scenario_gain_loss(portfolio_value, portfolio_std_dev, z_score, days):
    return portfolio_value * (portfolio_expected_return/252) *days + portfolio_value * portfolio_std_dev*z_score*np.sqrt(days)

simulations = 10000
scenario_returns = []

for i in range(simulations):
    z_score = random_z_score()
    scenario_returns.append(scenario_gain_loss(portfolio_value, portfolio_std_dev, z_score, days))

confidence_interval = var_conf_interval
var = -np.percentile(scenario_returns, (1 - confidence_interval) * 100)

tail_losses = [loss for loss in scenario_returns if loss <= -var]
cvar = -np.mean(tail_losses)
# Display results
col1, col2, col3, col4 = st.columns(4)

#with st.spinner('Calculating standard deviation...'):
#    time.sleep(5)
#with st.spinner('Calculating sharpe ratio...'):
#    time.sleep(5)
#with st.spinner('Running 10000 Monte Carlo simulations...'):
#    time.sleep(5)
#with st.spinner('Calculating VaR...'):
#    time.sleep(5)


with col1:
    with st.container():
        st.write('<div class="center-text">Volatility</div>', unsafe_allow_html=True)
        stddev = standard_deviation(initial_weights, cov_matrix)*100
        st.markdown(
            f'<div class="center-text">{stddev:,.2f}%</div>',
            unsafe_allow_html=True,
        )

with col2:
    with st.container():
        st.write('<div class="center-text">Sharpe Ratio</div>', unsafe_allow_html=True)
        s_ratio = sharpe_ratio(initial_weights, log_returns, risk_free_rate, cov_matrix)
        st.markdown(
            f'<div class="center-text">{s_ratio:,.2f}</div>',
            unsafe_allow_html=True,
        )

with col3:
    with st.container():
        st.write('<div class="center-text">VaR</div>', unsafe_allow_html=True)
        st.markdown(
            f'<div class="center-text">{var:.2f}</div>',
            unsafe_allow_html=True,
        )

with col4:
    with st.container():
        st.write('<div class="center-text">CVaR</div>', unsafe_allow_html=True)
        st.markdown(
            f'<div class="center-text">{cvar:.2f}</div>',
            unsafe_allow_html=True,
        )


st.divider()
total_value = portfolio_df['Total Value'].sum()

number_of_shares = [
    (weight * total_value) / price
    for weight, price in zip(optimal_weights, portfolio_df['Current Price'])
]
data = {
    'Ticker': tickers,
    'Optimised Weights': [f'{weight * 100:.2f}%' for weight in optimal_weights],
    'Number of Shares': [f'{shares:.0f}' for shares in number_of_shares],
}
weights_df = pd.DataFrame(data)
with st.expander('Optimised Portfolio'):
    st.subheader('Optimised Portfolio') 
    st.table(weights_df)  

    col1, col2, col3 = st.columns(3)
    with col1:
        with st.container():
            st.write('<div class="center-text">Volatility</div>', unsafe_allow_html=True)
            opti_stddev = standard_deviation(optimal_weights, cov_matrix)*100
            st.markdown(
                f'<div class="center-text">{opti_stddev:,.2f}%</div>',
                unsafe_allow_html=True,
            )

    with col2:
        with st.container():
            st.write('<div class="center-text">Sharpe Ratio</div>', unsafe_allow_html=True)
            opti_s_ratio = sharpe_ratio(optimal_weights, log_returns, risk_free_rate, cov_matrix)
            st.markdown(
                f'<div class="center-text">{opti_s_ratio:,.2f}</div>',
                unsafe_allow_html=True,
            )

    with col3:
        with st.container():
            st.write('<div class="center-text">Expected Returns</div>', unsafe_allow_html=True)
            er = expected_returns(optimal_weights, log_returns)*100
            st.markdown(
                f'<div class="center-text">{er:.2f}%</div>',
                unsafe_allow_html=True,
            )


st.divider()
from langchain_ollama import ChatOllama


def ai_report(df, other_info):
    other_info = f'{other_info}'
    llm = ChatOllama(
        model="llama3:latest",
        temperature=0.2
    )
    messages = [
    ("system", """You are a portfolio and equity analyst  writer and You have to generate reports like one. Never answer in first person. \
     Your job is to take portfolio data and convert them into insightful portfolio reports. You will recieve user's portfolio and your task is to create reports out of them. Give comprehensive explainations of all the figures and provide recommendations on the choice of stocks that the user has chosen. Make your answers detailed and in-depth.  \
     each section should contain a short paragraph on what the section is about, what qualities in a portfolio makes the section 'good' and whether the user portfolio satisfies the qualities. \
     for example: Industry composition: this helps us understand diversification. Diversification is important because it makes sure that our portfolio is spread out across different industries. our portfolio is good because it is sufficiently diversified (or) it is bad because it is not sufficiently diversified.\
     Answer in markdown.\
     The currency to be used is the local currency of the country\
     Each heading should have a rating out of 5. \
     Use the following format: \
     # Portfolio report \
     ## Overall portfolio summary \
     ## Portfolio composition \
     ## Industry composition \
     ## Risk analysis \
     ## Recommendations (if the )"""),
    ("human", f'My individual stock data is in {df} and my overall stock data is in {other_info}'),
    ]
    response = llm.stream(messages)  # Ensure `llm.stream()` yields tokens or chunks of content
    for chunk in response:
        yield chunk.content



ai_report_button = st.button('Generate AI report', use_container_width=True)

portfolio_info = f"""Total portfolio value: {total_value:,.2f}\
portfolio industry distribution: {portfolio_df['Industry'].value_counts()}\
Name of invested stocks: {portfolio_df['Company Name']}\
total portfolio return: {total_returns:,.2f}\
daily gain/loss: {daily_gain_loss:,.2f}\
risk metrics: \
standard deviation: {stddev:,.2f}%\
sharpe ratio: {s_ratio:,.2f}\
expected returns: {expected_returns(initial_weights, log_returns)*100:,.2f}%\
VaR: {var:,.2f} where the confidence interval is {var_conf_interval:,.2f} and the time period is {days} days\
CVaR: {cvar:,.2f}\
"""

if ai_report_button:
    with st.spinner('Generating AI report...'):
        report_placeholder = st.empty()  # Placeholder for streaming text
        report_text = ""  # Initialize an empty report text
        for chunk in ai_report(portfolio_df, portfolio_info):
            report_text += chunk  # Append each chunk to the text
            report_placeholder.markdown(report_text) 
