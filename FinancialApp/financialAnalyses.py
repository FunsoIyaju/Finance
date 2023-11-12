# Libraries
import numpy as np
import pandas as pd
import plotly.graph_objects as go
#import plotly.express as px
from plotly.subplots import make_subplots
#from datetime import datetime, timedelta
import yfinance as yf
import streamlit as st


# ==============================================================================
# HOT FIX FOR YFINANCE .INFO METHOD
# Ref: https://github.com/ranaroussi/yfinance/issues/1729
# ==============================================================================

import requests
import urllib


class YFinance:
    user_agent_key = "User-Agent"
    user_agent_value = ("Mozilla/5.0 (Windows NT 6.1; Win64; x64) "
                        "AppleWebKit/537.36 (KHTML, like Gecko) "
                        "Chrome/58.0.3029.110 Safari/537.36")

    def __init__(self, ticker):
        self.yahoo_ticker = ticker

    def __str__(self):
        return self.yahoo_ticker

    def _get_yahoo_cookie(self):
        cookie = None

        headers = {self.user_agent_key: self.user_agent_value}
        response = requests.get("https://fc.yahoo.com",
                                headers=headers,
                                allow_redirects=True)

        if not response.cookies:
            raise Exception("Failed to obtain Yahoo auth cookie.")

        cookie = list(response.cookies)[0]

        return cookie

    def _get_yahoo_crumb(self, cookie):
        crumb = None

        headers = {self.user_agent_key: self.user_agent_value}

        crumb_response = requests.get(
            "https://query1.finance.yahoo.com/v1/test/getcrumb",
            headers=headers,
            cookies={cookie.name: cookie.value},
            allow_redirects=True,
        )
        crumb = crumb_response.text

        if crumb is None:
            raise Exception("Failed to retrieve Yahoo crumb.")

        return crumb

    @property
    def info(self):
        # Yahoo modules doc informations :
        # https://cryptocointracker.com/yahoo-finance/yahoo-finance-api
        cookie = self._get_yahoo_cookie()
        crumb = self._get_yahoo_crumb(cookie)
        info = {}
        ret = {}

        headers = {self.user_agent_key: self.user_agent_value}

        yahoo_modules = ("assetProfile," 
                         "summaryDetail,"
                         "financialData,"
                         "indexTrend,"
                         "defaultKeyStatistics")

        url = ("https://query1.finance.yahoo.com/v10/finance/"
               f"quoteSummary/{self.yahoo_ticker}"
               f"?modules={urllib.parse.quote_plus(yahoo_modules)}"
               f"&ssl=true&crumb={urllib.parse.quote_plus(crumb)}")

        info_response = requests.get(url,
                                     headers=headers,
                                     cookies={cookie.name: cookie.value},
                                     allow_redirects=True)

        info = info_response.json()
        info = info['quoteSummary']['result'][0]

        for mainKeys in info.keys():
            for key in info[mainKeys].keys():
                if isinstance(info[mainKeys][key], dict):
                    try:
                        ret[key] = info[mainKeys][key]['raw']
                    except (KeyError, TypeError):
                        pass
                else:
                    ret[key] = info[mainKeys][key]

        return ret


def dashboard_header():
    st.set_page_config(page_title="Stock Analysis", page_icon=":bar_chart:", layout="wide")
    st.title(" :bar_chart: Stocks Financial Application")
    st.markdown('<style> div.block-container{padding-top:1rem;}</style>', unsafe_allow_html=True)



@st.cache_data
def getStockInfo(ticker, start_date, end_date):
    stock_df = yf.Ticker(ticker).history(start=start_date, end=end_date)
    stock_df.reset_index(inplace=True)  # Drop the indexes
    stock_df['Date'] = stock_df['Date'].dt.date  # Convert date-time to date
    return stock_df


@st.cache_data
def getHistoricalStockPrice(ticker, period="MAX", interval="1d"):
    stock_df = yf.Ticker(ticker).history(interval=interval, period=period)
    stock_df.reset_index(inplace=True)  # Drop the indexes
    stock_df['Date'] = stock_df['Date'].dt.date  # Convert date-time to date
    return stock_df


@st.cache_data
def getCompanyInfo(ticker):
    companySummary = YFinance(ticker).info
    # shareHolders = yf.Ticker(ticker).major_holders
    shareHolders = yf.Ticker(ticker).institutional_holders
    companyInfo = [companySummary, shareHolders]
    return companyInfo


@st.cache_data
def getFinancials(ticker, period="Annual"):
    financial = yf.Ticker(ticker)
    if period == "Annual":
        balance_sheet = financial.balance_sheet
        income_statement = financial.income_stmt
        cash_flow = financial.cashflow
    else:
        balance_sheet = financial.quarterly_balance_sheet
        income_statement = financial.quarterly_income_stmt
        cash_flow = financial.quarterly_cashflow
    financialInfo = [balance_sheet, income_statement, cash_flow]
    return financialInfo


@st.cache_data
def getOtherInfo(ticker, period="Annual"):
    others = yf.Ticker(ticker)
    _news = others.news
    _dividends = others.dividends
    otherInfo = [_news, _dividends]
    return otherInfo

def run_simulation(stock_price, time_horizon, n_simulation, seed):
    simulated_df_new = pd.DataFrame()
    volatility = stock_price['Close'].pct_change().std()
    for n_simu in range(0, n_simulation):
        stock_price_lst = []
        current_price = stock_price['Close'].iloc[-1]
        for i in range(0, time_horizon):
            # np.random.seed(seed)
            dy_return = np.random.normal(0, volatility, 1)[0]
            future_price = current_price * (1 + dy_return)
            stock_price_lst.append(future_price)
            current_price = future_price
        simulated_col = pd.Series(stock_price_lst)
        simulated_col.name = "Sim" + str(n_simu)
        simulated_df_new = pd.concat([simulated_df_new, simulated_col], axis=1)
    return simulated_df_new

# This is my streamlit application
def financialInfo(ticker, period="Annual"):
    @st.cache_resource
    def getFinancialInfo(ticker):
        financialData = yf.Ticker(ticker)
        return financialData

    financial = getFinancialInfo(ticker)
    if period == "Annual":
        balance_sheet = financial.balance_sheet
        income_statement = financial.income_stmt
        cash_flow = financial.cashflow
    else:
        balance_sheet = financial.quarterly_balance_sheet
        income_statement = financial.quarterly_income_stmt
        cash_flow = financial.quarterly_cashflow
    financial = [balance_sheet, income_statement, cash_flow]
    return financial

# Sets padding for figures
def set_padding(fig):
    fig.update_layout(margin=go.layout.Margin(
        r=10, #right margin
        b=10))

# Adds the range selector to given figure
def add_range_selector(fig):
    fig.update_layout(
        xaxis=dict(
            rangeselector=dict(
                buttons=[
                    dict(count=1, label='1M', step='month', stepmode='backward'),
                    dict(count=3, label='3M', step='month', stepmode='backward'),
                    dict(count=6, label='6M', step='month', stepmode='backward'),
                    dict(count=1, label='YTD', step='year', stepmode='todate'),
                    dict(count=1, label='1Y', step='year', stepmode='backward'),
                    dict(count=3, label='3Y', step='year', stepmode='backward'),
                    dict(count=5, label='5Y', step='year', stepmode='backward'),
                    dict(step='all')
                ]),
            type='date'),#end xaxis  definition
        xaxis2_type='date')

# Adds the volume chart to row 2, column 1
def add_volume_chart(fig):
    # Colours for the Bar chart
    colors = ['#9C1F0B' if row['Open'] - row['Close'] >= 0
          else '#2B8308' for index, row in stockprice.iterrows()]
    fig.add_trace(go.Bar(x='Date', y=stockprice['Volume'], showlegend=False, marker_color=colors), row=2, col=1)

# Main body of the financial dashboard #
# plotly package was reinstall

dashboard_header()

ticker_list = pd.read_html('https://en.wikipedia.org/wiki/List_of_S%26P_500_companies')[0]['Symbol']
st.sidebar.header("Choose your filter :")
ticker = st.sidebar.selectbox("Ticker", ticker_list)
#start = st.sidebar.date_input("Start Date", datetime.today().date() - timedelta(days=30))
#end = st.sidebar.date_input("End Date", datetime.today().date())
global stockprice
stockprice = getHistoricalStockPrice(ticker, interval="1d")
intervals = ["1d", "5d", "1wk", "1mo", "3mo"]
interval = st.sidebar.selectbox("Select Time Interval", intervals)
button = st.sidebar.button("Update Stock Data")
if button:
    stockprice = getHistoricalStockPrice(ticker, interval)


tab1, tab2, tab3, tab4, tab5 = st.tabs(["Summary", "Chart", "Financials", "Monte Carlo Simulation", "Analysis"])
with tab1:
    if ticker != '':
        infolst = getCompanyInfo(ticker)
        st.write('**Company Profile**')
        st.markdown('<div style="text-align: justify;">' + infolst[0]['longBusinessSummary'] + '</div><br>',
                    unsafe_allow_html=True)
        col1, col2 = st.columns([0.3, 0.7])
        with col1:
            st.write('**Key Statistics**')
            info_keys = {'previousClose': 'Previous Close',
                         'open': 'Open',
                         'bid': 'Bid',
                         'ask': 'Ask',
                         'marketCap': 'Market Cap',
                         'volume': 'Volume',
                         'currentPrice': 'Current Share Price',
                         'operatingMargins': 'Operating Margin',
                         'profitMargins': ' Profit Margin',
                         'ebitda': 'EBITDA'
                         }
            company_stats = {}  # Dictionary
            for key in info_keys:
                company_stats.update({info_keys[key]: infolst[0][key]})
            company_stats = pd.DataFrame({'Value': pd.Series(company_stats)})
            st.dataframe(company_stats, use_container_width=True)
        with col2:
            st.write('**Stock Chart**')
            #periods = ["1M", "3M", "6M", "YTD", "1Y", "3Y", "5Y", "MAX"]
            #period = st.selectbox("Time Duration", periods)
            #stockprice = getHistoricalStockPrice(ticker, interval="1d")
            st.area_chart(stockprice, x="Date", y="Close", use_container_width=True)

    st.write('**Share Holders**')
    st.dataframe(infolst[1], hide_index=True,  use_container_width=True)
    st.markdown('[Find more information on Wikipedia](https://en.wikipedia.org/)', unsafe_allow_html=True)
with tab2:
    #periods = ["1M", "3M", "6M", "YTD", "1Y", "3Y", "5Y", "MAX"]

    #stockprice = getHistoricalStockPrice(ticker, interval="1d")
    #col1, col2 = st.columns(2)
    #with col1:
        # period = st.selectbox("Select Time Duration", periods)
    #with col2:
    #interval = st.selectbox("Select Time Interval", intervals)
    #stockprice = getHistoricalStockPrice(ticker, period="MAX", interval="1d")
    if ticker != '':
        show_data = st.checkbox("Show Chart as Candle")
        if show_data:
            st.write('**Candlestick Chart**')
            fig = go.Figure()
            fig.add_candlestick(x=stockprice['Date'], open=stockprice['Open'], high=stockprice['High'], low=stockprice['Low'], close=stockprice['Close'])
            #fig.add_trace(go.Candlestick(x=stockprice['Date'], open=stockprice['Open'], high=stockprice['High'], low=stockprice['Low'], close=stockprice['Close']) )
            st.plotly_chart(fig, use_container_width=True)
        else:
            #st.line_chart(stockprice, x='Date', y='Close', use_container_width=True)

            # Construct a 2 x 1 Plotly figure
            fig = make_subplots(rows=2, cols=1, vertical_spacing=0.01, shared_xaxes=True)

            # Plot the Price chart
            fig.add_trace(go.Scatter(x=stockprice['Date'], y=stockprice['Close'], name='Price'), row=1, col=1)
            #fig = px.line(stockprice, x='Date', y='Close',
                          #title=ticker + ' - Stock Market Analysis with Time Period Selectors', height=500)

            # Add the volume chart
            #add_volume_chart(fig)

            # Adds the range selector
            add_range_selector(fig)

            # Set the color from white to black on range selector buttons
            fig.update_layout(xaxis=dict(rangeselector=dict(font=dict(color='black'))))

            # Add labels to y axes
            fig.update_yaxes(title_text="Price", row=1, col=1)
            #fig.update_yaxes(title_text="Volume", row=2, col=1)

            # Sets customized padding
            set_padding(fig)

            # Remove dates without values
            #fig.update_xaxes(rangebreaks=[dict(values=dt_breaks)])

            # Set the template and the title
            layout = go.Layout(title=ticker + ' - Price', height=500, legend_title='Legend')
            fig.update_layout(layout)
            #fig = px.line(stockprice, x='Date', y='Close', title=ticker + ' - Stock Market Analysis with Time Period Selectors', height=500)
            st.plotly_chart(fig, use_container_width=True)
            #fig.show()
with tab3:
    periodList = ["Annual", "Quarterly"]
    selected_period = st.selectbox("Select Period", periodList)
    financialData = financialInfo(ticker, selected_period)
    fin01, fin02, fin03 = st.tabs(["Balance Sheet", "Income Statement", "Cash Flow"])
    with fin01:
        st.dataframe(financialData[0], use_container_width=True)
    with fin02:
        st.dataframe(financialData[1], use_container_width=True)
    with fin03:
        st.dataframe(financialData[2], use_container_width=True)

with tab4:
    noSimulationList = [200, 500, 1000]
    timeHorizonList = [30, 60, 90]
    col1, col2 = st.columns(2)
    with col1:
        noSimulation = st.selectbox("Select No. of Simulation", noSimulationList)
    with col2:
        timeHorizon = st.selectbox("Select Time Horizon (Days)", timeHorizonList)

    #PriceDF = getStockInfo(ticker, start, end)
    df = run_simulation(stockprice, timeHorizon, noSimulation, 30)
    st.write('**Monte Carlo Simulation for ' + ticker + ' Stock Price in the next ' + str(timeHorizon) + ' days**')
    st.line_chart(df, use_container_width=True)
    var_at_risk = df.values[-1:]
    percentiles = [1, 5, 10]
    v99, v95, v90 = np.percentile(var_at_risk, percentiles) * -1
    st.write('**VAR for ' + ticker + ' at 95% Confidence Level**')
    #st.write(f"At 99% confidence level, loss will not exceed {v99:,.2f}")
    st.write(f"At 95% confidence level, loss will not exceed {v95:,.2f}")
    #st.write(f"At 90% confidence level, loss will not exceed {v90:,.2f}")
with tab5:
    moreInfo = getOtherInfo(ticker)
    st.write('**Dividend Payout Trend**')
    st.line_chart(moreInfo[1], use_container_width=True)
