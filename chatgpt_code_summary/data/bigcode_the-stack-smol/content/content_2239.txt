

import pandas as pd
import tweepy
from textblob  import TextBlob
from wordcloud import WordCloud
import plotly.graph_objs as go
import os
import re
import pystan
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import yfinance as yf
from fbprophet import Prophet
from fbprophet.plot import plot_plotly
from GoogleNews import GoogleNews
from ta.volatility import BollingerBands
from ta.trend import MACD
from ta.momentum import RSIIndicator
import datetime as datetime
import base64
import pandas as pd
import plotly.express as px
import datetime
import requests
from bs4 import BeautifulSoup

from datetime import date
from plotly import graph_objs








st.set_page_config( 
layout="wide",  
initial_sidebar_state="auto",
page_title= "Finance-Forcasting-Dashboard",  
page_icon= "Images/growth.png", 
)



    

col1, col2, col3 = st.beta_columns([1,2,1])
col1.write("")
col2.image("Images/LL.png", width = 500)
col3.write("")



st.set_option('deprecation.showPyplotGlobalUse', False)

main_bg = "Images/BACK.png"
main_bg_ext = "Images/BACK.png"


st.markdown(
    f"""
    <style>
    .reportview-container {{
        background: url(data:image/{main_bg_ext};base64,{base64.b64encode(open(main_bg, "rb").read()).decode()})
    }}
    </style>
    """,
    unsafe_allow_html=True
)
###############################Funtions############################

# load data from yahoo finance
def load_data(ticker):
    start = "2020-01-01"
    today = date.today().strftime("%Y-%m-%d")
    data = yf.download(ticker, start, today)
    data.reset_index(inplace=True)
    return data

# Plot raw data
def plot_raw_data():
    fig = graph_objs.Figure()
    fig.add_trace(graph_objs.Scatter(x=data['Date'], y=data['Open'], name="stock_open"))
    fig.add_trace(graph_objs.Scatter(x=data['Date'], y=data['Close'], name="stock_close"))
    fig.layout.update(title_text='Time Series data with Rangeslider', xaxis_rangeslider_visible=True)
    st.plotly_chart(fig)

def get_forecast(data):
    model = Prophet()
    model.fit(data)
    future = model.make_future_dataframe(periods=7)
    forecast = model.predict(future)
    
    return model, forecast

@st.cache
def read_data():
    url = "https://raw.githubusercontent.com/emrecanaltinsoy/forex_data/main/forex_usd_data.csv"
    data = pd.read_csv(url)
    cols = data.columns
    return data, cols[1:]


@st.cache
def get_range(data, date_range):
    start_index = data.index[data["date(y-m-d)"] == str(date_range[0])].tolist()[0]
    end_index = data.index[data["date(y-m-d)"] == str(date_range[1])].tolist()[0]
    data = data.iloc[start_index : end_index + 1]
    cols = data.columns
    dates = data["date(y-m-d)"]
    return data, dates


@st.cache
def scrape_currency():
    today = datetime.date.today()

    base_url = "https://www.x-rates.com/historical/?from=USD&amount=1&date"

    year = today.year
    month = today.month if today.month > 9 else f"0{today.month}"
    day = today.day if today.day > 9 else f"0{today.day}"

    URL = f"{base_url}={year}-{month}-{day}"

    page = requests.get(URL)

    soup = BeautifulSoup(page.content, "html.parser")

    table = soup.find_all("tr")[12:]

    currencies = [table[i].text.split("\n")[1:3][0] for i in range(len(table))]
    currencies.insert(0, "date(y-m-d)")
    currencies.insert(1, "American Dollar")
    rates = [table[i].text.split("\n")[1:3][1] for i in range(len(table))]
    rates.insert(0, f"{year}-{month}-{day}")
    rates.insert(1, "1")
    curr_data = {currencies[i]: rates[i] for i in range(len(rates))}
    curr_data = pd.DataFrame(curr_data, index=[0])

    cols = curr_data.columns

    return curr_data, cols[1:]


@st.cache
def train_model(data, currency, period):
    df_train = data[["date(y-m-d)", currency]]
    df_train = df_train.iloc[-365*2 :]
    df_train = df_train.rename(columns={"date(y-m-d)": "ds", currency: "y"})

    m = Prophet()
    m.fit(df_train)
    future = m.make_future_dataframe(periods=period)
    forecast = m.predict(future)

    return forecast, m

df_all, columns = read_data()

################################################################################



st.sidebar.image("Images/Menu.png", width = 330)


menu = ["Home","STOCKS Live Forcasting", "Crypto-Live Forcasting","View Historical Currency Charts", "Check Live Currency Exchange rates", "Forecast Currency Live Prices"]

choice = st.sidebar.selectbox("Menu", menu)
if choice == "Home":
    
  st.write("")

  st.write("""  <p style=" font-size: 15px; font-weight:normal; font-family:verdana"> Finance Dashboard is a special web service that allows you to view Cryptocurrencies,Stocks,and Live Currency Values by many useful methods (technical indicators, graphical patterns, sentimental analysis, and more). Trading and crypto investing requires constant analysis and monitoring. Traders need to track all their trades in order to improve results and find errors. If you don't use additional instruments, then trading will be unsystematic, and the results will be uncertain. Such a service will be useful and even extremely necessary for those who trade and invest in cryptocurrencies and Stocks. Competent selection of cryptocurrencies is at least half of investment success. Finance Dashboard has a simple interface and is great for quick analysis of the Stock market.   </p>
  """, unsafe_allow_html=True)
    
  st.write("")
  st.write("")
  st.write("")
  st.write("") 
  st.write("")


  st.write(""" <p style=" color:#E75480; font-size: 30px; font-weight:bold"> How does it work? </p>
  """, unsafe_allow_html=True)

  st.write("")




  st.image("Images/How.png", width = 1300)

  st.sidebar.write(" ")
  st.sidebar.write(" ")
    
  st.sidebar.image("Images/info.png", width = 300)


elif choice == "STOCKS Live Forcasting":
  st.title('Stocks Weekly Forecast')
  st.subheader('Enter the stock ticker:')
  ticker = st.text_input('example: GOOG')
  ticket = ticker.upper()
  if len(ticker)>0:
    data_load_state = st.text('Loading data...')
    data = load_data(ticker)
    if data.empty:
        data_load_state.text(f'No ticker named {ticker}')
        ticker = ''
    else:
        data_load_state.text('Loading data... done!')

        st.subheader(f'Company: {yf.Ticker(ticker).info["longName"]}')
        st.write(data.head())

        plot_raw_data()

        # prepare data for forecasting
        df_train = data[['Date','Close']]
        df_train = df_train.rename(columns={"Date": "ds", "Close": "y"})

        # train and forecast
        model, forecast = get_forecast(df_train)

        st.subheader('Forecast')
        
        # plot forecast
        st.write(f'Forecast plot for the next week')
        fig = plot_plotly(model, forecast)
        st.plotly_chart(fig)













elif choice == "View Historical Currency Charts":
    st.write("This app can be used to view historical **currency** charts!")

    date_range = st.date_input(
        "Choose date range",
        value=(
            datetime.date(2011, 1, 1),
            datetime.date(2011, 1, 1) + datetime.timedelta(df_all.shape[0] - 1),
        ),
        min_value=datetime.date(2011, 1, 1),
        max_value=datetime.date(2011, 1, 1) + datetime.timedelta(df_all.shape[0] - 1),
    )

    df, dates = get_range(df_all, date_range)

    selected_curr = st.multiselect("Select currencies", columns)

    ok = st.button("View")
    if ok:
        if selected_curr:
            # st.write(df[selected_curr])

            for curr in selected_curr:
                fig = px.line(
                    x=dates,
                    y=df[curr],
                )
                fig.update_layout(
                    xaxis_title="Date",
                    yaxis_title=curr,
                )
                st.write(fig)

elif choice == "Check Live Currency Exchange rates":
    st.write("This app can be used to check current **currency** data!")
    daily_df, columns = scrape_currency()
    base_curr = st.selectbox("Select the base currency", columns)
    selected_curr = st.multiselect("Select currencies", columns)
    if selected_curr:
        base = daily_df[base_curr].astype(float)

        selected = daily_df[selected_curr].astype(float)

        converted = selected / float(base)
        st.write(converted)

elif choice == "Forecast Currency Live Prices":
    currency = st.selectbox("Select the currency for prediction", columns)

    n_weeks = st.slider("Weeks of prediction", 4, 20, 8, 1)

    ok = st.button("Predict")
    if ok:
        train_state = st.text("Training the model...")
        pred, model = train_model(df_all, currency, period=n_weeks * 7)
        train_state.text("Model training completed!!")

        st.subheader("Forecast data")
        fig1 = plot_plotly(model, pred)
        st.plotly_chart(fig1)





elif choice == "Crypto-Live Forcasting":
  st.sidebar.header("Please select cryptocurrency")
  option = st.sidebar.selectbox("Ticker Symbol",("BTC-USD", "ETH-USD", "XRP-USD", "DOGE-USD", "ADA-USD", "BNB-USD", "LTC-USD",))
  today = datetime.date.today()
  before = today - datetime.timedelta(days=1400)
  start_date = st.sidebar.date_input('Start date', before)
  end_date = st.sidebar.date_input('End date', today)
  if start_date < end_date:
    st.sidebar.success("Start date:  `%s`\n\nEnd date: `%s` " % (start_date, end_date))
  else:
    st.sidebar.error("Error: End date must fall after start date.")

  @st.cache(allow_output_mutation = True)  
  def get_data(option, start_date, end_date):
    df = yf.download(option,start= start_date,end = end_date, progress=False)
    return df
  
  # Getting API_KEYS
  api_key = os.environ.get("Key")
  api_secret = os.environ.get("Secret")

  # Function for getting tweets
  # Create authentication
  @st.cache(allow_output_mutation = True)  
  def get_tweets(key, secret, search_term):
    authentication = tweepy.OAuthHandler(api_key, api_secret)
    api = tweepy.API(authentication)
    term = search_term+"-filter:retweets"
    # Create a cursor object 
    tweets = tweepy.Cursor(api.search, q = term, lang = "en",
                         since = today, tweet_mode = "extended").items(100)
    # Store the tweets
    tweets_text = [tweet.full_text for tweet in tweets]
    df = pd.DataFrame(tweets_text, columns = ["Tweets"]) 
    return df

  # Clean text
  @st.cache(allow_output_mutation = True) 
  def Clean(twt):
    twt = re.sub("#cryptocurrency", "cryptocurrency", twt)
    twt = re.sub("#Cryptocurrency", "Cryptocurrency", twt)
    twt = re.sub("#[A-Za-z0-9]+", "", twt)
    twt = re.sub("RT[\s]+", "", twt)
    twt = re.sub("\\n", "", twt)
    twt = re.sub("https?\://\S+", '', twt)
    twt = re.sub("<br />", "", twt)
    twt = re.sub("\d","", twt)
    twt = re.sub("it\'s", "it is", twt)
    twt = re.sub("can\'t", "cannot", twt)
    twt = re.sub("<(?:a\b[^>]*>|/a>)", "", twt)
    return twt

  # Subjectivity and Polarity
  @st.cache(allow_output_mutation = True)
  def subjectivity(text):
    return TextBlob(text).sentiment.subjectivity
  @st.cache(allow_output_mutation = True)
  def polarity(text):
    return TextBlob(text).sentiment.polarity


  # Create a function to get sentiment text
  @st.cache(allow_output_mutation = True)
  def sentiment(score):
    if score < 0:
      return "Negative"
    elif score == 0:
      return "Neutral"
    else:
      return "Positive" 

  if option == "BTC-USD":
    df = get_data(option, start_date, end_date)

    st.write(""" <p style=" color:#FFCC00; font-size: 30px; font-weight:bold"> Raw Data </p>
  """, unsafe_allow_html=True)

    st.write("    ")
    st.write(df)
    st.write(""" <p style=" color:#FFCC00; font-size: 30px; font-weight:bold"> Close Price </p>
  """, unsafe_allow_html=True)
    st.write("    ")
    st.line_chart(df["Close"])

    st.write(" ")


    # MACD

    st.write(" ")
    macd = MACD(df["Close"]).macd()

    st.write(""" <p style=" color:#FFCC00; font-size: 30px; font-weight:bold"> Moving Average Convergence Divergence </p>
  """, unsafe_allow_html=True)
    st.write(" ")

    st.area_chart(macd)


    # Bollinger Bands
    bb_bands = BollingerBands(df["Close"])
    bb = df
    bb["bb_h"] = bb_bands.bollinger_hband()
    bb["bb_l"] = bb_bands.bollinger_lband()
    bb = bb[["Close","bb_h","bb_l"]]

    st.write(""" <p style=" color:#FFCC00; font-size: 30px; font-weight:bold"> Bollinger Bands </p>
  """, unsafe_allow_html=True)
    st.line_chart(bb)


    st.write(" ")


    # Resistence Strength Indicator
    
    rsi = RSIIndicator(df["Close"]).rsi()
    st.write(""" <p style=" color:#FFCC00; font-size: 30px; font-weight:bold"> Resistence Strength Indicator </p>
  """, unsafe_allow_html=True)
    st.write(" ")
    st.line_chart(rsi)

    st.write("  ")

    st.write(""" <p style=" color:#FFCC00; font-size: 30px; font-weight:bold"> BTC-USD Forecast using Facebook Prophet </p>
  """, unsafe_allow_html=True) 
    
    st.write("  ")



    data = df.reset_index()
    period = st.slider("Days of prediction:", 1, 365)
   
    # Predict forecast with Prophet.
    df_train = data[["Date","Close"]]
    df_train = df_train.rename(columns={"Date": "ds", "Close": "y"})

    m = Prophet()
    m.fit(df_train)
    future = m.make_future_dataframe(periods=period)
    forecast = m.predict(future)
    
    #Plot
    st.write(f'Forecast plot for {period} days')
    fig1 = plot_plotly(m, forecast)
    st.plotly_chart(fig1)

  

    st.write(" ")

    st.write(""" <p style=" color:#FFCC00; font-size: 30px; font-weight:bold"> Latest News </p>
  """, unsafe_allow_html=True)

    st.write("  ")    

    news = GoogleNews()
    news = GoogleNews("en", "d")
    news.search("Bitcoin")
    news.get_page(1)
    result = news.result()
    st.write("1. " + result[1]["title"])
    st.info("1. " + result[1]["link"])
    st.write("2. " + result[2]["title"])
    st.info("2. " + result[2]["link"])
    st.write("3. " + result[3]["title"])
    st.info("3. " + result[3]["link"])
    st.write("4. " + result[4]["title"])
    st.info("4. " + result[4]["link"])
    st.write("5. " + result[5]["title"])
    st.info("5. " + result[5]["link"])

    


   


    



  elif option == "ETH-USD":
    df = get_data(option, start_date, end_date)

    st.write(""" <p style=" color:#FFCC00; font-size: 30px; font-weight:bold"> Raw Data </p>
  """, unsafe_allow_html=True)
    st.write(" ")

    st.write(df)

    st.write(" ")

    st.write(""" <p style=" color:#FFCC00; font-size: 30px; font-weight:bold"> Close Price </p>
  """, unsafe_allow_html=True)
    st.write("    ")
    st.line_chart(df["Close"])

    st.write(" ")

    # MACD

    st.write(" ")
    macd = MACD(df["Close"]).macd()

    st.write(""" <p style=" color:#FFCC00; font-size: 30px; font-weight:bold"> Moving Average Convergence Divergence </p>
  """, unsafe_allow_html=True)
    st.write(" ")

    st.area_chart(macd)


      # Bollinger Bands
    bb_bands = BollingerBands(df["Close"])
    bb = df
    bb["bb_h"] = bb_bands.bollinger_hband()
    bb["bb_l"] = bb_bands.bollinger_lband()
    bb = bb[["Close","bb_h","bb_l"]]

    st.write(""" <p style=" color:#FFCC00; font-size: 30px; font-weight:bold"> Bollinger Bands </p>
  """, unsafe_allow_html=True)
    st.line_chart(bb)


    st.write(" ")


    # Resistence Strength Indicator
    
    rsi = RSIIndicator(df["Close"]).rsi()
    st.write(""" <p style=" color:#FFCC00; font-size: 30px; font-weight:bold"> Resistence Strength Indicator </p>
  """, unsafe_allow_html=True)
    st.write(" ")
    st.line_chart(rsi)


    st.write("  ")

    
    st.write(""" <p style=" color:#FFCC00; font-size: 30px; font-weight:bold"> ETH-USD Forecast using Facebook Prophet </p>
  """, unsafe_allow_html=True)

    st.write("  ") 

    data = df.reset_index()
    period = st.slider("Days of prediction:", 1, 365)
   
    # Predict forecast with Prophet.
    df_train = data[["Date","Close"]]
    df_train = df_train.rename(columns={"Date": "ds", "Close": "y"})

    m = Prophet()
    m.fit(df_train)
    future = m.make_future_dataframe(periods=period)
    forecast = m.predict(future)
    
    st.write(f'Forecast plot for {period} days')
    fig1 = plot_plotly(m, forecast)
    st.plotly_chart(fig1)



    st.write(" ")

    st.write(""" <p style=" color:#FFCC00; font-size: 30px; font-weight:bold"> Latest News </p>
  """, unsafe_allow_html=True)

    st.write(" ")   

    news = GoogleNews()
    news = GoogleNews("en", "d")
    news.search("Etherium")
    news.get_page(1)
    result = news.result()
    st.write("1. " + result[1]["title"])
    st.info("1. " + result[1]["link"])
    st.write("2. " + result[2]["title"])
    st.info("2. " + result[2]["link"])
    st.write("3. " + result[3]["title"])
    st.info("3. " + result[3]["link"])
    st.write("4. " + result[4]["title"])
    st.info("4. " + result[4]["link"])
    st.write("5. " + result[5]["title"])
    st.info("5. " + result[5]["link"])

    
    
    
 

  elif option == "DOGE-USD":
    df = get_data(option, start_date, end_date)
    
    st.write(""" <p style=" color:#FFCC00; font-size: 30px; font-weight:bold"> Raw Data </p>
  """, unsafe_allow_html=True)
    st.write(" ")

    st.write(df)

    st.write(" ")

    st.write(""" <p style=" color:#FFCC00; font-size: 30px; font-weight:bold"> Close Price </p>
  """, unsafe_allow_html=True)
    st.write("    ")
    st.line_chart(df["Close"])

    st.write(" ")

    # MACD

    st.write(" ")
    macd = MACD(df["Close"]).macd()

    st.write(""" <p style=" color:#FFCC00; font-size: 30px; font-weight:bold"> Moving Average Convergence Divergence </p>
  """, unsafe_allow_html=True)
    st.write(" ")

    st.area_chart(macd)


      # Bollinger Bands
    bb_bands = BollingerBands(df["Close"])
    bb = df
    bb["bb_h"] = bb_bands.bollinger_hband()
    bb["bb_l"] = bb_bands.bollinger_lband()
    bb = bb[["Close","bb_h","bb_l"]]

    st.write(""" <p style=" color:#FFCC00; font-size: 30px; font-weight:bold"> Bollinger Bands </p>
  """, unsafe_allow_html=True)
    st.line_chart(bb)


    st.write(" ")


    # Resistence Strength Indicator
    
    rsi = RSIIndicator(df["Close"]).rsi()
    st.write(""" <p style=" color:#FFCC00; font-size: 30px; font-weight:bold"> Resistence Strength Indicator </p>
  """, unsafe_allow_html=True)
    st.write(" ")
    st.line_chart(rsi)

    st.write("  ")


    st.write(""" <p style=" color:#FFCC00; font-size: 30px; font-weight:bold"> DOGE-USD Forecast using Facebook Prophet </p>
  """, unsafe_allow_html=True) 
    
    st.write("  ")

    data = df.reset_index()
    period = st.slider("Days of prediction:", 1, 365)
   
    # Predict forecast with Prophet.
    df_train = data[["Date","Close"]]
    df_train = df_train.rename(columns={"Date": "ds", "Close": "y"})

    m = Prophet()
    m.fit(df_train)
    future = m.make_future_dataframe(periods=period)
    forecast = m.predict(future)
    
    st.write(f'Forecast plot for {period} days')
    fig1 = plot_plotly(m, forecast)
    st.plotly_chart(fig1)



    st.write(" ")

    st.write(""" <p style=" color:#FFCC00; font-size: 30px; font-weight:bold"> Latest News </p>
  """, unsafe_allow_html=True)

    st.write(" ")   

    news = GoogleNews()
    news = GoogleNews("en", "d")
    news.search("Dogecoin")
    news.get_page(1)
    result = news.result()
    st.write("1. " + result[1]["title"])
    st.info("1. " + result[1]["link"])
    st.write("2. " + result[2]["title"])
    st.info("2. " + result[2]["link"])
    st.write("3. " + result[3]["title"])
    st.info("3. " + result[3]["link"])
    st.write("4. " + result[4]["title"])
    st.info("4. " + result[4]["link"])
    st.write("5. " + result[5]["title"])
    st.info("5. " + result[5]["link"])

    st.write("  ")

   


  elif option == "XRP-USD":
    df = get_data(option, start_date, end_date)
    
    st.write(""" <p style=" color:#FFCC00; font-size: 30px; font-weight:bold"> Raw Data </p>
  """, unsafe_allow_html=True)
    st.write(" ")

    st.write(df)

    st.write(" ")

    st.write(""" <p style=" color:#FFCC00; font-size: 30px; font-weight:bold"> Close Price </p>
  """, unsafe_allow_html=True)
    st.write("    ")
    st.line_chart(df["Close"])

    st.write(" ")

    # MACD

    st.write(" ")
    macd = MACD(df["Close"]).macd()

    st.write(""" <p style=" color:#FFCC00; font-size: 30px; font-weight:bold"> Moving Average Convergence Divergence </p>
  """, unsafe_allow_html=True)
    st.write(" ")

    st.area_chart(macd)


      # Bollinger Bands
    bb_bands = BollingerBands(df["Close"])
    bb = df
    bb["bb_h"] = bb_bands.bollinger_hband()
    bb["bb_l"] = bb_bands.bollinger_lband()
    bb = bb[["Close","bb_h","bb_l"]]

    st.write(""" <p style=" color:#FFCC00; font-size: 30px; font-weight:bold"> Bollinger Bands </p>
  """, unsafe_allow_html=True)
    st.line_chart(bb)


    st.write(" ")


    # Resistence Strength Indicator
    
    rsi = RSIIndicator(df["Close"]).rsi()
    st.write(""" <p style=" color:#FFCC00; font-size: 30px; font-weight:bold"> Resistence Strength Indicator </p>
  """, unsafe_allow_html=True)
    st.write(" ")
    st.line_chart(rsi)


    st.write("  ")

    st.write(""" <p style=" color:#FFCC00; font-size: 30px; font-weight:bold"> DOGE-USD Forecast using Facebook Prophet </p>
  """, unsafe_allow_html=True) 
    
    st.write("  ")

    data = df.reset_index()
    period = st.slider("Days of prediction:", 1, 365)
   
    # Predict forecast with Prophet.
    df_train = data[["Date","Close"]]
    df_train = df_train.rename(columns={"Date": "ds", "Close": "y"})

    m = Prophet()
    m.fit(df_train)
    future = m.make_future_dataframe(periods=period)
    forecast = m.predict(future)
    
    st.write(f'Forecast plot for {period} days')
    fig1 = plot_plotly(m, forecast)
    st.plotly_chart(fig1)



    st.write(" ")

    st.write(""" <p style=" color:#FFCC00; font-size: 30px; font-weight:bold"> Latest News </p>
  """, unsafe_allow_html=True)

    st.write(" ")   

    news = GoogleNews()
    news = GoogleNews("en", "d")
    news.search("XRP")
    news.get_page(1)
    result = news.result()
    st.write("1. " + result[1]["title"])
    st.info("1. " + result[1]["link"])
    st.write("2. " + result[2]["title"])
    st.info("2. " + result[2]["link"])
    st.write("3. " + result[3]["title"])
    st.info("3. " + result[3]["link"])
    st.write("4. " + result[4]["title"])
    st.info("4. " + result[4]["link"])
    st.write("5. " + result[5]["title"])
    st.info("5. " + result[5]["link"])

    


  elif option == "ADA-USD":
    df = get_data(option, start_date, end_date)
    
    st.write(""" <p style=" color:#FFCC00; font-size: 30px; font-weight:bold"> Raw Data </p>
  """, unsafe_allow_html=True)
    st.write(" ")

    st.write(df)

    st.write(" ")

    st.write(""" <p style=" color:#FFCC00; font-size: 30px; font-weight:bold"> Close Price </p>
  """, unsafe_allow_html=True)
    st.write("    ")
    st.line_chart(df["Close"])

    st.write(" ")

    # MACD

    st.write(" ")
    macd = MACD(df["Close"]).macd()

    st.write(""" <p style=" color:#FFCC00; font-size: 30px; font-weight:bold"> Moving Average Convergence Divergence </p>
  """, unsafe_allow_html=True)
    st.write(" ")

    st.area_chart(macd)


      # Bollinger Bands
    bb_bands = BollingerBands(df["Close"])
    bb = df
    bb["bb_h"] = bb_bands.bollinger_hband()
    bb["bb_l"] = bb_bands.bollinger_lband()
    bb = bb[["Close","bb_h","bb_l"]]

    st.write(""" <p style=" color:#FFCC00; font-size: 30px; font-weight:bold"> Bollinger Bands </p>
  """, unsafe_allow_html=True)
    st.line_chart(bb)


    st.write(" ")


    # Resistence Strength Indicator
    
    rsi = RSIIndicator(df["Close"]).rsi()
    st.write(""" <p style=" color:#FFCC00; font-size: 30px; font-weight:bold"> Resistence Strength Indicator </p>
  """, unsafe_allow_html=True)
    st.write(" ")
    st.line_chart(rsi)


    st.write("  ")

    st.write(""" <p style=" color:#FFCC00; font-size: 30px; font-weight:bold"> ADA-USD Forecast using Facebook Prophet </p>
  """, unsafe_allow_html=True) 
    
    st.write("  ")

    data = df.reset_index()
    period = st.slider("Days of prediction:", 1, 365)
   
    # Predict forecast with Prophet.
    df_train = data[["Date","Close"]]
    df_train = df_train.rename(columns={"Date": "ds", "Close": "y"})

    m = Prophet()
    m.fit(df_train)
    future = m.make_future_dataframe(periods=period)
    forecast = m.predict(future)
    
    st.write(f'Forecast plot for {period} days')
    fig1 = plot_plotly(m, forecast)
    st.plotly_chart(fig1)



    st.write(" ")

    st.write(""" <p style=" color:#FFCC00; font-size: 30px; font-weight:bold"> Latest News </p>
  """, unsafe_allow_html=True)

    st.write(" ")   

    news = GoogleNews()
    news = GoogleNews("en", "d")
    news.search("cryptocurrency")
    news.get_page(1)
    result = news.result()
    st.write("1. " + result[1]["title"])
    st.info("1. " + result[1]["link"])
    st.write("2. " + result[2]["title"])
    st.info("2. " + result[2]["link"])
    st.write("3. " + result[3]["title"])
    st.info("3. " + result[3]["link"])
    st.write("4. " + result[4]["title"])
    st.info("4. " + result[4]["link"])
    st.write("5. " + result[5]["title"])
    st.info("5. " + result[5]["link"])

    


  elif option == "BNB-USD":
    df = get_data(option, start_date, end_date)
    
    st.write(""" <p style=" color:#FFCC00; font-size: 30px; font-weight:bold"> Raw Data </p>
  """, unsafe_allow_html=True)
    st.write(" ")

    st.write(df)

    st.write(" ")

    st.write(""" <p style=" color:#FFCC00; font-size: 30px; font-weight:bold"> Close Price </p>
  """, unsafe_allow_html=True)
    st.write("    ")
    st.line_chart(df["Close"])

    st.write(" ")

    # MACD

    st.write(" ")
    macd = MACD(df["Close"]).macd()

    st.write(""" <p style=" color:#FFCC00; font-size: 30px; font-weight:bold"> Moving Average Convergence Divergence </p>
  """, unsafe_allow_html=True)
    st.write(" ")

    st.area_chart(macd)


      # Bollinger Bands
    bb_bands = BollingerBands(df["Close"])
    bb = df
    bb["bb_h"] = bb_bands.bollinger_hband()
    bb["bb_l"] = bb_bands.bollinger_lband()
    bb = bb[["Close","bb_h","bb_l"]]

    st.write(""" <p style=" color:#FFCC00; font-size: 30px; font-weight:bold"> Bollinger Bands </p>
  """, unsafe_allow_html=True)
    st.line_chart(bb)


    st.write(" ")


    # Resistence Strength Indicator
    
    rsi = RSIIndicator(df["Close"]).rsi()
    st.write(""" <p style=" color:#FFCC00; font-size: 30px; font-weight:bold"> Resistence Strength Indicator </p>
  """, unsafe_allow_html=True)
    st.write(" ")
    st.line_chart(rsi)


    st.write("  ")

    st.write(""" <p style=" color:#FFCC00; font-size: 30px; font-weight:bold"> BNB-USD Forecast using Facebook Prophet </p>
  """, unsafe_allow_html=True) 
    
    st.write("  ")

    data = df.reset_index()
    period = st.slider("Days of prediction:", 1, 365)
   
    # Predict forecast with Prophet.
    df_train = data[["Date","Close"]]
    df_train = df_train.rename(columns={"Date": "ds", "Close": "y"})

    m = Prophet()
    m.fit(df_train)
    future = m.make_future_dataframe(periods=period)
    forecast = m.predict(future)
    
    st.write(f'Forecast plot for {period} days')
    fig1 = plot_plotly(m, forecast)
    st.plotly_chart(fig1)



    st.write(" ")

    st.write(""" <p style=" color:#FFCC00; font-size: 30px; font-weight:bold"> Latest News </p>
  """, unsafe_allow_html=True)

    st.write(" ")   

    news = GoogleNews()
    news = GoogleNews("en", "d")
    news.search("BNB")
    news.get_page(1)
    result = news.result()
    st.write("1. " + result[1]["title"])
    st.info("1. " + result[1]["link"])
    st.write("2. " + result[2]["title"])
    st.info("2. " + result[2]["link"])
    st.write("3. " + result[3]["title"])
    st.info("3. " + result[3]["link"])
    st.write("4. " + result[4]["title"])
    st.info("4. " + result[4]["link"])
    st.write("5. " + result[5]["title"])
    st.info("5. " + result[5]["link"])

   


  elif option == "LTC-USD":
    df = get_data(option, start_date, end_date)
    
    st.write(""" <p style=" color:#FFCC00; font-size: 30px; font-weight:bold"> Raw Data </p>
  """, unsafe_allow_html=True)
    st.write(" ")

    st.write(df)

    st.write(" ")

    st.write(""" <p style=" color:#FFCC00; font-size: 30px; font-weight:bold"> Close Price </p>
  """, unsafe_allow_html=True)
    st.write("    ")
    st.line_chart(df["Close"])

    st.write(" ")

    # MACD

    st.write(" ")
    macd = MACD(df["Close"]).macd()

    st.write(""" <p style=" color:#FFCC00; font-size: 30px; font-weight:bold"> Moving Average Convergence Divergence </p>
  """, unsafe_allow_html=True)
    st.write(" ")

    st.area_chart(macd)


      # Bollinger Bands
    bb_bands = BollingerBands(df["Close"])
    bb = df
    bb["bb_h"] = bb_bands.bollinger_hband()
    bb["bb_l"] = bb_bands.bollinger_lband()
    bb = bb[["Close","bb_h","bb_l"]]

    st.write(""" <p style=" color:#FFCC00; font-size: 30px; font-weight:bold"> Bollinger Bands </p>
  """, unsafe_allow_html=True)
    st.line_chart(bb)


    st.write(" ")


    # Resistence Strength Indicator
    
    rsi = RSIIndicator(df["Close"]).rsi()
    st.write(""" <p style=" color:#FFCC00; font-size: 30px; font-weight:bold"> Resistence Strength Indicator </p>
  """, unsafe_allow_html=True)
    st.write(" ")
    st.line_chart(rsi)


    st.write("  ")

    st.write(""" <p style=" color:#FFCC00; font-size: 30px; font-weight:bold"> LTC-USD Forecast using Facebook Prophet </p>
  """, unsafe_allow_html=True) 
    
    st.write("  ")

    data = df.reset_index()
    period = st.slider("Days of prediction:", 1, 365)
   
    # Predict forecast with Prophet.
    df_train = data[["Date","Close"]]
    df_train = df_train.rename(columns={"Date": "ds", "Close": "y"})

    m = Prophet()
    m.fit(df_train)
    future = m.make_future_dataframe(periods=period)
    forecast = m.predict(future)
    
    st.write(f'Forecast plot for {period} days')
    fig1 = plot_plotly(m, forecast)
    st.plotly_chart(fig1)



    st.write(" ")

    st.write(""" <p style=" color:#FFCC00; font-size: 30px; font-weight:bold"> Latest News </p>
  """, unsafe_allow_html=True)

    st.write(" ")   

    news = GoogleNews()
    news = GoogleNews("en", "d")
    news.search("Litecoin")
    news.get_page(1)
    result = news.result()
    st.write("1. " + result[1]["title"])
    st.info("1. " + result[1]["link"])
    st.write("2. " + result[2]["title"])
    st.info("2. " + result[2]["link"])
    st.write("3. " + result[3]["title"])
    st.info("3. " + result[3]["link"])
    st.write("4. " + result[4]["title"])
    st.info("4. " + result[4]["link"])
    st.write("5. " + result[5]["title"])
    st.info("5. " + result[5]["link"])


  # Sentiment Analysis

  st.write("  ")

  st.write(""" <p style=" color:#FFCC00; font-size: 30px; font-weight:bold"> How generally users feel about cryptocurrency? </p>
  """, unsafe_allow_html=True) 
    
  st.write("  ")


  df = get_tweets(api_key, api_secret, "#cryptocurrency")
  df["Tweets"] = df["Tweets"].apply(Clean)
  df["Subjectivity"] = df["Tweets"].apply(subjectivity)
  df["Polarity"] = df["Tweets"].apply(polarity)

  #WordCloud
  words = " ".join([twts for twts in df["Tweets"]])
  cloud = WordCloud(random_state = 21, max_font_size = 100).generate(words)
  plt.imshow(cloud, interpolation = "bilinear")
  plt.axis("off")
  st.pyplot()

    
  st.write(" ")

  st.write(""" <p style=" color:#FFCC00; font-size: 30px; font-weight:bold"> Sentiment Bar Plot  </p>
  """, unsafe_allow_html=True)

  st.write("  ") 

  # Get Sentiment tweets
  df["Sentiment"] = df["Polarity"].apply(sentiment)
  df["Sentiment"].value_counts().plot(kind = "bar", figsize = (10,5))
  plt.title("Sentiment Analysis Bar Plot")
  plt.xlabel("Sentiment")
  plt.ylabel("Number of Tweets")
  st.pyplot()
   
