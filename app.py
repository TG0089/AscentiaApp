import streamlit as st
import yfinance as yf
import pandas as pd
import gspread
from oauth2client.service_account import ServiceAccountCredentials
import streamlit_authenticator as stauth
from datetime import datetime
import numpy as np
import yaml

# ---------------------
# Authentication
# ---------------------
with open("credentials.yaml", "r") as file:
    config = yaml.safe_load(file)

authenticator = stauth.Authenticate(
    config['credentials'],
    cookie_name=config['cookie']['name'],
    key=config['cookie']['key'],
    cookie_expiry_days=config['cookie']['expiry_days']
)

# -----------------------------
# Streamlit Authenticator Login
# -----------------------------
# Call login and store result
login_result = authenticator.login("sidebar", key="login_form_1")

# Initialize default values
name = ""
authentication_status = None
username = ""

# Handle login result
if login_result is not None and isinstance(login_result, tuple):
    name, authentication_status, username = login_result
elif hasattr(authenticator, "authentication_status"):
    authentication_status = authenticator.authentication_status
    username = getattr(authenticator, "username", "")
    name = username

# ---------------------
# Authentication States
# ---------------------
if authentication_status:
    st.sidebar.success(f"Welcome, {name}!")
    authenticator.logout("Logout", "sidebar")

    # ---------------------
    # Google Sheets
    # ---------------------
    try:
        scope = ["https://spreadsheets.google.com/feeds", "https://www.googleapis.com/auth/drive"]
        creds = ServiceAccountCredentials.from_json_keyfile_name("service_account.json", scope)
        client = gspread.authorize(creds)
        sheet = client.open("Ascentia_Watchlists").sheet1
    except Exception as e:
        st.error(f"Error connecting to Google Sheets: {e}")
        sheet = None

    # ---------------------
    # Stock Search
    # ---------------------
    st.title("Ascentia - ASX Investment Analyzer (Full 10-Indicator)")
    ticker_input = st.text_input("Enter ASX Ticker (e.g., BHP.AX)", "BHP.AX").upper()

    if st.button("Analyze"):
        try:
            stock = yf.Ticker(ticker_input)
            hist = stock.history(period="1y")
            last_price = hist['Close'].iloc[-1]
            info = stock.info

            pe = info.get('trailingPE', np.nan)
            div_yield = info.get('dividendYield', np.nan)
            roe = info.get('returnOnEquity', np.nan)
            de = info.get('debtToEquity', np.nan)
            closes = hist['Close'].values
            volumes = hist['Volume'].values

            # --- Compute 10 indicators ---
            breakdown = []
            total_score = 0
            max_per_indicator = 9

            # 1) P/E
            if not np.isnan(pe):
                s = 9 if pe < 10 else 7 if pe < 15 else 5 if pe < 20 else 3 if pe < 30 else 1
            else:
                s = 5
            breakdown.append({'indicator':'P/E','score':s,'reason':f'P/E={pe}'})
            total_score += s

            # 2) Earnings growth (price % change 1yr)
            if len(closes) >= 252:
                pct = (closes[-1]-closes[0])/closes[0]*100
                s = 9 if pct>50 else 7 if pct>20 else 5 if pct>0 else 3 if pct>-20 else 1
            else:
                pct = 0
                s = 5
            breakdown.append({'indicator':'Earnings Growth','score':s,'reason':f'Price change ≈ {pct:.1f}%'})
            total_score += s

            # 3) ROE
            if not np.isnan(roe):
                roe_pct = roe*100 if roe<1 else roe
                s = 9 if roe_pct>20 else 7 if roe_pct>10 else 4 if roe_pct>5 else 2
            else:
                roe_pct = np.nan
                s = 5
            breakdown.append({'indicator':'ROE','score':s,'reason':f'{roe_pct:.1f}%' if not np.isnan(roe_pct) else 'N/A'})
            total_score += s

            # 4) Debt-to-Equity
            if not np.isnan(de):
                s = 9 if de<20 else 7 if de<50 else 4 if de<100 else 1
            else:
                s = 5
            breakdown.append({'indicator':'Debt/Equity','score':s,'reason':f'{de}'})
            total_score += s

            # 5) Dividend Yield
            if not np.isnan(div_yield):
                dy_pct = div_yield*100
                s = 9 if dy_pct>5 else 7 if dy_pct>3 else 5 if dy_pct>1.5 else 2
            else:
                dy_pct = np.nan
                s = 5
            breakdown.append({'indicator':'Dividend Yield','score':s,'reason':f'{dy_pct:.2f}%' if not np.isnan(dy_pct) else 'N/A'})
            total_score += s

            # 6) MA Crossover (50 vs 200)
            def ma(arr, n): return pd.Series(arr).rolling(n).mean().values
            ma_short, ma_long = ma(closes,50), ma(closes,200)
            s = 8 if ma_short[-1]>ma_long[-1] else 2
            breakdown.append({'indicator':'MA50/MA200','score':s,'reason':f'{ma_short[-1]:.2f} vs {ma_long[-1]:.2f}'})
            total_score += s

            # 7) RSI 14-day
            delta = np.diff(closes)
            up, down = delta.clip(min=0), -delta.clip(max=0)
            roll_up = pd.Series(up).rolling(14).mean().iloc[-1]
            roll_down = pd.Series(down).rolling(14).mean().iloc[-1]
            rs = roll_up / (roll_down+1e-9)
            rsi = 100 - 100/(1+rs)
            s = 9 if rsi<30 else 7 if rsi<45 else 5 if rsi<55 else 3 if rsi<70 else 1
            breakdown.append({'indicator':'RSI','score':s,'reason':f'{rsi:.1f}'})
            total_score += s

            # 8) Volume Trend
            vol_avg = pd.Series(volumes).rolling(30).mean().iloc[-1]
            vol_now = volumes[-1]
            ratio = vol_now / (vol_avg+1e-9)
            s = 8 if ratio>1.5 else 5 if ratio>0.7 else 2
            breakdown.append({'indicator':'Volume Trend','score':s,'reason':f'{ratio:.2f}'})
            total_score += s

            # 9) 12m Momentum
            pct12 = (closes[-1]-closes[0])/closes[0]*100 if len(closes)>=252 else 0
            s = 9 if pct12>50 else 7 if pct12>20 else 5 if pct12>0 else 3 if pct12>-20 else 1
            breakdown.append({'indicator':'12m Momentum','score':s,'reason':f'{pct12:.1f}%'})
            total_score += s

            # 10) Analyst Rec (fallback neutral)
            s = 5
            breakdown.append({'indicator':'Analyst Rec','score':s,'reason':'Neutral fallback'})
            total_score += s

            final_score = round(total_score / (max_per_indicator*10) * 100)

            st.subheader(f"{ticker_input} - {info.get('longName','N/A')}")
            st.write(f"**Last Price:** ${last_price:.2f}")
            st.write(f"**Investment Score:** {final_score}/100")
            st.write("**Recommendation:**", "STRONG BUY 🟢" if final_score>=75 else "NEUTRAL 🟡" if final_score>=50 else "AVOID 🔴")

            st.subheader("Indicator Breakdown")
            for b in breakdown:
                st.write(f"{b['indicator']}: {b['score']}/9 → {b['reason']}")

            if sheet is not None and st.button("Add to Watchlist"):
                sheet.append_row([username, ticker_input, str(datetime.now().date())])
                st.success(f"{ticker_input} added to your watchlist!")

        except Exception as e:
            st.error(f"Error fetching data: {e}")

    # ---------------------
    # User Watchlist
    # ---------------------
    if sheet is not None:
        user_watchlist = pd.DataFrame(sheet.get_all_records())
        user_watchlist = user_watchlist[user_watchlist['username']==username]
        st.subheader("My Watchlist")
        if not user_watchlist.empty:
            st.table(user_watchlist[['ticker','date_added']])
        else:
            st.info("Your watchlist is empty.")

elif authentication_status is False:
    st.error("Username/password is incorrect")
elif authentication_status is None:
    st.warning("Please enter your username and password")
   

























