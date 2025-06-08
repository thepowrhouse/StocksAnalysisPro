import numbers
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import ta
import yfinance as yf
from plotly.subplots import make_subplots

# Import the enhanced ML chart analyzer
from chart_analyzer import detect_chart_pattern


@st.cache_data(show_spinner=False)
def analyze_stock(ticker, exchange, analysis_period="1y", include_volume_analysis=True):
    """
    Performs technical analysis for a single stock.
    Results are cached to avoid re-downloading and re-computation on subsequent runs.
    The @st.cache_data decorator handles the caching mechanism.
    """
    # Construct the yfinance ticker
    if ticker.endswith('.NS') or ticker.endswith('.BO'):
        yf_ticker = ticker
    else:
        yf_ticker = f"{ticker}.NS" if exchange == "NSE" else f"{ticker}.BO"

    try:
        # Download data using yfinance
        data = yf.download(yf_ticker, period=analysis_period, progress=False)

        if data.empty:
            return {
                'Ticker': ticker, 'Exchange': exchange, 'Name': 'No Data', 'Current Price': np.nan,
                'RSI': np.nan, 'MACD Signal': 'No Data', 'Price vs MA50': 'No Data', 'Price vs MA200': 'No Data',
                'Volume Trend': 'No Data', 'Support': np.nan, 'Resistance': np.nan, 'ML Pattern': 'No Data',
                'Pattern Confidence': 'No Data', 'Technical Score': 'No Data', 'Recommendation': 'No Data Available',
                'Pattern_Info': {}
            }

        # --- Data processing and validation ---
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.droplevel(1)
        data = data.apply(pd.to_numeric, errors='coerce')

        close_data = data['Close']
        if isinstance(close_data, pd.DataFrame):
            close = close_data.iloc[:, 0].dropna()
        else:
            close = close_data.dropna()

        volume = None
        if include_volume_analysis and 'Volume' in data.columns:
            volume_data = data['Volume']
            if isinstance(volume_data, pd.DataFrame):
                volume = volume_data.iloc[:, 0].dropna()
            else:
                volume = volume_data.dropna()

        close = close.astype(float)
        if volume is not None:
            volume = volume.astype(float)

        if len(close) < 20:
            return {
                'Ticker': ticker, 'Exchange': exchange, 'Name': 'Insufficient Data', 'Current Price': np.nan,
                'RSI': np.nan, 'MACD Signal': 'N/A', 'Price vs MA50': 'N/A', 'Price vs MA200': 'N/A',
                'Volume Trend': 'N/A', 'Support': np.nan, 'Resistance': np.nan, 'ML Pattern': 'N/A',
                'Pattern Confidence': 'N/A', 'Technical Score': 'N/A', 'Recommendation': 'Insufficient Data',
                'Pattern_Info': {}
            }

        # --- Technical Indicator Calculations ---
        rsi = float(ta.momentum.RSIIndicator(close).rsi().iloc[-1]) if not np.isnan(
            ta.momentum.RSIIndicator(close).rsi().iloc[-1]) else np.nan
        macd_obj = ta.trend.MACD(close)
        macd = float(macd_obj.macd().iloc[-1])
        macd_signal = float(macd_obj.macd_signal().iloc[-1])
        macd_histogram = float(macd_obj.macd_diff().iloc[-1])
        ma50 = float(close.rolling(window=50).mean().iloc[-1]) if len(close) >= 50 else np.nan
        ma200 = float(close.rolling(window=200).mean().iloc[-1]) if len(close) >= 200 else np.nan
        current_price = close.iloc[-1]

        # Volume trend analysis
        volume_trend = "N/A"
        if include_volume_analysis and volume is not None and len(volume) >= 60:
            recent_volume = volume.tail(20).mean()
            older_volume = volume.tail(60).head(40).mean()
            if recent_volume > older_volume * 1.2:
                volume_trend = "Increasing"
            elif recent_volume < older_volume * 0.8:
                volume_trend = "Decreasing"
            else:
                volume_trend = "Stable"

        # Get stock info
        try:
            info = yf.Ticker(yf_ticker).info
            name = info.get('longName', info.get('shortName', ticker))
        except:
            name = ticker

        # --- ML Pattern Detection ---
        ml_pattern, ml_recommendation, support, resistance, ml_pattern_info = detect_chart_pattern(close, volume)

        # --- Pattern Confidence Calculation ---
        pattern_confidence = 0.0
        if ml_pattern not in ["No clear pattern", "Error in detection", "No Data"]:
            pattern_confidence = 0.7
            if len(close) > 100: pattern_confidence += 0.1
            if len(close) > 200: pattern_confidence += 0.1
            if include_volume_analysis and volume_trend == "Increasing": pattern_confidence += 0.1
            pattern_confidence = min(pattern_confidence, 1.0)

        # --- Comprehensive Technical Score Calculation ---
        technical_score = 0
        max_score = 0
        if not np.isnan(rsi):
            max_score += 1
            if rsi < 30:
                technical_score += 1
            elif rsi > 70:
                technical_score -= 1
        if not np.isnan(macd) and not np.isnan(macd_signal):
            max_score += 1
            if macd > macd_signal and macd_histogram > 0:
                technical_score += 1
            elif macd < macd_signal and macd_histogram < 0:
                technical_score -= 1
        if not np.isnan(ma50):
            max_score += 1
            if current_price > ma50:
                technical_score += 1
            else:
                technical_score -= 1
        if not np.isnan(ma200):
            max_score += 1
            if current_price > ma200:
                technical_score += 1
            else:
                technical_score -= 1
        if volume_trend != "N/A":
            max_score += 1
            if volume_trend == "Increasing":
                technical_score += 1
            elif volume_trend == "Decreasing":
                technical_score -= 1
        technical_score_pct = (technical_score / max_score) * 100 if max_score > 0 else 0

        # --- Generate Final Recommendation ---
        if ml_recommendation in ["buy", "strong buy"]:
            final_recommendation = "STRONG BUY" if technical_score_pct >= 50 else "BUY"
        elif ml_recommendation in ["sell", "strong sell"]:
            final_recommendation = "STRONG SELL" if technical_score_pct <= -50 else "SELL"
        else:
            if technical_score_pct >= 60:
                final_recommendation = "BUY"
            elif technical_score_pct <= -60:
                final_recommendation = "SELL"
            else:
                final_recommendation = "HOLD"

        # --- Format results into a dictionary ---
        support_val = round(float(support), 2) if pd.notna(support) else np.nan
        resistance_val = round(float(resistance), 2) if pd.notna(resistance) else np.nan

        return {
            'Ticker': ticker, 'Exchange': exchange,
            'Name': str(name)[:30] + "..." if len(str(name)) > 30 else str(name),
            'Current Price': round(float(current_price), 2),
            'RSI': round(float(rsi), 2) if not np.isnan(rsi) else np.nan,
            'MACD Signal': 'Bullish' if not np.isnan(macd) and not np.isnan(
                macd_signal) and macd > macd_signal else 'Bearish' if not np.isnan(macd) and not np.isnan(
                macd_signal) else 'N/A',
            'Price vs MA50': f"{((current_price / ma50 - 1) * 100):+.2f}%" if not np.isnan(ma50) else 'N/A',
            'Price vs MA200': f"{((current_price / ma200 - 1) * 100):+.2f}%" if not np.isnan(ma200) else 'N/A',
            'Volume Trend': volume_trend, 'Support': support_val, 'Resistance': resistance_val,
            'ML Pattern': ml_pattern,
            'Pattern Confidence': f"{pattern_confidence:.2%}" if pattern_confidence > 0 else 'N/A',
            'Technical Score': f"{technical_score_pct:.2f}%", 'Recommendation': final_recommendation,
            'Pattern_Info': ml_pattern_info
        }
    except Exception as e:
        error_message = str(e)
        error_type = type(e).__name__
        if "YFPricesMissingError" in error_type or "possibly delisted" in error_message:
            error_status = "Possibly Delisted"
        elif "HTTPError" in error_type or "HTTP Error 404" in error_message:
            error_status = "Not Found"
        else:
            error_status = "Error"

        return {
            'Ticker': ticker, 'Exchange': exchange, 'Name': error_status, 'Current Price': np.nan, 'RSI': np.nan,
            'MACD Signal': 'Error', 'Price vs MA50': 'Error', 'Price vs MA200': 'Error',
            'Volume Trend': 'Error', 'Support': np.nan, 'Resistance': np.nan, 'ML Pattern': 'Error',
            'Pattern Confidence': 'Error', 'Technical Score': 'Error', 'Recommendation': f"Error: {error_message}",
            'Pattern_Info': {}
        }


def safe_metric(label, value):
    # Only show delta for numeric RSI
    if label == "RSI" and isinstance(value, numbers.Number) and not np.isnan(value):
        delta = "Oversold" if value < 30 else "Overbought" if value > 70 else ""
        st.metric(label, value, delta=delta)
    elif isinstance(value, float) and np.isnan(value):
        # Handle NaN values
        st.metric(label, "N/A")
    else:
        st.metric(label, value)


def add_pattern_overlay(fig, data, pattern_info, pattern_name, row=1, col=1):
    """Enhanced function to add pattern overlay to the chart"""
    if not pattern_info or not isinstance(pattern_info, dict): return
    try:
        start_idx = pattern_info.get('position')
        length = pattern_info.get('template_len')
        template = pattern_info.get('normalized_template')
        segment = pattern_info.get('original_segment_prices')
        confidence = pattern_info.get('confidence', 0)

        if any(v is None for v in
               [start_idx, length, template, segment]) or length <= 0 or start_idx < 0 or start_idx + length > len(
            data): return

        pattern_dates = data['Date'].iloc[start_idx:start_idx + length]
        min_price, max_price = np.min(segment), np.max(segment)
        price_range = max_price - min_price
        scaled_pattern = min_price + (template * price_range) if price_range > 0 else np.full_like(template, min_price)

        fig.add_trace(go.Scatter(x=pattern_dates, y=scaled_pattern, mode='lines+markers', name=f"🔍 {pattern_name}",
                                 line=dict(color='magenta', width=4, dash='dot'),
                                 marker=dict(size=6, color='magenta', symbol='diamond'), opacity=0.8,
                                 hovertemplate=f"<b>{pattern_name}</b><br>Date: %{{x}}<br>Pattern Price: ₹%{{y:.2f}}<br>Confidence: {confidence:.1%}<br><extra></extra>"),
                      row=row, col=col)
        fig.add_trace(
            go.Scatter(x=[pattern_dates.iloc[0], pattern_dates.iloc[-1]], y=[scaled_pattern[0], scaled_pattern[-1]],
                       mode='markers', name="Pattern Boundaries",
                       marker=dict(size=12, color=['green', 'red'], symbol=['triangle-up', 'triangle-down'],
                                   line=dict(width=2, color='white')), showlegend=False,
                       hovertemplate="Pattern %{text}<br>Date: %{x}<br>Price: ₹%{y:.2f}<br><extra></extra>",
                       text=['Start', 'End']), row=row, col=col)
        fig.add_vrect(x0=pattern_dates.iloc[0], x1=pattern_dates.iloc[-1], fillcolor="purple", opacity=0.1,
                      layer="below", line_width=0, row=row, col=col)
        mid_date, mid_price = pattern_dates.iloc[len(pattern_dates) // 2], scaled_pattern[len(scaled_pattern) // 2]
        fig.add_annotation(x=mid_date, y=max_price + (max_price - min_price) * 0.1,
                           text=f"<b>{pattern_name}</b><br>Confidence: {confidence:.1%}", showarrow=True, arrowhead=2,
                           arrowsize=1, arrowwidth=2, arrowcolor="purple", ax=0, ay=-30,
                           bgcolor="rgba(240, 240, 240, 0.8)", bordercolor="purple", borderwidth=2,
                           font=dict(color="black", size=10), row=row, col=col)
    except Exception as e:
        st.warning(f"Could not overlay pattern: {str(e)}")


def add_support_resistance_overlay(fig, support_val, resistance_val, data, row=1, col=1):
    """Enhanced function to add support and resistance lines"""
    if pd.notna(support_val):
        fig.add_hline(y=support_val, line_dash="dash", line_color="green", line_width=2,
                      annotation_text=f"Support: ₹{support_val:.2f}", annotation_position="bottom right",
                      annotation=dict(bgcolor="rgba(240, 240, 240, 0.8)", bordercolor="green",
                                      font=dict(color="black")), row=row, col=col)
        fig.add_hrect(y0=support_val * 0.98, y1=support_val * 1.02, fillcolor="green", opacity=0.1, layer="below",
                      line_width=0, row=row, col=col)
    if pd.notna(resistance_val):
        fig.add_hline(y=resistance_val, line_dash="dash", line_color="red", line_width=2,
                      annotation_text=f"Resistance: ₹{resistance_val:.2f}", annotation_position="top right",
                      annotation=dict(bgcolor="rgba(240, 240, 240, 0.8)", bordercolor="red", font=dict(color="black")),
                      row=row, col=col)


# --- Main Streamlit App UI ---
st.set_page_config(page_title="Stock Analyzer", layout="wide")
st.title("🤖 ML-Enhanced Stock Analyzer (NSE/BSE)")

# Custom CSS for better styling
st.markdown(
    """
    <style>
        .main .block-container { padding-top: 2rem; padding-bottom: 1rem; max-width: 95% !important; }
        .stMetric { background-color: #f0f2f6; padding: 1rem; border-radius: 0.5rem; border-left: 4px solid #1f77b4; color: #333333 !important; }
        .stMetric label, .stMetric div { color: #333333 !important; }
        .pattern-card { background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); padding: 1rem; border-radius: 0.5rem; color: white; margin: 0.5rem 0; }
        .recommendation-buy { background-color: #28a745; color: white; padding: 0.5rem; border-radius: 0.25rem; font-weight: bold; }
        .recommendation-sell { background-color: #dc3545; color: white; padding: 0.5rem; border-radius: 0.25rem; font-weight: bold; }
        .recommendation-hold { background-color: #ffc107; color: black; padding: 0.5rem; border-radius: 0.25rem; font-weight: bold; }
        .pattern-highlight { background: linear-gradient(45deg, #ff6b6b, #4ecdc4); padding: 0.5rem; border-radius: 0.5rem; color: white; text-align: center; margin: 1rem 0; font-weight: bold; }
    </style>
    """,
    unsafe_allow_html=True,
)

# Set default values for analysis
analysis_period = "1y"
include_volume_analysis = True

# File upload section
st.header("📁 Upload Portfolio")
uploaded_file = st.file_uploader(
    "Upload your CSV file containing stock tickers",
    type=["csv"],
    help="CSV must contain 'Ticker' and 'Exchange' columns (Exchange: NSE or BSE)"
)

if uploaded_file is not None:
    try:
        df_portfolio = pd.read_csv(uploaded_file)

        if not all(col in df_portfolio.columns for col in ['Ticker', 'Exchange']):
            st.error("❌ CSV must have ['Ticker', 'Exchange'] columns")
            st.stop()

        st.header("🔍 Technical Analysis Results")
        progress_bar = st.progress(0)
        status_text = st.empty()

        results = []
        total_stocks = len(df_portfolio)

        for idx, row in df_portfolio.iterrows():
            ticker = str(row['Ticker']).strip().upper()
            exchange = str(row.get('Exchange', 'NSE')).strip().upper()
            if not ticker: continue

            # --- This is the key change ---
            # Call the cached analysis function. Streamlit will only run this
            # if the inputs have changed, otherwise it returns the cached result.
            result = analyze_stock(ticker, exchange, analysis_period, include_volume_analysis)
            results.append(result)

            progress = (idx + 1) / total_stocks
            status_text.text(f"Analyzing {ticker} ({idx + 1}/{total_stocks})...")
            progress_bar.progress(progress)

        # After processing all stocks, display results
        if results:
            df_results = pd.DataFrame(results)
            st.subheader("📈 Analysis Summary")
            st.dataframe(df_results.drop(columns=['Pattern_Info'], errors='ignore').style.format(precision=2).applymap(
                lambda x: "background-color: #28a745; color: white" if x == "STRONG BUY"
                else "background-color: #dc3545; color: white" if x == "STRONG SELL"
                else "background-color: #ffc107" if x in ["BUY", "SELL"]
                else "",
                subset=['Recommendation']
            ), use_container_width=True)

            # Portfolio summary metrics
            st.subheader("📊 Portfolio Summary")
            col1, col2, col3, col4 = st.columns(4)
            buy_rec = df_results[df_results['Recommendation'].str.contains('BUY', na=False)].shape[0]
            sell_rec = df_results[df_results['Recommendation'].str.contains('SELL', na=False)].shape[0]
            col1.metric("Stocks Analyzed", total_stocks)
            col2.metric("Buy Recommendations", buy_rec)
            col3.metric("Sell Recommendations", sell_rec)
            col4.metric("Hold Recommendations", total_stocks - buy_rec - sell_rec)

            # Detailed view for each stock - now with dropdown
            st.subheader("📊 Detailed Stock Analysis")
            stock_names = df_results['Name'].tolist()
            selected_stock_name = st.selectbox("Select a stock for detailed analysis:", stock_names)

            if selected_stock_name:
                selected_row = df_results[df_results['Name'] == selected_stock_name].iloc[0]

                pattern_info = selected_row.get('Pattern_Info', {})
                confidence = pattern_info.get('confidence', 0) if pattern_info else 0

                if selected_row['ML Pattern'] != 'No Data' and selected_row['ML Pattern'] != 'Error':
                    st.markdown(f"""
                        <div class="pattern-highlight">
                            🔍 <strong>Detected Pattern: {selected_row['ML Pattern']}</strong><br>
                            📊 Confidence Score: {confidence:.1%}<br>
                            💡 Recommendation: <span class="recommendation-{selected_row['Recommendation'].lower().replace('strong ', '')}">{selected_row['Recommendation']}</span>
                        </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown(f"""<div class="pattern-card"> ... </div>""", unsafe_allow_html=True)

                if pd.notna(selected_row['Current Price']):
                    chart_yf_ticker = f"{selected_row['Ticker']}.{'NS' if selected_row['Exchange'] == 'NSE' else 'BO'}"

                    # Re-download data for plotting. This is fast as it's just one stock.
                    # Alternatively, this could also be cached.
                    data = yf.download(chart_yf_ticker, period=analysis_period, progress=False)

                    if isinstance(data.columns, pd.MultiIndex):
                        data.columns = data.columns.droplevel(1)

                    data.reset_index(inplace=True)
                    if 'index' in data.columns:
                        data = data.rename(columns={'index': 'Date'})

                    required_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
                    data[required_cols] = data[required_cols].apply(pd.to_numeric, errors='coerce').dropna()

                    if not data.empty and 'Date' in data.columns:
                        fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.05,
                                            row_heights=[0.75, 0.25])
                        fig.add_trace(
                            go.Candlestick(x=data['Date'], open=data['Open'], high=data['High'], low=data['Low'],
                                           close=data['Close'], name='Price'), row=1, col=1)
                        fig.add_trace(go.Bar(x=data['Date'], y=data['Volume'], name='Volume',
                                             marker_color='rgba(0, 102, 204, 0.6)'), row=2, col=1)

                        add_support_resistance_overlay(fig, selected_row['Support'], selected_row['Resistance'], data,
                                                       row=1, col=1)

                        if pattern_info and selected_row['ML Pattern'] not in ['No Data', 'Error', 'N/A',
                                                                               'No clear pattern']:
                            add_pattern_overlay(fig, data, pattern_info, selected_row['ML Pattern'], row=1, col=1)

                        if len(data) >= 50:
                            fig.add_trace(
                                go.Scatter(x=data['Date'], y=data['Close'].rolling(window=50).mean(), mode='lines',
                                           name='MA50', line=dict(color='orange', width=1)), row=1, col=1)
                        if len(data) >= 200:
                            fig.add_trace(
                                go.Scatter(x=data['Date'], y=data['Close'].rolling(window=200).mean(), mode='lines',
                                           name='MA200', line=dict(color='blue', width=1)), row=1, col=1)

                        fig.update_layout(
                            title={'text': f"{selected_row['Name']} ({selected_row['Ticker']}) - Technical Analysis",
                                   'x': 0.5}, height=700, xaxis_rangeslider_visible=False)
                        st.plotly_chart(fig, use_container_width=True)

                        # Technical indicators summary
                        st.subheader("📊 Technical Indicators Summary")
                        col1, col2, col3, col4 = st.columns(4)
                        with col1:
                            safe_metric("Current Price", f"₹{selected_row['Current Price']}")
                            safe_metric("RSI", selected_row['RSI'])
                        with col2:
                            safe_metric("MACD Signal", selected_row['MACD Signal'])
                            safe_metric("Volume Trend", selected_row['Volume Trend'])
                        with col3:
                            safe_metric("Support Level",
                                        f"₹{selected_row['Support']}" if pd.notna(selected_row['Support']) else "N/A")
                            safe_metric("Price vs MA50", selected_row['Price vs MA50'])
                        with col4:
                            safe_metric("Resistance Level", f"₹{selected_row['Resistance']}" if pd.notna(
                                selected_row['Resistance']) else "N/A")
                            safe_metric("Price vs MA200", selected_row['Price vs MA200'])
                else:
                    st.warning("No data available to plot a chart because initial analysis failed.")
    except Exception as e:
        st.error(f"Error processing file: {e}")
else:
    st.info("Please upload a portfolio CSV file to begin analysis")
