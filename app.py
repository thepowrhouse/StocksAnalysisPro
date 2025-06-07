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
    """
    Enhanced function to add pattern overlay to the chart
    """
    if not pattern_info or not isinstance(pattern_info, dict):
        return

    try:
        # Get pattern information
        start_idx = pattern_info.get('position')
        length = pattern_info.get('template_len')
        template = pattern_info.get('normalized_template')
        segment = pattern_info.get('original_segment_prices')
        confidence = pattern_info.get('confidence', 0)

        # Validate all required data is available
        if any(v is None for v in [start_idx, length, template, segment]):
            return

        if length <= 0 or start_idx < 0 or start_idx + length > len(data):
            return

        # Get the dates for the pattern segment
        pattern_dates = data['Date'].iloc[start_idx:start_idx + length]

        # Scale the normalized template to match actual price range
        min_price, max_price = np.min(segment), np.max(segment)
        price_range = max_price - min_price

        if price_range > 0:
            scaled_pattern = min_price + (template * price_range)
        else:
            scaled_pattern = np.full_like(template, min_price)

        # Add pattern overlay trace
        fig.add_trace(go.Scatter(
            x=pattern_dates,
            y=scaled_pattern,
            mode='lines+markers',
            name=f"🔍 {pattern_name}",
            line=dict(
                color='magenta',
                width=4,
                dash='dot'
            ),
            marker=dict(
                size=6,
                color='magenta',
                symbol='diamond'
            ),
            opacity=0.8,
            hovertemplate=f"<b>{pattern_name}</b><br>" +
                          "Date: %{x}<br>" +
                          "Pattern Price: ₹%{y:.2f}<br>" +
                          f"Confidence: {confidence:.1%}<br>" +
                          "<extra></extra>"
        ), row=row, col=col)

        # Add pattern start and end markers
        fig.add_trace(go.Scatter(
            x=[pattern_dates.iloc[0], pattern_dates.iloc[-1]],
            y=[scaled_pattern[0], scaled_pattern[-1]],
            mode='markers',
            name="Pattern Boundaries",
            marker=dict(
                size=12,
                color=['green', 'red'],
                symbol=['triangle-up', 'triangle-down'],
                line=dict(width=2, color='white')
            ),
            showlegend=False,
            hovertemplate="Pattern %{text}<br>" +
                          "Date: %{x}<br>" +
                          "Price: ₹%{y:.2f}<br>" +
                          "<extra></extra>",
            text=['Start', 'End']
        ), row=row, col=col)

        # Add shaded region to highlight pattern area
        fig.add_vrect(
            x0=pattern_dates.iloc[0],
            x1=pattern_dates.iloc[-1],
            fillcolor="purple",
            opacity=0.1,
            layer="below",
            line_width=0,
            row=row, col=col
        )

        # Add annotation for pattern
        mid_date = pattern_dates.iloc[len(pattern_dates) // 2]
        mid_price = scaled_pattern[len(scaled_pattern) // 2]

        fig.add_annotation(
            x=mid_date,
            y=max_price + (max_price - min_price) * 0.1,
            text=f"<b>{pattern_name}</b><br>Confidence: {confidence:.1%}",
            showarrow=True,
            arrowhead=2,
            arrowsize=1,
            arrowwidth=2,
            arrowcolor="purple",
            ax=0,
            ay=-30,
            bgcolor="rgba(240, 240, 240, 0.8)",
            bordercolor="purple",
            borderwidth=2,
            font=dict(color="black", size=10),
            row=row, col=col
        )

    except Exception as e:
        st.warning(f"Could not overlay pattern: {str(e)}")


def add_support_resistance_overlay(fig, support_val, resistance_val, data, row=1, col=1):
    """
    Enhanced function to add support and resistance lines
    """
    if pd.notna(support_val):
        fig.add_hline(
            y=support_val,
            line_dash="dash",
            line_color="green",
            line_width=2,
            annotation_text=f"Support: ₹{support_val:.2f}",
            annotation_position="bottom right",
            annotation=dict(
                bgcolor="rgba(240, 240, 240, 0.8)",
                bordercolor="green",
                font=dict(color="black")
            ),
            row=row, col=col
        )

        # Add support zone (slightly below support line)
        fig.add_hrect(
            y0=support_val * 0.98,
            y1=support_val * 1.02,
            fillcolor="green",
            opacity=0.1,
            layer="below",
            line_width=0,
            row=row, col=col
        )

    if pd.notna(resistance_val):
        fig.add_hline(
            y=resistance_val,
            line_dash="dash",
            line_color="red",
            line_width=2,
            annotation_text=f"Resistance: ₹{resistance_val:.2f}",
            annotation_position="top right",
            annotation=dict(
                bgcolor="rgba(240, 240, 240, 0.8)",
                bordercolor="red",
                font=dict(color="black")
            ),
            row=row, col=col
        )

        # Add resistance zone (slightly above resistance line)
        fig.add_hrect(
            y0=resistance_val * 0.98,
            y1=resistance_val * 1.02,
            fillcolor="red",
            opacity=0.1,
            layer="below",
            line_width=0,
            row=row, col=col
        )


st.set_page_config(page_title="Stock Analyzer", layout="wide")
st.title("🤖 ML-Enhanced Stock Analyzer (NSE/BSE)")

# Custom CSS for better styling
st.markdown(
    """
    <style>
        .main .block-container {
            padding-top: 2rem;
            padding-bottom: 1rem;
            max-width: 95% !important;
        }
        .stMetric {
            background-color: #f0f2f6;
            padding: 1rem;
            border-radius: 0.5rem;
            border-left: 4px solid #1f77b4;
            color: #333333 !important;
        }
        .stMetric label, .stMetric div {
            color: #333333 !important;
        }
        .pattern-card {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            padding: 1rem;
            border-radius: 0.5rem;
            color: white;
            margin: 0.5rem 0;
        }
        .recommendation-buy {
            background-color: #28a745;
            color: white;
            padding: 0.5rem;
            border-radius: 0.25rem;
            font-weight: bold;
        }
        .recommendation-sell {
            background-color: #dc3545;
            color: white;
            padding: 0.5rem;
            border-radius: 0.25rem;
            font-weight: bold;
        }
        .recommendation-hold {
            background-color: #ffc107;
            color: black;
            padding: 0.5rem;
            border-radius: 0.25rem;
            font-weight: bold;
        }
        .pattern-highlight {
            background: linear-gradient(45deg, #ff6b6b, #4ecdc4);
            padding: 0.5rem;
            border-radius: 0.5rem;
            color: white;
            text-align: center;
            margin: 1rem 0;
            font-weight: bold;
        }
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

        # Validate required columns
        required_columns = ['Ticker', 'Exchange']
        if not all(col in df_portfolio.columns for col in required_columns):
            st.error(f"❌ CSV must have {required_columns} columns")
            st.stop()

        # Analysis section
        st.header("🔍 Technical Analysis Results")

        # Progress bar
        progress_bar = st.progress(0)
        status_text = st.empty()

        results = []
        total_stocks = len(df_portfolio)

        for idx, row in df_portfolio.iterrows():
            # Improved: Retrieve ticker and exchange directly from the current row
            ticker = str(row['Ticker']).strip().upper()
            exchange = str(row.get('Exchange', 'NSE')).strip().upper()

            # Improved: Skip row if the ticker is empty to prevent errors
            if not ticker:
                continue

            # Update progress
            progress = (idx + 1) / total_stocks
            status_text.text(f"Analyzing {ticker} ({idx + 1}/{total_stocks})")
            progress_bar.progress(progress)

            # Improved: Construct the yfinance ticker
            if ticker.endswith('.NS') or ticker.endswith('.BO'):
                yf_ticker = ticker
            else:
                yf_ticker = f"{ticker}.NS" if exchange == "NSE" else f"{ticker}.BO"

            try:
                # Download data
                data = yf.download(yf_ticker, period=analysis_period, progress=False)

                if data.empty:
                    results.append({
                        'Ticker': ticker,
                        'Exchange': exchange,
                        'Name': 'No Data',
                        'Current Price': np.nan,
                        'RSI': np.nan,
                        'MACD Signal': 'No Data',
                        'Price vs MA50': 'No Data',
                        'Price vs MA200': 'No Data',
                        'Volume Trend': 'No Data',
                        'Support': np.nan,
                        'Resistance': np.nan,
                        'ML Pattern': 'No Data',
                        'Pattern Confidence': 'No Data',
                        'Technical Score': 'No Data',
                        'Recommendation': 'No Data Available',
                        'Pattern_Info': {}  # Add empty dict for pattern info
                    })
                    continue

                # Handle MultiIndex columns if present
                if isinstance(data.columns, pd.MultiIndex):
                    data.columns = data.columns.droplevel(1)

                # Convert all columns to numeric
                data = data.apply(pd.to_numeric, errors='coerce')

                # Verify we have numeric data
                try:
                    # Check if data['Close'] is a DataFrame or Series
                    close_data = data['Close']
                    if isinstance(close_data, pd.DataFrame):
                        if not np.issubdtype(close_data.dtypes.iloc[0], np.number):
                            raise ValueError("Close data is not numeric")
                    else:
                        if not np.issubdtype(close_data.dtype, np.number):
                            raise ValueError("Close data is not numeric")
                except Exception as e:
                    results.append({
                        'Ticker': ticker,
                        'Exchange': exchange,
                        'Name': 'Invalid Data',
                        'Current Price': np.nan,
                        'RSI': np.nan,
                        'MACD Signal': 'N/A',
                        'Price vs MA50': 'N/A',
                        'Price vs MA200': 'N/A',
                        'Volume Trend': 'N/A',
                        'Support': np.nan,
                        'Resistance': np.nan,
                        'ML Pattern': 'N/A',
                        'Pattern Confidence': 'N/A',
                        'Technical Score': 'N/A',
                        'Recommendation': 'Invalid Data',
                        'Pattern_Info': {}  # Add empty dict for pattern info
                    })
                    continue

                # Extract price and volume data
                close_data = data['Close']
                # Handle if close_data is a DataFrame instead of a Series
                if isinstance(close_data, pd.DataFrame):
                    close = close_data.iloc[:, 0].dropna()
                else:
                    close = close_data.dropna()

                if include_volume_analysis:
                    volume_data = data['Volume']
                    # Handle if volume_data is a DataFrame instead of a Series
                    if isinstance(volume_data, pd.DataFrame):
                        volume = volume_data.iloc[:, 0].dropna()
                    else:
                        volume = volume_data.dropna()
                else:
                    volume = None

                # Convert to float to prevent type errors
                close = close.astype(float)
                volume = volume.astype(float) if include_volume_analysis and volume is not None else None

                if len(close) < 20:
                    results.append({
                        'Ticker': ticker,
                        'Exchange': exchange,
                        'Name': 'Insufficient Data',
                        'Current Price': np.nan,
                        'RSI': np.nan,
                        'MACD Signal': 'N/A',
                        'Price vs MA50': 'N/A',
                        'Price vs MA200': 'N/A',
                        'Volume Trend': 'N/A',
                        'Support': np.nan,
                        'Resistance': np.nan,
                        'ML Pattern': 'N/A',
                        'Pattern Confidence': 'N/A',
                        'Technical Score': 'N/A',
                        'Recommendation': 'Insufficient Data',
                        'Pattern_Info': {}  # Add empty dict for pattern info
                    })
                    continue

                # RSI
                rsi = float(ta.momentum.RSIIndicator(close).rsi().iloc[-1]) if not np.isnan(
                    ta.momentum.RSIIndicator(close).rsi().iloc[-1]) else np.nan

                # MACD
                macd_obj = ta.trend.MACD(close)
                macd = float(macd_obj.macd().iloc[-1])
                macd_signal = float(macd_obj.macd_signal().iloc[-1])
                macd_histogram = float(macd_obj.macd_diff().iloc[-1])

                # Moving averages
                ma50 = float(close.rolling(window=50).mean().iloc[-1]) if len(close) >= 50 else np.nan
                ma200 = float(close.rolling(window=200).mean().iloc[-1]) if len(close) >= 200 else np.nan

                current_price = close.iloc[-1]

                # Volume analysis
                volume_trend = "N/A"
                if include_volume_analysis and volume is not None and len(volume) >= 20:
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

                # ML Pattern Detection
                # Now detect_chart_pattern returns pattern_info as the 5th element
                ml_pattern, ml_recommendation, support, resistance, ml_pattern_info = detect_chart_pattern(close,
                                                                                                           volume)

                # Calculate pattern confidence based on multiple factors
                pattern_confidence = 0.0
                if ml_pattern != "No clear pattern" and ml_pattern != "Error in detection" and ml_pattern != "No Data":
                    # Base confidence from pattern detection
                    pattern_confidence = 0.7

                    # Adjust based on data quality
                    if len(close) > 100:
                        pattern_confidence += 0.1
                    if len(close) > 200:
                        pattern_confidence += 0.1

                    # Adjust based on volume confirmation
                    if include_volume_analysis and volume_trend == "Increasing":
                        pattern_confidence += 0.1

                pattern_confidence = min(pattern_confidence, 1.0)

                # Calculate comprehensive technical score
                technical_score = 0
                max_score = 0

                # RSI scoring
                if not np.isnan(rsi):
                    max_score += 1
                    if rsi < 30:
                        technical_score += 1  # Oversold - bullish
                    elif rsi > 70:
                        technical_score -= 1  # Overbought - bearish

                # MACD scoring
                if not np.isnan(macd) and not np.isnan(macd_signal):
                    max_score += 1
                    if macd > macd_signal and macd_histogram > 0:
                        technical_score += 1  # Bullish momentum
                    elif macd < macd_signal and macd_histogram < 0:
                        technical_score -= 1  # Bearish momentum

                # Moving average scoring
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

                # Volume scoring
                if volume_trend == "Increasing":
                    max_score += 1
                    technical_score += 1
                elif volume_trend == "Decreasing":
                    max_score += 1
                    technical_score -= 1

                # Normalize score to percentage
                if max_score > 0:
                    technical_score_pct = (technical_score / max_score) * 100
                else:
                    technical_score_pct = 0

                # Generate final recommendation (ML-first logic)
                if ml_recommendation in ["buy", "strong buy"]:
                    if technical_score_pct >= 50:
                        final_recommendation = "STRONG BUY"
                    else:
                        final_recommendation = "BUY"
                elif ml_recommendation in ["sell", "strong sell"]:
                    if technical_score_pct <= -50:
                        final_recommendation = "STRONG SELL"
                    else:
                        final_recommendation = "SELL"
                else:  # ml_recommendation is "hold" or something else
                    if technical_score_pct >= 60:
                        final_recommendation = "BUY"
                    elif technical_score_pct <= -60:
                        final_recommendation = "SELL"
                    else:
                        final_recommendation = "HOLD"

                # Append results
                # Ensure support and resistance are numeric before operations
                try:
                    support_val = round(float(support), 2) if support is not None and not np.isnan(support) else np.nan
                except (ValueError, TypeError):
                    support_val = np.nan

                try:
                    resistance_val = round(float(resistance), 2) if resistance is not None and not np.isnan(
                        resistance) else np.nan
                except (ValueError, TypeError):
                    resistance_val = np.nan

                result = {
                    'Ticker': ticker,
                    'Exchange': exchange,
                    'Name': str(name)[:30] + "..." if len(str(name)) > 30 else str(name),
                    'Current Price': round(float(current_price), 2),
                    'RSI': round(float(rsi), 2) if not np.isnan(rsi) else np.nan,
                    'MACD Signal': 'Bullish' if (not np.isnan(macd) and not np.isnan(
                        macd_signal) and macd > macd_signal) else 'Bearish' if (
                            not np.isnan(macd) and not np.isnan(macd_signal)) else 'N/A',
                    'Price vs MA50': f"{((current_price / ma50 - 1) * 100):+.2f}%" if not np.isnan(ma50) else 'N/A',
                    'Price vs MA200': f"{((current_price / ma200 - 1) * 100):+.2f}%" if not np.isnan(
                        ma200) else 'N/A',
                    'Volume Trend': volume_trend,
                    'Support': support_val,
                    'Resistance': resistance_val,
                    'ML Pattern': ml_pattern,
                    'Pattern Confidence': f"{pattern_confidence:.2%}" if pattern_confidence > 0 else 'N/A',
                    'Technical Score': f"{technical_score_pct:.2f}%",
                    'Recommendation': final_recommendation,
                    'Pattern_Info': ml_pattern_info  # Store pattern info for plotting
                }
                results.append(result)
            except Exception as e:
                error_message = str(e)
                error_type = type(e).__name__

                # Handle specific error cases
                if "YFPricesMissingError" in error_type or "possibly delisted" in error_message:
                    error_status = "Possibly Delisted"
                elif "HTTPError" in error_type or "HTTP Error 404" in error_message:
                    error_status = "Not Found"
                else:
                    error_status = "Error"

                results.append({
                    'Ticker': ticker,
                    'Exchange': exchange,
                    'Name': error_status,
                    'Current Price': np.nan,
                    'RSI': np.nan,
                    'MACD Signal': 'Error',
                    'Price vs MA50': 'Error',
                    'Price vs MA200': 'Error',
                    'Volume Trend': 'Error',
                    'Support': np.nan,
                    'Resistance': np.nan,
                    'ML Pattern': 'Error',
                    'Pattern Confidence': 'Error',
                    'Technical Score': 'Error',
                    'Recommendation': f"Error: {error_message}",
                    'Pattern_Info': {}  # Add empty dict for pattern info
                })

        # After processing all stocks, display results
        if results:
            # Convert results to DataFrame
            df_results = pd.DataFrame(results)

            # Display results table with styling
            st.subheader("📈 Analysis Summary")

            # The df_display and fillna lines were removed to prevent type errors.
            # Streamlit will handle rendering of np.nan values correctly.
            st.dataframe(df_results.drop(columns=['Pattern_Info'], errors='ignore').style.format(precision=2).applymap(
                lambda x: "background-color: #28a745; color: white"
                if x == "STRONG BUY"
                else "background-color: #dc3545; color: white"
                if x == "STRONG SELL"
                else "background-color: #ffc107"
                if x in ["BUY", "SELL"]
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

            # Create a list of stock names for the dropdown
            stock_names = df_results['Name'].tolist()
            selected_stock_name = st.selectbox("Select a stock for detailed analysis:", stock_names)

            if selected_stock_name:
                selected_row = df_results[df_results['Name'] == selected_stock_name].iloc[0]

                # Enhanced pattern display with confidence score
                pattern_info = selected_row.get('Pattern_Info', {})
                confidence = pattern_info.get('confidence', 0) if pattern_info else 0

                # Display pattern information with enhanced styling
                if selected_row['ML Pattern'] != 'No Data' and selected_row['ML Pattern'] != 'Error':
                    st.markdown(f"""
                        <div class="pattern-highlight">
                            🔍 <strong>Detected Pattern: {selected_row['ML Pattern']}</strong><br>
                            📊 Confidence Score: {confidence:.1%}<br>
                            💡 Recommendation: <span class="recommendation-{selected_row['Recommendation'].lower().replace('strong ', '')}">{selected_row['Recommendation']}</span>
                        </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                        <div class="pattern-card">
                            <h3>Pattern: {selected_row['ML Pattern']}</h3>
                            <p>Recommendation: <span class="recommendation-{selected_row['Recommendation'].lower().replace('strong ', '')}">{selected_row['Recommendation']}</span></p>
                        </div>
                    """, unsafe_allow_html=True)

                # Chart plotting logic with enhanced pattern overlay
                if pd.notna(selected_row['Current Price']):
                    chart_exchange = selected_row['Exchange']
                    chart_yf_ticker = f"{selected_row['Ticker']}.{'NS' if chart_exchange == 'NSE' else 'BO'}"

                    # Re-download data directly for plotting to ensure its integrity.
                    data = yf.download(chart_yf_ticker, period=analysis_period, progress=False)

                    # Handle MultiIndex columns if present
                    if isinstance(data.columns, pd.MultiIndex):
                        data.columns = data.columns.droplevel(1)

                    # Robust data validation for plotting
                    required_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
                    available_cols = [col for col in required_cols if col in data.columns]

                    if len(available_cols) < len(required_cols):
                        st.warning(
                            f"Chart could not be rendered for {selected_stock_name} due to missing columns. Available: {list(data.columns)}")
                        data = None
                    elif data[available_cols].isnull().all().all():
                        st.warning(
                            f"Chart could not be rendered for {selected_stock_name} due to all NaN values in OHLCV data.")
                        data = None
                    else:
                        data.reset_index(inplace=True)
                        if 'index' in data.columns:
                            data = data.rename(columns={'index': 'Date'})
                        # Ensure all relevant columns are numeric and drop rows with NaN in OHLCV
                        data[available_cols] = data[available_cols].apply(pd.to_numeric, errors='coerce')
                        data.dropna(subset=available_cols, inplace=True)

                        # Check if we still have data after cleaning
                        if data.empty:
                            st.warning(f"No valid data remaining for {selected_stock_name} after cleaning.")
                            data = None

                    # Check if the downloaded data is valid for plotting.
                    if isinstance(data, pd.DataFrame) and not data.empty and 'Date' in data.columns:
                        # Create a figure with subplots for price and volume
                        fig = make_subplots(
                            rows=2,
                            cols=1,
                            shared_xaxes=True,
                            vertical_spacing=0.05,
                            row_heights=[0.75, 0.25],
                            specs=[[{"secondary_y": False}], [{"secondary_y": False}]]
                        )

                        # 1. Price Candlestick Chart
                        fig.add_trace(go.Candlestick(
                            x=data['Date'],
                            open=data['Open'],
                            high=data['High'],
                            low=data['Low'],
                            close=data['Close'],
                            name='Price',
                            increasing_line_color='green',
                            decreasing_line_color='red'
                        ), row=1, col=1)

                        # 2. Volume Bar Chart
                        fig.add_trace(go.Bar(
                            x=data['Date'],
                            y=data['Volume'],
                            name='Volume',
                            marker_color='rgba(0, 102, 204, 0.6)'
                        ), row=2, col=1)

                        # 3. Add enhanced support/resistance overlay
                        add_support_resistance_overlay(
                            fig,
                            selected_row['Support'],
                            selected_row['Resistance'],
                            data,
                            row=1, col=1
                        )

                        # 4. Add enhanced pattern overlay
                        if pattern_info and selected_row['ML Pattern'] not in ['No Data', 'Error', 'N/A',
                                                                               'No clear pattern']:
                            add_pattern_overlay(
                                fig,
                                data,
                                pattern_info,
                                selected_row['ML Pattern'],
                                row=1, col=1
                            )

                        # 5. Add moving averages
                        if len(data) >= 50:
                            ma50 = data['Close'].rolling(window=50).mean()
                            fig.add_trace(go.Scatter(
                                x=data['Date'],
                                y=ma50,
                                mode='lines',
                                name='MA50',
                                line=dict(color='orange', width=1),
                                opacity=0.7
                            ), row=1, col=1)

                        if len(data) >= 200:
                            ma200 = data['Close'].rolling(window=200).mean()
                            fig.add_trace(go.Scatter(
                                x=data['Date'],
                                y=ma200,
                                mode='lines',
                                name='MA200',
                                line=dict(color='blue', width=1),
                                opacity=0.7
                            ), row=1, col=1)

                            # 6. Update layout for a professional look
                            fig.update_layout(
                                title={
                                    'text': f"{selected_row['Name']} ({selected_row['Ticker']}) - Technical Analysis",
                                    'x': 0.5,
                                    'xanchor': 'center',
                                    'font': {'size': 18, 'color': 'black'}  # Set title color
                                },
                                height=700,
                                xaxis_rangeslider_visible=False,
                                showlegend=True,
                                legend=dict(
                                    orientation="h",
                                    yanchor="bottom",
                                    y=1.02,
                                    xanchor="right",
                                    x=1,
                                    bgcolor='rgba(255, 255, 255, 0.95)',  # Make legend background more opaque
                                    bordercolor="black",
                                    borderwidth=1,
                                    font=dict(color="black")  # Set legend font color
                                ),
                                plot_bgcolor='rgba(240, 240, 240, 0.5)',
                                paper_bgcolor='white',
                                font=dict(size=12, color="black")  # Set global font color for better visibility
                            )

                            # Update axes for better visibility
                            fig.update_xaxes(
                                title_text="Date",
                                gridcolor='rgba(128, 128, 128, 0.3)',
                                title_font=dict(color='black'),
                                tickfont=dict(color='black'),
                                row=2, col=1
                            )
                            fig.update_yaxes(
                                title_text="Price (₹)",
                                gridcolor='rgba(128, 128, 128, 0.3)',
                                title_font=dict(color='black'),
                                tickfont=dict(color='black'),
                                row=1, col=1
                            )
                            fig.update_yaxes(
                                title_text="Volume",
                                gridcolor='rgba(128, 128, 128, 0.3)',
                                title_font=dict(color='black'),
                                tickfont=dict(color='black'),
                                row=2, col=1
                            )

                        # Display the chart without Streamlit's theme override
                        st.plotly_chart(fig, use_container_width=True)

                        # Add technical indicators summary
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

                        # Pattern details section
                        if pattern_info and selected_row['ML Pattern'] not in ['No Data', 'Error', 'N/A',
                                                                               'No clear pattern']:
                            st.subheader("🔍 Pattern Analysis Details")

                            pattern_col1, pattern_col2 = st.columns(2)

                            with pattern_col1:
                                st.info(f"""
                                **Pattern Type:** {selected_row['ML Pattern']}

                                **Pattern Confidence:** {confidence:.1%}

                                **Technical Score:** {selected_row['Technical Score']}

                                **Final Recommendation:** {selected_row['Recommendation']}
                                """)

                            with pattern_col2:
                                if 'position' in pattern_info and 'template_len' in pattern_info:
                                    pattern_start = pattern_info['position']
                                    pattern_length = pattern_info['template_len']
                                    pattern_end = pattern_start + pattern_length

                                    if pattern_end <= len(data):
                                        pattern_start_date = data['Date'].iloc[pattern_start].strftime('%Y-%m-%d')
                                        pattern_end_date = data['Date'].iloc[pattern_end - 1].strftime('%Y-%m-%d')

                                        st.success(f"""
                                        **Pattern Location:**

                                        **Start Date:** {pattern_start_date}

                                        **End Date:** {pattern_end_date}

                                        **Duration:** {pattern_length} days

                                        **Pattern Strength:** {'Strong' if confidence > 0.7 else 'Moderate' if confidence > 0.5 else 'Weak'}
                                        """)
                    else:
                        st.warning(
                            "Could not download valid chart data for the selected stock or 'Date' column is missing.")
                else:
                    st.warning("No data available to plot a chart because initial analysis failed.")
        else:
            st.warning("No results to display")
    except Exception as e:
        st.error(f"Error processing file: {e}")
else:
    st.info("Please upload a portfolio CSV file to begin analysis")
