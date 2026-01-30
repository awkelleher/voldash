"""
Volatility Trading Dashboard
Consolidates key metrics for options trading analysis
"""

import streamlit as st
import pandas as pd
import numpy as np
import vol_analysis as va
import variance_ratios as vr
from datetime import datetime, timedelta

# Page config
st.set_page_config(
    page_title="Vol Trading Dashboard",
    page_icon="📊",
    layout="wide"
)

# Load data
@st.cache_data
def load_data():
    """Load historical data with caching"""
    return va.load_historical_data()

@st.cache_data(ttl=10)  # Cache for only 10 seconds for live data
def load_live_vols():
    """Load live vol overrides if available"""
    try:
        import json
        import os
        if os.path.exists('live_vols.json'):
            with open('live_vols.json', 'r') as f:
                return json.load(f)
    except:
        pass
    return None

@st.cache_data
def load_price_data():
    """Load historical price data for variance ratio calculations"""
    try:
        import os
        if os.path.exists('all_commodity_prices.csv'):
            prices = pd.read_csv('all_commodity_prices.csv')
            prices['date'] = pd.to_datetime(prices['date'], format='mixed')
            return prices
    except:
        pass
    return None

# Custom CSS for better formatting
st.markdown("""
    <style>
    .metric-card {
        background-color: #f0f2f6;
        padding: 10px;
        border-radius: 5px;
        margin: 5px 0;
    }
    .stMetric label {
        font-size: 14px;
    }
    </style>
""", unsafe_allow_html=True)

# Title
st.title("📊 Volatility Trading Dashboard")

# Sidebar - Settings
st.sidebar.header("Settings")

# Load data
try:
    df = load_data()
    live_vols = load_live_vols()
    
    latest_date = df['date'].max()
    earliest_date = df['date'].min()
    
    # Show live data indicator if available
    if live_vols:
        from datetime import datetime
        live_time = datetime.fromisoformat(live_vols['timestamp'])
        st.sidebar.success(f"🔴 LIVE DATA: {live_time.strftime('%H:%M:%S')}")
        if st.sidebar.button("Clear Live Data"):
            import os
            if os.path.exists('live_vols.json'):
                os.remove('live_vols.json')
                st.rerun()
    
    # Date selector
    selected_date = st.sidebar.date_input(
        "Analysis Date",
        value=latest_date,
        min_value=earliest_date,
        max_value=latest_date
    )
    selected_date = pd.to_datetime(selected_date)
    
    # Commodity selector
    commodity = st.sidebar.selectbox(
        "Commodity",
        ['SOY', 'MEAL', 'OIL', 'CORN', 'WHEAT']
    )
    
    # Lookback period
    lookback = st.sidebar.slider(
        "Historical Lookback (trading days)",
        min_value=60,
        max_value=504,
        value=252
    )
    
    st.sidebar.markdown("---")
    st.sidebar.caption(f"Data range: {earliest_date.date()} to {latest_date.date()}")
    st.sidebar.caption(f"Total records: {len(df):,}")
    st.sidebar.caption(f"Trading days: {df['date'].nunique():,}")
    
except Exception as e:
    st.error(f"Error loading data: {str(e)}")
    st.stop()

# Main content
st.markdown(f"### {commodity} - {selected_date.date()}")

# Tab layout
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📈 Main Dashboard",
    "📊 Term Structure",
    "🔀 Spreads",
    "📉 Skew Analysis",
    "📐 Var Ratios"
])

# ============================================================================
# TAB 1: MAIN DASHBOARD
# ============================================================================
with tab1:
    # Check if we have live data for this commodity
    live_vol_override = None
    if live_vols and commodity in live_vols.get('vols', {}):
        live_vol_override = live_vols['vols'][commodity]
        st.info(f"🔴 Using LIVE market data: {live_vol_override:.2f}%")
    
    # Get front month stats
    front_vol_stats = va.calculate_percentile_rank(
        df, selected_date, 1, 'clean_vol', 
        lookback_days=lookback, commodity=commodity
    )
    
    fwd_vol_stats = va.calculate_percentile_rank(
        df, selected_date, 1, 'fwd_vol',
        lookback_days=lookback, commodity=commodity
    )
    
    if front_vol_stats and fwd_vol_stats:
        # Override with live data if available
        if live_vol_override:
            front_vol_stats['current'] = live_vol_override
            fwd_vol_stats['current'] = live_vol_override
        
        # Top metrics row
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                "Clean Vol (M1)",
                f"{front_vol_stats['current']:.2f}%",
                delta=f"{front_vol_stats['percentile']:.0f}th %ile"
            )
            st.caption(f"Median: {front_vol_stats['median']:.2f}% | Δ: {front_vol_stats['distance_from_median']:+.2f}%")
        
        with col2:
            st.metric(
                "Forward Vol (M1)",
                f"{fwd_vol_stats['current']:.2f}%",
                delta=f"{fwd_vol_stats['percentile']:.0f}th %ile"
            )
            st.caption(f"Median: {fwd_vol_stats['median']:.2f}%")
        
        # Calendar spread
        m1_m2_spread = va.calculate_calendar_spread(
            df, selected_date, 1, 2, 'fwd_vol', commodity=commodity
        )
        
        with col3:
            if m1_m2_spread:
                st.metric(
                    "M1-M2 Fwd Vol Spread",
                    f"{m1_m2_spread['current_spread']:.2f}%",
                    delta=f"{m1_m2_spread['percentile']:.0f}th %ile"
                )
                st.caption(f"Median: {m1_m2_spread['median_spread']:.2f}%")
        
        # Get current data for signal
        current_data = df[
            (df['date'] == selected_date) & 
            (df['commodity'] == commodity) & 
            (df['contract_month'] == 1)
        ]
        
        with col4:
            if len(current_data) > 0:
                row = current_data.iloc[0]
                if pd.notna(row['trading_dte']):
                    st.metric("Trading DTE", f"{int(row['trading_dte'])}")
                    if pd.notna(row['clean_dte']):
                        st.caption(f"Clean DTE: {row['clean_dte']:.1f}")
        
        st.markdown("---")
        
        # Your Forward Vol vs Predicted Realized Vol section
        st.subheader("🎯 Forward Vol vs Predicted Realized Vol")
        
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.markdown("**Your Model Input:**")
            predicted_rv = st.number_input(
                "Predicted Realized Vol (%)",
                min_value=0.0,
                max_value=100.0,
                value=fwd_vol_stats['current'] if fwd_vol_stats else 15.0,
                step=0.1,
                help="Enter your model's prediction for realized vol"
            )
            
            if fwd_vol_stats:
                fwd_vol = fwd_vol_stats['current']
                diff = fwd_vol - predicted_rv
                ratio = fwd_vol / predicted_rv if predicted_rv > 0 else 0
                
                st.markdown("**Analysis:**")
                st.write(f"Forward Vol: **{fwd_vol:.2f}%**")
                st.write(f"Your Prediction: **{predicted_rv:.2f}%**")
                st.write(f"Difference: **{diff:+.2f}%**")
                st.write(f"Ratio: **{ratio:.2f}x**")
                
                if diff > 2:
                    st.success("**SIGNAL: LONG VOL** ✅")
                    st.caption("Market IV is rich vs your prediction")
                elif diff < -2:
                    st.error("**SIGNAL: SHORT VOL** ❌")
                    st.caption("Market IV is cheap vs your prediction")
                else:
                    st.info("**SIGNAL: NEUTRAL** ⚖️")
                    st.caption("Market IV is fairly valued")
        
        with col2:
            # Historical comparison
            st.markdown("**Historical Context:**")
            
            # Get historical forward vols
            hist_data = df[
                (df['commodity'] == commodity) & 
                (df['contract_month'] == 1) &
                (df['date'] < selected_date)
            ].tail(lookback)[['date', 'fwd_vol']].dropna()
            
            if len(hist_data) > 0:
                chart_data = hist_data.set_index('date')
                # Add current point
                current_point = pd.DataFrame({
                    'fwd_vol': [fwd_vol_stats['current']]
                }, index=[selected_date])
                chart_data = pd.concat([chart_data, current_point])
                
                st.line_chart(chart_data, height=200)
                st.caption(f"Forward Vol over last {lookback} days")
    
    else:
        st.warning(f"No data available for {commodity} on {selected_date.date()}")

# ============================================================================
# TAB 2: TERM STRUCTURE
# ============================================================================
with tab2:
    st.subheader("IV Term Structure - Week Over Week Progression")
    
    term_structure = va.get_iv_term_structure(df, selected_date, commodity)
    
    if len(term_structure) > 0:
        # Filter to front 8 months
        term_structure_display = term_structure.head(8).copy()
        
        # Format for display
        term_structure_display['expiry'] = term_structure_display['expiry'].dt.strftime('%Y-%m-%d')
        term_structure_display = term_structure_display[[
            'contract_month', 'expiry', 'dirty_vol', 'clean_vol', 'fwd_vol', 'trading_dte'
        ]]
        term_structure_display.columns = [
            'Month', 'Expiry', 'Dirty Vol', 'Clean Vol', 'Fwd Vol', 'DTE'
        ]
        
        # Display table
        st.dataframe(
            term_structure_display.style.format({
                'Dirty Vol': '{:.2f}%',
                'Clean Vol': '{:.2f}%',
                'Fwd Vol': '{:.2f}%',
                'DTE': '{:.0f}'
            }),
            use_container_width=True
        )
        
        # Charts
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Clean Vol Curve**")
            chart_data = term_structure_display.set_index('Month')['Clean Vol']
            st.line_chart(chart_data)
        
        with col2:
            st.markdown("**Forward Vol Curve**")
            chart_data = term_structure_display.set_index('Month')['Fwd Vol']
            st.line_chart(chart_data)
    else:
        st.warning("No term structure data available")

# ============================================================================
# TAB 3: SPREADS
# ============================================================================
with tab3:
    st.subheader("Calendar & Cross-Commodity Spreads")
    
    # Calendar spreads
    st.markdown("#### Calendar Spreads (Forward Vol)")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # M1-M2
        spread_12 = va.calculate_calendar_spread(
            df, selected_date, 1, 2, 'fwd_vol', commodity=commodity
        )
        if spread_12:
            st.metric(
                "M1 - M2",
                f"{spread_12['current_spread']:.2f}%",
                delta=f"{spread_12['percentile']:.0f}th %ile"
            )
            st.caption(f"Median: {spread_12['median_spread']:.2f}% | Δ: {spread_12['distance_from_median']:+.2f}%")
        
        # M2-M3
        spread_23 = va.calculate_calendar_spread(
            df, selected_date, 2, 3, 'fwd_vol', commodity=commodity
        )
        if spread_23:
            st.metric(
                "M2 - M3",
                f"{spread_23['current_spread']:.2f}%",
                delta=f"{spread_23['percentile']:.0f}th %ile"
            )
            st.caption(f"Median: {spread_23['median_spread']:.2f}%")
    
    with col2:
        # M3-M4
        spread_34 = va.calculate_calendar_spread(
            df, selected_date, 3, 4, 'fwd_vol', commodity=commodity
        )
        if spread_34:
            st.metric(
                "M3 - M4",
                f"{spread_34['current_spread']:.2f}%",
                delta=f"{spread_34['percentile']:.0f}th %ile"
            )
            st.caption(f"Median: {spread_34['median_spread']:.2f}%")
    
    st.markdown("---")
    
    # Cross-commodity spreads
    st.markdown("#### Cross-Commodity IV Spreads")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # SOY vs MEAL
        soy_meal = va.calculate_cross_commodity_spread(
            df, selected_date, 'SOY', 'MEAL', 1, 'clean_vol'
        )
        if soy_meal:
            st.markdown("**SOY vs MEAL (M1 Clean Vol)**")
            st.metric(
                "Spread",
                f"{soy_meal['current_spread']:.2f}%",
                delta=f"{soy_meal['percentile']:.0f}th %ile"
            )
            st.caption(f"SOY: {soy_meal['commodity1_value']:.2f}% | MEAL: {soy_meal['commodity2_value']:.2f}%")
            st.caption(f"Median spread: {soy_meal['median_spread']:.2f}%")
    
    with col2:
        # CORN vs WHEAT
        corn_wheat = va.calculate_cross_commodity_spread(
            df, selected_date, 'CORN', 'WHEAT', 1, 'clean_vol'
        )
        if corn_wheat:
            st.markdown("**CORN vs WHEAT (M1 Clean Vol)**")
            st.metric(
                "Spread",
                f"{corn_wheat['current_spread']:.2f}%",
                delta=f"{corn_wheat['percentile']:.0f}th %ile"
            )
            st.caption(f"CORN: {corn_wheat['commodity1_value']:.2f}% | WHEAT: {corn_wheat['commodity2_value']:.2f}%")
            st.caption(f"Median spread: {corn_wheat['median_spread']:.2f}%")

# ============================================================================
# TAB 4: SKEW ANALYSIS
# ============================================================================
with tab4:
    st.subheader("Skew Analysis vs Historical")
    
    skew_summary = va.get_skew_summary(df, selected_date, 1, commodity=commodity)
    
    if len(skew_summary) > 0:
        # Display as table
        display_df = skew_summary.copy()
        display_df.columns = ['Current', 'Percentile', 'Median', 'Distance', 'Lookback', 'Strike']
        display_df = display_df[['Strike', 'Current', 'Median', 'Percentile', 'Distance']]
        
        st.dataframe(
            display_df.style.format({
                'Current': '{:.1f}',
                'Median': '{:.1f}',
                'Percentile': '{:.1f}%',
                'Distance': '{:+.1f}'
            }),
            use_container_width=True
        )
        
        # Visual comparison
        st.markdown("**Current vs Median Skew**")
        
        chart_data = pd.DataFrame({
            'Strike': display_df['Strike'],
            'Current': display_df['Current'],
            'Median': display_df['Median']
        }).set_index('Strike')
        
        st.line_chart(chart_data)
        
        # Percentile heatmap
        st.markdown("**Percentile Ranks**")
        percentile_data = display_df.set_index('Strike')['Percentile']
        
        # Color code based on percentile
        def percentile_color(val):
            if val > 75:
                return 'background-color: #ff6b6b'
            elif val > 60:
                return 'background-color: #ffd93d'
            elif val < 25:
                return 'background-color: #6bcf7f'
            elif val < 40:
                return 'background-color: #95e1d3'
            return ''
        
        styled_percentiles = display_df[['Strike', 'Percentile']].style.applymap(
            percentile_color, subset=['Percentile']
        ).format({'Percentile': '{:.1f}%'})
        
        st.dataframe(styled_percentiles, use_container_width=True)
        
        st.caption("🟢 Low percentile (<40%) | 🟡 High percentile (>60%) | 🔴 Very high (>75%)")
    
    else:
        st.warning("No skew data available")

# ============================================================================
# TAB 5: VARIANCE RATIOS
# ============================================================================
with tab5:
    st.subheader("Variance Ratios - Historical Averages")

    # Load price data
    prices_df = load_price_data()

    if prices_df is not None and len(prices_df) > 0:
        # Front options month selector
        col1, col2, col3 = st.columns([1, 1, 2])

        with col1:
            front_month = st.selectbox(
                "Front Options Month",
                ['H', 'K', 'N', 'Q', 'U', 'X', 'F'],
                index=0,
                help="Select the front options month to view variance ratios"
            )

            # Map options month to futures month for display
            futures_month = vr.OPTIONS_TO_FUTURES.get(front_month, front_month)
            st.caption(f"Underlying futures: {futures_month}")

        with col2:
            lookback_years = st.selectbox(
                "Lookback (Years)",
                [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, None],
                index=4,  # Default to 5 years
                format_func=lambda x: "All" if x is None else str(x),
                help="Number of years of historical data to use"
            )

        # Calculate variance ratio matrix
        with st.spinner("Calculating variance ratios..."):
            matrix_df, metadata = vr.get_variance_ratio_display(
                prices_df, front_month, commodity, lookback_years=lookback_years
            )

        if len(matrix_df) > 0:
            # Display metadata
            st.markdown(f"""
            **Historical Data:**
            - Years included: {len(metadata['years_included'])} ({min(metadata['years_included'])+2000} - {max(metadata['years_included'])+2000})
            - Total trading days: {metadata['total_trading_days']}
            """)

            # Style the dataframe
            def highlight_diagonal(val):
                """Highlight cells close to 1.0 (diagonal)"""
                if pd.isna(val):
                    return ''
                if 0.99 <= val <= 1.01:
                    return 'background-color: #90EE90'  # Light green for diagonal
                elif val < 0.7:
                    return 'background-color: #FFB6C1'  # Light red for low ratios
                elif val > 1.3:
                    return 'background-color: #ADD8E6'  # Light blue for high ratios
                return ''

            # Display the matrix
            st.dataframe(
                matrix_df.style.applymap(highlight_diagonal).format("{:.2f}"),
                use_container_width=True,
                height=450
            )

            st.caption("""
            **Reading the matrix:**
            - Rows = contract being measured
            - Columns = contract position on curve (VarRat1 = front, VarRat2 = 2nd, etc.)
            - Cell = Variance(row) / Variance(column)
            - Green = diagonal (1.0), Red = low ratio (<0.7), Blue = high ratio (>1.3)
            """)

            # Show the futures curve order
            with st.expander("Futures Curve Order"):
                curve = vr.get_futures_curve_for_front_month(futures_month, commodity)
                curve_display = " → ".join([f"{i+1}FM: {m}" for i, m in enumerate(curve[:8])])
                st.write(curve_display)

        else:
            st.warning(f"No variance ratio data available for {front_month} options on {commodity}")

    else:
        st.warning("Price data not available. Run update_from_hertz.py to load price data.")

# Footer
st.markdown("---")
st.caption("Vol Trading Dashboard | Data updated daily")
