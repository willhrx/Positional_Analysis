import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
from scipy.stats import norm
import time
from position_class import Position, Option, MarketConditions

# Page configuration
st.set_page_config(
    page_title="Options Position Analyzer",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    /* Main page styling */
    .main {
        padding-top: 2rem;
    }
    
    /* Custom metric cards */
    div[data-testid="metric-container"] {
        background-color: #f0f2f6;
        border: 1px solid #c9d1d9;
        padding: 1rem;
        border-radius: 0.5rem;
        box-shadow: 0 1px 3px rgba(0,0,0,0.1);
    }
    
    /* Improve sidebar styling */
    .css-1d391kg {
        padding-top: 3rem;
    }
    
    /* Custom headers */
    .custom-header {
        color: #1f2937;
        font-weight: 600;
        margin-bottom: 1rem;
    }
    
    /* Position summary cards */
    .position-card {
        background-color: #ffffff;
        padding: 1rem;
        border-radius: 0.5rem;
        border: 1px solid #e5e7eb;
        margin-bottom: 0.5rem;
    }
    
    /* Success/Error messages */
    .stSuccess {
        background-color: #d1fae5;
        color: #065f46;
    }
    
    .stError {
        background-color: #fee2e2;
        color: #991b1b;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'position' not in st.session_state:
    st.session_state.position = Position()
    st.session_state.show_position_form = False
    st.session_state.last_update = time.time()

# Helper functions
def format_currency(value):
    """Format value as currency with proper negative handling"""
    if value < 0:
        return f"-${abs(value):,.2f}"
    return f"${value:,.2f}"

def format_percentage(value):
    """Format value as percentage"""
    return f"{value:.1%}"

def create_gauge_chart(value, title, range_vals=[-1, 1]):
    """Create a gauge chart for Greeks visualization"""
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=value,
        title={'text': title},
        domain={'x': [0, 1], 'y': [0, 1]},
        gauge={
            'axis': {'range': range_vals},
            'bar': {'color': "dodgerblue"},
            'steps': [
                {'range': [range_vals[0], range_vals[1]], 'color': "lightgray"},
            ],
        }
    ))
    fig.update_layout(height=200, margin=dict(l=20, r=20, t=40, b=20))
    return fig

# Header
col1, col2, col3 = st.columns([2, 3, 1])
with col1:
    st.title("Options Position Analyzer")
with col3:
    if st.button("Refresh", key="refresh"):
        st.session_state.last_update = time.time()

st.markdown("---")

# Sidebar - Position Builder
with st.sidebar:
    st.header("Position Builder")
    
    # Market Conditions Section
    st.subheader("Market Conditions")
    
    spot_price = st.number_input(
        "Spot Price ($)",
        min_value=1.0,
        max_value=10000.0,
        value=100.0,
        step=0.5,
        help="Current price of the underlying asset"
    )
    
    col1, col2 = st.columns(2)
    with col1:
        volatility = st.slider(
            "Implied Vol (%)",
            min_value=5,
            max_value=200,
            value=25,
            step=1,
            help="Implied volatility as annual percentage"
        )
    
    with col2:
        risk_free_rate = st.slider(
            "Risk-Free Rate (%)",
            min_value=0.0,
            max_value=10.0,
            value=5.0,
            step=0.1,
            help="Annual risk-free interest rate"
        )
    
    # Create market conditions object
    market = MarketConditions(
        spot_price=spot_price,
        risk_free_rate=risk_free_rate / 100,
        volatility=volatility / 100
    )
    
    st.markdown("---")
    
    # Position Entry Section
    st.subheader("Add Positions")
    
    # Underlying shares
    with st.expander("Underlying Shares", expanded=False):
        shares = st.number_input(
            "Number of Shares",
            min_value=-10000,
            max_value=10000,
            value=0,
            step=100,
            help="Positive for long, negative for short"
        )
        
        share_price = st.number_input(
            "Entry Price ($)",
            min_value=0.01,
            value=spot_price,
            step=0.5
        )
        
        if st.button("Add Shares", key="add_shares"):
            if shares != 0:
                st.session_state.position.add_underlying(shares, share_price)
                st.success(f"Added {shares} shares at ${share_price}")
                time.sleep(0.5)
                st.rerun()
    
    # Options
    with st.expander("Options", expanded=True):
        col1, col2 = st.columns(2)
        
        with col1:
            option_type = st.selectbox(
                "Type",
                ["Call", "Put"],
                help="Call or Put option"
            )
            
            strike = st.number_input(
                "Strike Price ($)",
                min_value=1.0,
                value=100.0,
                step=1.0
            )
        
        with col2:
            position_type = st.selectbox(
                "Position",
                ["Long", "Short"],
                help="Long (buy) or Short (sell)"
            )
            
            contracts = st.number_input(
                "Contracts",
                min_value=1,
                value=1,
                step=1,
                help="Number of contracts (each = 100 shares)"
            )
        
        expiry_date = st.date_input(
            "Expiry Date",
            min_value=datetime.now().date(),
            value=(datetime.now() + timedelta(days=30)).date()
        )
        
        premium = st.number_input(
            "Premium per Share ($)",
            min_value=0.01,
            value=2.0,
            step=0.05,
            help="Option premium per share"
        )
        
        if st.button("Add Option", type="primary", key="add_option"):
            # Convert position type to position size
            position_size = contracts if position_type == "Long" else -contracts
            
            # Create option
            option = Option(
                strike=float(strike),
                expiry=datetime.combine(expiry_date, datetime.min.time()),
                option_type=option_type.lower(),
                position_size=position_size
            )
            
            st.session_state.position.add_option(option, premium)
            st.success(f"Added {position_type} {contracts} {option_type} @ ${strike}")
            time.sleep(0.5)
            st.rerun()
    
    # Position Summary
    st.markdown("---")
    st.subheader("Current Position")
    
    summary = st.session_state.position.summarize_position()
    
    if summary['underlying_shares'] != 0:
        shares_type = "Long" if summary['underlying_shares'] > 0 else "Short"
        st.info(f"**Shares:** {shares_type} {abs(summary['underlying_shares'])}")
    
    if summary['options']:
        for i, opt in enumerate(summary['options']):
            option_desc = (
                f"{opt['position_type']} {abs(opt['position'])} "
                f"{opt['type'].upper()} @ ${opt['strike']} "
                f"(exp: {opt['expiry']})"
            )
            col1, col2 = st.columns([4, 1])
            with col1:
                st.caption(option_desc)
            with col2:
                if st.button("❌", key=f"remove_{i}"):
                    st.session_state.position.remove_option(i)
                    st.rerun()
    else:
        st.caption("No options added yet")
    
    # Clear position button
    if st.button("🗑️ Clear All Positions", key="clear_all"):
        st.session_state.position = Position()
        st.success("Position cleared")
        time.sleep(0.5)
        st.rerun()

# Main content area
# Create tabs for different views
tab1, tab2, tab3, tab4 = st.tabs(["P&L Analysis", "Greeks", "Risk Scenarios", "Summary"])

with tab1:
    # P&L Analysis
    st.header("Profit & Loss Analysis")
    
    if not st.session_state.position.options and st.session_state.position.underlying_shares == 0:
        st.info("Add positions using the sidebar to begin analysis")
    else:
        # Calculate P&L across price range
        price_range = np.linspace(spot_price * 0.5, spot_price * 1.5, 200)
        pnls = st.session_state.position.calculate_pnl_range(price_range, market)
        
        # Create P&L chart
        fig_pnl = go.Figure()
        
        # Add P&L line
        fig_pnl.add_trace(go.Scatter(
            x=price_range,
            y=pnls,
            mode='lines',
            name='P&L',
            line=dict(color='blue', width=3),
            fill='tozeroy',
            fillcolor='rgba(0, 100, 255, 0.1)'
        ))
        
        # Add zero line
        fig_pnl.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.5)
        
        # Add current price line
        fig_pnl.add_vline(
            x=spot_price,
            line_dash="dash",
            line_color="red",
            opacity=0.5,
            annotation_text="Current Price",
            annotation_position="top"
        )
        
        # Add breakeven points
        breakevens = st.session_state.position.get_breakeven_points(market)
        for be in breakevens:
            fig_pnl.add_vline(
                x=be,
                line_dash="dot",
                line_color="green",
                opacity=0.5,
                annotation_text=f"BE: ${be:.2f}",
                annotation_position="bottom"
            )
        
        # Update layout
        fig_pnl.update_layout(
            title="Position P&L at Expiration",
            xaxis_title="Underlying Price ($)",
            yaxis_title="Profit/Loss ($)",
            height=500,
            hovermode='x unified',
            showlegend=True
        )
        
        st.plotly_chart(fig_pnl, use_container_width=True)
        
        # Key metrics
        col1, col2, col3, col4 = st.columns(4)
        
        # Calculate metrics
        current_pnl = st.session_state.position.calculate_pnl(market)
        max_pl = st.session_state.position.get_max_profit_loss(market)
        prob_profit = st.session_state.position.calculate_probability_of_profit(market)
        
        with col1:
            st.metric(
                "Current P&L",
                format_currency(current_pnl),
                delta=f"{current_pnl/abs(st.session_state.position.initial_cost):.1%}" if st.session_state.position.initial_cost != 0 else "N/A"
            )
        
        with col2:
            st.metric(
                "Max Profit",
                format_currency(max_pl['max_profit']),
                delta=f"at ${max_pl['max_profit_price']:.2f}"
            )
        
        with col3:
            st.metric(
                "Max Loss",
                format_currency(max_pl['max_loss']),
                delta=f"at ${max_pl['max_loss_price']:.2f}"
            )
        
        with col4:
            st.metric(
                "Probability of Profit",
                format_percentage(prob_profit),
                delta="at expiration"
            )
        
        # Breakeven analysis
        if breakevens:
            st.subheader("Breakeven Analysis")
            be_df = pd.DataFrame({
                'Breakeven Price': [f"${be:.2f}" for be in breakevens],
                'Distance from Current': [f"{(be/spot_price - 1)*100:.1f}%" for be in breakevens],
                'Required Move': [f"${be - spot_price:.2f}" for be in breakevens]
            })
            st.dataframe(be_df, hide_index=True, use_container_width=True)

with tab2:
    # Greeks Analysis
    st.header("Greeks Dashboard")
    
    if not st.session_state.position.options and st.session_state.position.underlying_shares == 0:
        st.info("👆 Add positions to view Greeks")
    else:
        # Calculate current Greeks
        greeks = st.session_state.position.calculate_total_greeks(market)
        
        # Display Greeks in gauge charts
        col1, col2, col3 = st.columns(3)
        
        with col1:
            fig_delta = create_gauge_chart(
                greeks['delta'],
                "Delta",
                [-100, 100]
            )
            st.plotly_chart(fig_delta, use_container_width=True)
            st.caption("Price sensitivity: $1 move = $" + f"{greeks['delta']:.2f}")
        
        with col2:
            fig_gamma = create_gauge_chart(
                greeks['gamma'],
                "Gamma",
                [-10, 10]
            )
            st.plotly_chart(fig_gamma, use_container_width=True)
            st.caption("Delta change rate: " + f"{greeks['gamma']:.4f}")
        
        with col3:
            fig_theta = create_gauge_chart(
                greeks['theta'],
                "Theta (per day)",
                [-50, 50]
            )
            st.plotly_chart(fig_theta, use_container_width=True)
            st.caption("Time decay: $" + f"{greeks['theta']:.2f}/day")
        
        col4, col5 = st.columns(2)
        
        with col4:
            st.metric(
                "Vega",
                f"{greeks['vega']:.2f}",
                delta="per 1% vol change",
                help="Sensitivity to volatility changes"
            )
        
        with col5:
            st.metric(
                "Rho",
                f"{greeks['rho']:.2f}",
                delta="per 1% rate change",
                help="Sensitivity to interest rate changes"
            )
        
        # Greeks over price range
        st.subheader("Greeks Profile")
        
        # Calculate Greeks across price range
        price_range_greeks = np.linspace(spot_price * 0.8, spot_price * 1.2, 50)
        deltas = []
        gammas = []
        
        for price in price_range_greeks:
            temp_market = MarketConditions(
                spot_price=price,
                risk_free_rate=market.risk_free_rate,
                volatility=market.volatility
            )
            temp_greeks = st.session_state.position.calculate_total_greeks(temp_market)
            deltas.append(temp_greeks['delta'])
            gammas.append(temp_greeks['gamma'])
        
        # Create subplots for Greeks
        fig_greeks = make_subplots(
            rows=1, cols=2,
            subplot_titles=("Delta Profile", "Gamma Profile")
        )
        
        # Delta profile
        fig_greeks.add_trace(
            go.Scatter(
                x=price_range_greeks,
                y=deltas,
                mode='lines',
                name='Delta',
                line=dict(color='blue', width=2)
            ),
            row=1, col=1
        )
        
        # Gamma profile
        fig_greeks.add_trace(
            go.Scatter(
                x=price_range_greeks,
                y=gammas,
                mode='lines',
                name='Gamma',
                line=dict(color='green', width=2)
            ),
            row=1, col=2
        )
        
        # Add current price lines
        fig_greeks.add_vline(
            x=spot_price,
            line_dash="dash",
            line_color="red",
            opacity=0.5,
            row=1, col=1
        )
        fig_greeks.add_vline(
            x=spot_price,
            line_dash="dash",
            line_color="red",
            opacity=0.5,
            row=1, col=2
        )
        
        fig_greeks.update_xaxes(title_text="Underlying Price ($)")
        fig_greeks.update_yaxes(title_text="Delta", row=1, col=1)
        fig_greeks.update_yaxes(title_text="Gamma", row=1, col=2)
        fig_greeks.update_layout(height=400, showlegend=False)
        
        st.plotly_chart(fig_greeks, use_container_width=True)

with tab3:
    # Risk Scenarios
    st.header("Risk Scenario Analysis")
    
    if not st.session_state.position.options and st.session_state.position.underlying_shares == 0:
        st.info("👆 Add positions to view risk scenarios")
    else:
        # Define scenarios
        scenarios = [
            ("Current", 0, 0, 0),
            ("-10% Price", -10, 0, 5),
            ("-5% Price", -5, 0, 2),
            ("+5% Price", 5, 0, -2),
            ("+10% Price", 10, 0, -5),
            ("Vol Spike (+10%)", 0, 10, 0),
            ("Vol Crush (-10%)", 0, -10, 0),
            ("Market Crash", -20, 20, 0),
            ("Market Rally", 20, -10, 0)
        ]
        
        # Calculate scenarios
        scenario_results = []
        current_value = st.session_state.position.calculate_position_value(market)
        
        for name, price_change, vol_change, rate_change in scenarios:
            scenario_market = MarketConditions(
                spot_price=spot_price * (1 + price_change/100),
                risk_free_rate=(risk_free_rate + rate_change) / 100,
                volatility=(volatility + vol_change) / 100
            )
            
            value = st.session_state.position.calculate_position_value(scenario_market)
            pnl = value - current_value
            greeks = st.session_state.position.calculate_total_greeks(scenario_market)
            
            scenario_results.append({
                'Scenario': name,
                'Spot': f"${scenario_market.spot_price:.2f}",
                'Vol': f"{scenario_market.volatility*100:.1f}%",
                'Value': format_currency(value),
                'P&L': format_currency(pnl),
                'Delta': f"{greeks['delta']:.2f}",
                'Gamma': f"{greeks['gamma']:.3f}"
            })
        
        # Display scenario table
        scenario_df = pd.DataFrame(scenario_results)
        
        # Style the dataframe
        def color_negative(val):
            if isinstance(val, str) and val.startswith('-$'):
                return 'color: red'
            return ''
        
        styled_df = scenario_df.style.applymap(color_negative, subset=['P&L'])
        st.dataframe(styled_df, hide_index=True, use_container_width=True)
        
        # Scenario visualization
        st.subheader("Scenario Impact Visualization")
        
        # Extract P&L values for chart
        pnl_values = []
        for result in scenario_results[1:]:  # Skip current scenario
            pnl_str = result['P&L'].replace('$', '').replace(',', '')
            pnl_values.append(float(pnl_str))
        
        fig_scenarios = go.Figure(data=[
            go.Bar(
                x=[s['Scenario'] for s in scenario_results[1:]],
                y=pnl_values,
                marker_color=['red' if v < 0 else 'green' for v in pnl_values]
            )
        ])
        
        fig_scenarios.update_layout(
            title="Scenario P&L Impact",
            xaxis_title="Scenario",
            yaxis_title="P&L Change ($)",
            height=400,
            showlegend=False
        )
        
        st.plotly_chart(fig_scenarios, use_container_width=True)

with tab4:
    # Position Summary
    st.header("Position Summary & Analytics")
    
    if not st.session_state.position.options and st.session_state.position.underlying_shares == 0:
        st.info("👆 Add positions to view summary")
    else:
        col1, col2 = st.columns([2, 3])
        
        with col1:
            st.subheader("Position Details")
            
            # Position composition
            summary = st.session_state.position.summarize_position()
            
            st.markdown("**Underlying:**")
            if summary['underlying_shares'] != 0:
                shares_info = f"{summary['underlying_shares']} shares "
                shares_info += f"({'Long' if summary['underlying_shares'] > 0 else 'Short'})"
                st.markdown(f"- {shares_info}")
            else:
                st.markdown("- No shares")
            
            st.markdown("**Options:**")
            if summary['options']:
                for opt in summary['options']:
                    opt_info = (
                        f"- {opt['position_type']} {abs(opt['position'])} "
                        f"{opt['type'].upper()} @ ${opt['strike']}"
                    )
                    st.markdown(opt_info)
            else:
                st.markdown("- No options")
            
            # Cost basis
            st.markdown("**Cost Basis:**")
            st.markdown(f"- Initial cost: {format_currency(abs(st.session_state.position.initial_cost))}")
            st.markdown(f"- Current value: {format_currency(st.session_state.position.calculate_position_value(market))}")
            
        with col2:
            st.subheader("Strategy Analysis")
            
            # Identify strategy type
            if summary['underlying_shares'] > 0 and len(summary['options']) == 1:
                opt = summary['options'][0]
                if opt['type'] == 'call' and opt['position'] < 0:
                    strategy = "Covered Call"
                elif opt['type'] == 'put' and opt['position'] > 0:
                    strategy = "Protective Put"
                else:
                    strategy = "Custom"
            elif summary['underlying_shares'] == 0 and len(summary['options']) == 2:
                if all(opt['strike'] == summary['options'][0]['strike'] for opt in summary['options']):
                    if all(opt['position'] > 0 for opt in summary['options']):
                        strategy = "Long Straddle"
                    elif all(opt['position'] < 0 for opt in summary['options']):
                        strategy = "Short Straddle"
                    else:
                        strategy = "Custom"
                else:
                    strategy = "Spread"
            elif len(summary['options']) == 4:
                strategy = "Possible Iron Condor/Butterfly"
            else:
                strategy = "Custom Strategy"
            
            st.info(f"**Detected Strategy:** {strategy}")
            
            # Risk metrics
            st.markdown("**Risk Metrics:**")
            max_pl = st.session_state.position.get_max_profit_loss(market)
            prob_profit = st.session_state.position.calculate_probability_of_profit(market)
            
            risk_reward = abs(max_pl['max_profit'] / max_pl['max_loss']) if max_pl['max_loss'] != 0 else float('inf')
            
            metrics_df = pd.DataFrame({
                'Metric': ['Max Profit', 'Max Loss', 'Risk/Reward Ratio', 'Probability of Profit'],
                'Value': [
                    format_currency(max_pl['max_profit']),
                    format_currency(max_pl['max_loss']),
                    f"{risk_reward:.2f}:1" if risk_reward != float('inf') else "Unlimited",
                    format_percentage(prob_profit)
                ]
            })
            
            st.dataframe(metrics_df, hide_index=True, use_container_width=True)

# Footer
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; color: #666;'>
        <p>Options Position Analyzer | Real-time Greeks & P&L Analysis</p>
        <p style='font-size: 0.8em;'>⚠️ For educational purposes only. Not financial advice.</p>
    </div>
    """,
    unsafe_allow_html=True
)
