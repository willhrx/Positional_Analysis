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
    
    # Time to Expiration Slider
    if st.session_state.position.options:
        st.markdown("---")
        st.subheader("Time Analysis")
        
        # Find the nearest expiration
        current_date = datetime.now()
        nearest_expiry = min(opt.expiry for opt in st.session_state.position.options)
        max_days = (nearest_expiry - current_date).days
        
        if max_days > 0:
            days_to_expiry = st.slider(
                "Days to Expiration",
                min_value=0,
                max_value=max_days,
                value=max_days,
                step=1,
                help=f"Slide to see how position changes as time passes. Current date: {current_date.strftime('%Y-%m-%d')}"
            )
            
            # Calculate the analysis date
            analysis_date = nearest_expiry - timedelta(days=days_to_expiry)
            
            # Store in session state for use in calculations
            st.session_state.analysis_date = analysis_date
            
            # Display the analysis date
            st.info(f"Analyzing position as of: {analysis_date.strftime('%Y-%m-%d')}")
            
            # Show time decay impact
            current_value = st.session_state.position.calculate_position_value(market, current_date)
            future_value = st.session_state.position.calculate_position_value(market, analysis_date)
            time_decay = future_value - current_value
            
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Days Remaining", days_to_expiry)
            with col2:
                st.metric("Time Decay Impact", format_currency(time_decay))
        else:
            st.warning("All options have expired")
            st.session_state.analysis_date = current_date
    else:
        # No analysis date if no options
        st.session_state.analysis_date = datetime.now()

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
tab1, tab2, tab3, tab4 = st.tabs(["P&L Analysis", "Greeks", "Time Decay Animation", "Summary"])

with tab1:
    # P&L Analysis
    st.header("Profit & Loss Analysis")
    
    if not st.session_state.position.options and st.session_state.position.underlying_shares == 0:
        st.info("Add positions using the sidebar to begin analysis")
    else:
        # Replace the P&L calculation section in tab1 with this updated version

        # Get analysis date (use current date if not set)
        analysis_date = st.session_state.get('analysis_date', datetime.now())
        
        # Calculate P&L across price range
        price_range = np.linspace(spot_price * 0.5, spot_price * 1.5, 200)
        pnls = st.session_state.position.calculate_pnl_range(price_range, market, analysis_date)
        
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
        breakevens = st.session_state.position.get_breakeven_points(market, analysis_date)
        for be in breakevens:
            fig_pnl.add_vline(
                x=be,
                line_dash="dot",
                line_color="green",
                opacity=0.5,
                annotation_text=f"BE: ${be:.2f}",
                annotation_position="bottom"
            )
        
        # Update layout with analysis date info
        title_text = "Position P&L at Expiration" if days_to_expiry <= 0 else f"Position P&L with {days_to_expiry} Days to Expiration"
        
        fig_pnl.update_layout(
            title=title_text,
            xaxis_title="Underlying Price ($)",
            yaxis_title="Profit/Loss ($)",
            height=500,
            hovermode='x unified',
            showlegend=True
        )
        
        st.plotly_chart(fig_pnl, use_container_width=True)
        
        # Key metrics (update these to use analysis_date)
        col1, col2, col3, col4 = st.columns(4)
        
        # Calculate metrics
        current_pnl = st.session_state.position.calculate_pnl(market, analysis_date)
        max_pl = st.session_state.position.get_max_profit_loss(market, analysis_date)
        prob_profit = st.session_state.position.calculate_probability_of_profit(market, analysis_date)
        
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
        analysis_date = st.session_state.get('analysis_date', datetime.now())
        
        # Calculate current Greeks at the analysis date
        greeks = st.session_state.position.calculate_total_greeks(market, analysis_date)
        
        # Display Greeks in gauge charts
        col1, col2, col3 = st.columns(3)
        
        with col1:
            fig_delta = create_gauge_chart(
                greeks['delta'],
                "Delta",
                [-1, 1]
            )
            st.plotly_chart(fig_delta, use_container_width=True)
            st.caption("Price sensitivity: $1 move = $" + f"{greeks['delta']:.2f}")
        
        with col2:
            fig_gamma = create_gauge_chart(
                greeks['gamma'],
                "Gamma",
                [-0.1, 0.1]
            )
            st.plotly_chart(fig_gamma, use_container_width=True)
            st.caption("Delta change rate: " + f"{greeks['gamma']:.4f}")
        
        with col3:
            fig_theta = create_gauge_chart(
                greeks['theta'],
                "Theta (per day)",
                [-0.5, 0.5]
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
            temp_greeks = st.session_state.position.calculate_total_greeks(temp_market, analysis_date)
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
    # Optional: Add this to create an animated time decay visualization
# Add this as a new tab or in the P&L Analysis section

    if st.checkbox("Show Time Decay Animation"):
        if st.session_state.position.options:
            # Create frames for animation
            current_date = datetime.now()
            nearest_expiry = min(opt.expiry for opt in st.session_state.position.options)
            days_to_expiry = (nearest_expiry - current_date).days
            
            if days_to_expiry > 0:
                # Create frames for different time points
                time_points = np.linspace(days_to_expiry, 0, min(20, days_to_expiry))
                
                frames = []
                for days in time_points:
                    analysis_date = nearest_expiry - timedelta(days=int(days))
                    pnls = st.session_state.position.calculate_pnl_range(price_range, market, analysis_date)
                    
                    frames.append(go.Frame(
                        data=[go.Scatter(
                            x=price_range,
                            y=pnls,
                            mode='lines',
                            line=dict(color='blue', width=3)
                        )],
                        name=str(int(days))
                    ))
                
                # Create figure with animation
                fig_anim = go.Figure(
                    data=[go.Scatter(
                        x=price_range,
                        y=frames[0].data[0].y,
                        mode='lines',
                        line=dict(color='blue', width=3)
                    )],
                    frames=frames
                )
                
                # Add play/pause buttons
                fig_anim.update_layout(
                    updatemenus=[{
                        'type': 'buttons',
                        'showactive': False,
                        'buttons': [
                            {
                                'label': 'Play',
                                'method': 'animate',
                                'args': [None, {
                                    'frame': {'duration': 200, 'redraw': True},
                                    'fromcurrent': True,
                                    'transition': {'duration': 100}
                                }]
                            },
                            {
                                'label': 'Pause',
                                'method': 'animate',
                                'args': [[None], {
                                    'frame': {'duration': 0, 'redraw': False},
                                    'mode': 'immediate',
                                    'transition': {'duration': 0}
                                }]
                            }
                        ]
                    }],
                    sliders=[{
                        'active': 0,
                        'yanchor': 'top',
                        'xanchor': 'left',
                        'currentvalue': {
                            'prefix': 'Days to Expiry: ',
                            'visible': True,
                            'xanchor': 'right'
                        },
                        'steps': [
                            {
                                'args': [[str(int(days))], {
                                    'frame': {'duration': 200, 'redraw': True},
                                    'mode': 'immediate',
                                    'transition': {'duration': 100}
                                }],
                                'label': str(int(days)),
                                'method': 'animate'
                            }
                            for days in time_points
                        ]
                    }],
                    title="P&L Time Decay Animation",
                    xaxis_title="Underlying Price ($)",
                    yaxis_title="Profit/Loss ($)",
                    height=600
                )
                
                # Add reference lines
                fig_anim.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.5)
                fig_anim.add_vline(x=spot_price, line_dash="dash", line_color="red", opacity=0.5)
                
                st.plotly_chart(fig_anim, use_container_width=True)

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
