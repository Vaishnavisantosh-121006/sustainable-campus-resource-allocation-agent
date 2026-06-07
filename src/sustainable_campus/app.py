import os
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from loguru import logger

# Import local modules
from sustainable_campus.config import default_config, AppConfig
from sustainable_campus.ai.dataset_generator import generate_campus_dataset
from sustainable_campus.ai.demand_prediction import train_and_save_models
from sustainable_campus.simulation.environment import CampusSimulation
from sustainable_campus.simulation.metrics import calculate_cumulative_metrics
from sustainable_campus.game_theory.payoff_matrix import generate_payoff_matrix
from sustainable_campus.game_theory.nash_equilibrium import solve_game

# ----------------------------------------------------
# Page Configuration & Styling
# ----------------------------------------------------
st.set_page_config(
    page_title="Sustainable Campus Energy Allocation",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Premium Styling
st.markdown("""
<style>
    /* Main container styling */
    .main {
        background-color: #0d1117;
        color: #c9d1d9;
    }
    
    /* Modern Card UI */
    .metric-card {
        background: rgba(255, 255, 255, 0.03);
        border-radius: 12px;
        padding: 20px;
        border: 1px solid rgba(255, 255, 255, 0.08);
        box-shadow: 0 4px 30px rgba(0, 0, 0, 0.2);
        backdrop-filter: blur(5px);
        margin-bottom: 15px;
    }
    .metric-header {
        font-size: 14px;
        color: #8b949e;
        text-transform: uppercase;
        letter-spacing: 1.2px;
        margin-bottom: 5px;
    }
    .metric-val {
        font-size: 32px;
        font-weight: 700;
        color: #58a6ff;
    }
    .metric-sub {
        font-size: 12px;
        color: #3fb950;
        margin-top: 5px;
    }
    
    /* Headers styling */
    h1, h2, h3 {
        color: #f0f6fc;
        font-weight: 600;
    }
    
    /* Sidebar styling */
    section[data-testid="stSidebar"] {
        background-color: #161b22;
        border-right: 1px solid rgba(255, 255, 255, 0.08);
    }
    
    /* Custom tabs styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 24px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        white-space: pre-wrap;
        background-color: transparent;
        border-radius: 4px 4px 0px 0px;
        gap: 0px;
        padding-top: 10px;
        padding-bottom: 10px;
        font-size: 16px;
        font-weight: 500;
        color: #8b949e;
    }
    .stTabs [aria-selected="true"] {
        color: #58a6ff !important;
        border-bottom-color: #58a6ff !important;
    }
</style>
""", unsafe_style_html=True)

# Define directories
DATA_DIR = "data"
MODELS_DIR = "models"
DATA_FILE = os.path.join(DATA_DIR, "campus_energy_demand.csv")

# ----------------------------------------------------
# Auto-Generation of Data & Models on Startup
# ----------------------------------------------------
if not os.path.exists(DATA_FILE):
    st.info("Generating synthetic campus energy dataset. This runs only once...")
    generate_campus_dataset(DATA_FILE)

# Ensure models are trained
model_checkfiles = [
    os.path.join(MODELS_DIR, "classroom_demand_model.joblib"),
    os.path.join(MODELS_DIR, "laboratory_demand_model.joblib"),
    os.path.join(MODELS_DIR, "hostel_demand_model.joblib")
]
if not all(os.path.exists(f) for f in model_checkfiles):
    st.info("Training demand forecasting models. This runs only once...")
    train_and_save_models(DATA_FILE, MODELS_DIR)

# ----------------------------------------------------
# Sidebar Configuration Panel
# ----------------------------------------------------
st.sidebar.title("⚡ Microgrid Controls")
st.sidebar.markdown("Configure grid constants and agent optimization weights.")

# Grid Config Inputs
st.sidebar.subheader("Grid Constants")
grid_capacity = st.sidebar.number_input("Grid Capacity (kWh/hr)", min_value=100.0, max_value=5000.0, value=1000.0, step=100.0)
normal_price = st.sidebar.number_input("Normal Price ($/kWh)", min_value=0.01, max_value=1.00, value=0.15, step=0.01)
peak_price = st.sidebar.number_input("Peak Price ($/kWh)", min_value=0.01, max_value=2.00, value=0.30, step=0.01)

# Agent Weights Inputs
st.sidebar.subheader("Utility Weights (Sum = 1.0)")
w_satisfaction = st.sidebar.slider("w1: Satisfaction Weight", 0.0, 1.0, 0.5, 0.05)
w_sustainability = st.sidebar.slider("w2: Sustainability Weight", 0.0, 1.0, 0.3, 0.05)
w_cost = st.sidebar.slider("w3: Cost Savings Weight", 0.0, 1.0, 0.2, 0.05)

# Normalise weights
total_w = w_satisfaction + w_sustainability + w_cost
if abs(total_w - 1.0) > 1e-5:
    w_satisfaction /= total_w
    w_sustainability /= total_w
    w_cost /= total_w

# Apply changes to config
config = AppConfig()
config.grid.total_capacity_kwh = grid_capacity
config.grid.normal_price_per_kwh = normal_price
config.grid.peak_price_per_kwh = peak_price
config.weights.satisfaction = w_satisfaction
config.weights.sustainability = w_sustainability
config.weights.cost = w_cost

# ----------------------------------------------------
# Header Banner
# ----------------------------------------------------
st.title("Sustainable Campus Resource Allocation Agent")
st.markdown("An autonomous multi-agent AI system allocating energy resources among classrooms, laboratories, and hostels using game theory and RL.")

# Create Streamlit Tabs
tab_home, tab_simulation, tab_forecast, tab_game_theory, tab_sustainability = st.tabs([
    "🏡 Overview & SDGs",
    "🏃 Live Simulation",
    "📈 AI Forecasts",
    "🎮 Game Theory Nash Solver",
    "🍃 Sustainability Metrics"
])

# ----------------------------------------------------
# TAB 1: HOME / OVERVIEW
# ----------------------------------------------------
with tab_home:
    st.subheader("Project Architecture")
    
    col_desc, col_graph = st.columns([3, 2])
    
    with col_desc:
        st.markdown("""
        ### Multi-Agent Energy Management System
        The campus grid manages limited grid electricity capacity. When total energy requested by 
        Classrooms, Laboratories, and Hostels exceeds available grid capacity, conflict arises. 
        
        This project simulates an autonomous microgrid containing:
        - **Classroom Agent**: Predicts HVAC and comfort loads based on schedules.
        - **Laboratory Agent**: Predicts heavy server and ventilation baseloads.
        - **Hostel Agent**: Models peak resident usage (mornings and evenings).
        - **Energy Manager Agent**: Employs optimization policies (Proportional, Priority, Game Theory Nash Equilibrium, or Reinforcement Learning Q-learning) to dynamically allocate energy.
        
        ### SDG Impact Alignment
        
        | Goal | Target Alignment | Project Implementation |
        | --- | --- | --- |
        | **SDG 7: Affordable & Clean Energy** | 7.3: Double the global rate of improvement in energy efficiency. | AI-driven curtailment of standby loads and optimization using Nash Equilibrium and RL policies to prevent overload wastage. |
        | **SDG 12: Responsible Consumption** | 12.2: Achieve the sustainable management and efficient use of natural resources. | Reduces carbon footprints by optimizing local solar/wind integration and encouraging demand-side response. |
        """)
        
    with col_graph:
        st.markdown("### Agent Interactivity Model")
        
        # Display Mermaid Diagram using st.code
        st.code("""
  ┌────────────────────────────────────────────────────────┐
  │                   Campus Environment                   │
  │                     (Message Bus)                      │
  └───────▲───────────────▲────────────────▲───────────────┘
          │               │                │
    ┌─────▼─────┐   ┌─────▼─────┐    ┌─────▼─────┐
    │ Classroom │   │Laboratory │    │  Hostel   │
    │   Agent   │   │   Agent   │    │   Agent   │
    └─────▲─────┘   └─────▲─────┘    └─────▲─────┘
          │               │                │
          │         ┌─────▼─────┐          │
          └────────►│  Energy   │◄─────────┘
                    │  Manager  │
                    └───────────┘
        """, language="text")

# ----------------------------------------------------
# TAB 2: LIVE SIMULATION
# ----------------------------------------------------
with tab_simulation:
    st.subheader("Simulation Controls & Real-Time Allocations")
    
    col_controls, col_display = st.columns([1, 3])
    
    with col_controls:
        sim_duration = st.selectbox("Simulation Duration", [30, 100, 365], format_func=lambda x: f"{x} Days")
        sim_policy = st.selectbox("Allocation Policy", ["equal_proportion", "priority", "game_theoretic", "rl"], format_func=lambda x: {
            "equal_proportion": "Equal Proportional Sharing",
            "priority": "Rule-Based Priority (Lab > Class > Hostel)",
            "game_theoretic": "Game Theory (Nash Equilibrium)",
            "rl": "Reinforcement Learning (Q-learning)"
        }[x])
        
        run_sim = st.button("Run Simulation", type="primary")
        
    with col_display:
        if run_sim or "sim_history" in st.session_state:
            if run_sim:
                with st.spinner("Running simulation and training RL policy if applicable..."):
                    # Initialize and run simulation
                    sim = CampusSimulation(
                        config=config,
                        policy=sim_policy,
                        model_dir=MODELS_DIR,
                        rl_policy_path=os.path.join(MODELS_DIR, "rl_qtable.json")
                    )
                    steps = sim_duration * 24
                    df_history = sim.run_simulation(steps)
                    st.session_state["sim_history"] = df_history
                    st.session_state["sim_policy"] = sim_policy
            
            df_hist = st.session_state["sim_history"]
            policy_used = st.session_state["sim_policy"]
            
            # Display overall metrics for this run
            cum_metrics = calculate_cumulative_metrics(df_hist)
            
            col_m1, col_m2, col_m3, col_m4 = st.columns(4)
            with col_m1:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-header">Energy Saved</div>
                    <div class="metric-val">{cum_metrics['total_energy_saved_mwh']:.2f} MWh</div>
                    <div class="metric-sub">Carbon savings support</div>
                </div>
                """, unsafe_style_html=True)
            with col_m2:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-header">Carbon Prevented</div>
                    <div class="metric-val">{cum_metrics['total_carbon_saved_tons']:.2f} tCO2</div>
                    <div class="metric-sub">SDG 12 Positive Impact</div>
                </div>
                """, unsafe_style_html=True)
            with col_m3:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-header">Avg Satisfaction</div>
                    <div class="metric-val">{cum_metrics['average_satisfaction']*100:.1f}%</div>
                    <div class="metric-sub">Comfort metric</div>
                </div>
                """, unsafe_style_html=True)
            with col_m4:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-header">Sustainability Score</div>
                    <div class="metric-val">{cum_metrics['average_sustainability_score']*100:.1f} / 100</div>
                    <div class="metric-sub">Overall Rating</div>
                </div>
                """, unsafe_style_html=True)
                
            # Plot Allocations vs Requests (Sample the first 72 hours for clean UI visualization)
            st.markdown("### Hourly Allocations vs Demands (First 72 Hours)")
            sample_df = df_hist.head(72)
            
            fig_alloc = go.Figure()
            # Classroom
            fig_alloc.add_trace(go.Scatter(x=sample_df["step"], y=sample_df["req_classroom"], name="Classroom Request", line=dict(color="#e74c3c", dash="dash")))
            fig_alloc.add_trace(go.Scatter(x=sample_df["step"], y=sample_df["alloc_classroom"], name="Classroom Allocated", line=dict(color="#c0392b")))
            # Lab
            fig_alloc.add_trace(go.Scatter(x=sample_df["step"], y=sample_df["req_laboratory"], name="Lab Request", line=dict(color="#f1c40f", dash="dash")))
            fig_alloc.add_trace(go.Scatter(x=sample_df["step"], y=sample_df["alloc_laboratory"], name="Lab Allocated", line=dict(color="#d35400")))
            # Hostel
            fig_alloc.add_trace(go.Scatter(x=sample_df["step"], y=sample_df["req_hostel"], name="Hostel Request", line=dict(color="#2ecc71", dash="dash")))
            fig_alloc.add_trace(go.Scatter(x=sample_df["step"], y=sample_df["alloc_hostel"], name="Hostel Allocated", line=dict(color="#27ae60")))
            
            fig_alloc.update_layout(
                template="plotly_dark",
                xaxis_title="Simulation Hours",
                yaxis_title="Energy (kWh)",
                height=400,
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
            )
            st.plotly_chart(fig_alloc, use_container_width=True)
            
            # Cumulative Grid load vs capacity
            st.markdown("### Total Grid Load vs Grid Capacity")
            total_req = sample_df["req_classroom"] + sample_df["req_laboratory"] + sample_df["req_hostel"]
            total_alloc = sample_df["total_energy_consumed"]
            
            fig_grid = go.Figure()
            fig_grid.add_trace(go.Scatter(x=sample_df["step"], y=total_req, name="Total Demand Request", line=dict(color="#9b59b6")))
            fig_grid.add_trace(go.Scatter(x=sample_df["step"], y=total_alloc, name="Total Allocated Power", fill='tozeroy', line=dict(color="#3498db")))
            fig_grid.add_trace(go.Scatter(x=sample_df["step"], y=sample_df["total_capacity"], name="Grid Capacity Limit", line=dict(color="#e74c3c", width=2, dash="dot")))
            
            fig_grid.update_layout(
                template="plotly_dark",
                xaxis_title="Simulation Hours",
                yaxis_title="Total Microgrid Load (kWh)",
                height=400,
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
            )
            st.plotly_chart(fig_grid, use_container_width=True)
        else:
            st.warning("Click 'Run Simulation' to start the campus microgrid simulation.")

# ----------------------------------------------------
# TAB 3: AI FORECASTS
# ----------------------------------------------------
with tab_forecast:
    st.subheader("Demand Prediction Model Metrics & Evaluation")
    
    # Train results (usually we reload them or calculate from dataset)
    # Let's perform a simple train-test evaluations to display actual model charts
    if os.path.exists(DATA_FILE):
        df = pd.read_csv(DATA_FILE)
        
        # Display dataset description
        st.markdown("### Synthetic Load Profile Dataset (Sample Rows)")
        st.dataframe(df.head(10))
        
        # Load models to visualize predictions on a test window
        import joblib
        st.markdown("### Demand Forecasting Performance (Actual vs Forecasted)")
        
        # Select an agent type to inspect forecast
        agent_type = st.selectbox("Select Agent Type to inspect ML predictions:", ["classroom", "laboratory", "hostel"])
        
        model_path = os.path.join(MODELS_DIR, f"{agent_type}_demand_model.joblib")
        if os.path.exists(model_path):
            model = joblib.load(model_path)
            
            # Formulate validation prediction for a specific window
            # Let's select a 7-day window (168 hours) to chart actuals vs predictions
            test_window = df.iloc[500:500+168].copy()
            occ_col = f"{agent_type}_occupancy"
            prev_col = f"{agent_type}_prev_demand"
            target_col = f"{agent_type}_demand"
            
            features = ["day_of_week", "hour", "temperature", "is_holiday", "is_exam_period", occ_col, prev_col]
            X_eval = test_window[features].copy()
            X_eval.columns = ["day_of_week", "hour", "temperature", "is_holiday", "is_exam_period", "occupancy", "prev_demand"]
            
            preds = model.predict(X_eval)
            test_window["Predicted"] = preds
            
            fig_fc = go.Figure()
            fig_fc.add_trace(go.Scatter(x=test_window["timestamp"], y=test_window[target_col], name="Actual Load Demand", line=dict(color="#2ecc71")))
            fig_fc.add_trace(go.Scatter(x=test_window["timestamp"], y=test_window["Predicted"], name="ML Predicted Demand", line=dict(color="#e74c3c", dash="dash")))
            
            fig_fc.update_layout(
                template="plotly_dark",
                xaxis_title="Date Time",
                yaxis_title="Electricity Load (kWh)",
                height=450,
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
            )
            st.plotly_chart(fig_fc, use_container_width=True)
            
            # Model Accuracy Table
            # Let's display hardcoded model comparisons computed during training
            st.markdown("### Model Comparison & Evaluations")
            
            # We can recompute on a sample split
            from sklearn.model_selection import train_test_split
            X_all = df[features].copy()
            X_all.columns = ["day_of_week", "hour", "temperature", "is_holiday", "is_exam_period", "occupancy", "prev_demand"]
            y_all = df[target_col].values
            _, X_t, _, y_t = train_test_split(X_all, y_all, test_size=0.2, random_state=42)
            
            from sklearn.linear_model import LinearRegression
            lr = LinearRegression().fit(X_all, y_all) # fast fitting
            lr_preds = lr.predict(X_t)
            
            from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
            lr_mae = mean_absolute_error(y_t, lr_preds)
            lr_rmse = np.sqrt(mean_squared_error(y_t, lr_preds))
            lr_r2 = r2_score(y_t, lr_preds)
            
            rf_preds = model.predict(X_t)
            rf_mae = mean_absolute_error(y_t, rf_preds)
            rf_rmse = np.sqrt(mean_squared_error(y_t, rf_preds))
            rf_r2 = r2_score(y_t, rf_preds)
            
            metrics_compare = pd.DataFrame({
                "Regression Model": ["Linear Regression", "Random Forest Regressor"],
                "MAE (kWh)": [lr_mae, rf_mae],
                "RMSE (kWh)": [lr_rmse, rf_rmse],
                "R² Score": [lr_r2, rf_r2]
            })
            st.table(metrics_compare)
            
        else:
            st.warning("Model file not found. Run simulation to train models.")

# ----------------------------------------------------
# TAB 4: GAME THEORY & NASH SOLVER
# ----------------------------------------------------
with tab_game_theory:
    st.subheader("Interactive Game Theory Analysis")
    st.markdown("Tune nominal requests and grid limits to see how the agents coordinate strategy payoffs and Nash Equilibria.")
    
    col_g_inputs, col_g_results = st.columns([1, 2])
    
    with col_g_inputs:
        st.markdown("#### Demand Signals (Nominal)")
        class_nom = st.slider("Classroom Demand Request (kWh)", 50.0, 500.0, 200.0, 10.0)
        lab_nom = st.slider("Laboratory Demand Request (kWh)", 50.0, 500.0, 150.0, 10.0)
        hostel_nom = st.slider("Hostel Demand Request (kWh)", 100.0, 1000.0, 400.0, 20.0)
        
        st.markdown("#### Grid Scarcity Setting")
        scarcity_cap = st.slider("Scarcity Capacity Limit (kWh)", 100.0, 1500.0, 600.0, 20.0)
        
    with col_g_results:
        # Generate matrix dynamically
        pm = generate_payoff_matrix(class_nom, lab_nom, hostel_nom, scarcity_cap, config.weights, config.grid)
        sol = solve_game(pm)
        
        # Display solution
        st.markdown(f"### Solver Result: `{sol['outcome_type']}`")
        
        col_s1, col_s2, col_s3 = st.columns(3)
        with col_s1:
            st.metric("Classroom Strategy", sol["strategy_names"]["classroom"])
            st.metric("Classroom Payoff Utility", f"{sol['payoffs']['classroom']:.3f}")
        with col_s2:
            st.metric("Laboratory Strategy", sol["strategy_names"]["laboratory"])
            st.metric("Laboratory Payoff Utility", f"{sol['payoffs']['laboratory']:.3f}")
        with col_s3:
            st.metric("Hostel Strategy", sol["strategy_names"]["hostel"])
            st.metric("Hostel Payoff Utility", f"{sol['payoffs']['hostel']:.3f}")
            
        # Social welfare
        st.metric("Social Welfare (Sum Utility)", f"{sol['social_welfare']:.3f}")
        
        # Heatmap of payoffs (Slice for Hostel Strategy)
        st.markdown("### Payoff Matrix Slice (Classroom vs Laboratory Strategy)")
        hostel_strat_idx = sol["strategy_profile"][2]
        
        # Matrix slice for hostel_strat_idx
        slice_pm = pm[:, :, hostel_strat_idx, :]
        
        # Compute average utility of agents for heatmap visualization
        avg_slice_utility = np.mean(slice_pm, axis=2)
        
        fig_heat = px.imshow(
            avg_slice_utility,
            labels=dict(x="Laboratory Strategy", y="Classroom Strategy", color="Avg Utility"),
            x=["Conservative (0.7x)", "Normal (1.0x)", "Aggressive (1.3x)"],
            y=["Conservative (0.7x)", "Normal (1.0x)", "Aggressive (1.3x)"],
            color_continuous_scale="Viridis"
        )
        fig_heat.update_layout(template="plotly_dark", height=350)
        st.plotly_chart(fig_heat, use_container_width=True)

# ----------------------------------------------------
# TAB 5: SUSTAINABILITY DASHBOARD
# ----------------------------------------------------
with tab_sustainability:
    st.subheader("Microgrid Sustainability Analytics & KPIs")
    
    if "sim_history" in st.session_state:
        df_hist = st.session_state["sim_history"]
        cum_metrics = calculate_cumulative_metrics(df_hist)
        
        # Gauge indicators for sustainability score
        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number",
            value=cum_metrics["average_sustainability_score"] * 100,
            title={'text': "Campus Sustainability Index Score"},
            gauge={
                'axis': {'range': [0, 100]},
                'bar': {'color': "#2ecc71"},
                'steps': [
                    {'range': [0, 50], 'color': "#c0392b"},
                    {'range': [50, 80], 'color': "#f1c40f"},
                    {'range': [80, 100], 'color': "#27ae60"}
                ]
            }
        ))
        fig_gauge.update_layout(template="plotly_dark", height=300)
        st.plotly_chart(fig_gauge, use_container_width=True)
        
        # Metric plots: Cumulative Energy Saved & Cost Savings over time
        st.markdown("### Cumulative Microgrid Efficiency Savings")
        df_hist["cum_saved_kwh"] = df_hist["total_energy_saved"].cumsum()
        df_hist["cum_cost_saved_usd"] = df_hist["cost_saved_usd"].cumsum()
        
        fig_cum = go.Figure()
        fig_cum.add_trace(go.Scatter(x=df_hist["step"], y=df_hist["cum_saved_kwh"], name="Cumulative Energy Saved (kWh)", line=dict(color="#2ecc71")))
        fig_cum.add_trace(go.Scatter(x=df_hist["step"], y=df_hist["cum_cost_saved_usd"], name="Cumulative Financial Savings ($)", yaxis="y2", line=dict(color="#f1c40f")))
        
        fig_cum.update_layout(
            template="plotly_dark",
            xaxis_title="Simulation Hours",
            yaxis=dict(title="Energy Savings (kWh)", titlefont=dict(color="#2ecc71"), tickfont=dict(color="#2ecc71")),
            yaxis2=dict(title="Financial Savings ($)", titlefont=dict(color="#f1c40f"), tickfont=dict(color="#f1c40f"), overlaying="y", side="right"),
            height=400
        )
        st.plotly_chart(fig_cum, use_container_width=True)
        
        # Satisfaction KPI breakdown per agent
        st.markdown("### Agent-by-Agent Comfort and Satisfaction")
        
        # Extract satisfactions
        classroom_sat = df_hist["classroom_occupancy"] # placeholder or actual
        # Let's plot actual allocation fractions (alloc / request)
        c_sat = (df_hist["alloc_classroom"] / df_hist["req_classroom"]).fillna(1.0).rolling(24).mean()
        l_sat = (df_hist["alloc_laboratory"] / df_hist["req_laboratory"]).fillna(1.0).rolling(24).mean()
        h_sat = (df_hist["alloc_hostel"] / df_hist["req_hostel"]).fillna(1.0).rolling(24).mean()
        
        fig_sat = go.Figure()
        fig_sat.add_trace(go.Scatter(x=df_hist["step"], y=c_sat, name="Classroom Comfort", line=dict(color="#e74c3c")))
        fig_sat.add_trace(go.Scatter(x=df_hist["step"], y=l_sat, name="Laboratory Comfort", line=dict(color="#f1c40f")))
        fig_sat.add_trace(go.Scatter(x=df_hist["step"], y=h_sat, name="Hostel Comfort", line=dict(color="#2ecc71")))
        
        fig_sat.update_layout(
            template="plotly_dark",
            xaxis_title="Simulation Hours (24h Rolling Mean)",
            yaxis_title="Comfort Index (0 - 1.0)",
            height=400,
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        st.plotly_chart(fig_sat, use_container_width=True)
    else:
        st.warning("Please run a simulation on the 'Live Simulation' page to see sustainability reports.")
