import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as ob
import os
from loguru import logger

from src.config import config
from src.data_generator import generate_synthetic_data
from src.prediction_model import train_and_evaluate_models, DemandPredictor
from src.game_theory import CampusGameSolver
from src.rl import CampusEnergyEnv, train_q_learning
from src.simulation import CampusSimulation

# Set page configuration
st.set_page_config(
    page_title="Sustainable Campus Resource Allocation",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling CSS for premium feel
st.markdown("""
<style>
    .main {
        background-color: #0d1117;
        color: #c9d1d9;
        font-family: 'Inter', sans-serif;
    }
    .stButton>button {
        background: linear-gradient(135deg, #1f6feb 0%, #115293 100%);
        color: white;
        border-radius: 8px;
        border: none;
        padding: 8px 16px;
        font-weight: 600;
        transition: all 0.3s ease;
    }
    .stButton>button:hover {
        background: linear-gradient(135deg, #388bfd 0%, #1f6feb 100%);
        transform: translateY(-2px);
        box-shadow: 0px 4px 15px rgba(31, 111, 235, 0.4);
    }
    .metric-card {
        background-color: #161b22;
        border: 1px solid #30363d;
        border-radius: 12px;
        padding: 20px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        text-align: center;
        transition: all 0.3s ease;
    }
    .metric-card:hover {
        border-color: #58a6ff;
        transform: translateY(-4px);
    }
    .kpi-value {
        font-size: 2rem;
        font-weight: 700;
        color: #58a6ff;
        margin-bottom: 5px;
    }
    .kpi-title {
        font-size: 0.9rem;
        color: #8b949e;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    h1, h2, h3 {
        color: #58a6ff;
        font-weight: 700;
    }
</style>
""", unsafe_allow_html=True)

# Ensure data and models directory exist and models are trained
DATA_PATH = "data/campus_demand.csv"
MODELS_DIR = "models"
POLICY_PATH = "models/q_table.json"

@st.cache_resource
def initialize_system():
    """Initializes synthetic dataset and trains ML models and RL policies on startup."""
    with st.spinner("Initializing system and training models. Please wait..."):
        # 1. Generate synthetic dataset
        if not os.path.exists(DATA_PATH):
            generate_synthetic_data(days=180, output_path=DATA_PATH)
            
        # 2. Train forecasting models
        train_and_evaluate_models(data_path=DATA_PATH, models_dir=MODELS_DIR)
        
        # 3. Train RL Q-table
        if not os.path.exists(POLICY_PATH):
            env = CampusEnergyEnv()
            train_q_learning(env, episodes=500, save_path=POLICY_PATH)
            
    st.success("System ready! ML demand models and RL allocation policies loaded successfully.")

# Initialize
initialize_system()

# Sidebar navigation
st.sidebar.image("https://img.icons8.com/color/96/energy-preservation.png", width=80)
st.sidebar.title("Campus Energy Admin")
st.sidebar.markdown("---")

page = st.sidebar.radio(
    "Navigate Menu",
    ["🏠 Home Overview", "⚡ Live Simulation", "🔮 AI Demand Forecasts", "🎮 Game Theory", "🌱 Sustainability KPIs"]
)

# Sidebar Weight adjustment controls
st.sidebar.markdown("### Utility Weights (Game Theory)")
w_satisfaction = st.sidebar.slider("Agent Satisfaction Weight (w1)", 0.0, 1.0, 0.5, 0.05)
w_sustainability = st.sidebar.slider("Sustainability Weight (w2)", 0.0, 1.0, 0.3, 0.05)
w_cost = st.sidebar.slider("Cost Penalty Weight (w3)", 0.0, 1.0, 0.2, 0.05)

# Validate weights sum to 1
weight_sum = w_satisfaction + w_sustainability + w_cost
if abs(weight_sum - 1.0) > 1e-5:
    st.sidebar.warning(f"Weights sum is {weight_sum:.2f}. We will normalize them automatically.")
    total = w_satisfaction + w_sustainability + w_cost
    w_satisfaction_norm = w_satisfaction / total
    w_sustainability_norm = w_sustainability / total
    w_cost_norm = w_cost / total
else:
    w_satisfaction_norm = w_satisfaction
    w_sustainability_norm = w_sustainability
    w_cost_norm = w_cost

# Apply to config
config.game.w_satisfaction = w_satisfaction_norm
config.game.w_sustainability = w_sustainability_norm
config.game.w_cost = w_cost_norm

st.sidebar.info(f"Normalized Weights:\n- Satisfaction: {w_satisfaction_norm:.2f}\n- Sustainability: {w_sustainability_norm:.2f}\n- Cost: {w_cost_norm:.2f}")

# ----------------- HOME PAGE -----------------
if page == "🏠 Home Overview":
    st.title("Sustainable Campus Resource Allocation Agent")
    st.markdown("""
    This project is an autonomous multi-agent AI system designed to intelligently distribute limited campus energy resources (solar and grid electricity) among classrooms, laboratories, and hostels.
    By combining dynamic game-theoretic negotiation and reinforcement learning policies, the system optimizes satisfaction, minimizes wastage, and reduces carbon footprint.
    """)
    
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("SDG 7: Affordable and Clean Energy")
        st.markdown("""
        - **Green Integration**: Prioritizes solar energy utilization (30% baseline solar capacity ratio) before requesting fossil-fuel grid energy.
        - **Grid Curtailment**: Intelligently scales back allocations during shortages, protecting academic infrastructure while maximizing residential stability.
        """)
        
    with col2:
        st.subheader("SDG 12: Responsible Consumption and Production")
        st.markdown("""
        - **Demand Optimization**: Classroom, Laboratory, and Hostel agents predict local hourly loads, preventing over-allocation and wastage.
        - **Carbon Footprint Tracking**: Continuous computation of actual carbon intensity vs. an unoptimized baseline, tracking equivalent trees planted.
        """)
        
    st.subheader("System Architecture")
    st.markdown("""
    ```mermaid
    graph TD
        ClassroomAgent[Classroom Agent] -->|1. Request energy| EnergyManager[Energy Manager Agent]
        LaboratoryAgent[Laboratory Agent] -->|1. Request energy| EnergyManager
        HostelAgent[Hostel Agent] -->|1. Request energy| EnergyManager
        
        Predictor[AI Demand Predictor] -.->|Inference| ClassroomAgent
        Predictor -.->|Inference| LaboratoryAgent
        Predictor -.->|Inference| HostelAgent
        
        EnergyManager -->|2. Allocate energy| ClassroomAgent
        EnergyManager -->|2. Allocate energy| LaboratoryAgent
        EnergyManager -->|2. Allocate energy| HostelAgent
        
        GameTheory[Game Theory Solver] -.->|Solve Nash Equilibrium| EnergyManager
        RLPolicy[RL Q-Learning Policy] -.->|Optimize Action| EnergyManager
    ```
    """, unsafe_allow_html=True)
    st.info("Tip: Install a markdown renderer or use the other dashboard sections to see visual representations of allocations!")

# ----------------- LIVE SIMULATION PAGE -----------------
elif page == "⚡ Live Simulation":
    st.title("Live Multi-Agent Campus Simulation")
    
    col_ctrl1, col_ctrl2, col_ctrl3, col_ctrl4 = st.columns(4)
    with col_ctrl1:
        sim_duration = st.selectbox("Simulation Duration", [30, 100, 365], index=0)
    with col_ctrl2:
        strategy = st.selectbox("Allocation Strategy", ["pro_rata", "game_theory", "rl"], index=1)
    with col_ctrl3:
        solar_ratio = st.slider("Solar Capacity Ratio", 0.0, 1.0, 0.3, 0.05)
    with col_ctrl4:
        model_type = st.selectbox("ML Forecasting Model", ["random_forest", "linear_regression"], index=0)
        
    if st.button("🚀 Run Simulation"):
        sim = CampusSimulation(
            duration_days=sim_duration,
            strategy=strategy,
            solar_ratio=solar_ratio,
            model_type=model_type
        )
        with st.spinner("Simulating multi-agent environment..."):
            history, summary = sim.run()
            
        st.success(f"Simulation completed for {sim_duration} days using {strategy.upper()} strategy!")
        
        # Save history for other pages
        st.session_state["sim_history"] = history
        st.session_state["sim_summary"] = summary
        
    if "sim_history" in st.session_state:
        history = st.session_state["sim_history"]
        summary = st.session_state["sim_summary"]
        
        df_hist = pd.DataFrame(history)
        
        # Display KPI cards
        col_kpi1, col_kpi2, col_kpi3, col_kpi4 = st.columns(4)
        with col_kpi1:
            st.markdown(f"""
            <div class="metric-card">
                <div class="kpi-value">{summary['sustainability_score']:.1f}%</div>
                <div class="kpi-title">Sustainability Score</div>
            </div>
            """, unsafe_allow_html=True)
        with col_kpi2:
            st.markdown(f"""
            <div class="metric-card">
                <div class="kpi-value">{summary['average_satisfaction']*100:.1f}%</div>
                <div class="kpi-title">Avg Agent Satisfaction</div>
            </div>
            """, unsafe_allow_html=True)
        with col_kpi3:
            st.markdown(f"""
            <div class="metric-card">
                <div class="kpi-value">{summary['carbon_reduction_kg']:.1f} kg</div>
                <div class="kpi-title">CO2 Emissions Saved</div>
            </div>
            """, unsafe_allow_html=True)
        with col_kpi4:
            st.markdown(f"""
            <div class="metric-card">
                <div class="kpi-value">${summary['cost_saved_usd']:.2f}</div>
                <div class="kpi-title">Utility Savings</div>
            </div>
            """, unsafe_allow_html=True)
            
        st.markdown("---")
        
        # Visualizations
        st.subheader("Energy Allocations vs Requests (First 72 Hours)")
        df_72h = df_hist.head(72)
        
        fig_alloc = ob.Figure()
        colors = {"classroom": "#58a6ff", "laboratory": "#7ee787", "hostel": "#ff7b72"}
        for agent in ["classroom", "laboratory", "hostel"]:
            fig_alloc.add_trace(ob.Scatter(
                x=df_72h.index, y=df_72h[f"{agent}_request"],
                mode='lines', name=f"{agent.capitalize()} Request",
                line=dict(color=colors[agent], dash='dash')
            ))
            fig_alloc.add_trace(ob.Scatter(
                x=df_72h.index, y=df_72h[f"{agent}_allocation"],
                mode='lines', name=f"{agent.capitalize()} Allocation",
                fill='tozeroy' if agent == 'classroom' else None, # overlay style
                line=dict(color=colors[agent])
            ))
        fig_alloc.update_layout(
            plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
            font_color="#c9d1d9", xaxis_title="Hour Index", yaxis_title="Energy (kWh)"
        )
        st.plotly_chart(fig_alloc, use_container_width=True)
        
        # Satisfaction Trends
        st.subheader("Agent Satisfaction and Hostel Efficiency Trends")
        fig_sat = px.line(
            df_hist.head(168), # 1 week
            y=["classroom_satisfaction", "laboratory_satisfaction", "hostel_satisfaction", "hostel_efficiency"],
            title="Satisfaction & Efficiency Over 1 Week (Hourly)",
            color_discrete_map={
                "classroom_satisfaction": "#58a6ff",
                "laboratory_satisfaction": "#7ee787",
                "hostel_satisfaction": "#ff7b72",
                "hostel_efficiency": "#d2a8ff"
            }
        )
        fig_sat.update_layout(plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)", font_color="#c9d1d9")
        st.plotly_chart(fig_sat, use_container_width=True)
        
        # Display sample interaction logs
        st.subheader("Sample Agent Communication Logs (First 5 hours)")
        # Fabricate structured representation of routed logs
        log_records = []
        for i, row in df_hist.head(5).iterrows():
            hour_idx = f"Day {int(row['day'])}, Hr {int(row['hour'])}"
            log_records.append({"Timestamp": hour_idx, "Sender": "classroom", "Recipient": "energy_manager", "Type": "REQUEST", "Payload": f"requested_energy: {row['classroom_request']:.2f}"})
            log_records.append({"Timestamp": hour_idx, "Sender": "laboratory", "Recipient": "energy_manager", "Type": "REQUEST", "Payload": f"requested_energy: {row['laboratory_request']:.2f}"})
            log_records.append({"Timestamp": hour_idx, "Sender": "hostel", "Recipient": "energy_manager", "Type": "REQUEST", "Payload": f"requested_energy: {row['hostel_request']:.2f}"})
            log_records.append({"Timestamp": hour_idx, "Sender": "energy_manager", "Recipient": "classroom", "Type": "ALLOCATION", "Payload": f"allocated: {row['classroom_allocation']:.2f}"})
            log_records.append({"Timestamp": hour_idx, "Sender": "energy_manager", "Recipient": "laboratory", "Type": "ALLOCATION", "Payload": f"allocated: {row['laboratory_allocation']:.2f}"})
            log_records.append({"Timestamp": hour_idx, "Sender": "energy_manager", "Recipient": "hostel", "Type": "ALLOCATION", "Payload": f"allocated: {row['hostel_allocation']:.2f}"})
            
        st.table(pd.DataFrame(log_records))
    else:
        st.info("Click the 'Run Simulation' button above to generate a new live agent allocation run.")

# ----------------- AI FORECASTS PAGE -----------------
elif page == "🔮 AI Demand Forecasts":
    st.title("AI Energy Demand Forecasting")
    
    st.markdown("""
    Demand Forecasting models predict classroom, laboratory, and hostel electricity consumption. 
    Predictive features: hour, day of the week, occupancy, and historical consumption.
    """)
    
    # Train forecasting models interface
    if st.button("🔄 Train and Evaluate Forecasting Models"):
        with st.spinner("Retraining Linear Regression & Random Forest regressors..."):
            metrics = train_and_evaluate_models(data_path=DATA_PATH, models_dir=MODELS_DIR)
        st.success("Models retrained and stored successfully!")
        st.session_state["ml_metrics"] = metrics
        
    if "ml_metrics" not in st.session_state:
        # Load default metrics if exists
        try:
            st.session_state["ml_metrics"] = train_and_evaluate_models(data_path=DATA_PATH, models_dir=MODELS_DIR)
        except Exception as e:
            st.warning("Please click button to train model features.")
            
    if "ml_metrics" in st.session_state:
        metrics = st.session_state["ml_metrics"]
        
        # Display comparison
        st.subheader("Model Performance Evaluation ($R^2$ Metric)")
        agent_names = ["classroom", "laboratory", "hostel"]
        r2_lr = [metrics[a]["linear_regression"]["r2"] for a in agent_names]
        r2_rf = [metrics[a]["random_forest"]["r2"] for a in agent_names]
        
        df_perf = pd.DataFrame({
            "Agent": [a.capitalize() for a in agent_names] * 2,
            "R2 Score": r2_lr + r2_rf,
            "Model Type": ["Linear Regression"] * 3 + ["Random Forest"] * 3
        })
        
        fig_r2 = px.bar(
            df_perf, x="Agent", y="R2 Score", color="Model Type", barmode="group",
            color_discrete_map={"Linear Regression": "#1f6feb", "Random Forest": "#7ee787"},
            title="Random Forest vs. Linear Regression Accuracy"
        )
        fig_r2.update_layout(plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)", font_color="#c9d1d9")
        st.plotly_chart(fig_r2, use_container_width=True)
        
        # Detailed stats table
        st.subheader("Model Performance Details")
        stats = []
        for agent in agent_names:
            for model in ["linear_regression", "random_forest"]:
                m_info = metrics[agent][model]
                stats.append({
                    "Agent": agent.capitalize(),
                    "Model": model.replace("_", " ").capitalize(),
                    "MAE (kWh)": f"{m_info['mae']:.3f}",
                    "RMSE (kWh)": f"{m_info['rmse']:.3f}",
                    "R2": f"{m_info['r2']:.4f}"
                })
        st.table(pd.DataFrame(stats))
        
        # Feature importance
        st.subheader("Random Forest Feature Importances")
        # Take classroom as example
        feat_importances = metrics["classroom"]["random_forest"]["feature_importances"]
        df_feat = pd.DataFrame({
            "Feature": list(feat_importances.keys()),
            "Importance": list(feat_importances.values())
        }).sort_values(by="Importance", ascending=True)
        
        fig_feat = px.bar(
            df_feat, x="Importance", y="Feature", orientation="h",
            color_discrete_sequence=["#58a6ff"],
            title="Classroom Demand Model Feature Importances"
        )
        fig_feat.update_layout(plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)", font_color="#c9d1d9")
        st.plotly_chart(fig_feat, use_container_width=True)

# ----------------- GAME THEORY PAGE -----------------
elif page == "🎮 Game Theory":
    st.title("Game-Theoretic Negotiation Analysis")
    
    st.markdown("""
    Agents compete or cooperate to negotiate resource allocation during peak demand and grid shortages. 
    Here you can inspect the dynamically generated payoff matrix and calculate pure Nash Equilibria.
    """)
    
    col_gt1, col_gt2 = st.columns(2)
    with col_gt1:
        condition = st.selectbox("Select Condition", ["normal", "peak", "scarcity"])
        capacity = st.number_input("Grid Capacity (kWh)", min_value=10.0, max_value=2000.0, value=200.0)
    with col_gt2:
        class_req = st.slider("Classroom Base Request (kWh)", 10.0, 300.0, 50.0)
        lab_req = st.slider("Lab Base Request (kWh)", 10.0, 300.0, 100.0)
        hostel_req = st.slider("Hostel Base Request (kWh)", 10.0, 300.0, 120.0)
        
    requests = {"classroom": class_req, "laboratory": lab_req, "hostel": hostel_req}
    
    solver = CampusGameSolver(
        w_satisfaction=w_satisfaction_norm,
        w_sustainability=w_sustainability_norm,
        w_cost=w_cost_norm
    )
    
    game_data = solver.generate_payoff_matrix(requests, capacity, condition)
    equilibria = solver.find_pure_nash_equilibria(game_data)
    
    st.subheader("Nash Equilibria strategy profiles")
    if equilibria:
        for eq in equilibria:
            st.success(f"⚖️ Pure Strategy Nash Equilibrium profile: Classroom: **{eq[0]}**, Lab: **{eq[1]}**, Hostel: **{eq[2]}**")
    else:
        st.warning("⚠️ No Pure Strategy Nash Equilibrium profiles found! Falling back to Social Welfare maximization.")
        
    # Visualizing Payoffs for specific slice (Hostel playing Cooperative vs Selfish)
    st.subheader("Payoff Comparison Heatmap (Slice: Hostel plays Cooperative / CONSERVE)")
    matrix = game_data["matrix"]
    
    # Slice matrix when Hostel (dim 2) plays CONSERVE (0)
    # columns represent Lab (0/1), rows represent Classroom (0/1)
    # Cell holds Classroom & Lab payoffs
    matrix_slice = matrix[:, :, 0, :] # shape: (2, 2, 3)
    
    # Format for plotting
    data_heatmap = [
        [f"Class: {matrix_slice[0,0,0]:.2f}, Lab: {matrix_slice[0,0,1]:.2f}", f"Class: {matrix_slice[0,1,0]:.2f}, Lab: {matrix_slice[0,1,1]:.2f}"],
        [f"Class: {matrix_slice[1,0,0]:.2f}, Lab: {matrix_slice[1,0,1]:.2f}", f"Class: {matrix_slice[1,1,0]:.2f}, Lab: {matrix_slice[1,1,1]:.2f}"]
    ]
    
    df_heatmap = pd.DataFrame(
        data_heatmap,
        columns=["Lab CONSERVE", "Lab MAXIMIZE"],
        index=["Classroom CONSERVE", "Classroom MAXIMIZE"]
    )
    st.markdown("Cells display `(Classroom Payoff, Lab Payoff)` values:")
    st.table(df_heatmap)
    
    # Dynamic utility comparisons
    st.subheader("Utility Weights Impact on Solver Profile Selection")
    st.markdown("""
    By altering weights on the sidebar (Satisfaction, Sustainability, Cost), 
    the system changes its negotiation outcome dynamically.
    """)
    # Display weights contribution
    df_weights = pd.DataFrame({
        "Objective KPI": ["Satisfaction Weight (w1)", "Sustainability Weight (w2)", "Cost Weight (w3)"],
        "Weight Value": [w_satisfaction_norm, w_sustainability_norm, w_cost_norm]
    })
    fig_w = px.pie(df_weights, values="Weight Value", names="Objective KPI", color_discrete_sequence=px.colors.sequential.Agsunset)
    fig_w.update_layout(plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)", font_color="#c9d1d9")
    st.plotly_chart(fig_w, use_container_width=True)

# ----------------- SUSTAINABILITY PAGE -----------------
elif page == "🌱 Sustainability KPIs":
    st.title("Sustainability Analytics & SDGs Impact")
    
    if "sim_history" in st.session_state:
        summary = st.session_state["sim_summary"]
        history = st.session_state["sim_history"]
        df_hist = pd.DataFrame(history)
        
        col_s1, col_s2 = st.columns(2)
        with col_s1:
            st.subheader("Energy Source Distribution & Usage")
            # Solar vs grid energy
            total_allocated = summary["total_energy_consumed_kwh"]
            capacity_cap = sum(df_hist["capacity"])
            
            # Estimate solar contribution
            solar_contrib = sum([min(alloc, cap * solar_ratio) for alloc, cap in zip(df_hist["total_allocated"], df_hist["capacity"])])
            grid_contrib = max(0.0, total_allocated - solar_contrib)
            
            df_source = pd.DataFrame({
                "Energy Source": ["Clean Solar Energy", "Grid Electricity"],
                "Consumption (kWh)": [solar_contrib, grid_contrib]
            })
            
            fig_source = px.pie(
                df_source, values="Consumption (kWh)", names="Energy Source",
                color_discrete_sequence=["#7ee787", "#ff7b72"],
                title="Solar vs. Grid Generation Mix"
            )
            fig_source.update_layout(plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)", font_color="#c9d1d9")
            st.plotly_chart(fig_source, use_container_width=True)
            
        with col_s2:
            st.subheader("Cumulative Carbon Footprint Comparison")
            # Create cumulative curves
            df_hist["cumulative_actual_emissions"] = df_hist["actual_emissions_kg"].cumsum()
            # Approximate baseline emissions
            solar_cap = df_hist["capacity"] * solar_ratio
            baseline_sol = np.minimum(df_hist["total_requested"], solar_cap)
            baseline_grd = np.maximum(0.0, df_hist["total_requested"] - baseline_sol)
            baseline_em = (baseline_sol * config.sim.carbon_intensity_solar) + (baseline_grd * config.sim.carbon_intensity_grid)
            df_hist["cumulative_baseline_emissions"] = baseline_em.cumsum()
            
            fig_carbon = px.line(
                df_hist, y=["cumulative_actual_emissions", "cumulative_baseline_emissions"],
                title="Cumulative Carbon Footprint (kg CO2-eq)",
                labels={"value": "CO2 emissions (kg)", "index": "Hour"},
                color_discrete_map={
                    "cumulative_actual_emissions": "#7ee787",
                    "cumulative_baseline_emissions": "#ff7b72"
                }
            )
            fig_carbon.update_layout(plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)", font_color="#c9d1d9")
            st.plotly_chart(fig_carbon, use_container_width=True)
            
        st.subheader("Environmental Impact Dashboard")
        col_i1, col_i2, col_i3 = st.columns(3)
        with col_i1:
            trees = summary["carbon_reduction_kg"] / 21.8 # 1 mature tree absorbs ~21.8kg CO2 per year
            st.metric("Equivalent Trees Planted / Year Offset", f"{trees:.2f} trees", help="Calculated using mature tree annual absorption factor of 21.8 kg CO2")
        with col_i2:
            coal_saved = summary["carbon_reduction_kg"] / 0.9 # ~0.9kg CO2 per kWh coal
            st.metric("Coal Consumed Prevented", f"{coal_saved:.1f} kg", help="Based on 0.9 kg CO2 per kWh grid generation equivalent")
        with col_i3:
            alloc_eff = summary["average_efficiency"] * 100
            st.metric("Resource Allocation Efficiency", f"{alloc_eff:.1f}%")
            
    else:
        st.info("Run a simulation on the 'Live Simulation' page to see detailed sustainability analysis.")
