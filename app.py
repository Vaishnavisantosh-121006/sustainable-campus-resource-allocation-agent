"""
Streamlit Web Dashboard for the Smart Grid Energy Allocation AI Agent.
Provides an interactive GUI to adjust building load demands and visualize game-theoretic outcomes.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import streamlit as st
import pandas as pd
from src.models import Zone
from src.game_theory import GameTheoryAllocator
from src.agent import WasteCollectionAgent  # Class remains named WasteCollectionAgent for compatibility
from src.environment import Environment

# Page configuration
st.set_page_config(
    page_title="AI Smart Grid Energy Allocator",
    page_icon="⚡",
    layout="wide"
)

# Custom Styling
st.markdown("""
    <style>
    .main-title {
        font-size: 38px;
        font-weight: bold;
        color: #0b3c5d;
        text-align: center;
        margin-bottom: 5px;
    }
    .subtitle {
        font-size: 18px;
        color: #555;
        text-align: center;
        margin-bottom: 25px;
    }
    .section-card {
        background-color: #f0f4f8;
        padding: 20px;
        border-radius: 10px;
        border-left: 5px solid #0b3c5d;
        margin-bottom: 20px;
    }
    .metric-value {
        font-size: 24px;
        font-weight: bold;
        color: #0b3c5d;
    }
    </style>
""", unsafe_allow_html=True)

st.markdown("<div class='main-title'>⚡ Smart Grid Energy Resource AI Agent</div>", unsafe_allow_html=True)
st.markdown("<div class='subtitle'>Game Theory & Microgrid Load Allocator (SDG 7 & 11)</div>", unsafe_allow_html=True)

# Sidebar - Grid Settings
st.sidebar.header("⚡ Grid Capacity Controls")
total_capacity = st.sidebar.slider("Available Grid Supply (MW)", min_value=1, max_value=5, value=3)

st.sidebar.markdown("---")
st.sidebar.markdown("### 🌍 SDG Alignment")
st.sidebar.info("**SDG 7:** Affordable and Clean Energy\n\n**SDG 11:** Sustainable Cities and Communities")

# Main Page - Interactive Zone Controls
st.markdown("### 📊 Customize Building Load Demands (Current State)")
cols = st.columns(5)

zones_data = [
    {"id": "A", "pop": 1200, "demand": 85, "priority": 5},
    {"id": "B", "pop": 900, "demand": 40, "priority": 2},
    {"id": "C", "pop": 1500, "demand": 95, "priority": 5},
    {"id": "D", "pop": 700, "demand": 60, "priority": 3},
    {"id": "E", "pop": 1100, "demand": 75, "priority": 4}
]

updated_zones = []

for i, z in enumerate(zones_data):
    with cols[i]:
        st.markdown(f"#### 🏢 Zone {z['id']}")
        demand = st.slider(f"Load Demand (%)", min_value=0, max_value=100, value=z['demand'], key=f"d_{z['id']}")
        priority = st.slider(f"Priority (1-5)", min_value=1, max_value=5, value=z['priority'], key=f"p_{z['id']}")
        pop = st.number_input(f"Occupancy", min_value=100, max_value=5000, value=z['pop'], step=100, key=f"pop_{z['id']}")
        
        updated_zones.append(Zone(z['id'], int(pop), int(demand), int(priority)))

# Execution Button
if st.button("🚀 Run AI Agent Energy Optimizer", type="primary"):
    
    # Initialize environment
    env = Environment(total_capacity=total_capacity)
    env.zones = updated_zones
    
    # Create Allocator & Agent
    allocator = GameTheoryAllocator(updated_zones, total_capacity)
    
    st.markdown("---")
    
    res_col1, res_col2 = st.columns([1, 1])
    
    with res_col1:
        st.markdown("### 🧠 Agent Cognitive Lifecycle Trace")
        
        # 1. Perception
        st.subheader("[1] Perception Stage")
        st.write("Observing building loads and grid capacity bounds...")
        perc_df = pd.DataFrame([{
            "Zone": zone.zone_id,
            "Demand Level (%)": zone.demand_level,
            "Priority": zone.priority,
            "Occupancy": zone.population
        } for zone in updated_zones])
        st.dataframe(perc_df, use_container_width=True, hide_index=True)
        
        # 2. Reasoning
        st.subheader("[2] Reasoning Stage")
        st.write("Calculating electrical utility indices...")
        util_data = {"Building Zone": [], "Utility Score": []}
        for zone in updated_zones:
            util_data["Building Zone"].append(f"Zone {zone.zone_id}")
            util_data["Utility Score"].append(int(round(zone.utility())))
        
        util_df = pd.DataFrame(util_data)
        st.bar_chart(util_df.set_index("Building Zone"), color="#0b3c5d")
        
    with res_col2:
        # 3. Decision-making
        st.subheader("[3] Decision-Making Stage (Game Solver)")
        st.write("Solving for the Pure Strategy Nash Equilibrium (PSNE)...")
        
        eq_profile = allocator.find_nash_equilibrium()
        allocations = allocator.allocate_resources(eq_profile)
        
        strat_cols = st.columns(5)
        for idx, zone in enumerate(updated_zones):
            with strat_cols[idx]:
                strat = eq_profile[zone.zone_id]
                color = "blue" if strat == "REQUEST" else "grey"
                st.markdown(f"**Zone {zone.zone_id}**\n\n<span style='color:{color};font-weight:bold;'>{strat}</span>", unsafe_allow_html=True)
        
        # 4. Action
        st.subheader("[4] Action Stage")
        st.write("Grid Power Blocks Routed:")
        
        allocated_zones = [z_id for z_id, count in allocations.items() if count > 0]
        allocated_zones.sort(key=lambda z_id: env.get_zone_by_id(z_id).utility(), reverse=True)
        
        mw_num = 1
        for z_id in allocated_zones:
            count = allocations[z_id]
            for _ in range(count):
                st.info(f"⚡ **1 MW Block {mw_num}** successfully routed to **Zone {z_id}**")
                mw_num += 1
        
        if mw_num == 1:
            st.warning("No power allocated (all buildings chose local storage/delay).")

    # Payoff Matrix Analysis Section
    st.markdown("### 📊 Grid Payoff Matrix Analysis")
    st.write("Payoffs computed for different profile decisions to demonstrate stability:")
    
    profiles = [
        {"Name": "Local Battery Usage (All Delay)", "profile": {z.zone_id: "DELAY" for z in updated_zones}},
        {"Name": "Nash Equilibrium (Optimized)", "profile": eq_profile},
        {"Name": "Surge Competition (All Request)", "profile": {z.zone_id: "REQUEST" for z in updated_zones}}
    ]
    
    payoff_rows = []
    for p in profiles:
        payoffs = allocator.calculate_payoffs(p["profile"])
        row = {"Strategic Choice": p["Name"]}
        for zone in updated_zones:
            row[f"Zone {zone.zone_id} Payoff"] = int(round(payoffs[zone.zone_id]))
        payoff_rows.append(row)
        
    payoff_df = pd.DataFrame(payoff_rows)
    st.dataframe(payoff_df, use_container_width=True, hide_index=True)
    
    st.info("💡 **Nash Equilibrium Insight:** The zones with lower demand/priority choose to DELAY to use battery backups, because attempting to REQUEST and failing results in a severe blackout penalty. Thus, (Request, Delay, Request, Delay, Request) is the only stable state.")
