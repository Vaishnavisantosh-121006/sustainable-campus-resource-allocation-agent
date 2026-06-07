import pandas as pd
from typing import Dict, Any

def calculate_cumulative_metrics(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Calculates overall sustainability and operational KPIs from simulation history dataframe.
    """
    if df.empty:
        return {}

    total_consumed_kwh = df["total_energy_consumed"].sum()
    total_saved_kwh = df["total_energy_saved"].sum()
    total_carbon_saved_kg = df["carbon_reduction_kg"].sum()
    total_cost_usd = df["total_cost_usd"].sum()
    total_cost_saved_usd = df["cost_saved_usd"].sum()
    
    avg_satisfaction = df["average_satisfaction"].mean()
    avg_efficiency = df["average_efficiency"].mean()
    avg_sustainability_score = df["sustainability_score"].mean()

    # Convert to standard readable metrics
    total_consumed_mwh = total_consumed_kwh / 1000.0
    total_saved_mwh = total_saved_kwh / 1000.0
    total_carbon_saved_tons = total_carbon_saved_kg / 1000.0

    return {
        "total_energy_consumed_mwh": float(total_consumed_mwh),
        "total_energy_saved_mwh": float(total_saved_mwh),
        "total_carbon_saved_tons": float(total_carbon_saved_tons),
        "total_cost_usd": float(total_cost_usd),
        "total_cost_saved_usd": float(total_cost_saved_usd),
        "average_satisfaction": float(avg_satisfaction),
        "average_efficiency": float(avg_efficiency),
        "average_sustainability_score": float(avg_sustainability_score),
        "steps_simulated": len(df)
    }
