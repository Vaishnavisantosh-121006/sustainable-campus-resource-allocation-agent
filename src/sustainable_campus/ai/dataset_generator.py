import os
import numpy as np
import pandas as pd
from loguru import logger

def generate_campus_dataset(file_path: str) -> pd.DataFrame:
    """
    Generates a realistic synthetic campus energy demand dataset with 8760 hourly steps (365 days).
    Features: day_of_week, hour, temperature, occupancy, is_holiday, is_exam_period, prev_demand.
    Targets: classroom_demand, laboratory_demand, hostel_demand.
    """
    logger.info(f"Generating synthetic dataset at {file_path}")
    
    # Ensure directory exists
    dir_name = os.path.dirname(file_path)
    if dir_name and not os.path.exists(dir_name):
        os.makedirs(dir_name)

    np.random.seed(42)
    n_hours = 8760
    
    # Baseline temporal features
    timestamps = pd.date_range(start="2026-01-01 00:00:00", periods=n_hours, freq="h")
    hours = timestamps.hour
    day_of_weeks = timestamps.dayofweek
    day_of_years = timestamps.dayofyear
    
    # Define holidays (summer break: days 150-240, winter break: days 350-365, plus weekends)
    is_holiday = []
    for day, dow in zip(day_of_years, day_of_weeks):
        is_break = (150 <= day <= 210) or (350 <= day <= 365)  # Summer & Winter break
        is_we = dow >= 5
        is_holiday.append(1 if (is_break or is_we) else 0)
    
    is_holiday = np.array(is_holiday)

    # Exam periods (days 120-135, days 320-335)
    is_exam_period = np.array([
        1 if ((120 <= day <= 135) or (320 <= day <= 335)) else 0
        for day in day_of_years
    ])

    # Temperature model (sinusoidal seasonal & daily variation)
    base_temp = 20.0 + 10.0 * np.sin(2 * np.pi * (day_of_years - 100) / 365.0)
    daily_temp = 5.0 * np.sin(2 * np.pi * (hours - 8) / 24.0)
    noise_temp = np.random.normal(0, 1.5, n_hours)
    temperatures = base_temp + daily_temp + noise_temp

    # Occupancies
    classroom_occupancies = []
    lab_occupancies = []
    hostel_occupancies = []

    for hr, hol, exam in zip(hours, is_holiday, is_exam_period):
        if hol == 1:
            # Holiday occupancy
            c_occ = int(np.random.poisson(10))  # empty classroom
            l_occ = int(np.random.poisson(15))  # few researchers
            h_occ = int(np.random.poisson(1000)) # hostel full
        else:
            # Weekday occupancy
            if 8 <= hr < 17:
                c_occ = int(np.random.poisson(280)) if not exam else int(np.random.poisson(320))
                l_occ = int(np.random.poisson(40))
                h_occ = int(np.random.poisson(200))
            elif 17 <= hr < 22:
                c_occ = int(np.random.poisson(50))
                l_occ = int(np.random.poisson(35))
                h_occ = int(np.random.poisson(800))
            else:
                c_occ = int(np.random.poisson(5))
                l_occ = int(np.random.poisson(10))
                h_occ = int(np.random.poisson(1100))
        
        classroom_occupancies.append(c_occ)
        lab_occupancies.append(l_occ)
        hostel_occupancies.append(h_occ)

    classroom_occupancies = np.array(classroom_occupancies)
    lab_occupancies = np.array(lab_occupancies)
    hostel_occupancies = np.array(hostel_occupancies)

    # Calculate actual demand targets using formulas + noise
    classroom_demands = []
    lab_demands = []
    hostel_demands = []
    
    # Running values for autoregressive features
    prev_c_dem = 80.0
    prev_l_dem = 50.0
    prev_h_dem = 150.0

    for i in range(n_hours):
        hr = hours[i]
        temp = temperatures[i]
        hol = is_holiday[i]
        exam = is_exam_period[i]
        c_occ = classroom_occupancies[i]
        l_occ = lab_occupancies[i]
        h_occ = hostel_occupancies[i]

        # Classroom Demand Formula
        temp_delta = abs(temp - 22.0)
        c_hvac = 40.0 + (temp_delta * 3.5)
        c_load = c_occ * 0.15 if (8 <= hr < 18 and hol == 0) else 5.0
        c_dem = (c_hvac + c_load) * (1.2 if exam else 1.0)
        c_dem = 0.7 * c_dem + 0.2 * prev_c_dem + np.random.normal(0, 5.0)
        c_dem = max(5.0, c_dem)
        classroom_demands.append(c_dem)
        prev_c_dem = c_dem

        # Laboratory Demand Formula
        active_machines = min(50, max(5, int(l_occ * 0.8)))
        l_load = 30.0 + active_machines * 1.2
        l_mult = 1.3 if exam else 1.0
        if not (8 <= hr < 22):
            l_mult *= 0.6
        l_dem = l_load * l_mult
        l_dem = 0.8 * l_dem + 0.15 * prev_l_dem + np.random.normal(0, 4.0)
        l_dem = max(10.0, l_dem)
        lab_demands.append(l_dem)
        prev_l_dem = l_dem

        # Hostel Demand Formula
        is_peak = (6 <= hr < 9) or (17 <= hr < 23)
        peak_mult = 1.8 if is_peak else 0.7
        h_load = h_occ * 0.2 * peak_mult * (1.3 if hol else 1.0)
        if exam and (23 <= hr or hr < 2):
            h_load *= 1.4
        h_hvac = 60.0 + (abs(temp - 21.0) * 3.0)
        h_dem = h_load + h_hvac
        h_dem = 0.75 * h_dem + 0.2 * prev_h_dem + np.random.normal(0, 10.0)
        h_dem = max(15.0, h_dem)
        hostel_demands.append(h_dem)
        prev_h_dem = h_dem

    df = pd.DataFrame({
        "timestamp": timestamps,
        "day_of_week": day_of_weeks,
        "hour": hours,
        "temperature": temperatures,
        "is_holiday": is_holiday,
        "is_exam_period": is_exam_period,
        
        "classroom_occupancy": classroom_occupancies,
        "classroom_prev_demand": [80.0] + classroom_demands[:-1],
        "classroom_demand": classroom_demands,
        
        "laboratory_occupancy": lab_occupancies,
        "laboratory_prev_demand": [50.0] + lab_demands[:-1],
        "laboratory_demand": lab_demands,
        
        "hostel_occupancy": hostel_occupancies,
        "hostel_prev_demand": [150.0] + hostel_demands[:-1],
        "hostel_demand": hostel_demands,
    })

    df.to_csv(file_path, index=False)
    logger.info(f"Dataset generated successfully with {n_hours} records.")
    return df

if __name__ == "__main__":
    generate_campus_dataset("data/campus_energy_demand.csv")
