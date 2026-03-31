import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl
import streamlit as st
import matplotlib.pyplot as plt
import pandas as pd
import random
import time

# ==========================================
# 1. FUZZY LOGIC CORE (Modular Setup)
# ==========================================
@st.cache_resource
def build_fuzzy_system():
    # Define Universes (Ranges)
    traffic = ctrl.Antecedent(np.arange(0, 51, 1), 'traffic_density')
    waiting = ctrl.Antecedent(np.arange(0, 61, 1), 'waiting_time')
    green = ctrl.Consequent(np.arange(0, 61, 1), 'green_time')

    # Define Membership Functions (Triangular)
    traffic['low'] = fuzz.trimf(traffic.universe, [0, 0, 25])
    traffic['medium'] = fuzz.trimf(traffic.universe, [10, 25, 40])
    traffic['high'] = fuzz.trimf(traffic.universe, [30, 50, 50])

    waiting['short'] = fuzz.trimf(waiting.universe, [0, 0, 25])
    waiting['medium'] = fuzz.trimf(waiting.universe, [15, 30, 45])
    waiting['long'] = fuzz.trimf(waiting.universe, [35, 60, 60])

    green['short'] = fuzz.trimf(green.universe, [0, 0, 25])
    green['medium'] = fuzz.trimf(green.universe, [15, 30, 45])
    green['long'] = fuzz.trimf(green.universe, [35, 60, 60])

    # Refined Rule Base
    rule1 = ctrl.Rule(traffic['high'] | waiting['long'], green['long'])
    rule2 = ctrl.Rule(traffic['medium'] & waiting['medium'], green['medium'])
    rule3 = ctrl.Rule(traffic['low'] & waiting['short'], green['short'])
    rule4 = ctrl.Rule(traffic['low'] & waiting['medium'], green['medium'])
    rule5 = ctrl.Rule(traffic['medium'] & waiting['short'], green['short'])

    # Build and Return Control System
    system = ctrl.ControlSystem([rule1, rule2, rule3, rule4, rule5])
    return ctrl.ControlSystemSimulation(system)

fuzzy_sim = build_fuzzy_system()

# ==========================================
# 2. STREAMLIT GUI & VISUALIZATION
# ==========================================
st.set_page_config(page_title="Smart Traffic AI", layout="wide")
st.title("🚦 Simulation-Based Smart Traffic Signal Control")
st.markdown("**Powered by Fuzzy Logic (scikit-fuzzy)**")

# Create Tabs for different features
tab1, tab2 = st.tabs(["🎛️ Manual Override Tester", "🔄 Automated Random Simulation"])

# --- TAB 1: MANUAL TESTER ---
with tab1:
    st.header("Test Specific Scenarios")
    col1, col2 = st.columns(2)
    
    with col1:
        input_traffic = st.slider("Traffic Density (Vehicles)", 0, 50, 25)
        input_wait = st.slider("Waiting Time (Seconds)", 0, 60, 30)
        
    with col2:
        # Compute Fuzzy Logic
        fuzzy_sim.input['traffic_density'] = input_traffic
        fuzzy_sim.input['waiting_time'] = input_wait
        fuzzy_sim.compute()
        output_green = fuzzy_sim.output['green_time']
        
        st.metric(label="Calculated Green Signal Time", value=f"{output_green:.2f} seconds")
        if output_green < 20:
            st.success("Short Green Light - Normal Flow")
        elif output_green < 40:
            st.warning("Medium Green Light - Moderate Congestion")
        else:
            st.error("Long Green Light - Heavy Traffic Clearing!")

# --- TAB 2: AUTOMATED SIMULATION & GRAPHS ---
with tab2:
    st.header("Simulate Multi-Cycle Traffic")
    cycles = st.slider("Number of signal cycles to simulate:", 5, 50, 15)
    
    if st.button("Run Simulation"):
        progress_bar = st.progress(0)
        
        sim_data = []
        for i in range(cycles):
            # Generate random simulation data
            t_density = random.randint(0, 50)
            w_time = random.randint(0, 60)
            
            # Process through Fuzzy Logic
            fuzzy_sim.input['traffic_density'] = t_density
            fuzzy_sim.input['waiting_time'] = w_time
            fuzzy_sim.compute()
            g_time = fuzzy_sim.output['green_time']
            
            sim_data.append({
                "Cycle": i + 1,
                "Traffic Density": t_density,
                "Waiting Time (s)": w_time,
                "Green Time (s)": round(g_time, 2)
            })
            progress_bar.progress((i + 1) / cycles)
            time.sleep(0.05) # Simulate processing time
            
        df = pd.DataFrame(sim_data)
        
        # Display Logs
        st.subheader("Simulation Logs")
        st.dataframe(df, use_container_width=True)
        
        # Display Visualizations
        st.subheader("System Performance Graphs")
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(df["Cycle"], df["Traffic Density"], label="Traffic Density", marker='o', color='red')
        ax.plot(df["Cycle"], df["Green Time (s)"], label="Green Time Given", marker='s', color='green')
        ax.set_xlabel("Signal Cycle")
        ax.set_ylabel("Value")
        ax.set_title("Dynamic Adjustment of Green Time vs Traffic Density")
        ax.legend()
        ax.grid(True)
        
        st.pyplot(fig)