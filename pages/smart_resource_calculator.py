import streamlit as st
from pathlib import Path
import config.environment as env
from scripts.smart_resource_calculator_train import predict_optimal_resources
from dotenv import load_dotenv
load_dotenv()

cols4 = st.columns([0.8, 10])

with cols4[0]:
    st.image(image = str(Path(env.ASSETS_DIR, "smart_resource_calculator.png")), width = 75, use_container_width = True)
with cols4[1]:
    st.header("Smart Resource Calculator", divider = "red", anchor = False)

cols1 = st.columns(3)

with cols1[0]:
    new_table_count = st.number_input("Number of Tables:", step = 1)
    new_cluster_health = st.selectbox("Cluster Health:", ['healthy', 'degraded', 'critical'])
with cols1[1]:
    new_data_volume = st.number_input("Data Volume (GB):", step = 1)
    new_job_complexity = st.selectbox("Job Complexity:", ['high', 'medium', 'low'])
with cols1[2]:
    new_expected_runtime = st.number_input("Expected Runtime (in mins):", step = 1)
    historical_runtime_input = st.number_input("Enter historical runtime (in mins): ", step = 1)


# Optional historical runtime input

if historical_runtime_input is None:
    new_historical_runtime = None
else:
    new_historical_runtime = float(historical_runtime_input)

cols2 = st.columns([10,1])
with cols2[0]:
    st.divider()
with cols2[1]:
    submit = st.button("Submit")

def predict():
    predicted_resources = predict_optimal_resources(new_table_count, new_cluster_health, new_data_volume, new_job_complexity, new_expected_runtime, new_historical_runtime)

    st.header("Recommended resource configuration for the given workload:", anchor = False)
    
    with st.container(border = True):
        cols3 = st.columns(2)
        cols3[0].subheader(f"Number of Executors: **{predicted_resources['executors']}**", anchor = False)
        cols3[0].subheader(f"Cores per Executor: **{predicted_resources['cores_per_executor']}**", anchor = False)
        cols3[1].subheader(f"Executor Memory: **{predicted_resources['executor_memory']} GB**", anchor = False)
        cols3[1].subheader(f"Driver Memory: **{predicted_resources['driver_memory']} GB**", anchor = False)

if submit:
    predict()


    