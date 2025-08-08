import math
import streamlit as st
from pathlib import Path
import config.environment as env
from dotenv import load_dotenv
load_dotenv()

CSS_FILE = Path(env.CSS_DIR, "main.css")
PAGES_DIR = env.PAGES_DIR

with open(CSS_FILE) as f:
    css = f.read()

st.markdown(f'<style>{css}</style>', unsafe_allow_html = True)

st.header("Data Migration Factory Apps", anchor = False, divider = "red")

apps = [
    {"name": "Code Analyzer",
     "description": "The Code Analyzer is a tool designed to analyze legacy SQL scripts, identifying performance bottlenecks, redundancies, and other potential issues. It plays a crucial role in database modernization projects by helping to improve the efficiency and maintainability of existing code before migration.",
     "page": "code_analyzer.py",
     "image_icon": "code_analyzer.png",
    },
    {"name": "Script Migrator",
     "description": "The Script Migrator application is designed to streamline the process of converting scripts between different database platforms or environments. Its core functionality centers on seamless conversion of scripts, specifically focusing on Data Manipulation Language (DML) statements such as INSERT, UPDATE, DELETE, and SELECT. This makes it a powerful tool for database modernization and migration projects.",
     "page": "script_migrator.py",
     "image_icon": "smart_resource_calculator.png",
    },
    {"name": "Script Validator",
     "description": "The Script Validator application is a quality assurance tool designed to analyze and assess the quality of database scripts, ensuring adherence to best practices and coding standards. It provides automated feedback, helping developers improve the efficiency, maintainability, and overall robustness of their code.",
     "page": "script_validator.py",
     "image_icon": "smart_resource_calculator.png",
    },
    {"name": "Data Migrator",
     "description": "The Data Migrator application facilitates seamless data transfer between diverse and disparate systems, regardless of their underlying technologies. It simplifies the often-complex process of data migration with its user-friendly setup, configuration, and data mapping capabilities.",
     "page": "data_migrator.py",
     "image_icon": "smart_resource_calculator.png",
    },
    {"name": "Data Anomaly Detector",
     "description": "The Data Anomaly Detector (DAD) is a system designed to identify discrepancies between source and target datasets, ensuring an apple-to-apple match between data and schema. Its core strength lies in its optimized RL-Hash Block Algorithm, which scales efficiently regardless of data volume.",
     "page": "data_anomaly_detector.py",
     "image_icon": "data_anomaly_detector.png",
    },
    {"name": "OpsMate Assistant",
     "description": "The AI-Driven Auto Triaging system is designed to automate and optimize the process of handling incoming requests and support tickets. It leverages artificial intelligence and machine learning to intelligently categorize, prioritize, and assign tickets, significantly improving response times and operational efficiency.",
     "page": "opsmate.py",
     "image_icon": "smart_resource_calculator.png",
    },
    {"name": "Chat with Document",
     "description": "Chat with a document to get answers to questions, summarize content, or extract information using RAG.",
     "page": "chat_with_document.py",
     "image_icon": "smart_resource_calculator.png",
    },
    {"name": "Smart Resource Calculator",
     "description": "The Smart Resource Calculator is an AI/ML-powered tool designed to optimize the allocation of computational and data processing resources. It intelligently recommends resources based on the demands of the task, ensuring efficient utilization and minimizing waste.",
     "page": "smart_resource_calculator.py",
     "image_icon": "smart_resource_calculator.png",
    },
]

no_of_apps = len(apps)
app_grid_cols = 4
app_grid_rows = math.ceil(no_of_apps/app_grid_cols)
tile_height = 285
image_width = 65


app_num = 0
for row in range(app_grid_rows):
    st_cols = st.columns(app_grid_cols)
    for col in range(app_grid_cols):
        if app_num > no_of_apps - 1:
            break
        with st_cols[col].container(border = True, height = tile_height):
            
            # Image
            st.image(image = str(Path(env.ASSETS_DIR, apps[app_num]["image_icon"])), width = image_width)
            
            # App Title
            st.subheader(apps[app_num]["name"], divider = "grey", anchor = False)
            
            # App Description
            desc_col = st.columns(1)
            with desc_col[0].container(border = False, height = int(0.18 * tile_height)):
                st.markdown(f'<span style="font-size: 16px; text-align: center;">{apps[app_num]["description"]}</span>', unsafe_allow_html = True)
            
            # App Launch Button
            if st.button("Launch", key = f"app_{app_num}"):
                st.switch_page(Path(PAGES_DIR, apps[app_num]["page"]))
                
            app_num += 1