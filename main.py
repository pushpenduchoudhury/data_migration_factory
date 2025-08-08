import uuid
import streamlit as st
from pathlib import Path
import config.environment as env
from dotenv import load_dotenv
load_dotenv()


# Set Page Config
st.set_page_config(
    page_title = "Data Migration Factory",
    page_icon = "☁️",
    layout = "wide",
    initial_sidebar_state = "expanded"
)

# Initialize chat session in streamlit
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())
    
st.markdown(
    f"""
<style>
    .st-emotion-cache-595tnf:before {{
        content: "Migration Factory";
        font-weight: bold;
        font-size: xx-large;
    }}
</style>""",
        unsafe_allow_html=True,
    )


pages = {
    "Home": [
        st.Page(Path(env.PAGES_DIR, "app.py"), title = "Apps", icon = ":material/home:", default = True),
    ],
    "Script Analysis and Conversion": [
        st.Page(Path(env.PAGES_DIR, "code_analyzer.py"), title = "Code Analyzer ⏳🚧", icon = ":material/code_blocks:"),
        st.Page(Path(env.PAGES_DIR, "script_migrator.py"), title = "Script Migrator ⏳🚧", icon = ":material/move_group:"),
        st.Page(Path(env.PAGES_DIR, "script_validator.py"), title = "Script Validator ⏳🚧", icon = ":material/list_alt_check:"),
    ],
    "Data Migration": [
        st.Page(Path(env.PAGES_DIR, "data_migrator.py"), title = "Data Migrator ⏳🚧", icon = ":material/database_upload:"),
        st.Page(Path(env.PAGES_DIR, "data_anomaly_detector.py"), title = "Data Anomaly Detector ⏳🚧", icon = ":material/frame_inspect:"),
    ],
    "Operations": [
        st.Page(Path(env.PAGES_DIR, "opsmate.py"), title = "OpsMate Assistant", icon = ":material/smart_toy:"),
        st.Page(Path(env.PAGES_DIR, "chat_with_document.py"), title = "Chat with Document", icon = ":material/auto_stories:"),
    ],
    "Administration": [
        st.Page(Path(env.PAGES_DIR, "smart_resource_calculator.py"), title = "Smart Resource Calculator", icon = ":material/calculate:"),
    ],
}

page = st.navigation(pages, position = "sidebar", expanded = True)
page.run()