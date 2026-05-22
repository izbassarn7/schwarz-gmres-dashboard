# Entry point for Streamlit Cloud deployment
# Streamlit Cloud main file: app.py
# Local run:  streamlit run scripts/dashboard.py
import runpy, os, sys
sys.path.insert(0, os.path.dirname(__file__))
runpy.run_path(os.path.join(os.path.dirname(__file__), "scripts", "dashboard.py"),
               run_name="__main__")
