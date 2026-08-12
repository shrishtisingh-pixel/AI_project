import streamlit as st
import pandas as pd
from datetime import datetime
import os

# Set page config at the very beginning
st.set_page_config(
    page_title="ShopFlow AI — Predictive & Grounded Chat Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load backend logic directly to avoid duplicating code
from backend.database import init_db, insert_prediction, get_recent_predictions, get_formatted_predictions_context
from backend.ml_service import ml_service
from backend.chat_service import get_chat_response
from backend.schemas import PredictionInput

# Initialize SQLite database and load the ML model once
@st.cache_resource
def startup_services():
    init_db()
    ml_service.load_model()
    return True

services_ready = startup_services()

# --- Custom Styling for Premium Aesthetics ---
st.markdown("""
<style>
    /* Dark glassmorphic styling */
    .stApp {
        background-color: #090d16;
        background-image: 
            radial-gradient(circle at 10% 20%, rgba(99, 102, 241, 0.05) 0%, transparent 40%),
            radial-gradient(circle at 90% 80%, rgba(236, 72, 153, 0.04) 0%, transparent 40%);
        background-attachment: fixed;
    }
    
    /* Title text styling */
    .gradient-text {
        background: linear-gradient(135deg, #6366f1, #a855f7, #ec4899);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 800;
        font-family: 'Outfit', sans-serif;
    }
    
    /* Card design */
    .glass-card {
        background: rgba(17, 25, 40, 0.75);
        backdrop-filter: blur(16px);
        -webkit-backdrop-filter: blur(16px);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 16px;
        padding: 20px;
        margin-bottom: 20px;
    }
    
    /* Custom metric value */
    .metric-value {
        font-size: 3rem;
        font-weight: 800;
        background: linear-gradient(135deg, #6366f1, #a855f7, #ec4899);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        line-height: 1.1;
    }
    
    /* Level Badge glow effects */
    .level-badge {
        display: inline-block;
        padding: 6px 14px;
        border-radius: 20px;
        font-size: 0.8rem;
        font-weight: 700;
        text-transform: uppercase;
        border: 1px solid rgba(255,255,255,0.15);
    }
    .badge-low { background-color: rgba(16, 185, 129, 0.1); color: #10b981; border-color: rgba(16, 185, 129, 0.3); }
    .badge-medium { background-color: rgba(59, 130, 246, 0.1); color: #3b82f6; border-color: rgba(59, 130, 246, 0.3); }
    .badge-high { background-color: rgba(245, 158, 11, 0.1); color: #f59e0b; border-color: rgba(245, 158, 11, 0.3); }
    .badge-veryhigh { background-color: rgba(239, 68, 68, 0.1); color: #ef4444; border-color: rgba(239, 68, 68, 0.3); }
    
    /* Suggestion block */
    .suggestion-box {
        background-color: rgba(255, 255, 255, 0.02);
        border-left: 3px solid #8b5cf6;
        padding: 12px;
        border-radius: 4px;
        font-size: 0.9rem;
        line-height: 1.5;
    }
</style>
""", unsafe_allowed_value_exceptions=True, unsafe_allowed_html=True)

# --- Header ---
st.write(
    f'<div style="display: flex; justify-content: space-between; align-items: center; background: rgba(17, 25, 40, 0.75); padding: 15px 25px; border-radius: 12px; border: 1px solid rgba(255, 255, 255, 0.08); margin-bottom: 25px;">'
    f'  <div style="font-size: 1.8rem; font-weight: 800; font-family: Outfit;">ShopFlow <span class="gradient-text">AI</span></div>'
    f'  <div style="font-size: 0.85rem; color: #9ca3af; display: flex; align-items: center; gap: 8px;">'
    f'    <div style="width: 8px; height: 8px; background-color: #10b981; border-radius: 50%; box-shadow: 0 0 8px #10b981;"></div>'
    f'    <span>{"Live Model Loaded" if ml_service.is_loaded else "Running Fallback Mode"}</span>'
    f'  </div>'
    f'</div>',
    unsafe_allowed_html=True
)

# Initialize Session State variables for Chat history
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Hello! I am your retail assistant, connected directly to your prediction database. Ask me about historical logs, store metrics, or comparison summaries!"}
    ]

# Keep track of prediction triggers
if "last_prediction" not in st.session_state:
    st.session_state.last_prediction = None

# Grid layout: Predictive form & Analysis Output
col1, col2 = st.columns([1.2, 1.8])

# --- Predictive Engine Form (Col 1) ---
with col1:
    st.markdown('<div class="glass-card">', unsafe_allowed_html=True)
    st.markdown("### ⚙️ Predictive Engine")
    st.markdown("Input operational parameters to estimate weekly store traffic.")
    
    with st.form("prediction_form"):
        store = st.selectbox(
            "Store Location",
            options=list(range(1, 11)),
            format_func=lambda x: {
                1: "Store 1 (Northern Small)",
                2: "Store 2 (Northern Med)",
                3: "Store 3 (Northern Flagship)",
                4: "Store 4 (Northern Small)",
                5: "Store 5 (Northern Med)",
                6: "Store 6 (Southern Large)",
                7: "Store 7 (Southern Small)",
                8: "Store 8 (Southern Med)",
                9: "Store 9 (Southern Large)",
                10: "Store 10 (Southern Med)"
            }.get(x, f"Store {x}"),
            index=2 # default Store 3
        )
        
        date = st.date_input("Target Date", value=datetime(2026, 11, 27))
        is_holiday = st.checkbox("Holiday Week Surge", value=True)
        temperature = st.slider("Temperature (°F)", min_value=-10, max_value=110, value=45, step=1)
        fuel_price = st.slider("Fuel Price ($/gal)", min_value=2.00, max_value=5.00, value=3.45, step=0.05)
        cpi = st.slider("Consumer Price Index (CPI)", min_value=250, max_value=320, value=285, step=1)
        unemployment = st.slider("Unemployment Rate (%)", min_value=1.0, max_value=10.0, value=4.2, step=0.1)
        
        submit_btn = st.form_submit_button("Generate Prediction", use_container_width=True)
        
    st.markdown('</div>', unsafe_allowed_html=True)

# Run logic on form submission
if submit_btn:
    payload = PredictionInput(
        store=store,
        date=date.strftime("%Y-%m-%d"),
        is_holiday=1 if is_holiday else 0,
        temperature=float(temperature),
        fuel_price=float(fuel_price),
        cpi=float(cpi),
        unemployment=float(unemployment)
    )
    
    # Run prediction
    result = ml_service.predict(payload)
    
    # Save to database
    insert_prediction(
        store=payload.store,
        date=payload.date,
        is_holiday=payload.is_holiday,
        temperature=payload.temperature,
        fuel_price=payload.fuel_price,
        cpi=payload.cpi,
        unemployment=payload.unemployment,
        predicted_traffic=result.predicted_traffic,
        traffic_level=result.traffic_level,
        suggested_action=result.suggested_action
    )
    
    st.session_state.last_prediction = result
    # Force rerun to sync chat and table context
    st.rerun()

# --- Live Traffic Analysis Output (Col 2) ---
with col2:
    st.markdown('<div class="glass-card" style="height: 100%;">', unsafe_allowed_html=True)
    st.markdown("### 📊 Live Traffic Analysis")
    st.markdown("Real-time inference and feature importance explanations.")
    
    if st.session_state.last_prediction:
        res = st.session_state.last_prediction
        
        # Display main metric values
        m_col1, m_col2 = st.columns([1, 1.2])
        with m_col1:
            st.markdown("##### Predicted Weekly Traffic")
            st.markdown(f'<div class="metric-value">{res.predicted_traffic:,}</div>', unsafe_allowed_html=True)
            
            # Badge
            lvl = res.traffic_level.lower()
            badge_class = "badge-inactive"
            if lvl == "low": badge_class = "badge-low"
            elif lvl == "medium": badge_class = "badge-medium"
            elif lvl == "high": badge_class = "badge-high"
            elif lvl == "very high": badge_class = "badge-veryhigh"
            
            st.markdown(f'<div class="level-badge {badge_class}">{res.traffic_level}</div>', unsafe_allowed_html=True)
            
        with m_col2:
            st.markdown("##### Suggested Operational Action")
            st.markdown(f'<div class="suggestion-box">{res.suggested_action}</div>', unsafe_allowed_html=True)
            
        st.write("---")
        
        # Explainability Progress Bars
        st.markdown("##### Top AI Explainability Drivers (MDI Weights)")
        total_imp = sum(f.importance for f in res.top_factors)
        
        for factor in res.top_factors:
            norm_weight = (factor.importance / total_imp) * 100 if total_imp > 0 else 0
            val_str = str(factor.value)
            if factor.feature == "Holiday Season": val_str = "Active" if factor.value == 1 else "None"
            elif factor.feature == "Fuel Prices": val_str = f"${factor.value:.2f}"
            elif factor.feature == "Unemployment": val_str = f"{factor.value:.1f}%"
            elif factor.feature == "Temperature": val_str = f"{factor.value:.0f}°F"
            elif "Baseline" in factor.feature and factor.value == 1: val_str = "Target Store"
            
            st.write(f"**{factor.feature}** ({val_str}) — {norm_weight:.0f}% importance weight")
            st.progress(norm_weight / 100.0)
            
    else:
        st.markdown(
            '<div style="text-align: center; padding: 60px 0; color: #9ca3af;">'
            '  <h4>Awaiting Input Parameters</h4>'
            '  <p>Fill out the form on the left and click <strong>Generate Prediction</strong>.</p>'
            '</div>',
            unsafe_allowed_html=True
        )
    st.markdown('</div>', unsafe_allowed_html=True)

# Grid layout: Grounded Chat Panel & Recent Audit Log
st.write("---")
col3, col4 = st.columns([1, 1])

# --- Conversational Assistant (Col 3) ---
with col3:
    st.markdown('<div class="glass-card" style="height: 520px; display: flex; flex-direction: column;">', unsafe_allowed_html=True)
    st.markdown("### 💬 Grounded Chat Assistant")
    st.markdown("Ask natural language questions grounded in current SQLite predictions log.")
    
    # Custom container for chat messages with specific height and scrolling
    chat_container = st.container(height=320)
    
    # Display message history
    with chat_container:
        for msg in st.session_state.messages:
            with st.chat_message(msg["role"]):
                st.write(msg["content"])
                
    # Fast prompt suggestion chips
    chip_col1, chip_col2, chip_col3 = st.columns(3)
    
    prompt_clicked = None
    with chip_col1:
        if st.button("Highest Traffic Store?", key="btn_chip_1", use_container_width=True):
            prompt_clicked = "Which store had the highest predicted traffic?"
    with chip_col2:
        if st.button("Analyze Store 3", key="btn_chip_2", use_container_width=True):
            prompt_clicked = "Show details for Store 3"
    with chip_col3:
        if st.button("Holiday Surge Levels", key="btn_chip_3", use_container_width=True):
            prompt_clicked = "Did holiday weeks get flagged with High or Very High traffic?"
            
    # Capture user input
    user_query = st.chat_input("Type an analytical question... (e.g. why is Store 3 high on Nov 27?)", key="chat_input")
    
    # Trigger message query
    final_query = user_query or prompt_clicked
    
    if final_query:
        # Append User message
        st.session_state.messages.append({"role": "user", "content": final_query})
        
        # Display User message immediately
        with chat_container:
            with st.chat_message("user"):
                st.write(final_query)
                
        # Generate chatbot reply
        with chat_container:
            with st.chat_message("assistant"):
                with st.spinner("Analyzing predictions..."):
                    reply = get_chat_response(final_query)
                    st.write(reply)
                    
        # Append Assistant reply
        st.session_state.messages.append({"role": "assistant", "content": reply})
        st.rerun()

    st.markdown('</div>', unsafe_allowed_html=True)

# --- Recent Log table (Col 4) ---
with col4:
    st.markdown('<div class="glass-card" style="height: 520px; overflow: hidden;">', unsafe_allowed_html=True)
    st.markdown("### 📋 Predictions History Log")
    st.markdown("Recent prediction runs logged to SQLite. Used as context grounding.")
    
    # Fetch recent predictions
    recent_preds = get_recent_predictions(20)
    
    if recent_preds:
        df_history = pd.DataFrame(recent_preds)
        
        # Rename and filter columns for a clean presentation
        df_display = df_history[[
            "id", "timestamp", "store", "date", "is_holiday", 
            "temperature", "fuel_price", "predicted_traffic", "traffic_level"
        ]].copy()
        
        df_display.columns = [
            "ID", "Logged Time", "Store ID", "Target Date", "Is Holiday", 
            "Temp (°F)", "Fuel Price", "Predicted Traffic", "Level"
        ]
        
        # Map Is Holiday to friendly format
        df_display["Is Holiday"] = df_display["Is Holiday"].map({1: "Yes", 0: "No"})
        
        # Display the data frame cleanly
        st.dataframe(
            df_display, 
            hide_index=True,
            use_container_width=True,
            column_config={
                "Predicted Traffic": st.column_config.NumberColumn(format="%d"),
                "Fuel Price": st.column_config.NumberColumn(format="$%.2f"),
                "Temp (°F)": st.column_config.NumberColumn(format="%d°F")
            }
        )
    else:
        st.markdown(
            '<div style="text-align: center; padding: 120px 0; color: #9ca3af;">'
            '  <h5>No predictions recorded yet</h5>'
            '  <p>Run a prediction on the left to write its results to SQLite.</p>'
            '</div>',
            unsafe_allowed_html=True
        )
    st.markdown('</div>', unsafe_allowed_html=True)
