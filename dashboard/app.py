
import streamlit as st

st.set_page_config(
    page_title="AI-IDS | Temporarily Disabled",
    page_icon="🛡️",
    layout="centered"
)

st.markdown(
    """
    <div style="text-align:center; padding:50px 10px;">
        <h1>🛡️ AI-Based Intrusion Detection System</h1>
        <h2 style="color:#e74c3c;">System Temporarily Disabled</h2>
        <p>
            This application is temporarily unavailable.
        </p>
        <p>
            The AI-IDS research system is currently under maintenance.
        </p>
        <p>Thank you for your understanding.</p>
    </div>
    """,
    unsafe_allow_html=True
)

st.stop()
