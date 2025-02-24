import streamlit as st
import pandas as pd
from chat_agent import TitanicChatAgent
from data_loader import load_titanic_data
from visualization import create_visualization

# Page configuration
st.set_page_config(
    page_title="Titanic Dataset Chatbot",
    page_icon="🚢",
    layout="wide"
)

# Initialize session state
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

# Load data
try:
    df = load_titanic_data()
    agent = TitanicChatAgent(df)
except Exception as e:
    st.error(f"Error loading data: {str(e)}")
    st.stop()

# Header
st.title("🚢 Titanic Dataset Analysis Chatbot")
st.markdown("""
Ask questions about the Titanic dataset or request visualizations!
Try asking about:
- Survival rates
- Passenger demographics
- Ticket fares
- Age distribution
""")

# Sidebar with example questions
st.sidebar.title("Try These Questions")
example_questions = [
    "How many passengers were there?",
    "What was the survival rate?",
    "Show me the age distribution",
    "Show me passenger classes",
    "What was the average fare?"
]

# Example question buttons
for question in example_questions:
    if st.sidebar.button(question):
        response = agent.get_response(question)
        st.session_state.chat_history.append({"role": "user", "content": question})
        st.session_state.chat_history.append({"role": "assistant", "content": response})

# Display chat history
for message in st.session_state.chat_history:
    if message["role"] == "user":
        st.write("🧑 You:", message["content"])
    else:
        st.write("🤖 Assistant:", end=" ")
        if isinstance(message["content"], dict) and message["content"].get('type') == 'visualization':
            try:
                fig = create_visualization(
                    df,
                    message["content"]["viz_type"],
                    message["content"]["params"]
                )
                st.plotly_chart(fig, use_container_width=True)
            except Exception as e:
                st.error(f"Failed to create visualization: {str(e)}")
        else:
            st.write(message["content"])

# User input
user_input = st.text_input(
    "Ask a question:",
    key="user_input",
    placeholder="e.g., What was the survival rate?"
)

if st.button("Send", type="primary") and user_input:
    response = agent.get_response(user_input)
    st.session_state.chat_history.append({"role": "user", "content": user_input})
    st.session_state.chat_history.append({"role": "assistant", "content": response})
    st.session_state.user_input = ""  # Clear input
    st.rerun()  # Refresh to show new messages