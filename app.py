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

def main():
    st.title("🚢 Titanic Dataset Analysis Chatbot")
    
    # Load data
    df = load_titanic_data()
    
    # Initialize chat agent
    chat_agent = TitanicChatAgent(df)
    
    # Sidebar with example questions
    st.sidebar.title("Example Questions")
    example_questions = [
        "What percentage of passengers were male on the Titanic?",
        "Show me a histogram of passenger ages",
        "What was the average ticket fare?",
        "How many passengers embarked from each port?"
    ]
    for question in example_questions:
        if st.sidebar.button(question):
            st.session_state.chat_history.append({"role": "user", "content": question})
            
            # Get response from agent
            response = chat_agent.get_response(question)
            st.session_state.chat_history.append({"role": "assistant", "content": response})
    
    # Chat interface
    st.write("### Chat with the Titanic Dataset")
    
    # Display chat history
    for message in st.session_state.chat_history:
        with st.container():
            if message["role"] == "user":
                st.write("🧑 You:", message["content"])
            else:
                st.write("🤖 Assistant:", message["content"])
                
                # If the response contains visualization data
                if isinstance(message["content"], dict) and "viz_type" in message["content"]:
                    fig = create_visualization(
                        df,
                        message["content"]["viz_type"],
                        message["content"]["params"]
                    )
                    st.plotly_chart(fig, use_container_width=True)
    
    # User input
    user_input = st.text_input("Ask a question about the Titanic dataset:", key="user_input")
    
    if st.button("Send") and user_input:
        st.session_state.chat_history.append({"role": "user", "content": user_input})
        
        # Get response from agent
        response = chat_agent.get_response(user_input)
        st.session_state.chat_history.append({"role": "assistant", "content": response})
        
        # Clear input
        st.session_state.user_input = ""

if __name__ == "__main__":
    main()
