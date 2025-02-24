import streamlit as st
import pandas as pd
from chat_agent import TitanicChatAgent
from data_loader import load_titanic_data, get_data_summary
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
if 'data_loaded' not in st.session_state:
    st.session_state.data_loaded = False
if 'error_count' not in st.session_state:
    st.session_state.error_count = 0

def initialize_data():
    """
    Initialize data and agent with proper error handling
    """
    try:
        with st.spinner('Loading dataset...'):
            df = load_titanic_data()
            chat_agent = TitanicChatAgent(df)
            summary = get_data_summary(df)
            st.session_state.data_loaded = True
            return df, chat_agent, summary
    except Exception as e:
        st.error(f"Failed to load data: {str(e)}")
        return None, None, None

def handle_user_query(agent, query):
    """
    Handle user queries with proper error management
    """
    try:
        return agent.get_response(query)
    except Exception as e:
        error_msg = str(e)
        if "insufficient_quota" in error_msg.lower():
            st.session_state.error_count += 1
            if st.session_state.error_count >= 3:
                return "⚠️ Multiple API errors detected. Please try again later or contact support for assistance."
            return "🔄 API quota exceeded. Please wait a moment before trying again."
        elif "rate_limit" in error_msg.lower():
            return "⏳ Too many requests. Please wait a few seconds before asking another question."
        else:
            return f"❌ Error: {str(e)}"

def main():
    # Header
    st.title("🚢 Titanic Dataset Analysis Chatbot")
    st.markdown("""
    Explore the Titanic dataset through natural conversation! Ask questions about:
    - Passenger statistics (survival rates, counts)
    - Demographics (age, gender, class)
    - Ticket information (fares, classes)
    - Visualizations (distributions, comparisons)
    """)

    try:
        # Initialize data
        if not st.session_state.data_loaded:
            df, chat_agent, summary = initialize_data()
            if not df:
                return
            st.session_state.df = df
            st.session_state.chat_agent = chat_agent
            st.session_state.summary = summary

        # Display dataset summary in sidebar
        st.sidebar.title("Dataset Overview")
        st.sidebar.markdown(f"""
        📊 **Quick Stats**
        - Total Passengers: {st.session_state.summary['total_passengers']}
        - Survival Rate: {st.session_state.summary['survival_rate']}
        - Average Age: {st.session_state.summary['avg_age']} years
        - Average Fare: {st.session_state.summary['avg_fare']}
        - Gender Ratio: {st.session_state.summary['gender_ratio']}
        """)

        # Example questions
        st.sidebar.title("Try These Questions")
        example_questions = [
            "What was the overall survival rate?",
            "Show me a histogram of passenger ages",
            "What was the average ticket fare?",
            "Show me a pie chart of passenger classes",
            "What was the gender distribution?",
            "How many passengers were there in total?"
        ]

        for question in example_questions:
            if st.sidebar.button(question):
                # Process the example question
                response = st.session_state.chat_agent.get_response(question)
                st.session_state.chat_history.append({"role": "user", "content": question})
                st.session_state.chat_history.append({"role": "assistant", "content": response})

        # Chat interface
        st.write("### Chat with the Titanic Dataset")

        # Display chat history
        for message in st.session_state.chat_history:
            with st.container():
                if message["role"] == "user":
                    st.write("🧑 You:", message["content"])
                else:
                    st.write("🤖 Assistant:", end=" ")

                    if isinstance(message["content"], dict):
                        if message["content"].get('type') == 'visualization':
                            try:
                                fig = create_visualization(
                                    st.session_state.df,
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
            "Ask a question about the Titanic dataset:",
            key="user_input",
            placeholder="e.g., What was the survival rate for first class passengers?"
        )

        if st.button("Send", type="primary") and user_input:
            with st.spinner('Processing your question...'):
                try:
                    # Get response from agent
                    response = handle_user_query(st.session_state.chat_agent, user_input)

                    # Add to chat history
                    st.session_state.chat_history.append({"role": "user", "content": user_input})
                    st.session_state.chat_history.append({"role": "assistant", "content": response})

                    # Clear input
                    st.session_state.user_input = ""

                    # Rerun to update display
                    st.rerun()
                except Exception as e:
                    st.error(f"Failed to process question: {str(e)}")

        # Reset button
        if st.session_state.error_count >= 3:
            if st.button("🔄 Reset Application"):
                for key in st.session_state.keys():
                    del st.session_state[key]
                st.rerun()

    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
        if st.button("🔄 Reset Application"):
            for key in st.session_state.keys():
                del st.session_state[key]
            st.rerun()

if __name__ == "__main__":
    main()