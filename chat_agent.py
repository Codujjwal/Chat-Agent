from langchain_experimental.agents import create_pandas_dataframe_agent
from langchain_community.chat_models import ChatOpenAI
from langchain.agents.agent_types import AgentType
import os
import re

class TitanicChatAgent:
    def __init__(self, df):
        """
        Initialize the Titanic chatbot agent with proper security settings
        """
        try:
            # the newest OpenAI model is "gpt-4o" which was released May 13, 2024
            self.agent = create_pandas_dataframe_agent(
                ChatOpenAI(
                    temperature=0, 
                    model="gpt-3.5-turbo",  # Using a more cost-effective model
                    request_timeout=30
                ),
                df,
                verbose=True,
                agent_type=AgentType.OPENAI_FUNCTIONS,
                allow_dangerous_code=True,  # Required for DataFrame operations
            )
            self.df = df
        except Exception as e:
            print(f"Error initializing agent: {str(e)}")
            raise

    def get_response(self, query):
        """
        Process the user query and return appropriate response
        """
        # Input validation
        if not query or not isinstance(query, str):
            return "Please provide a valid question about the Titanic dataset."

        # Check for visualization requests
        viz_patterns = {
            'histogram': r'histogram|distribution of|show .+ distribution',
            'bar': r'bar chart|compare .+ across|show .+ by',
            'pie': r'pie chart|percentage|proportion of'
        }

        try:
            # Check for visualization patterns
            for viz_type, pattern in viz_patterns.items():
                if re.search(pattern, query.lower()):
                    viz_response = self._handle_visualization_query(query, viz_type)
                    if viz_response:
                        return viz_response

            # Handle regular queries with error handling
            response = self.agent.run(query)
            return response
        except Exception as e:
            error_msg = str(e)
            if "insufficient_quota" in error_msg:
                return "I apologize, but I'm currently unable to process requests due to API limits. Please try again later or contact support to ensure your API key has sufficient credits."
            elif "rate_limit" in error_msg:
                return "I'm receiving too many requests right now. Please wait a moment and try again."
            else:
                return f"I apologize, but I couldn't process that query. Please try rephrasing your question or ask something else."

    def _handle_visualization_query(self, query, viz_type):
        """
        Handle queries that require visualization with proper error handling
        """
        try:
            if viz_type == "histogram":
                if "age" in query.lower():
                    return {
                        "viz_type": "histogram",
                        "params": {
                            "column": "Age",
                            "bins": 30,
                            "title": "Distribution of Passenger Ages"
                        }
                    }
                elif "fare" in query.lower():
                    return {
                        "viz_type": "histogram",
                        "params": {
                            "column": "Fare",
                            "bins": 50,
                            "title": "Distribution of Ticket Fares"
                        }
                    }

            elif viz_type == "bar":
                if "embarked" in query.lower():
                    return {
                        "viz_type": "bar",
                        "params": {
                            "column": "Embarked",
                            "title": "Passengers by Embarkation Port"
                        }
                    }

            elif viz_type == "pie":
                if "class" in query.lower():
                    return {
                        "viz_type": "pie",
                        "params": {
                            "column": "Pclass",
                            "title": "Distribution of Passenger Classes"
                        }
                    }
                elif "gender" in query.lower() or "male" in query.lower() or "female" in query.lower():
                    return {
                        "viz_type": "pie",
                        "params": {
                            "column": "Sex",
                            "title": "Gender Distribution"
                        }
                    }

            return None
        except Exception as e:
            return f"Error creating visualization: {str(e)}"