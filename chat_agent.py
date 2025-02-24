from langchain_experimental.agents import create_pandas_dataframe_agent
from langchain.chat_models import ChatOpenAI
from langchain.agents.agent_types import AgentType
import os
import re

class TitanicChatAgent:
    def __init__(self, df):
        """
        Initialize the Titanic chatbot agent with proper security settings
        """
        # the newest OpenAI model is "gpt-4o" which was released May 13, 2024
        self.agent = create_pandas_dataframe_agent(
            ChatOpenAI(temperature=0, model="gpt-4o"),
            df,
            verbose=True,
            agent_type=AgentType.OPENAI_FUNCTIONS,
            allow_dangerous_code=True,  # Required for DataFrame operations
        )
        self.df = df

    def get_response(self, query):
        """
        Process the user query and return appropriate response
        """
        # Input validation
        if not query or not isinstance(query, str):
            return {"type": "text", "content": "Please provide a valid question about the Titanic dataset."}

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
                    return self._handle_visualization_query(query, viz_type)

            # Handle regular queries with error handling
            response = self.agent.run(query)
            return {"type": "text", "content": response}
        except Exception as e:
            error_msg = f"I apologize, but I couldn't process that query. Error: {str(e)}"
            return {"type": "text", "content": error_msg}

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

            # If no specific visualization is matched, try to get a text response
            response = self.agent.run(query)
            return {"type": "text", "content": response}
        except Exception as e:
            return {"type": "text", "content": f"Error creating visualization: {str(e)}"}