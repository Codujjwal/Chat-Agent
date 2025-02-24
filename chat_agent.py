from langchain_experimental.agents import create_pandas_dataframe_agent
from langchain_community.chat_models import ChatOpenAI
from langchain.agents.agent_types import AgentType
from typing import Dict, Any, Union
import pandas as pd
import re
import os

class TitanicChatAgent:
    def __init__(self, df: pd.DataFrame):
        """
        Initialize the Titanic chatbot agent with proper security settings
        and error handling
        """
        try:
            # Check for API key
            if not os.getenv("OPENAI_API_KEY"):
                raise ValueError("OpenAI API key not found. Please provide a valid API key.")

            self.llm = ChatOpenAI(
                temperature=0,
                model="gpt-3.5-turbo",
                request_timeout=30
            )
            self.agent = create_pandas_dataframe_agent(
                self.llm,
                df,
                verbose=True,
                agent_type=AgentType.OPENAI_FUNCTIONS,
                handle_parsing_errors=True,
                max_iterations=5,  # Limit iterations for safety
                max_execution_time=30,  # Timeout in seconds
                allow_python_primitives=True,  # Allow basic Python operations
                allow_pandas_syntax=True,  # Enable Pandas operations
                allow_dangerous_code=True  # Required for DataFrame operations but with controlled scope
            )
            self.df = df

            # Verify agent initialization with basic query
            test_query = "Count total rows"
            self._verify_agent(test_query)

        except Exception as e:
            error_msg = str(e)
            if "insufficient_quota" in error_msg.lower():
                raise Exception("OpenAI API quota exceeded. Please ensure your API key has sufficient credits.")
            elif "invalid_api_key" in error_msg.lower():
                raise Exception("Invalid OpenAI API key. Please provide a valid API key.")
            else:
                raise Exception(f"Failed to initialize chat agent: {str(e)}")

    def _verify_agent(self, test_query: str) -> None:
        """
        Verify agent functionality with a test query
        """
        try:
            self.agent.run(test_query)
        except Exception as e:
            error_msg = str(e)
            if "insufficient_quota" in error_msg.lower():
                raise Exception("OpenAI API quota exceeded during verification. Please check your API key's quota.")
            raise Exception(f"Agent verification failed: {str(e)}")

    def get_response(self, query: str) -> Union[Dict[str, Any], str]:
        """
        Process the user query and return appropriate response with
        comprehensive error handling
        """
        if not query or not isinstance(query, str):
            return "Please provide a valid question about the Titanic dataset."

        try:
            # Check for visualization requests
            viz_response = self._check_visualization_request(query)
            if viz_response:
                return viz_response

            # Process the query with safety checks
            if self._is_safe_query(query):
                response = self.agent.run(query)
                # Clean and format the response
                formatted_response = self._format_response(response)
                return formatted_response
            else:
                return "I apologize, but I cannot process that query for security reasons. Please try a different question."

        except Exception as e:
            return self._handle_error(e)

    def _is_safe_query(self, query: str) -> bool:
        """
        Check if the query is safe to execute
        """
        # List of potentially dangerous keywords
        dangerous_keywords = [
            "exec", "eval", "delete", "drop", "truncate",
            "system", "os.", "subprocess", "import"
        ]
        query_lower = query.lower()
        return not any(keyword in query_lower for keyword in dangerous_keywords)

    def _check_visualization_request(self, query: str) -> Union[Dict[str, Any], None]:
        """
        Check if the query requires a visualization and handle accordingly
        """
        viz_patterns = {
            'histogram': {
                'pattern': r'histogram|distribution of|show .+ distribution',
                'matches': {
                    'age': {'column': 'Age', 'title': 'Distribution of Passenger Ages'},
                    'fare': {'column': 'Fare', 'title': 'Distribution of Ticket Fares'}
                }
            },
            'bar': {
                'pattern': r'bar chart|compare .+ across|show .+ by',
                'matches': {
                    'embarked': {'column': 'Embarked', 'title': 'Passengers by Embarkation Port'},
                    'class': {'column': 'Pclass', 'title': 'Passengers by Class'}
                }
            },
            'pie': {
                'pattern': r'pie chart|percentage|proportion of',
                'matches': {
                    'class': {'column': 'Pclass', 'title': 'Distribution of Passenger Classes'},
                    'gender': {'column': 'Sex', 'title': 'Gender Distribution'},
                    'survived': {'column': 'Survived', 'title': 'Survival Distribution'}
                }
            }
        }

        query_lower = query.lower()
        for viz_type, config in viz_patterns.items():
            if re.search(config['pattern'], query_lower):
                for key, params in config['matches'].items():
                    if key in query_lower:
                        return {
                            'type': 'visualization',
                            'viz_type': viz_type,
                            'params': params
                        }

        return None

    def _format_response(self, response: str) -> str:
        """
        Clean and format the response for better presentation
        """
        # Remove any markdown formatting
        response = re.sub(r'```.*?```', '', response, flags=re.DOTALL)
        # Clean up newlines and spaces
        response = re.sub(r'\s+', ' ', response).strip()
        # Capitalize first letter
        response = response[0].upper() + response[1:] if response else response

        return response

    def _handle_error(self, error: Exception) -> str:
        """
        Handle different types of errors and return user-friendly messages
        """
        error_msg = str(error)

        if "insufficient_quota" in error_msg:
            return ("I apologize, but I'm currently unable to process requests due to API limits. "
                   "Please try again later or contact support.")
        elif "rate_limit" in error_msg:
            return "I'm receiving too many requests right now. Please wait a moment and try again."
        elif "timeout" in error_msg:
            return "The request took too long to process. Please try a simpler question."
        elif "security" in error_msg.lower():
            return "I cannot process that query due to security restrictions. Please try a different question."
        else:
            return ("I apologize, but I couldn't process that query. Please try rephrasing your "
                   "question or ask something else.")