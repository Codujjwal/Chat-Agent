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
        Initialize the Titanic chatbot agent with local analysis capabilities
        """
        self.df = df
        self.cache = {}  # Simple response cache
        self.common_queries = {
            'survival_rate': self._get_survival_rate,
            'passenger_count': self._get_passenger_count,
            'class_distribution': self._get_class_distribution,
            'gender_distribution': self._get_gender_distribution,
            'age_stats': self._get_age_stats,
            'fare_stats': self._get_fare_stats,
        }

    def get_response(self, query: str) -> Union[Dict[str, Any], str]:
        """
        Process queries with local analysis first, fallback to API if needed
        """
        if not query or not isinstance(query, str):
            return "Please provide a valid question about the Titanic dataset."

        try:
            # Check cache first
            if query in self.cache:
                return self.cache[query]

            # Check for visualization requests
            viz_response = self._check_visualization_request(query)
            if viz_response:
                return viz_response

            # Try local analysis first
            local_response = self._handle_local_query(query)
            if local_response:
                self.cache[query] = local_response
                return local_response

            return "I can provide basic statistics and visualizations about the Titanic dataset. Please try asking about survival rates, passenger demographics, or request specific visualizations."

        except Exception as e:
            return self._handle_error(e)

    def _handle_local_query(self, query: str) -> str:
        """
        Handle queries using local data analysis
        """
        query_lower = query.lower()

        # Check for survival rate queries
        if 'survival' in query_lower or 'survived' in query_lower:
            return self._get_survival_rate()

        # Check for passenger count queries
        if 'how many' in query_lower and 'passengers' in query_lower:
            return self._get_passenger_count()

        # Check for class distribution queries
        if 'class' in query_lower and ('distribution' in query_lower or 'breakdown' in query_lower):
            return self._get_class_distribution()

        # Check for gender distribution queries
        if ('gender' in query_lower or 'male' in query_lower or 'female' in query_lower) and 'distribution' in query_lower:
            return self._get_gender_distribution()

        # Check for age statistics queries
        if 'age' in query_lower:
            return self._get_age_stats()

        # Check for fare statistics queries
        if 'fare' in query_lower or 'ticket' in query_lower or 'price' in query_lower:
            return self._get_fare_stats()

        return None

    def _get_survival_rate(self) -> str:
        survived = self.df['Survived'].mean() * 100
        return f"The overall survival rate was {survived:.1f}% of passengers."

    def _get_passenger_count(self) -> str:
        total = len(self.df)
        return f"There were {total} passengers in the dataset."

    def _get_class_distribution(self) -> str:
        class_counts = self.df['Pclass'].value_counts()
        return f"Passenger class distribution: " + ", ".join([f"{class_counts[i]} passengers in {i}st class" for i in sorted(class_counts.index)])

    def _get_gender_distribution(self) -> str:
        gender_counts = self.df['Sex'].value_counts()
        return f"Gender distribution: {gender_counts[0]} {gender_counts.index[0]}s and {gender_counts[1]} {gender_counts.index[1]}s"

    def _get_age_stats(self) -> str:
        avg_age = self.df['Age'].mean()
        min_age = self.df['Age'].min()
        max_age = self.df['Age'].max()
        return f"Passenger age statistics: Average age was {avg_age:.1f} years, youngest passenger was {min_age:.0f} years old, oldest was {max_age:.0f} years old."

    def _get_fare_stats(self) -> str:
        avg_fare = self.df['Fare'].mean()
        min_fare = self.df['Fare'].min()
        max_fare = self.df['Fare'].max()
        return f"Ticket fare statistics: Average fare was ${avg_fare:.2f}, cheapest ticket was ${min_fare:.2f}, most expensive was ${max_fare:.2f}."

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

    def _handle_error(self, error: Exception) -> str:
        """
        Handle different types of errors and return user-friendly messages
        """
        return "I apologize, but I couldn't process that query. Please try asking about basic statistics or request a visualization."