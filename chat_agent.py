import pandas as pd
from typing import Dict, Any, Optional

class TitanicChatAgent:
    def __init__(self, df: pd.DataFrame):
        """Simple agent for analyzing Titanic dataset"""
        self.df = df

    def get_response(self, query: str) -> Any:
        """Process user queries"""
        try:
            query = query.lower().strip()

            # Handle visualization requests first
            if 'show' in query or 'plot' in query or 'chart' in query:
                if 'age' in query:
                    return {
                        'type': 'visualization',
                        'viz_type': 'histogram',
                        'params': {
                            'column': 'Age',
                            'title': 'Distribution of Passenger Ages'
                        }
                    }
                elif 'class' in query:
                    return {
                        'type': 'visualization',
                        'viz_type': 'pie',
                        'params': {
                            'column': 'Pclass',
                            'title': 'Distribution of Passenger Classes'
                        }
                    }
                elif 'gender' in query or 'sex' in query:
                    return {
                        'type': 'visualization',
                        'viz_type': 'pie',
                        'params': {
                            'column': 'Sex',
                            'title': 'Gender Distribution'
                        }
                    }
                elif 'fare' in query:
                    return {
                        'type': 'visualization',
                        'viz_type': 'histogram',
                        'params': {
                            'column': 'Fare',
                            'title': 'Distribution of Ticket Fares'
                        }
                    }

            # Handle statistical queries
            if 'survival' in query or 'survived' in query:
                survived = self.df['Survived'].sum()
                total = len(self.df)
                return f"Out of {total} passengers, {survived} survived ({(survived/total)*100:.1f}%)"

            if 'how many' in query:
                return f"There were {len(self.df)} passengers in the dataset"

            if 'average age' in query or 'mean age' in query:
                avg_age = self.df['Age'].mean()
                return f"The average passenger age was {avg_age:.1f} years"

            if 'average fare' in query or 'mean fare' in query:
                avg_fare = self.df['Fare'].mean()
                return f"The average ticket fare was ${avg_fare:.2f}"

            return "Please ask about passenger statistics (survival rates, counts) or request a visualization of the data."

        except Exception as e:
            return f"I encountered an error: {str(e)}. Please try a different question."