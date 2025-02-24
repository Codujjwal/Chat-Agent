import pandas as pd
import numpy as np
from typing import Dict, Any

def load_titanic_data() -> pd.DataFrame:
    """
    Load and preprocess the Titanic dataset with proper error handling
    and pandas warning fixes
    """
    try:
        # Load dataset
        df = pd.read_csv('https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv')

        # Create a copy to avoid chained assignment warnings
        df = df.copy()

        # Handle missing values properly
        df.loc[:, 'Age'] = df['Age'].fillna(df['Age'].median())
        df.loc[:, 'Embarked'] = df['Embarked'].fillna(df['Embarked'].mode()[0])
        df.loc[:, 'Fare'] = df['Fare'].fillna(df['Fare'].median())

        # Create new features
        df.loc[:, 'FamilySize'] = df['SibSp'] + df['Parch'] + 1
        df.loc[:, 'IsAlone'] = (df['FamilySize'] == 1).astype(int)

        # Extract title from name
        df.loc[:, 'Title'] = df['Name'].str.extract(' ([A-Za-z]+)\.', expand=False)

        # Group rare titles
        rare_titles = ['Lady', 'Countess', 'Capt', 'Col', 'Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona']
        df.loc[:, 'Title'] = df['Title'].replace(rare_titles, 'Rare')

        # Convert categorical variables
        df.loc[:, 'Sex'] = df['Sex'].map({'female': 0, 'male': 1})
        df.loc[:, 'Embarked'] = df['Embarked'].map({'S': 0, 'C': 1, 'Q': 2})

        return df
    except Exception as e:
        raise Exception(f"Failed to load or process Titanic dataset: {str(e)}")

def get_data_summary(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Get summary statistics of the dataset
    """
    try:
        return {
            'total_passengers': len(df),
            'survival_rate': f"{(df['Survived'].mean() * 100):.1f}%",
            'avg_age': f"{df['Age'].mean():.1f}",
            'avg_fare': f"${df['Fare'].mean():.2f}",
            'gender_ratio': f"{(df['Sex'].mean() * 100):.1f}% male"
        }
    except Exception as e:
        raise Exception(f"Failed to generate data summary: {str(e)}")