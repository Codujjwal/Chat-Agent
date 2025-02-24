import plotly.express as px
import plotly.graph_objects as go
from typing import Dict, Any
import pandas as pd

def create_visualization(df: pd.DataFrame, viz_type: str, params: Dict[str, Any]):
    """
    Create visualizations with comprehensive error handling and validation
    """
    try:
        if viz_type == "histogram":
            return create_histogram(df, params)
        elif viz_type == "bar":
            return create_bar_chart(df, params)
        elif viz_type == "pie":
            return create_pie_chart(df, params)
        elif viz_type == "scatter":
            return create_scatter_plot(df, params)
        else:
            raise ValueError(f"Unsupported visualization type: {viz_type}")
    except Exception as e:
        raise Exception(f"Failed to create visualization: {str(e)}")

def create_histogram(df: pd.DataFrame, params: Dict[str, Any]):
    """
    Create an enhanced histogram visualization
    """
    fig = px.histogram(
        df,
        x=params['column'],
        nbins=params.get('bins', 30),
        title=params.get('title', f'Distribution of {params["column"]}'),
        color=params.get('color', None),
        marginal=params.get('marginal', 'box'),
        hover_data=df.columns
    )

    update_layout(fig, params)
    return fig

def create_bar_chart(df: pd.DataFrame, params: Dict[str, Any]):
    """
    Create an enhanced bar chart visualization
    """
    if params.get('aggregation', None):
        data = df.groupby(params['column'])[params['aggregation']['column']].agg(params['aggregation']['func'])
    else:
        data = df[params['column']].value_counts()

    fig = px.bar(
        x=data.index,
        y=data.values,
        title=params.get('title', f'{params["column"]} Distribution'),
        color=data.index if params.get('use_color', True) else None,
        text=data.values if params.get('show_values', True) else None
    )

    update_layout(fig, params)
    return fig

def create_pie_chart(df: pd.DataFrame, params: Dict[str, Any]):
    """
    Create an enhanced pie chart visualization
    """
    data = df[params['column']].value_counts()

    fig = px.pie(
        values=data.values,
        names=data.index,
        title=params.get('title', f'{params["column"]} Distribution'),
        hole=params.get('donut', 0.3)
    )

    update_layout(fig, params)
    return fig

def create_scatter_plot(df: pd.DataFrame, params: Dict[str, Any]):
    """
    Create an enhanced scatter plot visualization
    """
    fig = px.scatter(
        df,
        x=params['x'],
        y=params['y'],
        color=params.get('color', None),
        size=params.get('size', None),
        title=params.get('title', f'{params["x"]} vs {params["y"]}'),
        trendline=params.get('trendline', 'ols') if params.get('show_trendline', False) else None
    )

    update_layout(fig, params)
    return fig

def update_layout(fig, params: Dict[str, Any]):
    """
    Apply consistent layout updates to all visualizations
    """
    fig.update_layout(
        xaxis_title=params.get('xlabel', params.get('column', '')),
        yaxis_title=params.get('ylabel', 'Count'),
        showlegend=params.get('showlegend', True),
        template='plotly_white',
        height=params.get('height', 500),
        title_x=0.5,
        margin=dict(t=50, l=50, r=50, b=50)
    )