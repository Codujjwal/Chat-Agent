import plotly.express as px
import plotly.graph_objects as go

def create_visualization(df, viz_type, params):
    """
    Create visualizations based on the type and parameters
    """
    if viz_type == "histogram":
        return create_histogram(df, params)
    elif viz_type == "bar":
        return create_bar_chart(df, params)
    elif viz_type == "pie":
        return create_pie_chart(df, params)
    else:
        raise ValueError(f"Unsupported visualization type: {viz_type}")

def create_histogram(df, params):
    """
    Create a histogram visualization
    """
    fig = px.histogram(
        df,
        x=params['column'],
        nbins=params.get('bins', 30),
        title=params.get('title', f'Distribution of {params["column"]}'),
        color=params.get('color', None)
    )
    
    fig.update_layout(
        xaxis_title=params.get('xlabel', params['column']),
        yaxis_title=params.get('ylabel', 'Count'),
        showlegend=True
    )
    
    return fig

def create_bar_chart(df, params):
    """
    Create a bar chart visualization
    """
    if params.get('aggregation', None):
        data = df.groupby(params['column'])[params['aggregation']['column']].agg(params['aggregation']['func'])
    else:
        data = df[params['column']].value_counts()
    
    fig = px.bar(
        x=data.index,
        y=data.values,
        title=params.get('title', f'{params["column"]} Distribution'),
    )
    
    fig.update_layout(
        xaxis_title=params.get('xlabel', params['column']),
        yaxis_title=params.get('ylabel', 'Count'),
        showlegend=False
    )
    
    return fig

def create_pie_chart(df, params):
    """
    Create a pie chart visualization
    """
    data = df[params['column']].value_counts()
    
    fig = px.pie(
        values=data.values,
        names=data.index,
        title=params.get('title', f'{params["column"]} Distribution')
    )
    
    return fig
