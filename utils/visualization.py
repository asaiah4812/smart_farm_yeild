import plotly.graph_objects as go
import plotly.express as px
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from plotly.subplots import make_subplots

def create_yield_timeseries(df, date_column='harvest_date', yield_column='yield_amount', crop_column='crop_type'):
    """
    Create a time series plot of yield data
    """
    fig = px.line(df, x=date_column, y=yield_column, color=crop_column,
                 title='Crop Yield Over Time',
                 labels={yield_column: 'Yield (tons/hectare)', date_column: 'Date'})
    
    return fig

def create_feature_correlation_heatmap(df, features):
    """
    Create a heatmap of feature correlations
    """
    # Calculate correlation matrix
    corr_matrix = df[features].corr()
    
    fig = go.Figure(data=go.Heatmap(
        z=corr_matrix.values,
        x=corr_matrix.columns,
        y=corr_matrix.index,
        colorscale='RdBu_r',
        zmin=-1,
        zmax=1,
        colorbar=dict(title='Correlation')
    ))
    
    fig.update_layout(title='Feature Correlation Heatmap')
    
    return fig

def create_yield_distribution_by_crop(df, yield_column='yield_amount', crop_column='crop_type'):
    """
    Create a box plot of yield distribution by crop type
    """
    fig = px.box(df, x=crop_column, y=yield_column,
                title='Yield Distribution by Crop Type',
                labels={yield_column: 'Yield (tons/hectare)', crop_column: 'Crop Type'})
    
    return fig

def create_feature_vs_yield_scatter(df, feature, yield_column='yield_amount', crop_column='crop_type'):
    """
    Create a scatter plot of a feature vs yield
    """
    fig = px.scatter(df, x=feature, y=yield_column, color=crop_column,
                    trendline='ols',
                    title=f'{feature} vs Yield',
                    labels={yield_column: 'Yield (tons/hectare)', feature: feature.replace('_', ' ').title()})
    
    return fig

def create_prediction_actual_plot(y_actual, y_predicted, model_name):
    """
    Create a scatter plot of actual vs predicted values
    """
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=y_actual,
        y=y_predicted,
        mode='markers',
        name='Predictions'
    ))
    
    # Add perfect prediction line
    min_val = min(min(y_actual), min(y_predicted))
    max_val = max(max(y_actual), max(y_predicted))
    
    fig.add_trace(go.Scatter(
        x=[min_val, max_val],
        y=[min_val, max_val],
        mode='lines',
        name='Perfect Prediction',
        line=dict(dash='dash', color='red')
    ))
    
    fig.update_layout(
        title=f'Actual vs Predicted Yield ({model_name})',
        xaxis_title='Actual Yield (tons/hectare)',
        yaxis_title='Predicted Yield (tons/hectare)'
    )
    
    return fig