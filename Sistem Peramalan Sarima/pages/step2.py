# step2.py

import dash
from dash import html, dcc, callback, Input, Output
import dash_bootstrap_components as dbc
import plotly.graph_objects as go
import pandas as pd
from statsmodels.tsa.stattools import adfuller

dash.register_page(__name__, name='2-Stationarity', title='SARIMA | 2-Stationarity')

from assets.fig_layout import my_figlayout

layout = dbc.Container([
    dbc.Row([dbc.Col(html.H3('Stationarity Check and Differencing'), width=12, className='row-titles')]),
    
    dbc.Row([
        dbc.Col(html.Label("ADF Stats :") , width=4),
        dbc.Col(html.Label("p-value :"), width=4),
    ]),
    
    dbc.Row([
        dbc.Col(html.Div(id='adf-stat') , width=4),
        dbc.Col(html.Div(id='p-value'), width=4),
        dbc.Col(html.Div(id='stationarity-conclusion'), width=4)
    ]),
    
    dbc.Row([dbc.Col(dcc.Dropdown(
        id='diff-dropdown', options=[{'label': f'Order {i}', 'value': i} for i in range(6)],
        value=1, placeholder='Choose differencing order', persistence=True, persistence_type='session'), width=4)
    ], className='row-controls'),
    
    dbc.Row([
        dbc.Col(dcc.Graph(id='original-plot', className='my-graph'), width=6),
        dbc.Col(dcc.Graph(id='differenced-plot', className='my-graph'), width=6)
    ], className='row-plots'),
    
    dcc.Store(id='selected-region-data'),
    dcc.Store(id='diff-order-store'),
])

@callback(
    Output('adf-stat', 'children'), Output('p-value', 'children'),
    Output('stationarity-conclusion', 'children'), Output('original-plot', 'figure'),
    Output('differenced-plot', 'figure'), Output('diff-order-store', 'data'),
    Input('selected-region-data', 'data'), Input('diff-dropdown', 'value')
)
def update_stationarity_check(selected_region_data, diff_order):
    if selected_region_data:
        data = pd.read_json(selected_region_data)
        data.set_index('Tanggal', inplace=True)
    else:
        return "No data", "No data", "No data", go.Figure(), go.Figure()

    original_series = data['Rata-rata Suhu (°C)']
    adf_stat, p_value = adfuller(original_series)[:2]
    stationarity_conclusion = 'Stationary' if p_value < 0.05 else 'Non-stationary'
    
    original_fig = go.Figure(go.Scatter(x=original_series.index, y=original_series, mode='lines'), layout=my_figlayout)
    original_fig.update_layout(title="Original Data", xaxis_title="Date", yaxis_title="Avg Temperature (°C)")
    
    if diff_order:
        differenced_series = original_series.diff(periods=diff_order).dropna()
        adf_stat, p_value = adfuller(differenced_series)[:2]
        stationarity_conclusion = 'Stationary' if p_value < 0.05 else 'Non-stationary'
        
        differenced_fig = go.Figure(go.Scatter(x=differenced_series.index, y=differenced_series, mode='lines'), layout=my_figlayout)
        differenced_fig.update_layout(title=f"Differenced Data (Order={diff_order})", xaxis_title="Date", yaxis_title="Temperature Difference (°C)")
    else:
        differenced_fig = go.Figure(layout=my_figlayout).update_layout(title="No Differencing Applied", xaxis_title="Date", yaxis_title="Temperature Difference (°C)")

    return f"{adf_stat:.3f}", f"{p_value:.3f}", stationarity_conclusion, original_fig, differenced_fig, diff_order
