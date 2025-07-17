# step3.py

import dash
from dash import html, dcc, callback, Input, Output
import dash_bootstrap_components as dbc
import pandas as pd
import numpy as np

from assets.acf_pacf_plots import acf_pacf

dash.register_page(__name__, name='3-Identifikasi Model', title='SARIMA | 3-Identifikasi Model')

layout = dbc.Container([
    dbc.Row([dbc.Col(html.H3('Model Identification: ACF and PACF Analysis'), width=12, className='row-titles')]),
    
    dbc.Row([dbc.Col(dcc.Dropdown(
        id='acf-pacf-lag-dropdown', options=[{'label': f'{i} Lags', 'value': i} for i in range(1, 41)],
        value=20, placeholder='Choose number of lags', persistence=True, persistence_type='session'), width=4)
    ], className='row-controls'),
    
    dbc.Row([
        dbc.Col(dcc.Graph(id='acf-plot', className='my-graph'), width=6),
        dbc.Col(dcc.Graph(id='pacf-plot', className='my-graph'), width=6)
    ], className='row-plots'),
    
    dcc.Store(id='selected-region-data'),
    dcc.Store(id='diff-order-store'),
])

@callback(
    Output('acf-plot', 'figure'), Output('pacf-plot', 'figure'),
    Input('selected-region-data', 'data'), Input('diff-order-store', 'data'),
    Input('acf-pacf-lag-dropdown', 'value')
)
def update_acf_pacf_plots(selected_region_data, diff_order, num_lags):
    if selected_region_data:
        data = pd.read_json(selected_region_data)
        data.set_index('Tanggal', inplace=True)
    else:
        return go.Figure(), go.Figure()

    series = data['Rata-rata Suhu (°C)'].diff(periods=diff_order).dropna() if diff_order else data['Rata-rata Suhu (°C)']

    n = len(series)
    significance_line = 1.96 / np.sqrt(n)

    fig_acf, fig_pacf = acf_pacf(series.to_frame(name='Rata-rata Suhu (°C)'), 'Rata-rata Suhu (°C)', num_lags)
    fig_acf.add_hline(y=significance_line, line_dash="dash", line_color="red")
    fig_acf.add_hline(y=-significance_line, line_dash="dash", line_color="red")
    fig_pacf.add_hline(y=significance_line, line_dash="dash", line_color="red")
    fig_pacf.add_hline(y=-significance_line, line_dash="dash", line_color="red")

    return fig_acf, fig_pacf
