import dash
from dash import html, dcc, callback, Input, Output
import dash_bootstrap_components as dbc
import pandas as pd
import plotly.graph_objects as go

dash.register_page(__name__, name='1-Pemilihan Data Wilayah', title='SARIMA | 1-Pemilihan Data Wilayah')

from assets.fig_layout import my_figlayout, my_linelayout

# Load dataset
_data_airp = pd.read_excel('data/Suhu_Permukaan_Riau.xlsx', index_col=None, parse_dates=False)

# Check for missing values in 'Tahun' or 'Bulan'
if _data_airp['Tahun'].isnull().any() or _data_airp['Bulan'].isnull().any():
    raise ValueError("Missing values found in 'Tahun' or 'Bulan' columns.")

# Ensure 'Tahun' and 'Bulan' are integers
_data_airp['Tahun'] = _data_airp['Tahun'].astype(int)
_data_airp['Bulan'] = _data_airp['Bulan'].astype(int)

# Create 'Tanggal' column from 'Tahun' and 'Bulan'
_data_airp['Tanggal'] = pd.to_datetime(
    _data_airp['Tahun'].astype(str) + '-' + _data_airp['Bulan'].astype(str) + '-01',
    errors='coerce'
)

# Check for invalid dates
if _data_airp['Tanggal'].isnull().any():
    print("Invalid dates found. Please check your 'Tahun' and 'Bulan' values.")

# Set 'Tanggal' as the index
_data_airp.set_index('Tanggal', inplace=True)

# Get unique regions for radio items
regions = _data_airp['Kota/Kabupaten'].unique()

# Page layout
layout = dbc.Container([
    dbc.Row([dbc.Col([html.H3(['Dataset Anda'])], width=12, className='row-titles')]),
    dbc.Row([
        dbc.Col([], width=3),
        dbc.Col([html.P(['Pilih daerah:'], className='par')], width=2),
        dbc.Col([dcc.RadioItems(
            options=[{'label': region, 'value': region} for region in regions],
            value=regions[0],
            persistence=True,
            persistence_type='session',
            id='radio-dataset',
            labelStyle={'display': 'inline-block', 'margin-right': '10px'}
        )], width=4),
        dbc.Col([], width=3)
    ], className='row-content'),
    dbc.Row([
        dbc.Col([], width=2),
        dbc.Col([dcc.Loading(id='p1_1-loading', type='circle', children=dcc.Graph(id='fig-pg1', className='my-graph'))], width=8),
        dbc.Col([], width=2)
    ], className='row-content'),
    dcc.Store(id='selected-region-data')
])

# Callback to update the graph and store selected data
@callback(
    Output('fig-pg1', 'figure'),
    Output('selected-region-data', 'data'),
    Input('radio-dataset', 'value')
)
def plot_data(selected_region):
    _data = _data_airp[_data_airp['Kota/Kabupaten'] == selected_region]
    fig = go.Figure(layout=my_figlayout)
    fig.add_trace(go.Scatter(x=_data.index, y=_data['Rata-rata Suhu (°C)'], mode='lines'))
    fig.update_layout(title=f'Temperature Data for {selected_region}', xaxis_title='Date', yaxis_title='Average Temperature (°C)')
    
    # Convert data to JSON for storage
    data_json = _data.reset_index().to_json(date_format='iso')
    
    return fig, data_json