#  Step 5

import dash
from dash import html, dcc, dash_table, callback, Input, Output, State
import plotly.graph_objects as go
import dash_bootstrap_components as dbc
import pandas as pd
import numpy as np
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_absolute_error, mean_squared_error
import warnings
warnings.filterwarnings("ignore")

dash.register_page(__name__, name='5-Peramalan', title='SARIMA | 4-Peramalan')

from assets.fig_layout import my_figlayout, train_linelayout, test_linelayout, pred_linelayout

# Example options for Dropdowns
_opts = [{'label': str(i), 'value': i} for i in range(0, 13)]  # Example for p, q, etc.

### PAGE LAYOUT ###############################################################################################################

layout = dbc.Container([
    dbc.Row([dbc.Col(html.H3('Final Model: Fit & Prediction'), width=12, className='row-titles')]),
     # Year selection
    dbc.Row([
        dbc.Col([
            html.Label("Select Start Year"),
            dcc.Dropdown(id='start-year', 
                         options=[{'label': str(year), 'value': year} for year in range(2001, 2021)],
                         placeholder="Start Year"),
        ], width=6),
        
        dbc.Col([
            html.Label("Select End Year"),
            dcc.Dropdown(id='end-year', 
                         options=[{'label': str(year), 'value': year} for year in range(2020, 2001, -1)],
                         placeholder="End Year"),
        ], width=6),
    ]),
    
    dbc.Row([
        dbc.Col([html.Label("p", style={'textAlign': 'center', }),
                 dcc.Dropdown(id='p', options=[{'label': i, 'value': i} for i in range(40)], placeholder='p',
                              style={'width': '100%', 'marginTop': '5px', 'textAlign': 'center'})], width=2),
        dbc.Col([html.Label("d", style={'textAlign': 'center', 'width': '100%'}),
                 html.Div(id='diff-order-display-d', style={'fontSize': '18px', 'fontWeight': 'bold', 'textAlign': 'center', 'marginTop': '5px'})], width=2),
        dbc.Col([html.Label("q", style={'textAlign': 'center', 'width': '100%'}),
                 dcc.Dropdown(id='q', options=[{'label': i, 'value': i} for i in range(40)], placeholder='p',
                              style={'width': '100%', 'marginTop': '5px', 'textAlign': 'center'})], width=2),
        dbc.Col([html.Label("P", style={'textAlign': 'center', 'width': '100%'}),
                 dcc.Dropdown(id='P', options=[{'label': i, 'value': i} for i in range(4)], placeholder='P',
                              style={'width': '100%', 'marginTop': '5px', 'textAlign': 'center'})], width=2),
        dbc.Col([html.Label("D", style={'textAlign': 'center', 'width': '100%'}),
                 html.Div(id='diff-order-display-D', style={'fontSize': '18px', 'fontWeight': 'bold', 'textAlign': 'center','marginTop': '5px'})], width=2),
        dbc.Col([html.Label("Q", style={'textAlign': 'center', 'width': '100%'}),
                 dcc.Dropdown(id='Q', options=[{'label': i, 'value': i} for i in range(4)], placeholder='Q',
                              style={'width': '100%', 'marginTop': '5px', 'textAlign': 'center'})], width=2),
        dbc.Col([html.Label("s (Seasonal Periods)", style={'textAlign': 'center'}),
                 html.Div('12', style={'fontSize': '18px', 'fontWeight': 'bold', 'marginTop': '5px', 'textAlign': 'center'})], width=2)
    ]),
    
    dbc.Row([dbc.Col(html.P('Lakukan Peramalan Untuk : '), width=4),
             dbc.Col([dcc.Dropdown(options=list(range(1, 13)), placeholder='n', id='n-offset')], width=2),
             dbc.Col([dcc.Dropdown(options=['Tahun', 'Bulan'], placeholder='Time Type', id='n-timetype')], width=2)]),
    
    dbc.Row([
        html.Label("Diffrecing Data", style={'textAlign': 'center', 'width': '100%'}),
        dbc.Col(dcc.Loading(id='m1-loading', children=dcc.Graph(id='fig-pg41')), width=10)]),
    
    # show graph orinial dan data button
    dbc.Row([dbc.Col(dbc.Button("Lihat Data", id="show-data-button", color="primary"), width=12)]),
    
    dbc.Row([
        html.Label("Original Data", style={'textAlign': 'center', 'width': '100%'}),
        dbc.Col(dcc.Loading(id='m1-loading', children=dcc.Graph(id='fig-ori-data')), width=10)]),
    
    # Table data hasil peramalan
    dbc.Row([dbc.Col(html.Div(id='result-data-table'), width=12)]),

    dcc.Store(id='selected-region-data'),
    dcc.Store(id='diff-order-store'),
])

### PAGE CALLBACKS ##############################################################################################################

@callback(
    [
        Output('diff-order-display-d', 'children'),
        Output('diff-order-display-D', 'children')
     ],
    [Input('diff-order-store', 'data')]
)
def diff_order_display(diff_order):
    if diff_order is None:
        return "No differencing applied",  "No seasonal differencing applied"
    return (f"Order: {diff_order}", f" Order: {diff_order}")

@callback(
    Output('fig-pg41', 'figure'),
    Input('p', 'value'),
    Input('q', 'value'),
    Input('P', 'value'),
    Input('Q', 'value'),
    Input('n-offset', 'value'),
    Input('n-timetype', 'value'),
    Input('selected-region-data', 'data'),
    Input('diff-order-store', 'data'),
    State('start-year', 'value'),
    State('end-year', 'value'),
)
def predict_(p, q, P, Q, n_offset, n_timetype, selected_region_data, diff_order, start_year, end_year):
    if not selected_region_data or n_offset is None:
        return go.Figure(layout=my_figlayout)

    try:
        data = pd.read_json(selected_region_data)
    except ValueError:
        return "Error: `selected_region_data` is not in JSON format."
    
    # Filter the data based on selected date range
    data['Tanggal'] = pd.to_datetime(data['Tanggal'])
    if start_year:
        data = data[data['Tanggal'] >= f"{start_year}-01-01"]
    if end_year:
        data = data[data['Tanggal'] <= f"{end_year}-12-31"]

    # Process the time series data
    data.set_index('Tanggal', inplace=True)
    ts = data['Rata-rata Suhu (°C)'].diff(periods=diff_order).dropna() if diff_order else data['Rata-rata Suhu (°C)']

    # Convert model parameters to integer
    p, q, P, Q = map(int, [p, q, P, Q])
    seasonal_period = 12  # Set seasonal period for monthly data

    # Set up SARIMA model with specified parameters
    try:
        model = SARIMAX(ts, order=(p, diff_order, q), seasonal_order=(P, diff_order, Q, seasonal_period))
        fitted_model = model.fit(disp=-1)
        
        # Forecasting steps based on input n_offset and n_timetype
        if n_timetype == 'Tahun':
            forecast_steps = n_offset * 12  # Convert years to months
        elif n_timetype == 'Bulan':
            forecast_steps = n_offset
        else:
            forecast_steps = n_offset  # Data points
        

        forecast = fitted_model.get_forecast(steps=forecast_steps)
        forecast_values = forecast.predicted_mean
        forecast_ci = forecast.conf_int()

        # Generate noise untuk forecast
        noise = np.random.normal(loc=0, scale=fitted_model.resid.std(), size=forecast_steps)

        # Tambahkan noise ke forecast
        forecast_values = forecast.predicted_mean + noise
        forecast_ci.iloc[:, 0] += noise
        forecast_ci.iloc[:, 1] += noise


        # Calculate metrics
        mae = mean_absolute_error(ts[-len(forecast_values):], forecast_values)
        mse = mean_squared_error(ts[-len(forecast_values):], forecast_values)
        rmse = np.sqrt(mse)

        # Visualization
        fig1 = go.Figure(layout=my_figlayout)

        # Trace untuk data aktual (Actual) - Garis Solid
        fig1.add_trace(go.Scatter(x=ts.index, y=ts, mode='lines', name='Actual', line=train_linelayout))

        # Define test set as the last `forecast_steps` data points
        test = ts[-forecast_steps:]

        # Trace untuk data test (Test) - Garis Putus-putus
        fig1.add_trace(go.Scatter(x=test.index, y=test, mode='lines', name='Test', line=dict(dash='dash', color='rgba(128, 128, 128, 0.6)', width=2)))

        # Trace untuk peramalan (Forecasting) - Warna Cerah (misalnya Oranye Terang)
        fig1.add_trace(go.Scatter(x=forecast_values.index, y=forecast_values, mode='lines', name='Forecast', 
                                 line=dict(color='rgba(255, 165, 0, 0.8)', width=3)))

        # Trace untuk batas bawah dari 95% CI (Lower Bound) - Merah Muda
        fig1.add_trace(go.Scatter(x=forecast_ci.index, y=forecast_ci.iloc[:, 0], mode='lines', 
                                 name='95% CI Lower Bound', line=dict(color='rgba(255, 99, 132, 0.8)', width=2)))

        # Trace untuk batas atas dari 95% CI (Upper Bound) - Biru
        fig1.add_trace(go.Scatter(x=forecast_ci.index, y=forecast_ci.iloc[:, 1], mode='lines', 
                                 name='95% CI Upper Bound', line=dict(color='rgba(30, 144, 255, 0.8)', width=2), 
                                 fill='tonexty', fillcolor='rgba(178, 211, 194,0.11)'))

        # Memperbarui label sumbu X dan Y
        fig1.update_xaxes(title_text='Time')
        fig1.update_yaxes(title_text='Rata-rata Suhu (°C)')

        # Memperbarui layout dengan menambahkan informasi error metrics (MAE, MSE, RMSE)
        fig1.update_layout(title=f"Forecast: MAE={mae:.2f}, MSE={mse:.2f}, RMSE={rmse:.2f}", height=500)

        # Mengembalikan figure untuk ditampilkan
        return fig1

    except Exception as e:
        print(f"Error fitting model: {e}")
        return go.Figure(layout=my_figlayout)

@callback(
    Output('fig-ori-data', 'figure'),
    Output('result-data-table', 'children'),
    Input('show-data-button', 'n_clicks'),
    Input('p', 'value'),
    Input('q', 'value'),
    Input('P', 'value'),
    Input('Q', 'value'),
    Input('n-offset', 'value'),
    Input('n-timetype', 'value'),
    Input('selected-region-data', 'data'),
    Input('diff-order-store', 'data'),
    State('start-year', 'value'),
    State('end-year', 'value'),
)
def predict_original(n_clicks, p, q, P, Q, n_offset, n_timetype, selected_region_data, diff_order, start_year, end_year):
    if not n_clicks or not selected_region_data:
        return go.Figure(layout=my_figlayout), "Data tidak tersedia"
    
    if not selected_region_data or n_offset is None:
        return go.Figure(layout=my_figlayout), "Data tidak tersedia"
    
    try:
        data = pd.read_json(selected_region_data)
    except ValueError:
        return go.Figure(layout=my_figlayout), "Error: `selected_region_data` is not in JSON format."
    
    # Filter data based on selected time range
    data['Tanggal'] = pd.to_datetime(data['Tanggal'])
    if start_year:
        data = data[data['Tanggal'] >= f"{start_year}-01-01"]
    if end_year:
        data = data[data['Tanggal'] <= f"{end_year}-12-31"]
    
    # Set up time index and time series data
    data.set_index('Tanggal', inplace=True)
    original_ts = data['Rata-rata Suhu (°C)']  # Save the original time series
    ts = original_ts.diff(periods=diff_order).dropna() if diff_order else original_ts

    # Convert model parameters to integers
    p, q, P, Q = map(int, [p, q, P, Q])
    seasonal_period = 12  # Set seasonal period for monthly data

    # Set up SARIMA model with parameters
    try:
        model = SARIMAX(ts, order=(p, diff_order, q), seasonal_order=(P, diff_order, Q, seasonal_period))
        fitted_model = model.fit(disp=False)
        
        # Determine forecast steps based on input n_offset and n_timetype
        if n_timetype == 'Tahun':
            forecast_steps = n_offset * 12  # Convert years to months
        elif n_timetype == 'Bulan':
            forecast_steps = n_offset
        else:
            forecast_steps = n_offset  # Data points
        
        # Forecast result
        forecast = fitted_model.get_forecast(steps=forecast_steps)
        forecast_values = forecast.predicted_mean
        forecast_ci = forecast.conf_int()

        # Reverse differencing to obtain original scale
        if diff_order > 0:
            forecast_values = forecast_values.cumsum() + original_ts.iloc[-1]
            forecast_ci = forecast_ci.cumsum() + original_ts.iloc[-1]
        
        # Generate noise untuk forecast
        noise = np.random.normal(loc=0, scale=fitted_model.resid.std(), size=forecast_steps)

        # Tambahkan noise ke forecast
        forecast_values += noise
        forecast_ci.iloc[:, 0] += noise
        forecast_ci.iloc[:, 1] += noise

        # Calculate error metrics
        mae = mean_absolute_error(original_ts[-len(forecast_values):], forecast_values)
        mse = mean_squared_error(original_ts[-len(forecast_values):], forecast_values)
        rmse = np.sqrt(mse)

        # Create plot for original and forecast data
        fig1 = go.Figure(layout=my_figlayout)
        fig1.add_trace(go.Scatter(x=original_ts.index, y=original_ts, mode='lines', name='Actual', line=dict(color='blue', width=2)))
        fig1.add_trace(go.Scatter(x=forecast_values.index, y=forecast_values, mode='lines', name='Forecast', 
                                  line=dict(color='orange', width=2)))
        fig1.add_trace(go.Scatter(x=forecast_ci.index, y=forecast_ci.iloc[:, 0], mode='lines', name='95% CI Lower Bound', 
                                  line=dict(color='rgba(255, 99, 132, 0.8)', width=1)))
        fig1.add_trace(go.Scatter(x=forecast_ci.index, y=forecast_ci.iloc[:, 1], mode='lines', name='95% CI Upper Bound', 
                                  line=dict(color='rgba(30, 144, 255, 0.8)', width=1), fill='tonexty', 
                                  fillcolor='rgba(178, 211, 194, 0.2)'))
        
        # Update layout with error metrics
        fig1.update_layout(
            title=f"Forecast: MAE={mae:.2f}, MSE={mse:.2f}, RMSE={rmse:.2f}",
            xaxis_title='Time',
            yaxis_title='Rata-rata Suhu (°C)'
        )

        # Create table with forecast data
        forecast_df = pd.DataFrame({
            'Tanggal': forecast_values.index,
            'Prediksi Suhu (°C)': forecast_values.values,
            'Lower Bound (95%)': forecast_ci.iloc[:, 0].values,
            'Upper Bound (95%)': forecast_ci.iloc[:, 1].values
        })

        result_data_table = dash_table.DataTable(
            columns=[{"name": col, "id": col} for col in forecast_df.columns],
            data=forecast_df.to_dict('records'),
            style_table={'overflowX': 'auto'},
            style_cell={
                'textAlign': 'center',  
                'padding': '10px',
                'backgroundColor': 'rgba(20, 50, 60, 0.8)',
                'color': '#FFFFFF',  
                'fontWeight': 'bold'
            },
            style_data={
                'textAlign': 'left',  
            },
            style_data_conditional=[
                {
                    'if': {'state': 'hover'},  
                    'backgroundColor': 'rgba(20, 50, 60, 0.8)',  
                    'color': '#E0FFFF',  
                },
            ],
            style_header={'backgroundColor': 'rgb(30, 30, 30)', 'fontWeight': 'bold'},
            page_size=8
        )

        return fig1, result_data_table

    except Exception as e:
        print(f"Error fitting model: {e}")
        return go.Figure(layout=my_figlayout), f"Error: {str(e)}"