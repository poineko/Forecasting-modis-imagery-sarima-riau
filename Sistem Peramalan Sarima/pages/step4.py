# step 4

import dash
from dash import html, dcc, dash_table, callback, Input, Output, State
import dash_bootstrap_components as dbc
import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX  
import statsmodels.api as sm  
import plotly.graph_objects as go  
import numpy as np 
from sklearn.metrics import mean_absolute_error, mean_squared_error
from scipy import stats


from assets.fig_layout import my_figlayout, my_linelayout
from assets.acf_pacf_plots import acf_pacf_resid  # Impor fungsi untuk ACF dan PACF plots

dash.register_page(__name__, name='4-Estimasi Parameter', title='SARIMA | 4-Estimasi Parameter')

### PAGE LAYOUT ###############################################################################################################

layout = dbc.Container([
    # Title
    dbc.Row([dbc.Col(html.H3('Parameter Estimation: SARIMA Model Selection'), width=12)]),
    
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

    # Model parameter selection with dcc.Dropdown
    dbc.Row([
        dbc.Col([
            html.Label('Select Parameters for Model 1'),

            html.Div([
                html.Label("p (AR order)"),
                dcc.Dropdown(id='p1', options=[{'label': i, 'value': i} for i in range(40)], placeholder='p')
            ], style={'marginBottom': '10px'}),

            html.Div([
                html.Label("d (Differencing Order)"),
                html.Div(id='diff-order-display-d-1', style={'fontSize': '18px', 'fontWeight': 'bold', 'marginTop': '5px'})
            ], style={'marginBottom': '10px'}),

            html.Div([
                html.Label("q (MA order)"),
                dcc.Dropdown(id='q1', options=[{'label': i, 'value': i} for i in range(40)], placeholder='q')
            ], style={'marginBottom': '10px'}),

            html.Div([
                html.Label("P (Seasonal AR order)"),
                dcc.Dropdown(id='P1', options=[{'label': i, 'value': i} for i in range(4)], placeholder='P')
            ], style={'marginBottom': '10px'}),

            html.Div([
                html.Label("D (Seasonal Differencing Order)"),
                html.Div(id='diff-order-display-D-1', style={'fontSize': '18px', 'fontWeight': 'bold', 'marginTop': '5px'})
            ], style={'marginBottom': '10px'}),

            html.Div([
                html.Label("Q (Seasonal MA order)"),
                dcc.Dropdown(id='Q1', options=[{'label': i, 'value': i} for i in range(4)], placeholder='Q')
            ], style={'marginBottom': '10px'}),
        ], width=4),

        dbc.Col([
            html.Label('Select Parameters for Model 2'),

            html.Div([
                html.Label("p (AR order)"),
                dcc.Dropdown(id='p2', options=[{'label': i, 'value': i} for i in range(40)], placeholder='p')
            ], style={'marginBottom': '10px'}),

            html.Div([
                html.Label("d (Differencing Order)"),
                html.Div(id='diff-order-display-d-2', style={'fontSize': '18px', 'fontWeight': 'bold', 'marginTop': '5px'})
            ], style={'marginBottom': '10px'}),

            html.Div([
                html.Label("q (MA order)"),
                dcc.Dropdown(id='q2', options=[{'label': i, 'value': i} for i in range(40)], placeholder='q')
            ], style={'marginBottom': '10px'}),

            html.Div([
                html.Label("P (Seasonal AR order)"),
                dcc.Dropdown(id='P2', options=[{'label': i, 'value': i} for i in range(4)], placeholder='P')
            ], style={'marginBottom': '10px'}),

            html.Div([
                html.Label("D (Seasonal Differencing Order)"),
                html.Div(id='diff-order-display-D-2', style={'fontSize': '18px', 'fontWeight': 'bold', 'marginTop': '5px'})
            ], style={'marginBottom': '10px'}),

            html.Div([
                html.Label("Q (Seasonal MA order)"),
                dcc.Dropdown(id='Q2', options=[{'label': i, 'value': i} for i in range(4)], placeholder='Q')
            ], style={'marginBottom': '10px'}),
        ], width=4),

    dbc.Col([
        html.Label('Select Parameters for Model 3'),

        html.Div([
            html.Label("p (AR order)"),
            dcc.Dropdown(id='p3', options=[{'label': i, 'value': i} for i in range(40)], placeholder='p')
        ], style={'marginBottom': '10px'}),

        html.Div([
            html.Label("d (Differencing Order)"),
            html.Div(id='diff-order-display-d-3', style={'fontSize': '18px', 'fontWeight': 'bold', 'marginTop': '5px'})
        ], style={'marginBottom': '10px'}),

        html.Div([
            html.Label("q (MA order)"),
            dcc.Dropdown(id='q3', options=[{'label': i, 'value': i} for i in range(40)], placeholder='q')
        ], style={'marginBottom': '10px'}),

        html.Div([
            html.Label("P (Seasonal AR order)"),
            dcc.Dropdown(id='P3', options=[{'label': i, 'value': i} for i in range(4)], placeholder='P')
        ], style={'marginBottom': '10px'}),

        html.Div([
            html.Label("D (Seasonal Differencing Order)"),
            html.Div(id='diff-order-display-D-3', style={'fontSize': '18px', 'fontWeight': 'bold', 'marginTop': '5px'})
        ], style={'marginBottom': '10px'}),

        html.Div([
            html.Label("Q (Seasonal MA order)"),
            dcc.Dropdown(id='Q3', options=[{'label': i, 'value': i} for i in range(4)], placeholder='Q')
        ], style={'marginBottom': '10px'}),
    ], width=4)
]),


    # Button to trigger parameter estimation
    dbc.Row([dbc.Col(dbc.Button("Estimate Parameters", id="estimate-button", color="primary"), width=12)]),
    
    # Output area for SARIMAX Results
    dbc.Row([dbc.Col(html.Pre(id='sarimax-results', style={'white-space': 'pre-wrap'}), width=12)]),
    
    # Diagnostic check button
    dbc.Row([dbc.Col(dbc.Button("Check Diagnostic", id="diagnostic-button", color="secondary"), width=12)]),
    

    # Output area for diagnostic checks
    dbc.Row([
        dbc.Col(html.Div(id='diagnostic-results'), width=12)  # Container for diagnostic results
    ]),
    
    # Table for diagnostic metrics
    dbc.Row([dbc.Col(html.Div(id='diagnostic-table'), width=12)]),

    # # Output area for diagnostic checks
    # dbc.Row([
    #     dbc.Col(dcc.Graph(id='acf-residual-plot', className='my-graph', style={'display': 'none'}), width=6),
    #     dbc.Col(dcc.Graph(id='pacf-residual-plot', className='my-graph', style={'display': 'none'}), width=6)
    # ], className='row-plots'),
    # dbc.Row([dbc.Col(dcc.Graph(id='qq-plot', style={'display': 'none'}), width=12)]),
    
    # # Table for diagnostic metrics
    # dbc.Row([dbc.Col(html.Div(id='diagnostic-table', style={'display': 'none'}), width=12)]),
    
    # dcc.Store to receive data 
    dcc.Store(id='selected-region-data'),
    dcc.Store(id='diff-order-store'),
    dcc.Store(id='model-results-store'),  
    # dcc.Store(id='original-data-store'),
    
])
## PAGE CALLBACKS ###############################################################################################################

@callback(
    [
        Output('diff-order-display-d-1', 'children'),
        Output('diff-order-display-d-2', 'children'),
        Output('diff-order-display-d-3', 'children'),
        Output('diff-order-display-D-1', 'children'),
        Output('diff-order-display-D-2', 'children'),
        Output('diff-order-display-D-3', 'children')
     ],
    [Input('diff-order-store', 'data')]
)
def diff_order_display(diff_order):
    if diff_order is None:
        return "No differencing applied", "No differencing applied", "No differencing applied", \
               "No seasonal differencing applied", "No seasonal differencing applied", "No seasonal differencing applied"

    return (f"Selected Differencing Order: {diff_order}",
            f"Selected Differencing Order: {diff_order}",
            f"Selected Differencing Order: {diff_order}",
            f"Selected Differencing Order: {diff_order}",
            f"Selected Differencing Order: {diff_order}",
            f"Selected Differencing Order: {diff_order}")

@callback(
    Output('sarimax-results', 'children'),
    Output('model-results-store', 'data'), 
    Input('estimate-button', 'n_clicks'),
    State('diff-order-store', 'data'),
    State('start-year', 'value'),
    State('end-year', 'value'),
    State('p1', 'value'), State('q1', 'value'), State('P1', 'value'), State('Q1', 'value'),
    State('p2', 'value'), State('q2', 'value'), State('P2', 'value'), State('Q2', 'value'),
    State('p3', 'value'), State('q3', 'value'), State('P3', 'value'), State('Q3', 'value'),
    State('selected-region-data', 'data')
)
def estimate_parameters(n_clicks, diff_order, start_year, end_year, p1, q1, P1, Q1, p2, q2, P2, Q2, p3, q3, P3, Q3, selected_region_data):
    if not n_clicks or not selected_region_data:
        return "Please ensure data and parameters are loaded."
    
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
    
    # Model parameters to test
    models = [
        (p1, diff_order, q1, P1, diff_order, Q1, 12),
        (p2, diff_order, q2, P2, diff_order, Q2, 12),
        (p3, diff_order, q3, P3, diff_order, Q3, 12)
    ]

    # SARIMAX Results output
    results_text = ""
    model_results = [] 
    for idx, (p, d, q, P, D, Q, s) in enumerate(models):
        try:
            model = SARIMAX(ts, order=(p, d, q), seasonal_order=(P, D, Q, s))
            results = model.fit(disp=False)
            results_text += f"Model {idx+1}: SARIMA({p},{d},{q})x({P},{D},{Q},{s})\n"
            results_text += results.summary().tables[0].as_text() + "\n\n"

            residuals = results.resid.tolist()  # Store residuals as a list
            fitted_values = results.fittedvalues.tolist()  # Store fitted values as a list
            
            # Save model info, residuals, and fitted values in JSON-serializable format
            model_results.append({
                "model": f"SARIMA({p},{d},{q})x({P},{D},{Q},{s})",
                "aic": results.aic,
                "bic": results.bic,
                "params": results.params.to_dict(),
                "residuals": residuals,
                "fitted_values": fitted_values
            })
        
        except MemoryError:
            results_text += f"Model {idx+1}: SARIMA({p},{d},{q})x({P},{D},{Q},{s}) could not be estimated due to memory constraints.\n\n"
        except Exception as e:
            results_text += f"Model {idx+1}: Error occurred: {str(e)}\n\n"

    return results_text, model_results

@callback(
    Output('diagnostic-results', 'children'),
    Output('diagnostic-table', 'children'),
    Input('diagnostic-button', 'n_clicks'),
    State('start-year', 'value'),
    State('end-year', 'value'),
    State('diff-order-store', 'data'),
    State('selected-region-data', 'data'),
    State('model-results-store', 'data'),
)
def check_diagnostics(n_clicks, start_year, end_year, diff_order, selected_region_data, model_results):
    if n_clicks is None or not model_results:
        return html.Div("No diagnostics available."), html.Div("")

    # Load data
    data = pd.read_json(selected_region_data)
    data['Tanggal'] = pd.to_datetime(data['Tanggal'])
    data.set_index('Tanggal', inplace=True)

    if start_year:
        data = data[data.index >= f"{start_year}-01-01"]
    if end_year:
        data = data[data.index <= f"{end_year}-12-31"]

    ts = data['Rata-rata Suhu (°C)'].diff(periods=diff_order).dropna() if diff_order else data['Rata-rata Suhu (°C)']

    # Initialize diagnostics list
    diagnostics = []
    result_divs = []

    for model_result in model_results:
        residuals = model_result['residuals']
        residuals_df = pd.DataFrame(residuals, columns=['Residuals'])

        # Calculate model metrics
        fitted_values = model_result['fitted_values']

        if diff_order > 0:
            fitted_values = pd.Series(fitted_values).cumsum() + ts.iloc[0]

        # Calculate model metrics
        metrics = {
            'Model': model_result['model'],
            'AIC': f"{model_result['aic']:.3f}",  # AIC with 3 decimal places
            'p-value (Ljung-Box)': f"{sm.stats.acorr_ljungbox(residuals, lags=[10], return_df=True)['lb_pvalue'].values[0]:.3f}",  # p-value with 3 decimal places
            'MAE': f"{mean_absolute_error(ts, fitted_values):.2f}",  # MAE with 2 decimal places
            'MSE': f"{mean_squared_error(ts, fitted_values):.2f}",  # MSE with 2 decimal places
            'RMSE': f"{mean_squared_error(ts, fitted_values, squared=False):.2f}"  # RMSE with 2 decimal places
        }
        diagnostics.append(metrics)

        # Create ACF and PACF plots
        n = len(residuals_df)
        significance_line = 1.96 / np.sqrt(n)

        fig_acf, fig_pacf = acf_pacf_resid(residuals_df, 'Residuals', num_lags=40)
        fig_acf.add_hline(y=significance_line, line_dash="dash", line_color="red")
        fig_acf.add_hline(y=-significance_line, line_dash="dash", line_color="red")
        fig_pacf.add_hline(y=significance_line, line_dash="dash", line_color="red")
        fig_pacf.add_hline(y=-significance_line, line_dash="dash", line_color="red")
        
        # Generate QQ-plot for each model
        qq_data = stats.probplot(residuals, dist="norm")
        fig_qq = go.Figure(layout=my_figlayout)
        fig_qq.add_trace(go.Scatter(x=qq_data[0][0], y=qq_data[0][1], mode='markers', name=model_result['model']))
        fig_qq.add_trace(go.Scatter(x=qq_data[0][0], y=qq_data[0][0], line=dict(color='red'), name='Normalitas'))


        # Append results for the current model in row layout
        result_divs.append(
            html.Div([
                html.H4(f"Model: {model_result['model']}"),
                dbc.Row([
                    dbc.Col(dcc.Graph(figure=fig_acf, id=f'acf-plot-{model_result["model"]}'), width=6),
                    dbc.Col(dcc.Graph(figure=fig_pacf, id=f'pacf-plot-{model_result["model"]}'), width=6),
                ], className='row-plots'),
                dcc.Graph(figure=fig_qq, id=f'qq-plot-{model_result["model"]}'),
                html.Hr()
            ])
        )

    # Diagnostic Table with centered columns, left-aligned rows, and customized hover effect
    diagnostic_table = dash_table.DataTable(
        columns=[{"name": col, "id": col} for col in diagnostics[0].keys()],
        data=diagnostics,
        style_table={'overflowX': 'auto'},
        style_cell={
            'textAlign': 'center',  # Center-aligns column content
            'padding': '10px',
            'backgroundColor': 'rgba(20, 50, 60, 0.8)',
            'color': '#FFFFFF',  # White text for readability on darker backgrounds
            'fontWeight': 'bold'
        },
        style_data={
            'textAlign': 'left',  # Left-aligns row data
        },
        style_data_conditional=[
            {
                'if': {'state': 'hover'},  # Apply style on hover
                'backgroundColor': 'rgba(20, 50, 60, 0.8)',  # Dark teal shade to complement the gradient background
                'color': '#E0FFFF',  # Light text color to ensure readability on hover
            },
        ],
        style_header_conditional=[
            {
                'if': {'state': 'hover'},  # Apply style on hover
                'backgroundColor': 'rgba(20, 50, 60, 0.8)',  # Dark teal shade to complement the gradient background
                'color': '#E0FFFF',  # Light text color to ensure readability on hover
            },
        ],
        page_size=8
    )
    # Tampilkan tabel jika ada data diagnostik
    # table_style = {'display': 'block'} if diagnostics else {'display': 'none'}

    # Combine everything into a single Div
    return html.Div([
        *result_divs,
        html.Hr(),
    ]), diagnostic_table