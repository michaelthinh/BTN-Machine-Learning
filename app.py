import dash
from dash import dcc, html
import plotly.graph_objs as go
from dash.dependencies import Input, Output, State
from datetime import timedelta
from trading_data import getDataFromCoin
from constant import coin_labels, algorithms, timeframes, day_number, windowSize
import pandas as pd
from model.factory import ModelPredictServiceFactory
from model.utils import ROCCalculator

# initialize
app = dash.Dash()
server = app.server

# implement ui
app.layout = html.Div(
	style={'font-family': 'Helvetica Neue'},
	children=[
    #header
    html.Div(
              children=[
                html.Span("Đồ án nhóm cuối kì ứng dụng Machine Learning"),
                html.Span("Nhóm 04")
              ],
              className='header',
          ),

    # Tool bar
    html.Div(
        style={
            "padding": "12px 20px",
        },
        children=[
            html.Div(
                style={"display": "flex", "gap": "20px", "align-items": "center"},
                children=[
                    dcc.Dropdown(
                        id="coin-dropdown",
                        options=coin_labels,
                        value=coin_labels[0]['value'],
                        clearable=False,
                        style={"width": "200px"},
                    ),
                    dcc.Dropdown(
                        id="algorithm-dropdown",
                        options=algorithms,
                        value=algorithms[0]['value'],
                        clearable=False,
                        style={"width": "200px"},
                    ),
                    html.H5("Timeframe:", style={"marginLeft": "20px"}),
                    dcc.Dropdown(
                        id="timeframe",
                        options=list(timeframes.values()),
                        value=list(timeframes.values())[0]['value'],
                        clearable=False,
                        style={"width": "200px"},
                    ),
                    html.H5("Number of timeframe:", style={"marginLeft": "20px"}),
                    dcc.Dropdown(
                        id="day-number",
                        options=day_number,
                        value=day_number[0],
                        clearable=False,
                        style={"width": "200px"},
                    ),
                ],
            ),
            dcc.Dropdown(
                placeholder="Select attributes",
                multi=True,
                id="feature-dropdown",
                options=[
                    {"label": "Close Price", "value": "close"},
                    {"label": "ROC", "value": "ROC"},
                ],
                value=["close", "ROC"],
                clearable=False,
                style={"width": "auto", "width": "250px"},
            ),
        ],
    ),
    # Title
    html.Div(
        style={"display": "flex", "alignItems":"center"},
        children=[
            html.H1(
                "Trading Price Analysis Dashboard",
                style={"textAlign": "left", "margin": "20px"},
            ),
            # Loading process
            dcc.Loading(id='loading-indicator', type='default', children=[
                html.Div(id='loading-placeholder', style={"textAlign": "center"})
            ]),
        ]
    ),
    # Graph presentation
    html.Div(
        children=[
            dcc.Graph(
                id="candlestick-graph",
            )
        ],
        style={"border": "solid 1px gray", "marginTop": "10px"},
    ),
    
    # footer
    html.Div(
            children=[
                html.P('Các thành viên:'),
                html.Div(
                    children=[
                        html.Div('Mai Cường Thịnh'),
                        html.Div('Nguyễn Hữu Thiện'),
                        html.Div('Ngô Nguyễn Quang Tú'),
                        html.Div('Lưu Tuấn Khanh'),
                    ]
                )
            ],
            className='footer',
        ),
    dcc.Interval(id="interval-component", interval=2 * 1000, n_intervals=0),
    dcc.Interval(id='hide-loading-interval', interval=1* 1000, n_intervals=0, max_intervals=1)
])

# TRADING PRICE
@app.callback(
    Output('loading-placeholder', 'children'),
    [
        Input("coin-dropdown", "value"),
        Input("algorithm-dropdown", "value"),
        Input("feature-dropdown", "value"),
        Input("day-number", "value"),
        Input("timeframe", "value"),
        Input('hide-loading-interval', 'n_intervals')
    ],
    [State('loading-placeholder', 'children')]
)
def update_loading_state(coin, algorithm, features, day_number, timeframe, n_intervals, current_state):
    ctx = dash.callback_context

    if not ctx.triggered:
        return ''

    input_id = ctx.triggered[0]['prop_id'].split('.')[0]

    if input_id in ["coin-dropdown", "algorithm-dropdown", "feature-dropdown", "day-number", "timeframe"]:
        return 'Loading...'
    elif input_id == 'hide-loading-interval' and n_intervals > 0:
        return ''

    return current_state

@app.callback(
    Output('hide-loading-interval', 'n_intervals'),
    [
        Input("coin-dropdown", "value"),
        Input("algorithm-dropdown", "value"),
        Input("feature-dropdown", "value"),
        Input("day-number", "value"),
        Input("timeframe", "value")
    ]
)
def start_loading_interval(coin, algorithm, features, day_number, timeframe):
    return 0

@app.callback(
    Output("candlestick-graph", "figure"),
    [
        Input("coin-dropdown", "value"),
        Input("algorithm-dropdown", "value"),
        Input("feature-dropdown", "value"),
        Input("day-number", "value"),
        Input("timeframe", "value"),
        Input("interval-component", "n_intervals")
    ]
)
def update_trading_price_graph(coin, algorithm, features, day_number, timeframe, n_intervals):
    df = getDataFromCoin(coin, timeframe, day_number)
    figure = go.Figure(
        data=[
            go.Candlestick(
                x=df.timestamp,
                open=df.open,
                high=df.high,
                low=df.low,
                close=df.close,
                name='Trading Price'
            )
        ]
    )

    if timeframe == timeframes["day"]["value"] and day_number >= windowSize:
        df['ROC'] = ROCCalculator().fromClose(df['close'])
        predictService = ModelPredictServiceFactory.getModelPredictService(
            modelName=algorithm, features=features, coin=coin
        )
        prediction = predictService.execute(df)
        addPredictCandle(figure, df.timestamp.max() + timedelta(1), prediction)

    figure.update_layout(
        title=f"Trading Price Analysis ({coin})",
        yaxis_title="Trading Price (USD)",
        xaxis_title="Date",
        xaxis_rangeslider_visible=False,
    )

    return figure

def addPredictCandle(figure, date, candel_df: pd.DataFrame):
    new_candle = go.Candlestick(
        x=[date],
        open=candel_df["open"],
        high=candel_df["high"],
        low=candel_df["low"],
        close=candel_df["close"],
        increasing=dict(line=dict(color="blue")),
        decreasing=dict(line=dict(color="#3C3B6E")),
        name="Predicted Trading Price",
    )
    figure.add_trace(new_candle)

if __name__ == "__main__":
    app.run_server(debug=True)