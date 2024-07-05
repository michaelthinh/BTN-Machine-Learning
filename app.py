import dash
from dash import dcc
from dash import html
import plotly.graph_objs as go
from dash.dependencies import Input, Output
from datetime import date, timedelta
from trading_data import getDataFromCoin
from constant import coins, algorithms, timeframes, day_number

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

    # tool bar
    html.Div(
      style={'padding': '12px 20px',},
			children=[
				html.Div(
        style={"display": "flex", "gap": "20px", "align-items": "center"},
        children=[
            dcc.Dropdown(
                id='coin-dropdown',
                options=coins, 
                value='btcusd', 
                clearable=False,
                style={"width": "200px"}),
            dcc.Dropdown(
                id='algorithm-dropdown',
                options=algorithms,
                value='lstm',
                clearable=False,
                style={"width": "200px"}),

            html.H5("Timeframe:",style={"marginLeft": "20px"}),
            dcc.Dropdown(
                id='timeframe',
                options=timeframes,
                value=60,
                clearable=False,
                style={"width": "200px"},
            ),
            html.H5("Number of timeframe:",style={"marginLeft": "20px"}),
            dcc.Dropdown(
                id='day-number',
                options=day_number,
                value=20,
                clearable=False,
                style={"width": "200px"},
            ),
        ]),
        dcc.Dropdown(
            placeholder='Select attributes',
            multi=True,
            id='price-type-dropdown',
            options=[
                {'label': 'Close Price', 'value': 'Close'},
                {'label': 'ROC', 'value': 'ROC'},
            ],
            value=['Close', 'ROC'],
            clearable=False,
						style={"width": "auto", 'min-width': '200px'}
            ),
      ]
    ),
    # title
    html.H1("Trading Price Analysis Dashboard", style={"textAlign": "left", "margin": "20px"}),
    # graph presentation
    html.Div(
        children = [
            # dcc.Loading(
                dcc.Graph(
                    id='candlestick-graph',
                )
            # ),
        ],
        style={"border": "solid 1px gray", "marginTop": "10px"}  
    ),
	dcc.Interval(id='interval-component', 
                 interval=2 * 1000, 
                 n_intervals=0),
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
        )
])

# TRADING PRICE
@app.callback(
    Output('candlestick-graph', 'figure'),
    [
        Input('coin-dropdown', 'value'),
        Input('algorithm-dropdown', 'value'),
        Input('price-type-dropdown', 'value'),
        Input('day-number', 'value'),
        Input('timeframe', 'value'),
		Input('interval-component', 'n_intervals')
    ]
)
def update_trading_price_graph(coin, algorithm, price_type, day_number, timeframe, n_intervals):
    # GET dữ liệu dựa trên coin được chọn theo ngày
    df = getDataFromCoin(coin, timeframe, day_number)
    # Tạo biểu đồ nến
    figure = go.Figure(
                data = [
                    go.Candlestick(
                        x = df.timestamp,
                        open = df.open,
                        high = df.high,
                        low = df.low,
                        close = df.close
                        )])
    # Thêm dự đoán vào biểu đồ
    if(timeframe == 86400):
        addPredictCandle(figure, df.timestamp.max() + timedelta(1))

    figure.update_layout(
        title=f'Trading Price Analysis ({coin})',
        yaxis_title='Trading Price (USD)',
        xaxis_title='Date',
        xaxis_rangeslider_visible=False
    )

    return figure

def addPredictCandle(figure, date):
	# Thêm cây nến mới với màu sắc khác (ví dụ: màu xanh dương)
	new_candle = go.Candlestick(
			x=[date],  # Ngày của cây nến mới
			open=[63000.00],  # Giá mở cửa của cây nến mới
			high=[68000.00],  # Giá cao nhất của cây nến mới
			low=[61000.00],  # Giá thấp nhất của cây nến mới
			close=[67500.00],  # Giá đóng cửa của cây nến mới
			increasing=dict(line=dict(color='blue')),
			decreasing=dict(line=dict(color='yellow')),
			name='Predicted Trading Price'
	)

	figure.add_trace(new_candle)

# start app
if __name__=='__main__':
	app.run_server(debug=True)
