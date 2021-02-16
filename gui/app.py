import os
import pathlib
import re

import dash
import dash_core_components as dcc
import dash_html_components as html
import pandas as pd
from dash.dependencies import Input, Output, State
import cufflinks as cf

# Initialize app

FONT_SIZE = 15
app = dash.Dash(
    __name__,
    meta_tags=[
        {"name": "viewport", "content": "width=device-width, initial-scale=1.0"}
    ],
)
server = app.server

# Load data

APP_PATH = str(pathlib.Path(__file__).parent.resolve())

df_lat_lon = pd.read_csv(
    os.path.join(APP_PATH, os.path.join("data", "lat_lon_counties.csv"))
)
df_lat_lon["FIPS "] = df_lat_lon["FIPS "].apply(lambda x: str(x).zfill(5))

YEARS = [2003, 2004, 2005, 2006, 2007, 2008, 2009, 2010, 2011, 2012, 2013, 2014, 2015]

DEFAULT_COLORSCALE = [
    "#f2fffb",
    "#98ffe0",
    "#6df0c8",
    "#59dab2",
    "#31c194",
    "#25a27b",
    "#188463",
]

DEFAULT_OPACITY = 0.8

mapbox_access_token = "pk.eyJ1IjoibXl1bmdzdW5raW0iLCJhIjoiY2tmdjBuang5MG9jcDJwbzhvMW5kMnB3OSJ9.WTXKogSwvY4j5k7onkTlzQ"
mapbox_style = "mapbox://styles/plotlymapbox/cjvprkf3t1kns1cqjxuxmwixz"

# App layout
app.layout = html.Div(
    id="root",
    children=[
        html.Div(
            id="header",
            children=[
                # html.Img(id="logo", src=app.get_asset_url("dash-logo.png")),
                html.H4(children="숙박업소 객실 수요 추정 모델"),
                html.P(
                    id="description",
                    children="† 본 대시보드는 전력 사용량을 토대로 숙박업소의 객실 수요량을 추정하고 예상 수입을 산출하는 모델입니다."
                ),
            ],
        ),
        html.Div(
            id="app-container",
            children=[
                html.Div(
                    id="left-column",
                    children=[
                        html.Div(
                            id="slider-container",
                            children=[
                                html.P(
                                    id="slider-text",
                                    children="분석을 원하는 업체의 ID를 입력하시오.",
                                ),
                                dcc.Input(
                                            id="PNU 코드",
                                            type='text',
                                            placeholder="예시: 호텔A",
                                            style={'width': 500},
                                ),
                                html.Button('Submit', id='button', n_clicks = 0),
                            ],
                        ),
                        html.Div(
                            id="heatmap-container",
                            children=[
                                html.P(
                                    children="모니터링 현황",
                                    id="heatmap-title",
                                    style={
                                           "margin-left": 25,
                                           }
                                ),
                                dcc.Graph(
                                    style={'width': 970, 'height':10,
                                           "margin-left": "auto",
                                           "margin-right": "auto",
                                            "marginBottom": 30,
                                           },
                                    id="county-choropleth",
                                    figure=dict(
                                        data=[dict(x=0, y=0)],
                                        layout=dict(
                                        ),
                                    ),
                                ),
                            ],
                        ),
                    ],
                ),
                html.Div(
                    id="graph-container",
                    children=[
                        html.P(id="chart-selector", children="분석 결과"),
                        dcc.Dropdown(
                            options=[
                                {
                                    "label": "예상 수입",
                                    "value": "예상 수입",
                                },
                                {
                                    "label": "투숙객 수",
                                    "value": "투숙객 수",
                                },
                            ],
                            value="예상 수입",# 기본값 설정
                            id="chart-dropdown",
                        ),
                        dcc.Graph(
                            style={"marginBottom": 0,'height':100},
                            id="selected-data",
                            figure=dict(
                                data=[dict(x=0, y=0)],
                                layout=dict(
                                    paper_bgcolor="#F4F4F8",
                                    plot_bgcolor="#F4F4F8",
                                    autofill=True,
                                    margin=dict(t=75, r=50, b=10, l=50),
                                ),
                            ),
                        ),
                    ],
                ),
            ],
        ),
    ],
)


@app.callback(
    Output("county-choropleth", "figure"),
    [Input("button", "n_clicks")],
)

def display_map(n_clicks):
    if n_clicks == 0:
        return dict(
            data=[dict(x=0, y=0)],
            layout=dict(
                title="Waiting for submit...",
                paper_bgcolor="#1f2630",
                plot_bgcolor="#1f2630",
                type="fill",
                showline = False,
                zeroline = False,
                showgrid = False,
                showticklabels = False,
                font=dict(color="#2cfec1"),
                margin=dict(t=75, r=50, b=100, l=75),
                showarrow=False,
                xaxis=dict(
                    tickmode='array',
                    tickvals=[1,2],
                    ticktext=[' ',' '],
                    showgrid= False,
                    zeroline=False
                    ),
                yaxis=dict(
                    tickmode='array',
                    tickvals=[1, 2],
                    ticktext=[' ', ' '],
                    showgrid=False,
                    zeroline=False
                )
            ))

    import numpy as np
    from plotly.subplots import make_subplots
    import plotly.graph_objects as go

    df = pd.read_csv('data/consump_data.csv')
    df.set_index('datetime', inplace=True)
    # fig = df.iplot(kind="line", title=None, asFigure=True)
    fig = make_subplots(rows=2, cols=1)
    x_val_1 = [1, 2, 3,4,5,6,7,8]
    x_val_2 = [9, 10,11]
    x_val = x_val_1 + x_val_2
    y_1 = [40, 50, 60, 15, 21, 31, 11, 21]
    y_2 = [30, 43, 20]
    y = y_1 + y_2
    trace1 = go.Scatter(x=x_val_1, y=y_1,mode='lines+markers',
                        line=dict(width=5),
                        marker=dict(symbol='circle',size=12),
                        name="과거 전력 사용량")
    trace2 = go.Scatter(x=x_val[-4:], y=y[-4:],yaxis="y2",mode='lines+markers',
                        line=dict(width=5,dash='dash'),
                        marker=dict(symbol='circle',size=12),
                        name="예상 전력 사용량")
    # trace1 = go.Bar(x=x_val_1, y=[40, 50, 60, 15, 21, 31, 11, 21],
    #                     name="과거 전력 사용량")
    # trace2 = go.Bar(x=x_val_2, y=[30, 43, 20], yaxis="y2",
    #                     name="예상 전력 사용량")

    fig.append_trace(
        trace1,
        row=1, col=1
    )
    fig.append_trace(
        trace2,
        row=1, col=1
    )
    y_1 = [380, 290, 580, 190, 210, 340, 150, 260]
    y_2 = [350, 430, 250]
    y = y_1 + y_2
    trace1 = go.Scatter(x=x_val_1, y=y_1, mode='lines+markers',
                        line=dict(width=5),
                        marker=dict(symbol='x',size=12),
                        name="과거 투숙객 수", marker_color='#AB63FA')
    trace2 = go.Scatter(x=x_val[-4:],  y=y[-4:], yaxis="y2", mode='lines+markers',
                        line=dict(width=5,dash='dash'),
                        marker=dict(symbol='x',size=12),
                        name="예상 투숙객 수", marker_color='#FFA15A')

    # trace1 = go.Bar(x=x_val_1, y=[380, 290, 580, 190, 210, 340, 150, 260],
    #                     name="과거 방문객 수", marker_color = '#AB63FA')
    # trace2 = go.Bar(x=x_val_2, y=[350, 430, 250], yaxis="y2",
    #                     name="예상 방문객 수", marker_color = '#FFA15A')

    fig.append_trace(
        trace1,
        row=2, col=1
    )
    fig.append_trace(
        trace2,
        row=2, col=1
    )
    # Update xaxis properties
    tick_text = ['2020-11-17', '            ', '2020-11-19', '            ', '2020-11-21', '            ',
                               '2020-11-23', '            ', '2020-11-25', '            ', '2020-11-27',
                               '            ', '2020-11-29', '            ', '2020-11-31']
    fig.update_xaxes(title_text="날짜",
                     tickvals=x_val,
                     ticktext=tick_text,
                     showgrid=False,
                     zeroline=False,
                     tickfont=dict(size=FONT_SIZE), title_font={"size": FONT_SIZE+1},
                     row=2, col=1)
    fig.update_xaxes(title_text="",
                     tickvals=x_val,
                     ticktext=tick_text,
                     showgrid=False,
                     zeroline=False,
    tickfont = dict(size=FONT_SIZE), title_font = {"size": FONT_SIZE+3},
                     row=1, col=1)
    # Update yaxis properties
    fig.update_yaxes(title_text="전력 사용량 [kWh]", row=1, col=1,tickfont=dict(size=FONT_SIZE),title_font = {"size": FONT_SIZE+1})
    fig.update_yaxes(title_text="투숙객 수", row=2, col=1,tickfont=dict(size=FONT_SIZE),title_font = {"size": FONT_SIZE+1})

    fig.update_layout(legend=dict(font=dict(size=FONT_SIZE)),
                      legend_title=dict(font=dict(size=FONT_SIZE+5))
                      , title="과거 및 예상 전력 사용량과 투숙객 수")

    fig_layout = fig["layout"]
    fig_data = fig["data"]
    fig_layout["paper_bgcolor"] = "#1f2630"
    fig_layout["plot_bgcolor"] = "#1f2630"
    fig_layout["font"]["color"] = "#2cfec1"
    fig_layout["title"]["font"]["color"] = "#2cfec1"
    fig_layout["xaxis"]["tickfont"]["color"] = "#2cfec1"

    fig_layout["yaxis"]["tickfont"]["color"] = "#2cfec1"
    fig_layout["xaxis"]["gridcolor"] = "#5b5b5b"
    fig_layout["yaxis"]["gridcolor"] = "#5b5b5b"
    fig_layout["margin"]["t"] = 75
    fig_layout["margin"]["r"] = 50
    fig_layout["margin"]["b"] = 100
    fig_layout["margin"]["l"] = 50


    return fig

"""
@app.callback(Output("heatmap-title", "children"), [Input("button", "value")])
def update_map_title(year):
    return "전력 사용량 및 방문객 수"
"""
@app.callback(
    Output("selected-data", "figure"),
    [Input("button", "n_clicks"),
     Input("chart-dropdown", "value")],
)
def display_selected_data(n_clicks, chart_dropdown):
    if n_clicks == 0:
        return dict(
            data=[dict(x=0, y=0)],
            layout=dict(
                title="Waiting for submit...",
                paper_bgcolor="#1f2630",
                plot_bgcolor="#1f2630",
                font=dict(color="#2cfec1"),
                margin=dict(t=75, r=50, b=100, l=75),
                xaxis=dict(
                    tickmode='array',
                    tickvals=[1, 2],
                    ticktext=[' ', ' '],
                    showgrid=False,
                    zeroline=False
                ),
                yaxis=dict(
                    tickmode='array',
                    tickvals=[1, 2],
                    ticktext=[' ', ' '],
                    showgrid=False,
                    zeroline=False
                )
            ),
        )
    import pandas as pd
    import plotly.graph_objects as go
    import numpy as np
    if chart_dropdown == "예상 수입":
        title = "과거 및 예상 기대 수입"
        xticks = ['한달 전', '일주일 전', '2일 전', '1일 전', '오늘', '내일', '2일 후']
        colors = ['#19D3F3'] * len(xticks)
        colors[-3:] = ['#FF6692'] * 3
        from plotly.subplots import make_subplots
        fig = make_subplots(rows=1, cols=1)
        trace1 =go.Bar(x=xticks[:-3], y=[365000, 210000, 140000, 230000, 200000, 270000, 250000][:-3], marker_color=colors[:-3],
                       name="과거")
        trace2 = go.Bar(x=xticks[-3:-2], y=[365000, 210000, 140000, 230000, 200000, 270000, 250000][-3:-2],
                        marker_color='#D6C8C7',
                        name="현재")

        trace3 =go.Bar(x=xticks[-2:], y=[365000, 210000, 140000, 230000, 200000, 270000, 250000][-2:], marker_color=colors[-2:],
                       name="예측")
        fig.append_trace(
            trace1,
            row=1, col=1
        )
        fig.append_trace(
            trace2,
            row=1, col=1
        )
        fig.append_trace(
            trace3,
            row=1, col=1
        )

        # Change the bar mode
        fig.update_layout(barmode='group',title=title)
        fig.update_layout(legend=dict(font=dict(size=FONT_SIZE)),
                          legend_title=dict(font=dict(size=FONT_SIZE+5)))
        fig.update_yaxes(title_text="수입 [₩]",tickfont=dict(size=FONT_SIZE), title_font={"size": FONT_SIZE+1},)
        fig.update_xaxes(title_text="날짜",tickfont=dict(size=FONT_SIZE), title_font={"size": FONT_SIZE+1},)
    else:
        title = "과거 투숙객 수 및 오늘 예상 투숙객 수"
        xticks = ['한달 전','일주일 전', '2일 전', '1일 전','오늘 예상 투숙객 수']
        colors = ['#636EFA'] * len(xticks)
        colors[-1] = '#EF553B'
        fig = go.Figure(data=[
            go.Bar(x=xticks, y=[280, 210, 130, 240, 198],marker_color=colors),
        ])
        # Change the bar mode
        fig.update_layout(barmode='group', title=title)
        fig.update_yaxes(title_text="투숙객 수")



    # fig.update(layout_showlegend=False)
    """
    fig.update_layout(
        xaxis=dict(
            tickmode='array',
            tickvals=[1, 3],
            ticktext=['전력사용량', '방문객수']
        )
    )
    """
    fig_layout = fig["layout"]
    fig_data = fig["data"]
    fig_layout["paper_bgcolor"] = "#1f2630"
    fig_layout["plot_bgcolor"] = "#1f2630"
    fig_layout["font"]["color"] = "#2cfec1"
    fig_layout["title"]["font"]["color"] = "#2cfec1"
    fig_layout["xaxis"]["tickfont"]["color"] = "#2cfec1"
    fig_layout["yaxis"]["tickfont"]["color"] = "#2cfec1"
    fig_layout["xaxis"]["gridcolor"] = "#5b5b5b"
    fig_layout["yaxis"]["gridcolor"] = "#5b5b5b"
    fig_layout["margin"]["t"] = 75
    fig_layout["margin"]["r"] = 50
    fig_layout["margin"]["b"] = 100
    fig_layout["margin"]["l"] = 50

    return fig


if __name__ == "__main__":
    app.run_server(debug=True)
