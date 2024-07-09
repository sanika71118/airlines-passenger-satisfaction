import dash
from dash import dcc, html
from dash.dependencies import Input, Output, State
import pandas as pd
import plotly.express as px
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import numpy as np
from numpy.linalg import svd, cond

df = pd.read_csv("Airlines_train.csv")

app = dash.Dash(__name__)
server = app.server

# phase 3 layout
app.layout = html.Div([
    html.H1("Airlines Passenger Satisfaction Dashboard", style={'text-align': 'center', 'margin-bottom': '30px'}),
    dcc.Tabs(id='tabs', value='tab1', children=[
        dcc.Tab(label='Data Exploration', value='tab1', children=[
            html.Div([
                html.Label("Select Gender:", style={'margin-right': '10px'}),
                dcc.Dropdown(
                    id='gender-dropdown',
                    options=[{'label': 'Male', 'value': 'Male', 'title': 'Select Male'}, {'label': 'Female', 'value': 'Female', 'title': 'Select Female'}],
                    value=['Female'],
                    multi=True,
                    clearable=False
                ),
            ], style={'margin-bottom': '20px'}),

            html.Div([
                html.Label("Select Customer Type:", style={'margin-right': '10px'}),
                dcc.Checklist(
                    id='customer-type-checklist',
                    options=[
                        {'label': cust_type, 'value': cust_type, 'title': f'Select {cust_type}'}
                        for cust_type in df['Customer Type'].unique()
                    ],
                    value=[df['Customer Type'].unique()[0]],
                    inline=True
                ),
            ], style={'margin-bottom': '20px'}),

            html.Div([
                html.Label("Select Class:", style={'margin-right': '10px'}),
                dcc.RadioItems(
                    id='class-radio',
                    options=[{'label': cls, 'value': cls, 'title': f'Select {cls}'} for cls in df['Class'].unique()],
                    value=df['Class'].unique()[0],
                    labelStyle={'display': 'inline-block', 'margin-right': '10px'}
                ),
            ], style={'margin-bottom': '20px'}),

            html.Div([
                html.Label("Select Flight Distance Range:", style={'margin-right': '10px'}),
                dcc.RangeSlider(
                    id='flight-distance-slider',
                    min=df['Flight Distance'].min(),
                    max=df['Flight Distance'].max(),
                    value=[df['Flight Distance'].min(), df['Flight Distance'].max()],
                    marks={dist: str(dist) for dist in range(0, df['Flight Distance'].max() + 1, 1000)},
                    step=1000
                ),
            ], style={'margin-bottom': '20px'}),

            html.Div([
                html.Label("Select Graph Type:", style={'margin-right': '10px'}),
                dcc.Dropdown(
                    id='graph-type-dropdown',
                    options=[
                        {'label': 'Countplot Bar', 'value': 'countplot_bar'},
                        {'label': 'Pie Chart', 'value': 'pie_chart'},
                        {'label': 'Histogram', 'value': 'histogram'},
                    ],
                    value='countplot_bar',
                    clearable=False
                ),
            ], style={'margin-bottom': '20px'}),

            dcc.Graph(id='dynamic-graph'),

            html.Br()
        ]),
        dcc.Tab(label='Feature Selection', value='tab2', children=[
            html.Div([
                html.Label("Select Columns:", style={'margin-right': '10px'}),
                dcc.Dropdown(
                    id='norm-columns-dropdown',
                    options=[{'label': col, 'value': col} for col in df.columns if
                             col not in ['id', 'X', 'Gender', 'Customer Type', 'Type of Travel', 'Class',
                                         'satisfaction', 'Age']],
                    value=[df.columns[7]],
                    multi=True,
                    clearable=False
                ),
            ], style={'margin-bottom': '20px'}),

            html.Button('Normalize Data', id='normalize-button', n_clicks=0),

            dcc.Loading(id="loading-normalize", type="default", children=[
                html.Div(id='normalization-output'),
            html.Br(),
            html.Br()
            ]),

        dcc.Tab(label='Submit Feedback', value='tab3', children=[
            html.Div([
                dcc.Textarea(
                    id='feedback-textarea',
                    style={'width': '100%', 'height': 100},
                    placeholder='Enter your feedback here...'
                ),
            html.Button('Submit Feedback', id='submit-feedback', n_clicks=0),
                html.Div(id='feedback-output'),
            dcc.Download(id='download-feedback')])
            ]),

        ]),
    ]),
])

# phase 4 callback
@app.callback(
    Output('dynamic-graph', 'figure'),
    [Input('gender-dropdown', 'value'),
     Input('customer-type-checklist', 'value'),
     Input('class-radio', 'value'),
     Input('flight-distance-slider', 'value'),
     Input('graph-type-dropdown', 'value')]
)
def update_graph(gender, selected_customer_types, flight_class, flight_dist_range, graph_type):
    filtered_df = df[(df['Gender'].isin(gender)) &
                     (df['Customer Type'].isin(selected_customer_types)) &
                     (df['Class'] == flight_class) &
                     (df['Flight Distance'].between(flight_dist_range[0], flight_dist_range[1]))]

    if graph_type == 'countplot_bar':
        fig = px.bar(filtered_df, x='Age', y='Arrival Delay in Minutes', color='satisfaction', barmode='group')
        fig.update_layout(title="Age vs Arrival Delay (Countplot Bar)", xaxis_title="Age",
                          yaxis_title="Arrival Delay")

    elif graph_type == 'pie_chart':
        pie_df = filtered_df.groupby('satisfaction').size().reset_index(name='count')
        fig = px.pie(pie_df, values='count', names='satisfaction', title='Satisfaction Distribution (Pie Chart)')

    elif graph_type == 'histogram':
        fig = px.histogram(filtered_df, x='Age', y='Arrival Delay in Minutes', color='satisfaction',
                           marginal='box', nbins=10)
        fig.update_layout(title="Age vs Arrival Delay (Histogram)", xaxis_title="Age",
                          yaxis_title="Arrival Delay (Minutes)")

    return fig



@app.callback(
    Output('normalization-output', 'children'),
    [Input('normalize-button', 'n_clicks')],
    [State('norm-columns-dropdown', 'value')]
)
def normalize_and_select_features(n_clicks, selected_columns):
    if n_clicks > 0:
        features = df[selected_columns]
        scaler = StandardScaler()
        scaled_features = scaler.fit_transform(features)

        cm = pd.DataFrame(scaled_features, columns=features.columns).corr()

        pca = PCA(n_components=0.95)
        pca.fit(scaled_features)
        explained_variance = pca.explained_variance_ratio_

        components = pca.n_components_
        features_to_be_removed = scaled_features.shape[1] - components

        transformed_features = pca.transform(scaled_features)
        u_reduced, s_reduced, v_reduced = svd(transformed_features, full_matrices=False)
        cond_num_reduced = cond(transformed_features)

        cm_reduced = pd.DataFrame(transformed_features).corr()

        return html.Div([
            html.H3(' Results'),
            html.P(f'The number of features to be removed are {features_to_be_removed}'),
            html.P(f'The new singular values are {np.round(s_reduced, 2)}'),
            html.P(f'The new conditional number is {np.round(cond_num_reduced, 2)}'),
        ])

    return html.Div()
@app.callback(
    Output('feedback-output', 'children'),
    Output('download-feedback', 'data'),
    Input('submit-feedback', 'n_clicks'),
    State('feedback-textarea', 'value')
)
def handle_feedback(n_clicks, feedback):
    if n_clicks > 0:
        if feedback:
            filename = 'feedback.txt'
            with open(filename, 'w') as f:
                f.write(feedback)
            return "Feedback submitted successfully.", dcc.send_file(filename)
        else:
            return "Please enter some feedback before submitting.", None
    return "", None



# phase 5 run 

if __name__ == '__main__':
        app.run_server(debug=False, port=8030, host='0.0.0.0')
