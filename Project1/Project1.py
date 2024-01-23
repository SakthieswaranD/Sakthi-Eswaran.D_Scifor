#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import pandas as pd
import dash
import dash_html_components as html
import dash_core_components as dcc
import plotly.express as px
from dash.dependencies import Input,Output
df=pd.read_csv(r"world_population_data1.csv")
app = dash.Dash(__name__)
app.layout = html.Div([
    html.H1('Interactive Dashboard',style={'text-align': 'center'}),
    html.H2('Select the year',style={'text-align': 'center'}),
    dcc.Dropdown(options=['2023', '2022', '2020', '2015','2010', '2000', '1990', '1980', '1970'],value='2020',id='input'),
    dcc.Dropdown(options=['2023', '2022', '2020', '2015','2010', '2000', '1990', '1980', '1970'],value='2020',id='input1',disabled=True),
    html.Br(),
    html.Br(),
    dcc.Graph(id='output',style={'width': '50%','display': 'inline-block'}),
    dcc.Graph(id='output1',style={'width': '50%','display': 'inline-block'}),
    dcc.Graph(id='output2',style={'width': '50%','display': 'inline-block'}),
    dcc.Graph(id='output4',style={'width': '50%','display': 'inline-block'}),
    dcc.Graph(id='output3',style={'width': '50%','display': 'inline-block'}),
    html.H1('Comparison between the population growth',style={'text-align': 'center'}),
    dcc.Dropdown(options=list(df['country']),value='India',id='input5'),
    dcc.Dropdown(options=list(df['country']),value='India',id='input6'),
    dcc.Graph(id='output5',style={'width': '50%','display': 'inline-block'}),
    
    dcc.Graph(id='output6',style={'width': '50%','display': 'inline-block'})
    
    
    
])

@app.callback(
    Output('output','figure'),
    [Input('input', 'value')]
)
def update_output(value):
        dff=df.iloc[:9,:]
        fig = px.bar(dff, x='country', y=value,title='Top 10 populated countries')
        return (fig)
@app.callback(
    Output('output1','figure'),
    [Input('input', 'value')]
)
def update_output1(value):
        dff=df.iloc[-9:,:]
        fig= px.bar(dff, x='country', y=value,title='Least 10 populated countries')
        fig.update_layout(xaxis = {"categoryorder":"total ascending"})
        
        return (fig)
    
@app.callback(
    Output('output2','figure'),
    [Input('input', 'value')]
)
def update_output2(value):
        fig=px.choropleth(df,locations='cca3',
                 color=value,
                 hover_name='country',title='Population  of the countries')
        return(fig)

@app.callback(
    Output('output3','figure'),
    [Input('input', 'value')]
)
def update_output3(value):
        df3=df.groupby('continent')[value].sum().reset_index(name ='Total_population')
        fig=px.pie(df3,values='Total_population',names='continent',title='Population proportion of Continents')

        return(fig)
@app.callback(
    Output('output4','figure'),
    [Input('input', 'value')]
)
def update_output4(value):
        df4=df
        df4['density']=df4[value]/df['area(sqkm)']
    
        fig=px.choropleth(df4,locations='cca3',
                 color='density',
                 hover_name='country',
                 range_color=(0,600),title='Population density of the countries /squarekm')
        return(fig)
@app.callback(
    Output('output5','figure'),
    [Input('input5', 'value')]
)
def update_output5(value):
    
                    A=[1970,1980,1990,2000,2010,2015,2020,2022,2023]
                    B=[df['1970'][df['country']==value].values[0],df['1980'][df['country']==value].values[0],
                       df['1990'][df['country']==value].values[0],df['2000'][df['country']==value].values[0],
                       df['2010'][df['country']==value].values[0],df['2015'][df['country']==value].values[0],
                       df['2020'][df['country']==value].values[0],df['2022'][df['country']==value].values[0],
                       df['2023'][df['country']==value].values[0]]
                    
                    fig=px.line(x=A,y=B,title='Lineplot of country population over years',markers=True)
                    
        
    
                    return(fig)
@app.callback(
    Output('output6','figure'),
    [Input('input6', 'value')]
)
def update_output6(value):
    
                    A=[1970,1980,1990,2000,2010,2015,2020,2022,2023]
                    B=[df['1970'][df['country']==value].values[0],df['1980'][df['country']==value].values[0],
                       df['1990'][df['country']==value].values[0],df['2000'][df['country']==value].values[0],
                       df['2010'][df['country']==value].values[0],df['2015'][df['country']==value].values[0],
                       df['2020'][df['country']==value].values[0],df['2022'][df['country']==value].values[0],
                       df['2023'][df['country']==value].values[0]]
                    
                    fig=px.line(x=A,y=B,title='Lineplot of country population over years',markers=True)
                    
        
    
                    return(fig)





if __name__ == "__main__":
    app.run_server(debug=True,use_reloader=False)


# In[ ]:




