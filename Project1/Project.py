#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#Import the neccessary libraries
import pandas as pd
import dash
from dash import html,dcc
import plotly.express as px
from dash.dependencies import Input,Output
df=pd.read_csv(r"world_population_data1.csv")# Created dataframe for the given csv file
app = dash.Dash(__name__)                    #created the object for Dash class
app.layout = html.Div([                      #creating layout for the dash app
    html.H1('Interactive Dashboard',style={'text-align': 'center'}), #created heading level1
    html.H3('Select the year',),#created heading level2
    dcc.Dropdown(options=['2023', '2022', '2020', '2015','2010', '2000', '1990', '1980', '1970'],value='2020',id='input'),#created a dropdown
    html.Br(),#created a linebreak
    html.Br(),#created a linebreak
    dcc.Graph(id='output',style={'width': '50%','display': 'inline-block'}),#created a interactive graph component
    dcc.Graph(id='output1',style={'width': '50%','display': 'inline-block'}),#created a interactive graph component
    dcc.Graph(id='output2',style={'width': '50%','display': 'inline-block'}),#created a interactive graph component
    dcc.Graph(id='output4',style={'width': '50%','display': 'inline-block'}),#created a interactive graph component
    dcc.Graph(id='output3',style={'width': '50%','display': 'inline-block'}),#created a interactive graph component
    html.H2('Comparison between the countries in  population growth',style={'text-align': 'center'}),#created heading level1
    dcc.Dropdown(options=list(df['country']),value='India',id='input5',style={'width': '50%','display': 'inline-block'}),#created a dropdown
    dcc.Dropdown(options=list(df['country']),value='India',id='input6',style={'width': '50%','display': 'inline-block'}),#created a dropdown
    dcc.Graph(id='output5',style={'width': '50%','display': 'inline-block'}),#created a interactive graph component
    
    dcc.Graph(id='output6',style={'width': '50%','display': 'inline-block'})#created a interactive graph component
    
    
    
])

@app.callback(#created a decorator for the callback function update_output
    Output('output','figure'),
    [Input('input', 'value')]
)
def update_output(value):#the callback function
    
        dff=df.sort_values(value,ascending=False).iloc[:9,:]#created a newdataframe for our plot
        fig = px.bar(dff, x='country', y=value,title='Top 10 populated countries')#created barplot
        return (fig)
@app.callback(#created a decorator for the callback function update_output1
    Output('output1','figure'),
    [Input('input', 'value')]
)
def update_output1(value):
        dff=df.sort_values(value).iloc[:9,:]#created a newdataframe for our plot
        fig= px.bar(dff, x='country', y=value,title='Least 10 populated countries')#created barplot
       
        
        return (fig)
    
@app.callback(#created a decorator for the callback function update_output2
    Output('output2','figure'),
    [Input('input', 'value')]
)
def update_output2(value):
        fig=px.choropleth(df,locations='cca3',#created a choropleth map
                 color=value,
                 hover_name='country',title='Population  of the countries')
        return(fig)

@app.callback(#created a decorator for the callback function update_output3
    Output('output3','figure'),
    [Input('input', 'value')]
)
def update_output3(value):
        df3=df.groupby('continent')[value].sum().reset_index(name ='Total_population')#created a newdataframe for our plot
        fig=px.pie(df3,values='Total_population',names='continent',title='Population proportion of Continents')

        return(fig)
@app.callback(#created a decorator for the callback function update_output4
    Output('output4','figure'),
    [Input('input', 'value')]
)
def update_output4(value):
        df4=df#created a newdataframe for our plot
        df4['density']=df4[value]/df['area(sqkm)']
    
        fig=px.choropleth(df4,locations='cca3',#created a choropleth map
                 color='density',
                 hover_name='country',
                 range_color=(0,600),title='Population density of the countries /squarekm')
        return(fig)
@app.callback(#created a decorator for the callback function update_output5
    Output('output5','figure'),
    [Input('input5', 'value')]
)
def update_output5(value):
    
                    A=[1970,1980,1990,2000,2010,2015,2020,2022,2023]#created a list of years of population data
                    B=[df['1970'][df['country']==value].values[0],df['1980'][df['country']==value].values[0],
                       df['1990'][df['country']==value].values[0],df['2000'][df['country']==value].values[0],
                       df['2010'][df['country']==value].values[0],df['2015'][df['country']==value].values[0],
                       df['2020'][df['country']==value].values[0],df['2022'][df['country']==value].values[0],
                       df['2023'][df['country']==value].values[0]]#created a list of values matching the value and the years
                    
                    fig=px.line(x=A,y=B,title='Lineplot of country population over years',markers=True)#created a line plot
                    
        
    
                    return(fig)
@app.callback(#created a decorator for the callback function update_output6
    Output('output6','figure'),
    [Input('input6', 'value')]
)
def update_output6(value):
    
                    A=[1970,1980,1990,2000,2010,2015,2020,2022,2023]#created a list of years of population data
                    B=[df['1970'][df['country']==value].values[0],df['1980'][df['country']==value].values[0],
                       df['1990'][df['country']==value].values[0],df['2000'][df['country']==value].values[0],
                       df['2010'][df['country']==value].values[0],df['2015'][df['country']==value].values[0],
                       df['2020'][df['country']==value].values[0],df['2022'][df['country']==value].values[0],
                       df['2023'][df['country']==value].values[0]]#created a list of values matching the value and the years
                    
                    fig=px.line(x=A,y=B,title='Lineplot of country population over years',markers=True)#created a line plot
                    
        
    
                    return(fig)





if __name__ == "__main__":
    app.run_server(debug=True,use_reloader=False)#started dash server


# In[ ]:




