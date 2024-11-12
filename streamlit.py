import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression
import xgboost as xgb
from sklearn.model_selection import train_test_split
import geopandas as gpd
import plotly.express as px
import plotly.graph_objects as go  # Import this for heatmap
from wordcloud import WordCloud

# Caching data loading function to optimize performance
@st.cache
def load_data():
    df = pd.read_csv('layoffs_data.csv')  # Adjust path if necessary
    df['Laid_Off_Count'].fillna(df['Laid_Off_Count'].mean(), inplace=True)
    df['Funds_Raised'].fillna(df['Funds_Raised'].mean(), inplace=True)
    df['Percentage'].fillna(df['Percentage'].mean(), inplace=True)
    df['Date'] = pd.to_datetime(df['Date'])
    
    # Standardize country names if necessary (Optional, if you have country name variations)
    df['Country'] = df['Country'].replace({'United States': 'United States of America'})
    
    return df

# Load data
df = load_data()

# Streamlit Title
st.title('Layoffs Analysis Dashboard')

# Display raw data if the checkbox is checked
if st.checkbox('Show raw data'):
    st.write(df.head())

# Filter data by user input (interactive filters)
# Filter data by user input (interactive filters)
st.sidebar.header('Data Filters')
industry_filter = st.sidebar.selectbox('Select Industry', df['Industry'].unique())

# Restrict the country filter to only 'United States' and 'India'
country_filter = st.sidebar.selectbox('Select Country', ['United States of America', 'India'])

# Filter data based on selections
df_filtered = df[(df['Industry'] == industry_filter) & (df['Country'] == country_filter)]


# Top 5 countries with most layoffs
st.subheader('Top 5 Countries with Most Layoffs')

top_countries = (
    df.groupby('Country')['Laid_Off_Count']
    .sum()
    .sort_values(ascending=False)
    .head(5)
    .reset_index()
)

fig = px.bar(
    top_countries,
    x='Country',
    y='Laid_Off_Count',
    title='Top 5 Countries with Most Layoffs(2020-2024)',
    labels={'Laid_Off_Count': 'Total Layoffs'},
    color='Country'
)
st.plotly_chart(fig)

# Company and Location Insights
st.subheader('Layoffs by Country')

# Group by Country and sum layoffs
country_layoffs = df.groupby('Country')['Laid_Off_Count'].sum().reset_index()

# Sort by layoffs in descending order
country_layoffs = country_layoffs.sort_values(by='Laid_Off_Count', ascending=False)

# Create a horizontal bar chart with custom color scale
fig = px.bar(
    country_layoffs,
    x='Laid_Off_Count',
    y='Country',
    title='Total Layoffs by Country',
    labels={'Laid_Off_Count': 'Total Layoffs'},
    color='Laid_Off_Count',  # Color by the number of layoffs
    color_continuous_scale='Viridis',  # Custom color scale for better visuals
    orientation='h'  # Horizontal bar chart
)

# Customize layout for better presentation
fig.update_layout(
    title_font=dict(size=22, family='Arial', color='darkblue'),
    xaxis_title='Total Layoffs',
    yaxis_title='Country',
    xaxis=dict(
        tickformat=",.0f",  # Add thousands separator to x-axis
        tickfont=dict(color='darkred'),  # Change color of x-axis labels
        title_font=dict(color='purple')  # Change color of x-axis title
    ),
    yaxis=dict(
        tickfont=dict(color='darkgreen'),  # Change color of y-axis labels
        title_font=dict(color='teal')  # Change color of y-axis title
    ),
    plot_bgcolor='rgb(245, 245, 245)',  # Light background
    paper_bgcolor='rgb(255, 255, 255)',  # White paper background
    margin=dict(l=100, r=50, t=50, b=50),  # Add space around the chart
    showlegend=False,  # Remove legend to reduce clutter
    hoverlabel=dict(bgcolor='white', font_size=12, font_family='Arial')  # Hover label style
)

# Display the chart
st.plotly_chart(fig)


# Filter data for the years 2022 and 2023
df_recent_years = df[(df['Date'].dt.year.isin([2022, 2023])) & (df['Industry'] != 'Other') ]

# Group by Industry and sum layoffs for 2022 and 2023
top_industries_recent = (
    df_recent_years.groupby('Industry')['Laid_Off_Count']
    .sum()
    .sort_values(ascending=False)
    .head(5)
    .reset_index()
)

# Plot the top 5 industries
st.subheader('Top 5 Industries with Most Layoffs in 2022 and 2023')

# Horizontal bar chart
fig = px.bar(
    top_industries_recent,
    x='Laid_Off_Count',
    y='Industry',
    title='Top 5 Industries with Most Layoffs (2022-2023)',
    labels={'Laid_Off_Count': 'Total Layoffs'},
    color='Industry',
    orientation='h'  # Set orientation to horizontal
)

st.plotly_chart(fig)




st.subheader('Number of Layoffs per year')

# Group data by year for yearly layoffs
df_time_series = df_filtered.groupby(df_filtered['Date'].dt.year)['Laid_Off_Count'].sum().reset_index()
df_time_series.rename(columns={'Date': 'Year'}, inplace=True)

# Create a yearly layoffs line graph
fig = px.line(
    df_time_series,
    x='Year',
    y='Laid_Off_Count',
    title='Yearly Layoffs in India And USA based on Industry',
    labels={'Year': 'Year', 'Laid_Off_Count': 'Total Layoffs'},
)

# Ensure x-axis treats years as discrete categories
fig.update_xaxes(type='category')

st.plotly_chart(fig)

# Group by company and sum the layoffs
top_companies = df.groupby('Company')['Laid_Off_Count'].sum().sort_values(ascending=False).head(10).reset_index()

df['Date'] = pd.to_datetime(df['Date'])  # Ensure Date column is in datetime format
df_filtered = df[df['Date'].dt.year >= 2020]

# Group by company and aggregate the total layoffs, industry, location, and percentage
top_companies_data = (
    df_filtered.groupby(['Company', 'Industry', 'Location_HQ', 'Country'])['Laid_Off_Count']
    .sum()
    .reset_index()
)

# Sort by total layoffs and select the top 20 companies
top_companies_data = top_companies_data.sort_values(by='Laid_Off_Count', ascending=False).head(50)

# Add a new column for the percentage of layoffs (if necessary, based on your dataset)
top_companies_data['Percentage'] = top_companies_data['Laid_Off_Count'] / top_companies_data['Laid_Off_Count'].sum() * 100

# Create an attractive dataframe with custom styling
st.subheader('Top 50 Companies with Most Layoffs (Since 2020)')

# Display the table with custom styles
st.dataframe(
    top_companies_data.style
    .format({'Laid_Off_Count': "{:,.0f}", 'Percentage': "{:.2f}%"})  # Format columns
    .highlight_max(subset='Laid_Off_Count', color='lightgreen')  # Highlight the largest layoffs
    .highlight_min(subset='Percentage', color='lightcoral')  # Highlight the lowest percentage
    .bar(subset=['Percentage'], color='#d65f5f')  # Bar chart for the percentage column
    .set_properties(**{'background-color': '#f5f5f5', 'color': 'black', 'border-color': 'black'})
    .set_table_styles([
        {'selector': 'thead th', 'props': [('background-color', '#4CAF50'), ('color', 'white')]},
        {'selector': 'tbody tr:nth-child(even)', 'props': [('background-color', '#f2f2f2')]},
        {'selector': 'tbody tr:nth-child(odd)', 'props': [('background-color', 'white')]},
    ])
)





# Create a bar chart to display top 10 companies with the highest layoffs
# st.subheader('Top 10 Companies with the Most Layoffs')

# fig = px.bar(
#     top_companies,
#     x='Company',
#     y='Laid_Off_Count',
#     title='Top 10 Companies with the Most Layoffs since Covid-19',
#     labels={'Laid_Off_Count': 'Total Layoffs', 'Company': 'Company'},
#     color='Company'
# )

# # Display the chart
# st.plotly_chart(fig)
# fig = px.treemap(
#     top_companies,
#     path=['Company'],
#     values='Laid_Off_Count',
#     color='Laid_Off_Count',
#     title='Layoffs by Company (Top 10)',
#     color_continuous_scale='RdBu'
# )
# st.plotly_chart(fig)


st.subheader('Top 10 Companies Based on Industry(All Countries)')

# Group by company and sum the layoffs for the selected industry
top_companies_filtered = df_filtered[df_filtered['Industry'] == industry_filter]  # Filter based on selected industry
top_companies_filtered = top_companies_filtered.groupby(['Company', 'Industry'])['Laid_Off_Count'].sum().reset_index()

# Sort by total layoffs and select the top companies
top_companies_filtered = top_companies_filtered.sort_values(by='Laid_Off_Count', ascending=False).head(10)

# Create a treemap for layoffs by company based on the industry filter
fig = px.treemap(
    top_companies_filtered,
    path=['Industry', 'Company'],  # Display industry and company in the hierarchy
    values='Laid_Off_Count',
    color='Laid_Off_Count',
    title=f'Layoffs by Company in {industry_filter} Industry',
    color_continuous_scale='RdBu'
)

# Display the treemap chart
st.plotly_chart(fig)


# country_summary = df.groupby('Country')['Laid_Off_Count'].sum().reset_index()

# # Create choropleth heatmap
# fig_heatmap_countries = px.choropleth(
#     country_summary,
#     locations='Country',
#     locationmode='country names',
#     color='Laid_Off_Count',
#     color_continuous_scale='Reds',
#     title='Layoffs Heatmap by Country',
#     labels={'Laid_Off_Count': 'Total Layoffs'},
#     hover_name='Country',
# )

# fig_heatmap_countries.update_geos(
#     showcountries=True,
#     showcoastlines=True,
#     showland=True,
#     fitbounds="locations"
# )

# st.plotly_chart(fig_heatmap_countries)

# Sentiment and Employee Impact (Word Cloud)
# Layoffs by Company Stage
# Layoffs by Company Stage (Filtered)
st.subheader('Layoffs by Company Stage')

# Filter the stages we are interested in
valid_stages = ['Acquired', 'Post-IPO', 'Private Equity', 'Seed']
stage_layoffs = df[df['Stage'].isin(valid_stages)]  # Only keep these stages

# Group by Stage and sum the layoffs
stage_layoffs = stage_layoffs.groupby('Stage')['Laid_Off_Count'].sum().reset_index()

# Create the pie chart
fig = px.pie(stage_layoffs, 
             names='Stage', 
             values='Laid_Off_Count', 
             title='Layoffs by Company Stage', 
             color='Stage', 
             color_discrete_map={'Acquired': 'lightblue', 'Post-IPO': 'lightgreen', 'Private Equity': 'lightcoral', 'Seed': 'lightgoldenrodyellow'})

st.plotly_chart(fig)


st.subheader('Impact of Funding on Layoffs')

# Create the scatter plot with red color and border around the dots
fig = px.scatter(df, 
                 x='Funds_Raised', 
                 y='Laid_Off_Count', 
                 title='Impact of Funding on Layoffs', 
                 labels={'Funds_Raised': 'Funds Raised', 'Laid_Off_Count': 'Number of Layoffs'})

# Update marker properties: red color and border
fig.update_traces(marker=dict(color='red', 
                              line=dict(color='black', width=1)))  # Adding black border with width 1

# Display the plot
st.plotly_chart(fig)




