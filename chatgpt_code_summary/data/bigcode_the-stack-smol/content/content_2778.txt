import plotly.graph_objects as go
import pandas as pd

from .Colors import COLOR_DISCRETE_MAP
from Classification import CATEGORIES


def all_categories_grouping(row: pd.Series) -> str:
    """
    Merge Category, Fuel and segment to a single string for unique categorization
    """
    if row['Fuel'] == 'Battery Electric':
        return row['Category'] + ' / ' + row['Fuel']
    else:
        try:
            result = row['Fuel'] + ' / ' + row['Segment'] + ' / ' + row['Euro Standard']
        except:  # For Off Road type with no Segment nor Euro Standard
            result = row['Fuel']
        return result


def activity_horizontal_bar_chart(stock_and_mileage_df: pd.DataFrame.groupby, output_folder):
    """
    Horizontal bar chart representing mean activity and other activities per unique categorization

    :param stock_and_mileage_df: Dataframe of the  vehicles registration list
    :param output_folder: output folder name where to store resulting chart
    :return: an html file containing the horizontal bar chart of the mean activity
    """
    data = stock_and_mileage_df.copy()

    # Delete off road data
    data = data[data['Category'] != 'Off Road']

    # Create single column classification
    data['segmentation'] = data.apply(lambda row: all_categories_grouping(row), axis=1)

    horizontal_plot = go.Figure()

    # Add Activity statistics and stock traces
    horizontal_plot.add_trace(go.Scatter(y=data['segmentation'], x=data['Max_Activity'], mode='markers',
                                         name='Activitat màxima', marker_color='rgb(288, 26, 28)'
                                         ))

    horizontal_plot.add_trace(go.Scatter(y=data['segmentation'], x=data['Min_Activity'], mode='markers',
                                         name='Activitat mínima', marker_color='rgb(229, 196, 148)'
                                         ))
    horizontal_plot.add_trace(go.Scatter(y=data['segmentation'], x=data['Std_Activity'], mode='markers',
                                         name="Desviació standard de l'activitat", marker=dict(
            color='rgb(800, 800, 800)',
            opacity=0)
                                         ))
    horizontal_plot.add_trace(go.Scatter(y=data['segmentation'], x=data['Stock'], mode='markers',
                                         name="Estoc", marker=dict(
            color='rgb(800, 800, 800)',
            opacity=0)
                                         ))
    horizontal_plot.add_trace(go.Scatter(y=data['segmentation'], x=data['Mean_Lifetime_Activity'], mode='markers',
                                         name="Lifetime cumulative activity mitja", marker=dict(
            color='rgb(800, 800, 800)',
            opacity=0)
                                         ))

    # For each category add the mean activity bar chart (to diferenciate by same colors as Stock distribution Pie Chart)
    for category in CATEGORIES:
        horizontal_plot.add_trace(go.Bar(
            y=data[data['Category'] == category]['segmentation'], x=data[data['Category'] == category]['Mean_Activity'],
            orientation='h', marker_color=COLOR_DISCRETE_MAP[category],
            name=f'Activitat mitjana {category}'
        ))

    # Update plot information
    horizontal_plot.update_layout(
        title="Activitat mitjana anual segons classificació del parc de vehicles d'Andorra",
        title_x=0.5,
        height=4000,
        width=1500,
        template='plotly_white',
        xaxis_title='Activitat mitja (km/any)',
        yaxis_title='Tipologia de vehicle',
        hovermode="y unified",
        hoverlabel=dict(namelength=100),
        xaxis_range=[0, stock_and_mileage_df['Max_Activity'].max()*1.05],
        xaxis=dict(
            tickmode='array',
            tickvals=[0, 5000, 15000, 25000, 50000, 100000, 150000, 200000],
            ticktext=['0', '5k', '15k', '25k', '50k', '100k', '150k', '200k'])
    )
    horizontal_plot.update_xaxes(showgrid=True, zeroline=True)
    horizontal_plot.show()

    # Save plot to html file
    filename = output_folder + "Activitat mitjana anual segons classificació del parc de vehicles d'Andorra.html"
    horizontal_plot.write_html(filename)