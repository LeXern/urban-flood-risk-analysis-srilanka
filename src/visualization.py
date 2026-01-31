"""
Visualization Module

Functions for creating maps, charts, and interactive dashboards.
Uses Folium for interactive maps and Plotly for charts.
"""

import numpy as np
import geopandas as gpd
import pandas as pd
import folium
from folium import plugins
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt
import branca.colormap as cm
from typing import Optional, List, Tuple, Dict
from pathlib import Path


def create_vulnerability_map(
    admin_gdf: gpd.GeoDataFrame,
    value_column: str = 'vulnerability_score',
    title: str = 'Flood Vulnerability',
    colormap: str = 'YlOrRd'
) -> folium.Map:
    """
    Create interactive choropleth map of vulnerability scores.
    
    Parameters
    ----------
    admin_gdf : gpd.GeoDataFrame
        Admin boundaries with vulnerability scores
    value_column : str
        Column containing values to visualize
    title : str
        Legend title
    colormap : str
        Color palette name
    
    Returns
    -------
    folium.Map
        Interactive map object
    
    Example
    -------
    >>> m = create_vulnerability_map(districts, 'vulnerability_score')
    >>> m.save('vulnerability_map.html')
    """
    # calculate center
    bounds = admin_gdf.total_bounds
    center = [(bounds[1] + bounds[3]) / 2, (bounds[0] + bounds[2]) / 2]
    
    # create base map
    m = folium.Map(
        location=center,
        zoom_start=9,
        tiles='cartodbpositron'
    )
    
    # ensure we have an id column for choropleth
    gdf = admin_gdf.copy()
    if 'id' not in gdf.columns:
        gdf['id'] = gdf.index.astype(str)
    
    # add choropleth
    folium.Choropleth(
        geo_data=gdf.to_json(),
        data=gdf,
        columns=['id', value_column],
        key_on='feature.properties.id',
        fill_color=colormap,
        fill_opacity=0.7,
        line_opacity=0.2,
        legend_name=title,
        nan_fill_color='gray'
    ).add_to(m)
    
    # add tooltips
    style_function = lambda x: {
        'fillColor': 'transparent',
        'color': 'transparent'
    }
    
    highlight_function = lambda x: {
        'fillColor': '#000000',
        'color': '#000000',
        'fillOpacity': 0.1,
        'weight': 2
    }
    
    tooltip_fields = [col for col in gdf.columns if col != 'geometry'][:5]
    
    folium.features.GeoJson(
        gdf,
        style_function=style_function,
        highlight_function=highlight_function,
        tooltip=folium.GeoJsonTooltip(
            fields=tooltip_fields,
            aliases=tooltip_fields,
            localize=True
        )
    ).add_to(m)
    
    # add controls
    plugins.Fullscreen().add_to(m)
    folium.LayerControl().add_to(m)
    
    return m


def create_building_density_map(
    admin_gdf: gpd.GeoDataFrame,
    buildings_gdf: Optional[gpd.GeoDataFrame] = None,
    sample_size: int = 5000
) -> folium.Map:
    """
    Create map showing building density and optionally individual buildings.
    
    Parameters
    ----------
    admin_gdf : gpd.GeoDataFrame
        Admin boundaries with building_density column
    buildings_gdf : gpd.GeoDataFrame, optional
        Building footprints (sample shown as markers)
    sample_size : int
        Max buildings to show as markers
    
    Returns
    -------
    folium.Map
        Interactive map
    """
    # create base with choropleth
    m = create_vulnerability_map(
        admin_gdf,
        value_column='building_density',
        title='Building Density',
        colormap='Blues'
    )
    
    # add building markers if provided
    if buildings_gdf is not None and len(buildings_gdf) > 0:
        # sample if too many
        if len(buildings_gdf) > sample_size:
            buildings_sample = buildings_gdf.sample(sample_size)
        else:
            buildings_sample = buildings_gdf
        
        # create marker cluster
        marker_cluster = plugins.MarkerCluster(name='Buildings')
        
        for idx, row in buildings_sample.iterrows():
            centroid = row.geometry.centroid
            folium.CircleMarker(
                location=[centroid.y, centroid.x],
                radius=3,
                color='red',
                fill=True,
                fillOpacity=0.6
            ).add_to(marker_cluster)
        
        marker_cluster.add_to(m)
    
    return m


def create_time_series_chart(
    data: pd.DataFrame,
    x_column: str,
    y_column: str,
    title: str = 'Time Series',
    x_label: str = 'Year',
    y_label: str = 'Rainfall (mm)'
) -> go.Figure:
    """
    Create interactive time series chart.
    
    Parameters
    ----------
    data : pd.DataFrame
        Data with time and value columns
    x_column : str
        Column for x-axis (time)
    y_column : str
        Column for y-axis (values)
    title : str
        Chart title
    x_label : str
        X-axis label
    y_label : str
        Y-axis label
    
    Returns
    -------
    go.Figure
        Plotly figure
    
    Example
    -------
    >>> fig = create_time_series_chart(annual_max, 'year', 'max_rainfall')
    >>> fig.write_html('rainfall_trend.html')
    """
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=data[x_column],
        y=data[y_column],
        mode='lines+markers',
        name=y_label,
        line=dict(color='#667eea', width=2),
        marker=dict(size=6)
    ))
    
    # add trend line
    z = np.polyfit(range(len(data)), data[y_column], 1)
    p = np.poly1d(z)
    
    fig.add_trace(go.Scatter(
        x=data[x_column],
        y=p(range(len(data))),
        mode='lines',
        name='Trend',
        line=dict(color='red', dash='dash', width=2)
    ))
    
    fig.update_layout(
        title=title,
        xaxis_title=x_label,
        yaxis_title=y_label,
        template='plotly_white',
        hovermode='x unified'
    )
    
    return fig


def create_vulnerability_ranking_chart(
    admin_gdf: gpd.GeoDataFrame,
    name_column: str,
    value_column: str = 'vulnerability_score',
    top_n: int = 15,
    title: str = 'Top Vulnerable Districts'
) -> go.Figure:
    """
    Create horizontal bar chart of top vulnerable areas.
    
    Parameters
    ----------
    admin_gdf : gpd.GeoDataFrame
        Admin data with vulnerability scores
    name_column : str
        Column with area names
    value_column : str
        Column with vulnerability scores
    top_n : int
        Number of top areas to show
    title : str
        Chart title
    
    Returns
    -------
    go.Figure
        Plotly figure
    """
    # get top N
    top_areas = admin_gdf.nlargest(top_n, value_column)
    
    # create colors based on score
    colors = px.colors.sample_colorscale(
        'YlOrRd',
        [x / max(top_areas[value_column]) for x in top_areas[value_column]]
    )
    
    fig = go.Figure(go.Bar(
        x=top_areas[value_column],
        y=top_areas[name_column],
        orientation='h',
        marker_color=colors
    ))
    
    fig.update_layout(
        title=title,
        xaxis_title='Vulnerability Score',
        yaxis_title='',
        template='plotly_white',
        yaxis={'categoryorder': 'total ascending'},
        height=max(400, top_n * 25)
    )
    
    return fig


def create_correlation_heatmap(
    data: pd.DataFrame,
    columns: List[str],
    title: str = 'Variable Correlation'
) -> go.Figure:
    """
    Create correlation heatmap between variables.
    
    Parameters
    ----------
    data : pd.DataFrame
        Data containing variables
    columns : list
        Columns to include in correlation
    title : str
        Chart title
    
    Returns
    -------
    go.Figure
        Plotly heatmap figure
    """
    # calculate correlation
    corr_matrix = data[columns].corr()
    
    fig = go.Figure(data=go.Heatmap(
        z=corr_matrix.values,
        x=columns,
        y=columns,
        colorscale='RdBu_r',
        zmid=0,
        text=np.round(corr_matrix.values, 2),
        texttemplate='%{text}',
        textfont={'size': 12},
        hoverongaps=False
    ))
    
    fig.update_layout(
        title=title,
        template='plotly_white',
        width=600,
        height=500
    )
    
    return fig


def create_static_map(
    admin_gdf: gpd.GeoDataFrame,
    value_column: str = 'vulnerability_score',
    title: str = 'Flood Vulnerability',
    cmap: str = 'YlOrRd',
    figsize: Tuple[int, int] = (12, 10)
) -> plt.Figure:
    """
    Create static matplotlib map for reports.
    
    Parameters
    ----------
    admin_gdf : gpd.GeoDataFrame
        Admin boundaries with values
    value_column : str
        Column to visualize
    title : str
        Map title
    cmap : str
        Matplotlib colormap
    figsize : tuple
        Figure size
    
    Returns
    -------
    plt.Figure
        Matplotlib figure
    """
    fig, ax = plt.subplots(1, 1, figsize=figsize)
    
    admin_gdf.plot(
        column=value_column,
        ax=ax,
        legend=True,
        legend_kwds={
            'label': value_column,
            'orientation': 'vertical',
            'shrink': 0.8
        },
        cmap=cmap,
        edgecolor='black',
        linewidth=0.5
    )
    
    ax.set_title(title, fontsize=16, fontweight='bold')
    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')
    ax.set_aspect('equal')
    
    plt.tight_layout()
    
    return fig


def generate_dashboard_html(
    admin_gdf: gpd.GeoDataFrame,
    output_path: str = 'dashboard.html',
    title: str = 'Flood Vulnerability Dashboard'
) -> None:
    """
    Generate complete HTML dashboard.
    
    Parameters
    ----------
    admin_gdf : gpd.GeoDataFrame
        Admin data with vulnerability scores and other metrics
    output_path : str
        Output file path
    title : str
        Dashboard title
    
    Example
    -------
    >>> generate_dashboard_html(districts, 'outputs/dashboard.html')
    """
    # create map
    vuln_map = create_vulnerability_map(admin_gdf)
    
    # create ranking chart
    name_col = [c for c in admin_gdf.columns if 'name' in c.lower()]
    name_col = name_col[0] if name_col else admin_gdf.columns[0]
    
    ranking_fig = create_vulnerability_ranking_chart(
        admin_gdf, name_col, 'vulnerability_score'
    )
    
    # combine into HTML
    html_template = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="utf-8">
        <title>{title}</title>
        <style>
            body {{
                font-family: Arial, sans-serif;
                margin: 0;
                padding: 20px;
                background-color: #f5f5f5;
            }}
            .header {{
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                color: white;
                padding: 30px;
                text-align: center;
                border-radius: 10px;
                margin-bottom: 20px;
            }}
            .section {{
                background: white;
                padding: 20px;
                margin: 20px 0;
                border-radius: 10px;
                box-shadow: 0 2px 8px rgba(0,0,0,0.1);
            }}
            .section h2 {{
                color: #667eea;
                margin-top: 0;
            }}
            iframe {{
                width: 100%;
                height: 500px;
                border: none;
                border-radius: 8px;
            }}
        </style>
    </head>
    <body>
        <div class="header">
            <h1>{title}</h1>
            <p>Sri Lanka Urban Flood Risk Analysis</p>
        </div>
        
        <div class="section">
            <h2>Vulnerability Map</h2>
            {vuln_map._repr_html_()}
        </div>
        
        <div class="section">
            <h2>Top Vulnerable Areas</h2>
            {ranking_fig.to_html(include_plotlyjs='cdn', full_html=False)}
        </div>
        
        <div class="section">
            <h2>Methodology</h2>
            <p><strong>Vulnerability Index = 0.4 × Rainfall + 0.3 × Building Density + 0.3 × (1 - Elevation)</strong></p>
        </div>
    </body>
    </html>
    """
    
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(html_template)
    
    print(f"Dashboard saved to: {output_path}")


if __name__ == "__main__":
    print("Visualization module loaded successfully")
    print("\nAvailable functions:")
    print("  - create_vulnerability_map()")
    print("  - create_building_density_map()")
    print("  - create_time_series_chart()")
    print("  - create_vulnerability_ranking_chart()")
    print("  - create_correlation_heatmap()")
    print("  - create_static_map()")
    print("  - generate_dashboard_html()")
