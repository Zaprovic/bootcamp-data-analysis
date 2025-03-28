
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import geopandas as gpd
from scipy.interpolate import griddata
import numpy as np
import xarray as xr

# Load data

df_entomologico = pd.read_csv('../data/entomologico.csv')
df_cases = pd.read_csv('../data/cases.csv')


def read_as_df(file_name):
    ds = xr.open_dataset(file_name)
    df = ds.to_dataframe()
    coordinates = ['lat', 'lon']
    for coordinate in coordinates:
        position_index = df.index.names.index(coordinate)
        df = df.reset_index(position_index)
    return df


# Data loading
df_humedad_relativa = read_as_df("../data/humedad-relativa.nc").reset_index()
df_precipitacion = read_as_df("../data/precipitacion.nc").reset_index()
df_temp_max = read_as_df("../data/temp-maxima.nc").reset_index()
df_temp_min = read_as_df("../data/temp-minima.nc").reset_index()

# Clean data
df_humedad_relativa['time'] = pd.to_datetime(df_humedad_relativa['time'])
df_precipitacion['time'] = pd.to_datetime(df_precipitacion['time'])
df_temp_max['time'] = pd.to_datetime(df_temp_max['time'])
df_temp_min['time'] = pd.to_datetime(df_temp_min['time'])

# Streamlit widgets for user input
st.title('Interactive Dashboard: Dengue Stratification in Cauca, Colombia')

st.sidebar.title('Filters')
municipio = st.sidebar.selectbox(
    "Select Municipio", ['Patía', 'Piamonte', 'Miranda'])
species_filter = st.sidebar.multiselect("Select Species", options=df_entomologico['scientificName'].unique(
), default=['Aedes aegypti', 'Aedes albopictus'])

# Filter data based on user input
df_entomologico_filtered = df_entomologico[df_entomologico['Municipio'] == municipio]
df_entomologico_filtered = df_entomologico_filtered[df_entomologico_filtered['scientificName'].isin(
    species_filter)]

# Plotting functions


def plot_meteorological_data():
    fig, axs = plt.subplots(5, 1, figsize=(15, 20))
    x_order = ['Enero', 'Febrero', 'Marzo', 'Abril', 'Mayo', 'Junio',
               'Julio', 'Agosto', 'Septiembre', 'Octubre', 'Noviembre', 'Diciembre']

    axs[0].plot(df_temp_max['time'],
                df_temp_max['air_temperature'], label='Max Temp')
    axs[0].set_title('Temperature Maximum')

    axs[1].plot(df_temp_min['time'],
                df_temp_min['air_temperature'], label='Min Temp')
    axs[1].set_title('Temperature Minimum')

    axs[2].plot(df_humedad_relativa['time'],
                df_humedad_relativa['relative_humidity'], label='Humidity')
    axs[2].set_title('Relative Humidity')

    axs[3].plot(df_precipitacion['time'],
                df_precipitacion['precipitation'], label='Precipitation')
    axs[3].set_title('Precipitation')

    axs[4].set_xlabel('Time')
    axs[4].set_ylabel('Precipitation / Temperature')

    plt.tight_layout()
    st.pyplot(fig)


def plot_species_distribution():
    sc = df_entomologico_filtered['scientificName'].value_counts()
    fig, ax = plt.subplots(figsize=(6, 5))

    colors = plt.cm.viridis(np.linspace(0, 0.50, len(sc)))

    wedges, texts, autotexts = ax.pie(sc.values, labels=None, autopct='%1.1f%%', colors=colors, textprops={
                                      'fontsize': 10, 'weight': 'bold', 'color': 'white'}, startangle=90)

    legend_labels = [f"{name} ({count} ejemplares)" for name,
                     count in zip(sc.index, sc.values)]
    ax.legend(wedges, legend_labels, title="Especies",
              loc="center left", bbox_to_anchor=(1, 0, 0.5, 1), fontsize=9)
    ax.set_title('Distribución de Especies Recolectadas', fontsize=14)

    st.pyplot(fig)


def plot_geographic_distribution():
    df_patia = df_entomologico_filtered[df_entomologico_filtered['Municipio'] == 'Patía']
    gdf_patia = gpd.GeoDataFrame(df_patia, geometry=gpd.points_from_xy(
        df_patia.decimalLongitude, df_patia.decimalLatitude))
    ax = gdf_patia.plot(marker='o', color='red', markersize=5, alpha=0.6)
    plt.title('Distribución en Patía', fontsize=14)
    plt.xlabel('Longitud')
    plt.ylabel('Latitud')
    plt.tight_layout()
    st.pyplot()


# Display selected plots
plot_meteorological_data()
plot_species_distribution()
plot_geographic_distribution()
