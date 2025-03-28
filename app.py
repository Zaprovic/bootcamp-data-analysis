import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import geopandas as gpd
import xarray as xr
from scipy.interpolate import griddata

# Set page configuration
st.set_page_config(
    page_title="Estratificación del Dengue - Cauca, Colombia",
    page_icon="🦟",
    layout="wide",
)

# Function to read NetCDF files (from original script)


@st.cache_data
def read_as_df(file_name):
    # Read file
    ds = xr.open_dataset(file_name)
    # Convert to dataframe
    df = ds.to_dataframe()

    # Unstack index (coordinates)
    # Get position of the coordinates in index
    coordinates = ["lat", "lon"]
    # For each coordinate, reset index:
    for coordinate in coordinates:
        position_index = df.index.names.index(coordinate)
        df = df.reset_index(position_index)

    return df


# Load data function with caching to improve performance


@st.cache_data
def load_data():
    df_entomologico = pd.read_csv("data/entomologico.csv")
    df_cases = pd.read_csv("data/cases.csv")

    df_meteorologico_patia = pd.read_csv("data/meteorologicos-patia.csv")
    df_meteorologico_miranda = pd.read_csv("data/meteorologicos-miranda.csv")
    df_meteorologico_piamonte = pd.read_csv("data/meteorologicos-piamonte.csv")

    # Process entomological data
    df_entomologico["locality"] = df_entomologico["locality"].str.replace(
        r"^Casa ubicada en el barrio (.+) de (.+)$", r"\1 (\2)", regex=True
    )
    df_entomologico["locality"] = (
        df_entomologico["locality"].str.extract(r"^([^(]+)", expand=False).str.strip()
    )
    df_entomologico.rename(columns={"locality": "Barrio"}, inplace=True)
    df_entomologico.rename(columns={"county": "Municipio"}, inplace=True)
    df_entomologico["id"] = df_entomologico["id"].str.replace(
        "INS:ProyectoDengue67217:", "", regex=False
    )
    df_entomologico.drop(
        columns=[
            "organismRemarks",
            "eventID",
            "verbatimElevation",
            "higherClassification",
            "genus",
            "subgenus",
            "specificEpithet",
            "vernacularName",
            "eventDate",
            "locationID",
        ],
        inplace=True,
    )

    # Extract male and female counts
    df_entomologico["Machos"] = (
        df_entomologico["sex"].str.extract(r"(\d+)\s*[Mm]achos?").fillna(0).astype(int)
    )
    df_entomologico["Hembras"] = (
        df_entomologico["sex"].str.extract(r"(\d+)\s*[Hh]embras?").fillna(0).astype(int)
    )
    macho_only_mask = df_entomologico["sex"].str.match(r"^[Mm]acho$", na=False)
    hembra_only_mask = df_entomologico["sex"].str.match(r"^[Hh]embra$", na=False)
    df_entomologico.loc[macho_only_mask, "Machos"] = 1
    df_entomologico.loc[hembra_only_mask, "Hembras"] = 1
    df_entomologico.drop(columns=["sex"], inplace=True)

    # Process case data
    df_cases = df_cases.drop(
        columns=[
            "OBJECTID",
            "Loc_name",
            "Match_addr",
            "Pertenencia etnica",
            "fec_consulta",
            "ini_sintomas",
            "locationID",
            "Ocupación ",
        ]
    )
    df_cases.rename(columns={"county": "Barrio"}, inplace=True)
    df_cases["id"] = df_cases.index

    # Process meteorological data
    # Process Patía
    df_meteorologico_patia["Temperatura media (°C)"] = df_meteorologico_patia[
        "Temperatura media (°C)"
    ].str.replace(",", ".")
    df_meteorologico_patia["Temperatura máxima (°C)"] = df_meteorologico_patia[
        "Temperatura máxima (°C)"
    ].str.replace(",", ".")
    df_meteorologico_patia["Temperatura mínima (°C)"] = df_meteorologico_patia[
        "Temperatura mínima (°C)"
    ].str.replace(",", ".")
    df_meteorologico_patia["Humedad (%)"] = df_meteorologico_patia[
        "Humedad (%)"
    ].str.replace(",", ".")
    df_meteorologico_patia["Precipitación (mm)"] = df_meteorologico_patia[
        "Precipitación (mm)"
    ].str.replace(",", ".")
    df_meteorologico_patia["Índice de calor (°C)"] = df_meteorologico_patia[
        "Índice de calor (°C)"
    ].str.replace(",", ".")
    df_meteorologico_patia["Fecha"] = pd.to_datetime(
        df_meteorologico_patia["Fecha"], format="mixed"
    )

    # Process Piamonte
    df_meteorologico_piamonte["Temperatura media (°C)"] = df_meteorologico_piamonte[
        "Temperatura media (°C)"
    ].str.replace(",", ".")
    df_meteorologico_piamonte["Temperatura máxima (°C)"] = df_meteorologico_piamonte[
        "Temperatura máxima (°C)"
    ].str.replace(",", ".")
    df_meteorologico_piamonte["Temperatura mínima (°C)"] = df_meteorologico_piamonte[
        "Temperatura mínima (°C)"
    ].str.replace(",", ".")
    df_meteorologico_piamonte["Humedad (%)"] = df_meteorologico_piamonte[
        "Humedad (%)"
    ].str.replace(",", ".")
    df_meteorologico_piamonte["Precipitación (mm)"] = df_meteorologico_piamonte[
        "Precipitación (mm)"
    ].str.replace(",", ".")
    df_meteorologico_piamonte["Índice de calor (°C)"] = df_meteorologico_piamonte[
        "Índice de calor (°C)"
    ].str.replace(",", ".")
    df_meteorologico_piamonte["Fecha"] = pd.to_datetime(
        df_meteorologico_piamonte["Fecha"], dayfirst=True
    )

    # Process Miranda
    df_meteorologico_miranda["Temperatura media (°C)"] = df_meteorologico_miranda[
        "Temperatura media (°C)"
    ].str.replace(",", ".")
    df_meteorologico_miranda["Temperatura máxima (°C)"] = df_meteorologico_miranda[
        "Temperatura máxima (°C)"
    ].str.replace(",", ".")
    df_meteorologico_miranda["Temperatura mínima (°C)"] = df_meteorologico_miranda[
        "Temperatura mínima (°C)"
    ].str.replace(",", ".")
    df_meteorologico_miranda["Humedad (%)"] = df_meteorologico_miranda[
        "Humedad (%)"
    ].str.replace(",", ".")
    df_meteorologico_miranda["Precipitación (mm)"] = df_meteorologico_miranda[
        "Precipitación (mm)"
    ].str.replace(",", ".")
    df_meteorologico_miranda["Índice de calor (°C)"] = df_meteorologico_miranda[
        "Índice de calor (°C)"
    ].str.replace(",", ".")
    df_meteorologico_miranda["Fecha"] = pd.to_datetime(
        df_meteorologico_miranda["Fecha"], format="mixed"
    )

    # Handle NaN values and convert to float
    meteo_dfs = [
        df_meteorologico_patia,
        df_meteorologico_miranda,
        df_meteorologico_piamonte,
    ]
    meteo_cols = [
        "Temperatura media (°C)",
        "Temperatura máxima (°C)",
        "Temperatura mínima (°C)",
        "Humedad (%)",
        "Precipitación (mm)",
        "Índice de calor (°C)",
    ]

    for df in meteo_dfs:
        df.dropna(subset=["Fecha"], inplace=True)
        df.sort_values(by=["Fecha"], ascending=True, inplace=True)
        for col in meteo_cols:
            # Explicit conversion to float
            df[col] = df[col].astype(float)

    # Load NetCDF files
    try:
        df_humedad_relativa = read_as_df("data/humedad-relativa.nc").reset_index()
        df_precipitacion = read_as_df("data/precipitacion.nc").reset_index()
        df_temp_max = read_as_df("data/temp-maxima.nc").reset_index()
        df_temp_min = read_as_df("data/temp-minima.nc").reset_index()

        df_humedad_relativa["time"] = pd.to_datetime(df_humedad_relativa["time"])
        df_precipitacion["time"] = pd.to_datetime(df_precipitacion["time"])
        df_temp_max["time"] = pd.to_datetime(df_temp_max["time"])
        df_temp_min["time"] = pd.to_datetime(df_temp_min["time"])
    except Exception as e:
        st.error(f"Error loading NetCDF files: {e}")
        df_humedad_relativa = df_precipitacion = df_temp_max = df_temp_min = None

    return (
        df_entomologico,
        df_cases,
        df_meteorologico_patia,
        df_meteorologico_miranda,
        df_meteorologico_piamonte,
        df_humedad_relativa,
        df_precipitacion,
        df_temp_max,
        df_temp_min,
    )


# Group meteorological data by month


@st.cache_data
def group_meteorological_data_by_month(
    df_meteorologico_patia, df_meteorologico_miranda, df_meteorologico_piamonte
):
    meses = {
        1: "Enero",
        2: "Febrero",
        3: "Marzo",
        4: "Abril",
        5: "Mayo",
        6: "Junio",
        7: "Julio",
        8: "Agosto",
        9: "Septiembre",
        10: "Octubre",
        11: "Noviembre",
        12: "Diciembre",
    }

    try:
        # Group data for Patía
        df_meteorologico_patia_grouped = (
            df_meteorologico_patia.groupby(df_meteorologico_patia["Fecha"].dt.month)
            .agg(
                {
                    "Temperatura media (°C)": "mean",
                    "Temperatura máxima (°C)": "max",
                    "Temperatura mínima (°C)": "min",
                    "Humedad (%)": "mean",
                    "Precipitación (mm)": "mean",
                    "Índice de calor (°C)": "mean",
                }
            )
            .reset_index()
        )

        # Group data for Piamonte
        df_meteorologico_piamonte_grouped = (
            df_meteorologico_piamonte.groupby(
                df_meteorologico_piamonte["Fecha"].dt.month
            )
            .agg(
                {
                    "Temperatura media (°C)": "mean",
                    "Temperatura máxima (°C)": "max",
                    "Temperatura mínima (°C)": "min",
                    "Humedad (%)": "mean",
                    "Precipitación (mm)": "mean",
                    "Índice de calor (°C)": "mean",
                }
            )
            .reset_index()
        )

        # Group data for Miranda
        df_meteorologico_miranda_grouped = (
            df_meteorologico_miranda.groupby(df_meteorologico_miranda["Fecha"].dt.month)
            .agg(
                {
                    "Temperatura media (°C)": "mean",
                    "Temperatura máxima (°C)": "max",
                    "Temperatura mínima (°C)": "min",
                    "Humedad (%)": "mean",
                    "Precipitación (mm)": "mean",
                    "Índice de calor (°C)": "mean",
                }
            )
            .reset_index()
        )

        # Map month numbers to names
        df_meteorologico_patia_grouped["Fecha"] = df_meteorologico_patia_grouped[
            "Fecha"
        ].map(meses)
        df_meteorologico_piamonte_grouped["Fecha"] = df_meteorologico_piamonte_grouped[
            "Fecha"
        ].map(meses)
        df_meteorologico_miranda_grouped["Fecha"] = df_meteorologico_miranda_grouped[
            "Fecha"
        ].map(meses)

        return (
            df_meteorologico_patia_grouped,
            df_meteorologico_piamonte_grouped,
            df_meteorologico_miranda_grouped,
        )

    except Exception as e:
        st.error(f"Error en el procesamiento de datos meteorológicos: {e}")
        # Return empty dataframes with the expected columns in case of error
        empty_df = pd.DataFrame(
            columns=[
                "Fecha",
                "Temperatura media (°C)",
                "Temperatura máxima (°C)",
                "Temperatura mínima (°C)",
                "Humedad (%)",
                "Precipitación (mm)",
                "Índice de calor (°C)",
            ]
        )
        return empty_df.copy(), empty_df.copy(), empty_df.copy()


# Function to create location diversity data


@st.cache_data
def create_location_diversity(df_entomologico):
    # First, calculate total mosquitoes (males + females) for each record
    df_entomologico["total_mosquitoes"] = (
        df_entomologico["Machos"] + df_entomologico["Hembras"]
    )

    # Create the base location_diversity with summary statistics
    location_diversity = (
        df_entomologico.groupby(["Barrio", "Municipio"])
        .agg(
            {
                "total_mosquitoes": "sum",  # Sum the total mosquitoes instead of counting records
                "scientificName": pd.Series.nunique,
            }
        )
        .reset_index()
        .rename(
            columns={
                # Keep the same column name for compatibility
                "total_mosquitoes": "Total de individuos",
                "scientificName": "Especies únicas",
            }
        )
    )

    # Create a pivot table with sums for each species at each location
    species_pivot = df_entomologico.pivot_table(
        index=["Barrio", "Municipio"],
        columns="scientificName",
        values="total_mosquitoes",  # Use the total mosquitoes column
        aggfunc="sum",  # Sum instead of count
        fill_value=0,
    )

    # Reset index to prepare for merge
    species_pivot = species_pivot.reset_index()

    # Merge the two DataFrames
    location_diversity = location_diversity.merge(
        species_pivot, on=["Barrio", "Municipio"]
    )

    location_diversity = location_diversity.sort_values(
        "Total de individuos", ascending=False
    )

    # Filter by municipality
    location_diversity_patia = location_diversity[
        location_diversity["Municipio"] == "Patía"
    ].drop(["Especies únicas"], axis=1)
    location_diversity_piamonte = location_diversity[
        location_diversity["Municipio"] == "Piamonte"
    ].drop(["Especies únicas"], axis=1)
    location_diversity_miranda = location_diversity[
        location_diversity["Municipio"] == "Miranda"
    ].drop(["Especies únicas"], axis=1)

    location_diversity = location_diversity.drop(["Especies únicas"], axis=1)

    return (
        location_diversity,
        location_diversity_patia,
        location_diversity_piamonte,
        location_diversity_miranda,
    )


# Function to group geographic data


@st.cache_data
def group_by_coordinates(df, value_column, sort_ascending=False):
    return (
        df.groupby(["lon", "lat"])
        .agg({value_column: "mean"})
        .reset_index()
        .sort_values(value_column, ascending=sort_ascending)
    )


# Main app structure


def main():
    st.title("🦟 Estratificación del Dengue en el Departamento del Cauca, Colombia")

    # Create sidebar for navigation
    st.sidebar.title("Navegación")
    page = st.sidebar.radio(
        "Seleccione una sección:",
        [
            "Introducción",
            "Datos Meteorológicos",
            "Datos Entomológicos",
            "Casos de Dengue",
            "Análisis Geográfico",
        ],
    )

    # Load all data
    (
        df_entomologico,
        df_cases,
        df_meteorologico_patia,
        df_meteorologico_miranda,
        df_meteorologico_piamonte,
        df_humedad_relativa,
        df_precipitacion,
        df_temp_max,
        df_temp_min,
    ) = load_data()

    # Process data for each section as needed
    if page == "Introducción":
        show_introduction(df_entomologico, df_cases)

    elif page == "Datos Meteorológicos":
        show_meteorological_data(
            df_meteorologico_patia, df_meteorologico_miranda, df_meteorologico_piamonte
        )

    elif page == "Datos Entomológicos":
        show_entomological_data(df_entomologico)

    elif page == "Casos de Dengue":
        show_dengue_cases(df_cases)

    elif page == "Análisis Geográfico":
        if (
            df_humedad_relativa is not None
            and df_precipitacion is not None
            and df_temp_max is not None
            and df_temp_min is not None
        ):
            show_geographic_analysis(
                df_entomologico,
                df_humedad_relativa,
                df_precipitacion,
                df_temp_max,
                df_temp_min,
            )
        else:
            st.error(
                "Los archivos NetCDF necesarios no pudieron cargarse correctamente."
            )


def show_introduction(df_entomologico, df_cases):
    st.markdown(
        """
    # Introducción al Análisis del Dengue en Cauca

    Este dashboard interactivo presenta un análisis detallado de la estratificación del dengue en el departamento del Cauca, Colombia.
    El análisis se centra en tres municipios principales: **Patía**, **Miranda** y **Piamonte**.

    ## Componentes del análisis:
    
    1. **Datos meteorológicos**: Temperatura, humedad y precipitaciones que influyen en la proliferación del vector.
    2. **Datos entomológicos**: Diversidad y distribución de especies de mosquitos.
    3. **Casos de dengue**: Distribución geográfica y temporal de los casos reportados.
    4. **Análisis geográfico**: Visualización espacial de variables relevantes.

    """
    )

    # Display a summary of available data
    st.subheader("Datos disponibles para análisis")
    col1, col2, col3 = st.columns(3)

    with col1:
        st.info(f"Datos entomológicos: {df_entomologico.shape[0]} registros")

    with col2:
        st.info(f"Casos de dengue: {df_cases.shape[0]} casos reportados")

    with col3:
        total_mosquitos = (
            df_entomologico["Machos"].sum() + df_entomologico["Hembras"].sum()
        )
        st.info(f"Total de mosquitos: {total_mosquitos} ejemplares")


def show_meteorological_data(
    df_meteorologico_patia, df_meteorologico_miranda, df_meteorologico_piamonte
):
    st.header("Análisis de Datos Meteorológicos")

    # Group data by month
    df_patia_grouped, df_piamonte_grouped, df_miranda_grouped = (
        group_meteorological_data_by_month(
            df_meteorologico_patia, df_meteorologico_miranda, df_meteorologico_piamonte
        )
    )

    # Create a variable selector
    variable = st.selectbox(
        "Seleccione la variable meteorológica a visualizar:",
        [
            "Temperatura media (°C)",
            "Temperatura máxima (°C)",
            "Temperatura mínima (°C)",
            "Humedad (%)",
            "Precipitación (mm)",
            "Índice de calor (°C)",
        ],
    )

    # Create multi-select for municipalities
    municipios = st.multiselect(
        "Seleccione municipios a mostrar",
        ["Patía", "Miranda", "Piamonte"],
        default=["Patía", "Miranda", "Piamonte"],
    )

    # Create the figure
    fig, ax = plt.subplots(figsize=(12, 3))

    # Common x-axis
    x_order = [
        "Enero",
        "Febrero",
        "Marzo",
        "Abril",
        "Mayo",
        "Junio",
        "Julio",
        "Agosto",
        "Septiembre",
        "Octubre",
        "Noviembre",
        "Diciembre",
    ]

    # Plot selected data
    if "Patía" in municipios:
        ax.plot(
            df_patia_grouped["Fecha"],
            df_patia_grouped[variable],
            marker="o",
            label="Patía",
        )

    if "Piamonte" in municipios:
        ax.plot(
            df_piamonte_grouped["Fecha"],
            df_piamonte_grouped[variable],
            marker="s",
            label="Piamonte",
        )

    if "Miranda" in municipios:
        ax.plot(
            df_miranda_grouped["Fecha"],
            df_miranda_grouped[variable],
            marker="^",
            label="Miranda",
        )

    ax.set_ylabel(variable)
    ax.set_title(f"{variable} por Mes")
    ax.grid(True, linestyle="--", alpha=0.7)
    ax.legend()

    # Display the plot
    st.pyplot(fig)

    # Display statistics
    st.subheader("Estadísticas descriptivas")
    col1, col2, col3 = st.columns(3)

    with col1:
        if "Patía" in municipios:
            st.write("**Patía**")
            st.write(f"Promedio: {df_patia_grouped[variable].mean():.2f}")
            st.write(f"Máximo: {df_patia_grouped[variable].max():.2f}")
            st.write(f"Mínimo: {df_patia_grouped[variable].min():.2f}")

    with col2:
        if "Miranda" in municipios:
            st.write("**Miranda**")
            st.write(f"Promedio: {df_miranda_grouped[variable].mean():.2f}")
            st.write(f"Máximo: {df_miranda_grouped[variable].max():.2f}")
            st.write(f"Mínimo: {df_miranda_grouped[variable].min():.2f}")

    with col3:
        if "Piamonte" in municipios:
            st.write("**Piamonte**")
            st.write(f"Promedio: {df_piamonte_grouped[variable].mean():.2f}")
            st.write(f"Máximo: {df_piamonte_grouped[variable].max():.2f}")
            st.write(f"Mínimo: {df_piamonte_grouped[variable].min():.2f}")

    # Raw data table with filter
    st.subheader("Datos meteorológicos")
    show_raw = st.checkbox("Mostrar datos crudos")

    if show_raw:
        if "Patía" in municipios:
            st.write("**Patía**")
            st.dataframe(df_meteorologico_patia)

        if "Miranda" in municipios:
            st.write("**Miranda**")
            st.dataframe(df_meteorologico_miranda)

        if "Piamonte" in municipios:
            st.write("**Piamonte**")
            st.dataframe(df_meteorologico_piamonte)


def show_entomological_data(df_entomologico):
    st.header("Análisis de Datos Entomológicos")

    # Create location diversity data
    (
        location_diversity,
        location_diversity_patia,
        location_diversity_piamonte,
        location_diversity_miranda,
    ) = create_location_diversity(df_entomologico)

    # Show species distribution overall
    st.subheader("Distribución de especies por municipio")

    # Select municipality
    municipio = st.selectbox(
        "Seleccione un municipio:", ["Todos", "Patía", "Miranda", "Piamonte"]
    )

    if municipio == "Todos":
        filtered_data = df_entomologico
    else:
        filtered_data = df_entomologico[df_entomologico["Municipio"] == municipio]

    # Calculate species counts
    # Calculate species counts based on individualCount sum
    sc = (
        filtered_data.groupby("scientificName")["individualCount"]
        .sum()
        .sort_values(ascending=False)
    )

    # Create a pie chart with matplotlib
    fig, ax = plt.subplots(figsize=(5, 3))
    colors = plt.cm.viridis(np.linspace(0, 0.50, len(sc)))

    # Calculate percentages for the legend
    total = sc.sum()
    percentages = [(count / total) * 100 for count in sc.values]

    wedges, texts = ax.pie(sc.values, labels=None, colors=colors, startangle=90)

    legend_labels = [
        f"{name} ({count} ejemplares, {perc:.1f}%)"
        for name, count, perc in zip(sc.index, sc.values, percentages)
    ]
    ax.legend(
        wedges,
        legend_labels,
        title="Especies",
        loc="center left",
        bbox_to_anchor=(1, 0, 0.5, 1),
        fontsize=9,
    )

    plt.title(
        f'Distribución de Especies Recolectadas en {municipio if municipio != "Todos" else "Todos los municipios"}',
        fontsize=14,
    )
    plt.tight_layout()

    # Show the plot
    st.pyplot(fig)

    # Show species distribution by neighborhood
    st.subheader("Distribución de especies por barrio")

    # Select which diversity data to show based on municipality
    if municipio == "Patía":
        diversity_data = location_diversity_patia
    elif municipio == "Miranda":
        diversity_data = location_diversity_miranda
    elif municipio == "Piamonte":
        diversity_data = location_diversity_piamonte
    else:
        diversity_data = location_diversity

    # Sort data
    diversity_data_sorted = diversity_data.sort_values(
        "Total de individuos", ascending=True
    )

    # Create the stacked bar chart
    if not diversity_data_sorted.empty:
        fig, ax = plt.subplots(figsize=(20, 6))

        # Get species columns
        species_cols = ["Aedes aegypti", "Aedes albopictus", "Culex quinquefasciatus"]
        species_cols = [
            col for col in species_cols if col in diversity_data_sorted.columns
        ]

        # Create color palette
        species_colors = sns.color_palette("viridis", len(species_cols))

        # Create stacked bar chart
        bottom = np.zeros(len(diversity_data_sorted))

        for i, species in enumerate(species_cols):
            if species in diversity_data_sorted.columns:
                counts = diversity_data_sorted[species].values
                barrios = diversity_data_sorted["Barrio"].values

                bars = ax.bar(
                    barrios,
                    counts,
                    bottom=bottom,
                    color=species_colors[i],
                    label=species,
                    edgecolor="white",
                    linewidth=0.5,
                )
                bottom += counts

                # Add count labels
                for j, (count, b) in enumerate(zip(counts, bottom - counts / 2)):
                    if count > 5:
                        ax.text(
                            j,
                            bottom[j] - counts[j] / 2,
                            str(int(count)),
                            ha="center",
                            va="center",
                            color="white",
                            fontweight="bold",
                        )

        ax.set_title(
            f'Distribución de especies por barrio en {municipio if municipio != "Todos" else "todos los municipios"}',
            fontsize=14,
        )
        ax.set_ylabel("Cantidad de ejemplares", fontsize=12)
        ax.set_xlabel("Barrio", fontsize=12)
        ax.legend(title="Especies")
        plt.grid(axis="y", linestyle="--", alpha=0.3)
        plt.xticks(rotation=0, ha="right")

        plt.tight_layout()
        st.pyplot(fig)

        # Add summary table of totals by neighborhood
        st.subheader(
            f"Total de ejemplares por barrio en {municipio if municipio != 'Todos' else 'todos los municipios'}"
        )

        # Create a summary dataframe with just Barrio and total counts
        summary_df = diversity_data_sorted[["Barrio", "Total de individuos"]].copy()

        # Display as a formatted table with metrics
        col1, col2 = st.columns([3, 1])
        with col1:
            # Format the dataframe for better display
            summary_styled = summary_df.style.format({"Total de individuos": "{:.0f}"})
            st.dataframe(summary_styled, use_container_width=True)

        with col2:
            # Show total specimens for this municipality
            total_specimens = summary_df["Total de individuos"].sum()
            st.metric("Total de ejemplares", f"{int(total_specimens)}")
    else:
        st.write("No hay datos disponibles para el municipio seleccionado.")

    # Raw data with filters
    st.subheader("Datos entomológicos filtrados")

    # Multi-select filter for species
    species = st.multiselect(
        "Filtrar por especies:", df_entomologico["scientificName"].unique(), default=[]
    )

    # Filter data
    if species:
        filtered_data = filtered_data[filtered_data["scientificName"].isin(species)]

    # Show filtered data
    st.dataframe(filtered_data)


def show_dengue_cases(df_cases):
    st.header("Análisis de Casos de Dengue")

    # Group data by municipality
    casos_dengue_municipio = (
        df_cases.groupby("Barrio")
        .agg(
            {
                "id": "count",
            }
        )
        .reset_index()
        .rename(columns={"id": "Casos"})
    )

    # Create pie chart
    fig, ax = plt.subplots(figsize=(5, 5))
    colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(casos_dengue_municipio)))

    wedges, texts, autotexts = ax.pie(
        casos_dengue_municipio["Casos"],
        labels=None,
        autopct="%1.1f%%",
        colors=colors,
        startangle=90,
        shadow=False,
        textprops={"fontsize": 12, "weight": "bold", "color": "white"},
    )

    legend_labels = [
        f"{mun} ({cases} casos)"
        for mun, cases in zip(
            casos_dengue_municipio["Barrio"], casos_dengue_municipio["Casos"]
        )
    ]
    ax.legend(
        wedges, legend_labels, loc="center left", bbox_to_anchor=(1, 0.5), fontsize=10
    )

    ax.set_title("Distribución de casos de dengue por municipio", fontsize=14, pad=20)
    plt.tight_layout()

    # Show the plot
    st.pyplot(fig)

    # Show case data
    st.subheader("Datos de casos de dengue")

    # Add filtering options
    barrio_filter = st.selectbox(
        "Filtrar por barrio:", ["Todos"] + list(df_cases["Barrio"].unique())
    )

    # Apply filter
    if barrio_filter != "Todos":
        filtered_cases = df_cases[df_cases["Barrio"] == barrio_filter]
    else:
        filtered_cases = df_cases

    # Show filtered data
    st.dataframe(filtered_cases)

    # Show summary statistics
    st.subheader("Resumen estadístico")
    st.write(f"Total de casos registrados: {len(df_cases)}")
    st.write(f"Municipios afectados: {df_cases['Barrio'].nunique()}")


def show_geographic_analysis(
    df_entomologico, df_humedad_relativa, df_precipitacion, df_temp_max, df_temp_min
):
    st.header("Análisis Geográfico")

    # Group data by coordinates for better visualization
    df_humedad_relativa_grouped = group_by_coordinates(
        df_humedad_relativa, "relative_humidity"
    )
    df_precipitacion_grouped = group_by_coordinates(df_precipitacion, "precipitation")
    df_temp_max_grouped = group_by_coordinates(df_temp_max, "air_temperature")
    df_temp_min_grouped = group_by_coordinates(df_temp_min, "air_temperature")

    # Select map type
    map_type = st.selectbox(
        "Seleccione el tipo de visualización:",
        ["Distribución de mosquitos", "Variables meteorológicas"],
    )

    if map_type == "Distribución de mosquitos":
        # Select municipality
        municipio = st.selectbox(
            "Seleccione un municipio:", ["Patía", "Miranda", "Piamonte"]
        )

        # Filter data by municipality
        df_filtered = df_entomologico[df_entomologico["Municipio"] == municipio]

        # Use appropriate colors for each municipality
        if municipio == "Patía":
            color = "red"
        elif municipio == "Miranda":
            color = "blue"
        else:  # Piamonte
            color = "green"

        # Create GeoDataFrame
        gdf = gpd.GeoDataFrame(
            df_filtered,
            geometry=gpd.points_from_xy(
                df_filtered.decimalLongitude, df_filtered.decimalLatitude
            ),
        )

        # Use columns to constrain the width of the figure
        col1, col2, col3 = st.columns([1, 2, 1])

        with col2:
            # Create matplotlib figure for the map - SIGNIFICANTLY REDUCED SIZE
            fig, ax = plt.subplots(figsize=(3, 2.5))
            gdf.plot(ax=ax, marker="o", color=color, markersize=4, alpha=0.6)
            ax.set_title(f"Distribución en {municipio}", fontsize=10)
            ax.set_xlabel("Longitud", fontsize=8)
            ax.set_ylabel("Latitud", fontsize=8)
            ax.tick_params(axis="both", which="major", labelsize=7)
            plt.tight_layout()

            # Show the plot
            st.pyplot(fig)

    else:  # Variables meteorológicas
        # Create tabs for different meteorological variables
        tab1, tab2 = st.tabs(["Temperatura", "Humedad y Precipitación"])

        with tab1:
            # Create figure for temperature data
            fig, ax = plt.subplots(1, 2, figsize=(15, 6))

            # Plot max temperature
            lon_max = df_temp_max_grouped["lon"].values
            lat_max = df_temp_max_grouped["lat"].values
            values_max = df_temp_max_grouped["air_temperature"].values

            # Create grid for interpolation
            grid_lon_max = np.linspace(lon_max.min(), lon_max.max(), 400)
            grid_lat_max = np.linspace(lat_max.min(), lat_max.max(), 400)
            grid_lon_max, grid_lat_max = np.meshgrid(grid_lon_max, grid_lat_max)

            # Interpolate values
            grid_values_max = griddata(
                (lon_max, lat_max),
                values_max,
                (grid_lon_max, grid_lat_max),
                method="cubic",
            )

            # Create plot
            im0 = ax[0].pcolormesh(
                grid_lon_max,
                grid_lat_max,
                grid_values_max,
                cmap="coolwarm",
                shading="auto",
            )
            fig.colorbar(im0, ax=ax[0], label="Temperatura (°C)")
            ax[0].set_title("Temperatura máxima promedio")
            ax[0].set_xlabel("Longitud")
            ax[0].set_ylabel("Latitud")

            # Plot min temperature
            lon_min = df_temp_min_grouped["lon"].values
            lat_min = df_temp_min_grouped["lat"].values
            values_min = df_temp_min_grouped["air_temperature"].values

            # Create grid for interpolation
            grid_lon_min = np.linspace(lon_min.min(), lon_min.max(), 400)
            grid_lat_min = np.linspace(lat_min.min(), lat_min.max(), 400)
            grid_lon_min, grid_lat_min = np.meshgrid(grid_lon_min, grid_lat_min)

            # Interpolate values
            grid_values_min = griddata(
                (lon_min, lat_min),
                values_min,
                (grid_lon_min, grid_lat_min),
                method="cubic",
            )

            # Create plot
            im1 = ax[1].pcolormesh(
                grid_lon_min,
                grid_lat_min,
                grid_values_min,
                cmap="viridis",
                shading="auto",
            )
            fig.colorbar(im1, ax=ax[1], label="Temperatura (°C)")
            ax[1].set_title("Temperatura mínima promedio")
            ax[1].set_xlabel("Longitud")
            ax[1].set_ylabel("Latitud")

            plt.tight_layout()
            st.pyplot(fig)

        with tab2:
            # Create figure for humidity and precipitation
            fig, ax = plt.subplots(1, 2, figsize=(15, 6))

            # Plot humidity
            lon_hum = df_humedad_relativa_grouped["lon"].values
            lat_hum = df_humedad_relativa_grouped["lat"].values
            values_hum = df_humedad_relativa_grouped["relative_humidity"].values

            # Create grid for interpolation
            grid_lon_hum = np.linspace(lon_hum.min(), lon_hum.max(), 400)
            grid_lat_hum = np.linspace(lat_hum.min(), lat_hum.max(), 400)
            grid_lon_hum, grid_lat_hum = np.meshgrid(grid_lon_hum, grid_lat_hum)

            # Interpolate values
            grid_values_hum = griddata(
                (lon_hum, lat_hum),
                values_hum,
                (grid_lon_hum, grid_lat_hum),
                method="cubic",
            )

            # Create plot
            im0 = ax[0].pcolormesh(
                grid_lon_hum,
                grid_lat_hum,
                grid_values_hum,
                cmap="coolwarm",
                shading="auto",
            )
            fig.colorbar(im0, ax=ax[0], label="Humedad (%)")
            ax[0].set_title("Humedad relativa promedio")
            ax[0].set_xlabel("Longitud")
            ax[0].set_ylabel("Latitud")

            # Plot precipitation
            lon_prec = df_precipitacion_grouped["lon"].values
            lat_prec = df_precipitacion_grouped["lat"].values
            values_prec = df_precipitacion_grouped["precipitation"].values

            # Create grid for interpolation
            grid_lon_prec = np.linspace(lon_prec.min(), lon_prec.max(), 400)
            grid_lat_prec = np.linspace(lat_prec.min(), lat_prec.max(), 400)
            grid_lon_prec, grid_lat_prec = np.meshgrid(grid_lon_prec, grid_lat_prec)

            # Interpolate values
            grid_values_prec = griddata(
                (lon_prec, lat_prec),
                values_prec,
                (grid_lon_prec, grid_lat_prec),
                method="cubic",
            )

            # Create plot
            im1 = ax[1].pcolormesh(
                grid_lon_prec,
                grid_lat_prec,
                grid_values_prec,
                cmap="viridis",
                shading="auto",
            )
            fig.colorbar(im1, ax=ax[1], label="Precipitación (mm)")
            ax[1].set_title("Precipitación promedio")
            ax[1].set_xlabel("Longitud")
            ax[1].set_ylabel("Latitud")

            plt.tight_layout()
            st.pyplot(fig)


if __name__ == "__main__":
    main()
