#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
#  AnalisisWindCF30years.py
#
#  2023 Antonio Jiménez-Garrote <agarrote@ujaen.es>
#
#
'''
Este programa recoge las lineas de codigo necesarias para el analisis propuesto
por David para los datos de CF eolicos de 30 anyos.
'''

import sys
from WorkingDirectory_py3 import WorkingDirectory
WorkDir = WorkingDirectory('archivoConfiguracion.ini')
sys.path.append(WorkDir.getDirectorioLibrerias())

#-------------------------------------------------------------------LIBRARIES---
import numpy as np
import pandas as pd
import geopandas as gpd
from datetime import datetime, timedelta
from scipy.optimize import curve_fit
from matplotlib import pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cf
from matplotlib.colorbar import ColorbarBase
from matplotlib.colors import Normalize
import ArtistElsevier_py3 as Artist
from REE_py3 import CargaArchivoGeneracionESIOS
from SOWISP_py3 import SOWISP_HighResolution
from WindUtils_py3 import Correlacion, ArrayCorrelacionEspacial
#-------------------------------------------------------------------------------

#-------------------------------------------------------------------CONSTANTS---
YEAR_INI = 1990
YEAR_FIN = 2020
FECHA_CONGELAMIENTO = '2020-01-01 00:00:00' # date_format %Y-%m-%d %H:%M:%S
TUPLA_PROVINCIAS = ('ACoruna', 'Albacete', 'Almeria', 'Araba', 'Asturias',
    'Avila', 'Barcelona', 'Bizkaia', 'Burgos', 'Cadiz', 'Cantabria', 'Castellon',
    'CiudadReal', 'Cuenca', 'Granada', 'Guadalajara', 'Huelva', 'Huesca', 'Jaen',
    'LaRioja', 'Leon', 'Lleida', 'Lugo', 'Malaga', 'Murcia', 'Navarra', 'Ourense',
    'Palencia', 'Pontevedra', 'Salamanca', 'Segovia', 'Sevilla', 'Soria',
    'Tarragona', 'Teruel', 'Toledo', 'Valencia', 'Valladolid', 'Zamora',
    'Zaragoza'
)
DICT_PROVINCIAS_IGN = {
    'ACoruna': 'A Coruña', 'Almeria': 'Almería', 'Araba': 'Araba/Álava',
    'Avila': 'Ávila', 'Caceres': 'Cáceres', 'Cadiz': 'Cádiz',
    'Castellon': 'Castelló/Castellón', 'CiudadReal': 'Ciudad Real',
    'Jaen': 'Jaén', 'LaRioja': 'La Rioja', 'Leon': 'León', 'Malaga': 'Málaga',
    'Valencia': 'València/Valencia'
}
NOMBRE_ESPANYA = 'Spain'
NOMBRE_COL_PESOS = 'Peso'
NOMBRE_COL_FECHA_ML = 'Fecha'
NOMBRE_COL_CF_ML = 'prediccion'
NOMBRE_COL_FECHA = 'Date_UTC'
NOMBRE_COL_CF = '{}_CF'
NOMBRE_COL_LAT = 'lat'
NOMBRE_COL_LON = 'lon'
# NOMBRE_COL_POT_INS = '{}_PotIns_[MW]'
# NOMBRE_COL_MWH = '{}_Generacion_[MWh]'
DICCIONARIO_ESTACIONES = {'01': 'Winter', '02': 'Winter', '03': 'Spring', '04': 'Spring', '05': 'Spring', '06': 'Summer', '07': 'Summer', '08': 'Summer', '09': 'Autumn', '10': 'Autumn', '11': 'Autumn', '12': 'Winter'}
#-------------------------------------------------------------------------------

#-------------------------------------------------------------------FUNCTIONS---
def LeeCsvMadrid(archivoCsv, columnaFechas = NOMBRE_COL_FECHA_ML, columnaCF = NOMBRE_COL_CF_ML):
    df = pd.read_csv(archivoCsv, usecols = [columnaFechas, columnaCF], index_col = columnaFechas, parse_dates = True)
    return df.copy()

def Merge2DataFrames(df1, df2, metodo):
    dfMerge = pd.merge(df1, df2, how = metodo, left_index = True, right_index = True)
    return dfMerge.copy()

def SumaYearDiciembres(date):
    year = date.year
    mes = date.month
    if mes == 12:
        year += 1
    return year

def IsDJFM(date):
    mes = date.month
    if np.isin(mes, (12, 1, 2, 3)).item() == True:
        return 'Si'
    else:
        return 'No'

def CalculaCentroidesGeoDf(geoDataFrame):
    '''
    Voy a proceder de manera analoga a la pagina oficial de GeoPandas:
    https://geopandas.org/docs/user_guide/data_structures.html
    '''
    geoDf = geoDataFrame.copy()
    geoDf.rename(columns = {'geometry': 'geometriaETRS89'}, inplace = True)
    '''
    Aparece el siguiente Warning cuando se usa la funcion centroid:
    Geometry is in a geographic CRS. Results from 'centroid' are likely
    incorrect. Use 'GeoSeries.to_crs()' to re-project geometries to a projected
    CRS before this operation.
    No se definir la proyeccion de las simulaciones a un Sistema de Coordenadas
    de Referencia en el que aparezcan las latitudes como unidad. Voy a optar por
    cambiar la proyeccion a WGS84 que es la tipica de los GPS.
    '''
    geoDf['geometriaWGS84'] = geoDf['geometriaETRS89'].to_crs('EPSG:4326').copy()
    geoDf['centroidesWGS84'] = geoDf['geometriaWGS84'].centroid.copy()
    centroidesGeopandas = gpd.GeoDataFrame(
        geoDf[['NAMEUNIT', 'centroidesWGS84']],
        geometry = 'centroidesWGS84',
        crs = 'EPSG:4326'
    )
    centroidesGeopandas.set_index('NAMEUNIT', inplace = True)
    return centroidesGeopandas.copy()

def ExtraeLatLonProvinciaGeoDfCentroides(geoDf, nombreIndex):
    nombreCol = geoDf.columns.item()
    return geoDf.loc[nombreIndex, nombreCol].y, geoDf.loc[nombreIndex, nombreCol].x

def DivideArrayCorrelacionEspacialEn2FlattenArrays(arrayCorrDist2D):
    arrayFlattenCorr = np.array([], dtype = float)
    arrayFlattenDist = np.array([], dtype = float)
    for iterator in range(1, arrayCorrDist2D.shape[1]):
        arrayFlattenCorr = np.concatenate((arrayFlattenCorr.copy(), arrayCorrDist2D[iterator - 1, iterator:]))
        arrayFlattenDist = np.concatenate((arrayFlattenDist.copy(), arrayCorrDist2D[iterator:, iterator - 1]))
    return arrayFlattenCorr, arrayFlattenDist

def FuncionExponecial(d, alpha, beta):
    return np.exp(-alpha * np.power(d, beta))
#-------------------------------------------------------------------------------

def main():
    dictDfRegiones = {}
    dfNacional = pd.DataFrame()
    latlon_centroids_NUTS3 = pd.DataFrame()
    cf_NUTS3 = pd.DataFrame()
    # dfNacionalActualizado = pd.DataFrame()
    # primeraProvincia = True

    shape_NUTS3 = gpd.read_file(WorkDir.getDirectorioAdicionalFromClave('shapefile'))
    shape_centroids_NUTS3 = CalculaCentroidesGeoDf(shape_NUTS3)

    sowisp = SOWISP_HighResolution(WorkDir.getFileFromClave('sowisp'))
    listRegiones = list(TUPLA_PROVINCIAS)
    listRegiones.append(NOMBRE_ESPANYA)
    sowisp.FiltraPorListaDeRegiones(listRegiones, inplace = True)
    dfPotIns = sowisp.SerieTemporalPotenciaInstaladaMW(datetime(YEAR_INI, 1, 1), datetime(YEAR_FIN, 12, 31, 23)) # por si se quiere modelizar el comportamiento historico de la generacion
    dfPesosCongelados = pd.DataFrame(
        dfPotIns.loc[FECHA_CONGELAMIENTO, list(TUPLA_PROVINCIAS)].values / dfPotIns.loc[FECHA_CONGELAMIENTO, list(TUPLA_PROVINCIAS)].sum(),
        index = pd.Index(TUPLA_PROVINCIAS, name = 'Provincia'),
        columns = [NOMBRE_COL_PESOS]
    )

    # tablaNAO = pd.read_table('/mnt/DATOS/Doctorado/ComparativaModelosEolicosRegionales/Datos/nao_reanalysis_t10trunc_1948-present.txt', sep = '\s+', names = ['Y', 'm', 'd', 'idxNAO'])
    # dfNAO = pd.DataFrame(tablaNAO['idxNAO'].values, index = pd.Index([datetime(tablaNAO.iloc[idx, 0], tablaNAO.iloc[idx, 1], tablaNAO.iloc[idx, 2]) for idx in tablaNAO.index.values], name = 'Date'), columns = ['idxNAO'])

    # tablaNAO = pd.read_table('/mnt/DATOS/Doctorado/ComparativaModelosEolicosRegionales/Datos/nao_station_monthly.txt', sep = '\s+', skiprows = 1, index_col = 0)
    # dfNAO = pd.DataFrame(
    #     tablaNAO.stack().values,
    #     index = pd.Index(
    #         pd.date_range(datetime(tablaNAO.index[0], 1, 1), datetime(tablaNAO.index[-1], 12, 31), freq = 'MS').to_pydatetime(),
    #         name = 'Date'
    #     ),
    #     columns = ['idxNAO']
    # )

    # tablaNAO = pd.read_table('/media/agarrote/DATOS/Doctorado/ComparativaModelosEolicosRegionales/Datos/nao_station_djfm.txt', sep = '\s+', skiprows = 1, names = ['year', 'idxNAO'], index_col = 0)

    # mean_djfm = {}
    for nombreProvincia in TUPLA_PROVINCIAS:
        print(nombreProvincia)
        dictDfRegiones[nombreProvincia] = pd.DataFrame()
        for year in range(YEAR_INI, YEAR_FIN + 1):
            dfModeloCFanyo = LeeCsvMadrid(WorkDir.getFileFromClave('wind_cf30').format(year, nombreProvincia, year))
            dfModeloCFanyo.rename(columns = {NOMBRE_COL_FECHA_ML: NOMBRE_COL_FECHA, NOMBRE_COL_CF_ML: NOMBRE_COL_CF.format(nombreProvincia)}, inplace = True)
            dictDfRegiones[nombreProvincia] = pd.concat([dictDfRegiones[nombreProvincia].copy(), dfModeloCFanyo.copy()])

        # df_copy = dictDfRegiones[nombreProvincia].copy()
        # df_copy['year'] = [int(SumaYearDiciembres(date)) for date in df_copy.index.to_pydatetime()]
        # df_copy['isDJFM'] = [IsDJFM(date) for date in df_copy.index.to_pydatetime()]
        # df_copy.where(df_copy['isDJFM'] == 'Si', np.nan, inplace = True)
        # df_copy.dropna(inplace = True)
        # df_copy_group = df_copy.groupby('year').mean()
        # mean_djfm[nombreProvincia] = pd.DataFrame(df_copy_group[NOMBRE_COL_CF.format(nombreProvincia)].values, columns = [NOMBRE_COL_CF.format(nombreProvincia)], index = df_copy_group.index.astype(int))

        try:
            latProv, lonProv = ExtraeLatLonProvinciaGeoDfCentroides(shape_centroids_NUTS3, nombreProvincia)
        except KeyError:
            latProv, lonProv = ExtraeLatLonProvinciaGeoDfCentroides(shape_centroids_NUTS3, DICT_PROVINCIAS_IGN[nombreProvincia])
        latlon_centroids_NUTS3 = pd.concat([latlon_centroids_NUTS3.copy(), pd.DataFrame(np.array([latProv, lonProv]).reshape(1, 2), index = [nombreProvincia], columns = [NOMBRE_COL_LAT, NOMBRE_COL_LON])])

        if nombreProvincia == TUPLA_PROVINCIAS[0]:
            metodo = 'outer'
        else:
            metodo = 'inner'
        dfNacional = Merge2DataFrames(
            dfNacional.copy(),
            pd.DataFrame(
                dictDfRegiones[nombreProvincia][NOMBRE_COL_CF.format(nombreProvincia)].values * dfPesosCongelados.loc[nombreProvincia, NOMBRE_COL_PESOS],
                columns = [f'{NOMBRE_COL_CF.format(nombreProvincia)}_weight'],
                index = dictDfRegiones[nombreProvincia].index
            ),
            metodo
        )
        cf_NUTS3 = Merge2DataFrames(
            cf_NUTS3.copy(),
            pd.DataFrame(
                dictDfRegiones[nombreProvincia][NOMBRE_COL_CF.format(nombreProvincia)].values.copy(),
                columns = [nombreProvincia],
                index = dictDfRegiones[nombreProvincia].index
            ),
            metodo
        )
    dictDfRegiones[NOMBRE_ESPANYA] = pd.DataFrame(dfNacional.sum(axis = 1), columns = [NOMBRE_COL_CF.format(NOMBRE_ESPANYA)])
    # df_copy = dictDfRegiones[NOMBRE_ESPANYA].copy()
    # df_copy['year'] = [int(SumaYearDiciembres(date)) for date in df_copy.index.to_pydatetime()]
    # df_copy['isDJFM'] = [IsDJFM(date) for date in df_copy.index.to_pydatetime()]
    # df_copy.where(df_copy['isDJFM'] == 'Si', np.nan, inplace = True)
    # df_copy.dropna(inplace = True)
    # df_copy_group = df_copy.groupby('year').mean()
    # mean_djfm[NOMBRE_ESPANYA] = pd.DataFrame(df_copy_group[NOMBRE_COL_CF.format(NOMBRE_ESPANYA)].values, columns = [NOMBRE_COL_CF.format(NOMBRE_ESPANYA)], index = df_copy_group.index.astype(int))

    fig = plt.figure(figsize = (14.0 / 2.54, 9.0 / 2.54), clear = True)
    ax = fig.add_subplot(1, 1, 1)
    ax.boxplot(
        [dictDfRegiones[NOMBRE_ESPANYA].loc[str(year), NOMBRE_COL_CF.format(NOMBRE_ESPANYA)].values for year in range(YEAR_INI, YEAR_FIN + 1)],
        positions = range(YEAR_INI, YEAR_FIN + 1),
        whis = [5, 95],
        widths = 0.5,
        showfliers = False,
        showmeans = True,
        patch_artist = True,
        boxprops = {'facecolor': 'tab:blue'},
        medianprops = {'color': 'black'},
        meanprops = {'marker': 'o', 'markerfacecolor': 'black', 'markeredgecolor': 'None', 'markersize': 4},
    )
    ax.set_xticks(range(YEAR_INI, YEAR_FIN + 1, 5), range(YEAR_INI, YEAR_FIN + 1, 5))
    ax.set_xlabel('Year', fontsize = 10, fontname = 'DejaVu Serif')
    ax.set_ylabel('Hourly wind CF [-]', fontsize = 10, fontname = 'DejaVu Serif')
    ax.tick_params(axis = 'both', labelsize = 8, direction = 'in')
    ax.grid(True, linestyle = '--', alpha = 0.7)
    fig.savefig(f'{WorkDir.getDirectorioFiguras()}BoxplotNacionalAnual_1990-2020.jpg', dpi = 600, bbox_inches = 'tight', pad_inches = 0.05)
    plt.close()

    hourly_national_wind_cf = dictDfRegiones[NOMBRE_ESPANYA].copy()
    hourly_national_wind_cf['Season'] = [DICCIONARIO_ESTACIONES[str(date.month).zfill(2)] for date in hourly_national_wind_cf.index]
    hourly_national_wind_cf['Year_Season'] = [f'{SumaYearDiciembres(date)}_{DICCIONARIO_ESTACIONES[str(date.month).zfill(2)]}' for date in hourly_national_wind_cf.index]
    fig = plt.figure(figsize = (14.0 / 2.54, 9.0 / 2.54), clear = True)
    ax = fig.add_subplot(1, 1, 1)
    ax.plot(
        range(YEAR_INI, YEAR_FIN + 1),
        dictDfRegiones[NOMBRE_ESPANYA].resample('1Y').mean().loc[:, NOMBRE_COL_CF.format(NOMBRE_ESPANYA)].values,
        'o--',
        color = 'black',
        markersize = 4,
        label = 'Annual'
    )
    ax.axhline(dictDfRegiones[NOMBRE_ESPANYA].mean().item(), color = 'black', alpha = 0.6)
    for season, color in zip(np.unique(list(DICCIONARIO_ESTACIONES.values())), ('tab:brown', 'tab:green', 'tab:orange', 'tab:gray')):
        if season == 'Winter':
            values = hourly_national_wind_cf.where(hourly_national_wind_cf['Season'] == season).groupby('Year_Season').mean().loc[:, NOMBRE_COL_CF.format(NOMBRE_ESPANYA)].values[:-1].copy()
        else:
            values = hourly_national_wind_cf.where(hourly_national_wind_cf['Season'] == season).groupby('Year_Season').mean().loc[:, NOMBRE_COL_CF.format(NOMBRE_ESPANYA)].values.copy()
        ax.plot(
            range(YEAR_INI, YEAR_FIN + 1),
            values,
            'o--',
            color = color,
            markersize = 4,
            label = season
        )
        ax.axhline(hourly_national_wind_cf.groupby('Season').mean().loc[season].item(), color = color, alpha = 0.6)
    ax.set_xlabel('Year', fontsize = 10, fontname = 'DejaVu Serif')
    ax.set_ylabel('Annual wind CF [-]', fontsize = 10, fontname = 'DejaVu Serif')
    ax.tick_params(axis = 'both', labelsize = 8, direction = 'in')
    ax.grid(True, linestyle = '--', alpha = 0.7)
    ax.legend(loc = 'upper center', ncol = 5, fontsize = 8, columnspacing = .7)
    fig.savefig(f'{WorkDir.getDirectorioFiguras()}SeasonYearlyCF_1990-2020.jpg', dpi = 600, bbox_inches = 'tight', pad_inches = 0.05)
    plt.close()

    # mean_cf_value = dictDfRegiones[NOMBRE_ESPANYA].mean().item()
    fig = plt.figure(figsize = (14.0 / 2.54, 9.0 / 2.54), clear = True)
    ax = fig.add_subplot(1, 1, 1)
    ax.plot(
        range(YEAR_INI, YEAR_FIN + 1),
        dictDfRegiones[NOMBRE_ESPANYA].resample('1Y').mean().loc[:, NOMBRE_COL_CF.format(NOMBRE_ESPANYA)].values / dictDfRegiones[NOMBRE_ESPANYA].mean().item(),
        'o-',
        color = 'black',
        markersize = 4,
        label = 'Annual'
    )
    for season, color in zip(np.unique(list(DICCIONARIO_ESTACIONES.values())), ('tab:brown', 'tab:green', 'tab:orange', 'tab:gray')):
        if season == 'Winter':
            values = hourly_national_wind_cf.where(hourly_national_wind_cf['Season'] == season).groupby('Year_Season').mean().loc[:, NOMBRE_COL_CF.format(NOMBRE_ESPANYA)].values[:-1].copy() / hourly_national_wind_cf.groupby('Season').mean().loc[season].item()
        else:
            values = hourly_national_wind_cf.where(hourly_national_wind_cf['Season'] == season).groupby('Year_Season').mean().loc[:, NOMBRE_COL_CF.format(NOMBRE_ESPANYA)].values.copy() / hourly_national_wind_cf.groupby('Season').mean().loc[season].item()
        ax.plot(
            range(YEAR_INI, YEAR_FIN + 1),
            values,
            'o-',
            color = color,
            markersize = 4,
            label = season
        )
    ax.set_xlabel('Year', fontsize = 10, fontname = 'DejaVu Serif')
    ax.set_ylabel('Annual anomaly wind CF [-]', fontsize = 10, fontname = 'DejaVu Serif')
    ax.tick_params(axis = 'both', labelsize = 8, direction = 'in')
    ax.grid(True, linestyle = '--', alpha = 0.7)
    ax.legend(loc = 'upper center', ncol = 5, fontsize = 8, columnspacing = .7)
    fig.savefig(f'{WorkDir.getDirectorioFiguras()}SeasonYearlyAnomalyCF_1990-2020.jpg', dpi = 600, bbox_inches = 'tight', pad_inches = 0.05)
    plt.close()

    hourly_national_wind_ramps = pd.DataFrame(
        hourly_national_wind_cf['Spain_CF'].values[1:] - hourly_national_wind_cf['Spain_CF'].values[:-1],
        columns = ['Spain_CF'],
        index = hourly_national_wind_cf.index[1:]
    )
    hourly_national_wind_ramps = pd.merge(hourly_national_wind_ramps, pd.DataFrame(hourly_national_wind_cf['Season']), how = 'inner', left_index = True, right_index = True)
    fig = plt.figure(figsize = (14.0 / 2.54, 21.0 / 2.54), clear = True)
    for iterator, season, color in zip(range(5), ('Spring', 'Summer', 'Autumn', 'Winter', 'Annual'), ('tab:green', 'tab:orange', 'tab:brown', 'tab:gray', 'black')):
        ax = fig.add_subplot(3, 2, iterator + 1)
        if season == 'Annual':
            values = hourly_national_wind_ramps.loc[:, NOMBRE_COL_CF.format(NOMBRE_ESPANYA)].values.copy()
        else:
            values = hourly_national_wind_ramps.where(hourly_national_wind_ramps['Season'] == season).dropna().loc[:, NOMBRE_COL_CF.format(NOMBRE_ESPANYA)].values.copy()
        horas, intervalos = np.histogram(values, bins = np.arange(-0.10, 0.11, 0.01), range = (-0.10, 0.10))
        ax.stairs(
            horas * (8760.0 / len(values)),
            intervalos,
            fill = True,
            color = color
        )
        ax.set_ylim([0, 2600])
        ax.set_title(season, fontsize = 10, fontname = 'DejaVu Serif')
        if ((iterator == 3) | (iterator == 4)):
            ax.set_xlabel('Hourly ramps CF [-]', fontsize = 10, fontname = 'DejaVu Serif')
        if ((iterator == 0) | (iterator == 2) | (iterator == 4)):
            ax.set_ylabel('N hours / year [-]', fontsize = 10, fontname = 'DejaVu Serif')
        ax.tick_params(axis = 'both', labelsize = 8, direction = 'in')
        ax.grid(True, linestyle = '--', alpha = 0.7)
    fig.savefig(f'{WorkDir.getDirectorioFiguras()}SeasonRampsHourlyCF.jpg', dpi = 600, bbox_inches = 'tight', pad_inches = 0.05)
    plt.close()

    cmap = plt.get_cmap('bwr')
    rango_mean = (0.13, 0.37)
    rango_sd = (0.01, 0.27)
    fig = plt.figure(0, figsize = (19.0 / 2.54, 9.0 / 2.54), clear = True)
    ax_mean = fig.add_subplot(1, 2, 1, projection = ccrs.PlateCarree())
    ax_mean.set_extent([-11.0, 4.5, 35.0, 45.0], ccrs.PlateCarree())
    ax_mean.coastlines(resolution = '50m')
    ax_mean.background_img(name = 'dem_gray', resolution = 'high')
    #ax.add_feature(cf.BORDERS)
    for nombreProvincia in tuple(dictDfRegiones.keys())[:-1]:
        try:
            nombreProvinciaLocalizable = DICT_PROVINCIAS_IGN[nombreProvincia]
        except KeyError:
            nombreProvinciaLocalizable = nombreProvincia
        mean_value = dictDfRegiones[nombreProvincia].mean().item()
        print(f'{nombreProvincia} - mean: {np.round(mean_value, 2)}')
        cbar_mean = ax_mean.add_geometries(
            geoms = shape_NUTS3[shape_NUTS3['NAMEUNIT'] == nombreProvinciaLocalizable]['geometry'].values,
            crs = ccrs.PlateCarree(),
            facecolor = cmap((mean_value - rango_mean[0]) / (rango_mean[-1] - rango_mean[0])),
            edgecolor = 'black',
            linewidth = 0.5
        )
        # ax_mean.set_title(f'Mean {rango_mean[0]}-{np.round(dictDfRegiones[NOMBRE_ESPANYA].mean().item(), 2)}-{rango_mean[-1]}', fontsize = 8)
    ax_sd = fig.add_subplot(1, 2, 2, projection = ccrs.PlateCarree())
    ax_sd.set_extent([-11.0, 4.5, 35.0, 45.0], ccrs.PlateCarree())
    ax_sd.coastlines(resolution = '50m')
    ax_sd.background_img(name = 'dem_gray', resolution = 'high')
    #ax.add_feature(cf.BORDERS)
    for nombreProvincia in tuple(dictDfRegiones.keys())[:-1]:
        try:
            nombreProvinciaLocalizable = DICT_PROVINCIAS_IGN[nombreProvincia]
        except KeyError:
            nombreProvinciaLocalizable = nombreProvincia
        sd_value = dictDfRegiones[nombreProvincia].std().item()
        print(f'{nombreProvincia} - sd: {np.round(sd_value, 2)}')
        cbar_sd = ax_sd.add_geometries(
            geoms = shape_NUTS3[shape_NUTS3['NAMEUNIT'] == nombreProvinciaLocalizable]['geometry'].values,
            crs = ccrs.PlateCarree(),
            facecolor = cmap((sd_value - rango_sd[0]) / (rango_sd[-1] - rango_sd[0])),
            edgecolor = 'black',
            linewidth = 0.5
        )
        # ax_sd.set_title(f'Std {rango_sd[0]}-{np.round(dictDfRegiones[NOMBRE_ESPANYA].std().item(), 2)}-{rango_sd[-1]}', fontsize = 8)
    ax_cbar_mean = fig.add_axes([ax_mean.get_position().x0, ax_mean.get_position().y0 - 0.05, 0.35, 0.02])
    cbar = ColorbarBase(ax_cbar_mean, cmap = cmap, norm = Normalize(vmin = rango_mean[0], vmax = rango_mean[-1]), orientation = 'horizontal')
    cbar.set_label(f'mean CF [-]', fontsize = 10, family = 'DejaVu Serif')
    cbar.ax.tick_params(labelsize = 8)
    ax_cbar_sd = fig.add_axes([ax_sd.get_position().x0, ax_sd.get_position().y0 - 0.05, 0.35, 0.02])
    cbar = ColorbarBase(ax_cbar_sd, cmap = cmap, norm = Normalize(vmin = rango_sd[0], vmax = rango_sd[-1]), orientation = 'horizontal')
    cbar.set_label(f'std CF [-]', fontsize = 10, family = 'DejaVu Serif')
    cbar.ax.tick_params(labelsize = 8)
    # fig.tight_layout()
    fig.savefig(f'{WorkDir.getDirectorioFiguras()}mean_sd_windCF.jpg', dpi = 600, bbox_inches = 'tight', pad_inches = 0.05)
    plt.close(0)

    cf_NUTS3['Season'] = [DICCIONARIO_ESTACIONES[str(date.month).zfill(2)] for date in cf_NUTS3.index]
    # cmap = plt.get_cmap('bwr')
    # rango = (0.09, 0.41)
    # fig = plt.figure(0, figsize = (19.0 / 2.54, 14.0 / 2.54), clear = True)
    for iterator, season in zip(range(5), ('Spring', 'Summer', 'Autumn', 'Winter', 'Annual')):
        # ax = fig.add_subplot(3, 2, iterator + 1, projection = ccrs.PlateCarree())
        # ax.set_extent([-11.0, 4.5, 35.0, 45.0], ccrs.PlateCarree())
        # ax.coastlines(resolution = '50m')
        # ax.background_img(name = 'dem_gray', resolution = 'high')
        #ax.add_feature(cf.BORDERS)
        for nombreProvincia in tuple(dictDfRegiones.keys())[:-1]:
            try:
                nombreProvinciaLocalizable = DICT_PROVINCIAS_IGN[nombreProvincia]
            except KeyError:
                nombreProvinciaLocalizable = nombreProvincia
            if iterator != 4:
                mean_value = cf_NUTS3.where(cf_NUTS3['Season'] == season).dropna().loc[:, nombreProvincia].mean()
            else:
                mean_value = dictDfRegiones[nombreProvincia].mean().item()
            print(f'{season} \t {nombreProvincia} \t {np.round(mean_value, 2)}')
    #         ax.add_geometries(
    #             geoms = shape_NUTS3[shape_NUTS3['NAMEUNIT'] == nombreProvinciaLocalizable]['geometry'].values,
    #             crs = ccrs.PlateCarree(),
    #             facecolor = cmap((mean_value - rango[0]) / (rango[-1] - rango[0])),
    #             edgecolor = 'black',
    #             linewidth = 0.5
    #         )
    #     ax.set_title(season)
    # ax_cbar = fig.add_axes([ax.get_position().x0, ax.get_position().y0 - 0.05, 0.9, 0.02])
    # cbar = ColorbarBase(ax_cbar, cmap = cmap, norm = Normalize(vmin = rango[0], vmax = rango[-1]), orientation = 'horizontal')
    # cbar.set_label(f'mean CF [-]', fontsize = 10, family = 'DejaVu Serif')
    # cbar.ax.tick_params(labelsize = 8)
    # fig.savefig(f'{WorkDir.getDirectorioFiguras()}mean_NUTS3_season_windCF.jpg', dpi = 600, bbox_inches = 'tight', pad_inches = 0.05)
    # plt.close(0)

    cmap = plt.get_cmap('bwr')
    rango = (0.0, 0.28)
    fig = plt.figure(0, figsize = (19.0 / 2.54, 14.0 / 2.54), clear = True)
    for iterator, season in zip(range(5), ('Spring', 'Summer', 'Autumn', 'Winter', 'Annual')):
        ax = fig.add_subplot(3, 2, iterator + 1, projection = ccrs.PlateCarree())
        ax.set_extent([-11.0, 4.5, 35.0, 45.0], ccrs.PlateCarree())
        ax.coastlines(resolution = '50m')
        ax.background_img(name = 'dem_gray', resolution = 'high')
        #ax.add_feature(cf.BORDERS)
        for nombreProvincia in tuple(dictDfRegiones.keys())[:-1]:
            try:
                nombreProvinciaLocalizable = DICT_PROVINCIAS_IGN[nombreProvincia]
            except KeyError:
                nombreProvinciaLocalizable = nombreProvincia
            if iterator != 4:
                sd_value = cf_NUTS3.where(cf_NUTS3['Season'] == season).dropna().loc[:, nombreProvincia].std()
            else:
                sd_value = dictDfRegiones[nombreProvincia].std().item()
            # print(f'{season} \t {nombreProvincia} \t {np.round(sd_value, 2)}')
            ax.add_geometries(
                geoms = shape_NUTS3[shape_NUTS3['NAMEUNIT'] == nombreProvinciaLocalizable]['geometry'].values,
                crs = ccrs.PlateCarree(),
                facecolor = cmap((sd_value - rango[0]) / (rango[-1] - rango[0])),
                edgecolor = 'black',
                linewidth = 0.5
            )
        ax.set_title(season)
    ax_cbar = fig.add_axes([ax.get_position().x0, ax.get_position().y0 - 0.05, 0.9, 0.02])
    cbar = ColorbarBase(ax_cbar, cmap = cmap, norm = Normalize(vmin = rango[0], vmax = rango[-1]), orientation = 'horizontal')
    cbar.set_label(f'std CF [-]', fontsize = 10, family = 'DejaVu Serif')
    cbar.ax.tick_params(labelsize = 8)
    fig.savefig(f'{WorkDir.getDirectorioFiguras()}sd_NUTS3_season_windCF.jpg', dpi = 600, bbox_inches = 'tight', pad_inches = 0.05)
    plt.close(0)

    cf_NUTS3['Season'] = [DICCIONARIO_ESTACIONES[str(date.month).zfill(2)] for date in cf_NUTS3.index]
    fig = plt.figure(0, figsize = (19.0 / 2.54, 14.0 / 2.54), clear = True)
    corr_distance_NUTS3 = ArrayCorrelacionEspacial(cf_NUTS3, latlon_centroids_NUTS3)
    corr_NUTS3_all, dist_NUTS3 = DivideArrayCorrelacionEspacialEn2FlattenArrays(corr_distance_NUTS3)
    dictPopt_all = curve_fit(FuncionExponecial, dist_NUTS3, corr_NUTS3_all, p0 = (0.1, 0.1))[0]
    scale_length_all = np.power(dictPopt_all[0], -1.0 / dictPopt_all[1])
    for iterator, season, color in zip(range(4), ('Spring', 'Summer', 'Autumn', 'Winter'), ('tab:green', 'tab:orange', 'tab:brown', 'tab:gray')):
        ax = fig.add_subplot(2, 2, iterator + 1)
        cf_NUTS3_season = cf_NUTS3.where(cf_NUTS3['Season'] == season).dropna().loc[:, cf_NUTS3.columns[:-1]].copy()
        corr_distance_NUTS3_season = ArrayCorrelacionEspacial(cf_NUTS3_season, latlon_centroids_NUTS3)
        corr_NUTS3_season, dist_NUTS3 = DivideArrayCorrelacionEspacialEn2FlattenArrays(corr_distance_NUTS3_season)
        dictPopt_season = curve_fit(FuncionExponecial, dist_NUTS3, corr_NUTS3_season, p0 = (0.1, 0.1))[0]
        scale_length_season = np.power(dictPopt_season[0], -1.0 / dictPopt_season[1])
        ax.plot(dist_NUTS3, corr_NUTS3_all, marker = 'o', markersize = 0.5, linestyle = '', color = 'black')
        ax.plot(np.arange(0, 900, 1), FuncionExponecial(np.arange(0, 900, 1), dictPopt_all[0], dictPopt_all[1]), color = 'black', label = f'{scale_length_all.round(1)} km')
        ax.plot(dist_NUTS3, corr_NUTS3_season, marker = 'o', markersize = 0.5, linestyle = '', color = color)
        ax.plot(np.arange(0, 900, 1), FuncionExponecial(np.arange(0, 900, 1), dictPopt_season[0], dictPopt_season[1]), color = color, label = f'{scale_length_season.round(1)} km')
        ax.set_title(season, fontsize = 10, fontname = 'DejaVu Serif')
        if ((iterator == 2) | (iterator == 3)):
            ax.set_xlabel('Distance [km]', fontsize = 10, fontname = 'DejaVu Serif')
        else:
            ax.set_xticklabels([])
        if ((iterator == 0) | (iterator == 2)):
            ax.set_ylabel('Correlation [-]', fontsize = 10, fontname = 'DejaVu Serif')
        else:
            ax.set_yticklabels([])
        ax.tick_params(axis = 'both', labelsize = 8, direction = 'in')
        ax.grid(True, linestyle = '--', alpha = 0.7)
        ax.legend(loc = 'upper right', fontsize = 8)
    fig.savefig(f'{WorkDir.getDirectorioFiguras()}CorrelacionEspacial.jpg', dpi = 600, bbox_inches = 'tight', pad_inches = 0.05)




    '''
    NAO
    '''

    corrs = {}
    for region in mean_djfm.keys():
        df_merge = Merge2DataFrames(tablaNAO, mean_djfm[region], 'inner')
        corrs[region] = Correlacion(df_merge[NOMBRE_COL_CF.format(region)].values, df_merge["idxNAO"].values)
        print(f'Correlacion NAO-{region}: \t{np.round(corrs[region], 2)}')

    vars_rel_pos = {}
    vars_rel_neg = {}
    for region in dictDfRegiones.keys():
        mean_cf_nao_pos = dictDfRegiones[region].loc[np.isin(dictDfRegiones[region].index.year, [1992, 1995, 2015])].mean().item()
        mean_cf_nao_neg = dictDfRegiones[region].loc[np.isin(dictDfRegiones[region].index.year, [1996, 2010, 2013])].mean().item()
        mean_cf = dictDfRegiones[region].mean().item()
        vars_rel_pos[region] = (mean_cf_nao_pos - mean_cf) / mean_cf * 100.0
        vars_rel_neg[region] = (mean_cf_nao_neg - mean_cf) / mean_cf * 100.0
        print(f'Var. rel. CF promedio {region} con NAO+ extrema: {np.round(vars_rel_pos[region], 2)} %')
        print(f'Var. rel. CF promedio {region} con NAO- extrema: {np.round(vars_rel_neg[region], 2)} %')

    fig, ax = plt.subplots(figsize = (14.0 / 2.54, 7.0 / 2.54), clear = True)
    ax.hist(dictDfRegiones[region], bins = np.arange(0.0, 1.1, 0.1), density = True, histtype = 'step', label = 'normal')
    ax.hist(dictDfRegiones[region].loc[np.isin(dictDfRegiones[region].index.year, [1992, 1995, 2015])], bins = np.arange(0.0, 1.1, 0.1), density = True, histtype = 'step', label = 'NAO+')
    ax.hist(dictDfRegiones[region].loc[np.isin(dictDfRegiones[region].index.year, [1996, 2010, 2013])], bins = np.arange(0.0, 1.1, 0.1), density = True, histtype = 'step', label = 'NAO-')
    ax.legend()
    fig.savefig('/home/agarrote/Escritorio/kk_hist_NAO+-.png', dpi = 600)

    cmap = plt.get_cmap('Greens')
    cmap = cmap.reversed()
    rango = (-1.0, 0.0)
    fig = plt.figure(0, figsize = (14.0 / 2.54, 9.0 / 2.54), clear = True)
    ax = fig.add_subplot(1, 1, 1, projection = ccrs.PlateCarree())
    ax.set_extent([-11.0, 4.5, 35.0, 45.0], ccrs.PlateCarree())
    ax.coastlines(resolution = '50m')
    #ax.add_feature(cf.BORDERS)
    for nombreProvincia in tuple(corrs.keys())[:-1]:
        try:
            nombreProvinciaLocalizable = DICT_PROVINCIAS_IGN[nombreProvincia]
        except KeyError:
            nombreProvinciaLocalizable = nombreProvincia
        ax.add_geometries(
            geoms = shape_NUTS3[shape_NUTS3['NAMEUNIT'] == nombreProvinciaLocalizable]['geometry'].values,
            crs = ccrs.PlateCarree(),
            facecolor = cmap((corrs[nombreProvincia] - rango[0]) / (rango[-1] - rango[0])),
            edgecolor = 'black',
            linewidth = 0.5
        )
    ax_cbar = fig.add_axes([ax.get_position().x0, ax.get_position().y0 - 0.05, 0.8, 0.02])
    cbar = ColorbarBase(ax_cbar, cmap = cmap, norm = Normalize(vmin = rango[0], vmax = rango[-1]), orientation = 'horizontal')
    cbar.set_label(f'Correlation [-]', fontsize = 10, family = 'DejaVu Serif')
    cbar.ax.tick_params(labelsize = 8)
    fig.savefig(f'{WorkDir.getDirectorioFiguras()}correlaciones_NAO.jpg', dpi = 600, bbox_inches = 'tight', pad_inches = 0.05)
    plt.close(0)

    cmap = plt.get_cmap('bwr')
    rango = (-10.0, 10.0)
    fig = plt.figure(0, figsize = (14.0 / 2.54, 9.0 / 2.54), clear = True)
    for iterator, diccionario, title in zip((1, 2), (vars_rel_neg, vars_rel_pos), ('NAO-', 'NAO+')):
        ax = fig.add_subplot(1, 2, iterator, projection = ccrs.PlateCarree())
        ax.set_extent([-11.0, 4.5, 35.0, 45.0], ccrs.PlateCarree())
        ax.coastlines(resolution = '50m')
        #ax.add_feature(cf.BORDERS)
        for nombreProvincia in tuple(corrs.keys())[:-1]:
            try:
                nombreProvinciaLocalizable = DICT_PROVINCIAS_IGN[nombreProvincia]
            except KeyError:
                nombreProvinciaLocalizable = nombreProvincia
            ax.add_geometries(
                geoms = shape_NUTS3[shape_NUTS3['NAMEUNIT'] == nombreProvinciaLocalizable]['geometry'].values,
                crs = ccrs.PlateCarree(),
                facecolor = cmap((diccionario[nombreProvincia] - rango[0]) / (rango[-1] - rango[0])),
                edgecolor = 'black',
                linewidth = 0.5
            )
            ax.set_title(title, fontsize = 10, fontname = 'DejaVu Serif')
    ax_cbar = fig.add_axes([0.1, ax.get_position().y0 - 0.05, 0.8, 0.02])
    cbar = ColorbarBase(ax_cbar, cmap = cmap, norm = Normalize(vmin = rango[0], vmax = rango[-1]), orientation = 'horizontal')
    cbar.set_label(r'$\left(\overline{\rm{CF}_{\rm{NAO}}} - \overline{\rm{CF}}\right) / \overline{\rm{CF}}$ $[\%]$', fontsize = 10, family = 'DejaVu Serif')
    cbar.ax.tick_params(labelsize = 8)
    fig.savefig(f'{WorkDir.getDirectorioFiguras()}variaciones_relativas_NAO+-.jpg', dpi = 600, bbox_inches = 'tight', pad_inches = 0.05)
    plt.close(0)

    '''
    fig = plt.figure(0, (19.0 / 2.54, 12.0 / 2.54), dpi = 600, clear = True)
    ax = fig.add_axes([0.12, 0.14, 0.85, 0.78])
    ax.plot_date(dictDfRegiones[NOMBRE_ESPANYA].resample('Y').mean().index.to_pydatetime() - timedelta(weeks = int(52 / 2.0)), dictDfRegiones[NOMBRE_ESPANYA].resample('Y').mean()[NOMBRE_COL_CF.format(NOMBRE_ESPANYA)].values, '-', label = 'Model')
    # ax.plot_date(kk.resample('Y').mean().loc['2007':].index.to_pydatetime(), kk.resample('Y').mean().loc['2007':, NOMBRE_COL_CF.format(NOMBRE_ESPANYA)].values, '-', label = 'Model_NoCongelado')
    # ax.plot_date(dfREE.resample('Y').mean().index.to_pydatetime(), dfREE.resample('Y').mean()[NOMBRE_COL_CF.format(NOMBRE_ESPANYA)].values, '-', color = 'black', label = 'REE')
    ax.set_xlabel('Year', fontsize = 10)
    ax.set_ylabel('Wind CF [-]', fontsize = 10)
    ax.legend()
    fig.savefig(f'{WorkDir.getDirectorioFiguras()}SerieTemporalNacionalAnual_{dictDfRegiones[NOMBRE_ESPANYA].index.year[0]}-{dictDfRegiones[NOMBRE_ESPANYA].index.year[-1]}.jpg', dpi = 600)
    plt.close(0)

    listaArraysParaBoxplot = []
    for year in range(YEAR_INI, YEAR_FIN + 1):
        listaArraysParaBoxplot.append(dictDfRegiones[NOMBRE_ESPANYA].loc[str(year), NOMBRE_COL_CF.format(NOMBRE_ESPANYA)].values.copy())

    fig = plt.figure(0, (19.0 / 2.54, 12.0 / 2.54), dpi = 600, clear = True)
    ax = fig.add_axes([0.12, 0.14, 0.85, 0.78])
    ax.boxplot(listaArraysParaBoxplot, whis = [5, 95], patch_artist = True, positions = dictDfRegiones[NOMBRE_ESPANYA].index.year.unique().values, manage_ticks = False, flierprops = dict(markersize = 0.5, markerfacecolor = 'tab:red', markeredgecolor = 'tab:red'), zorder = 5)
    # ax.plot(dfMergeCompleto.dropna().loc['2019', 'W100(m/s)'].values, dfMergeCompleto.dropna().loc['2019', 'CF'].values, marker = 'o', markersize = 0.5, color = 'tab:green', ls = '')
    # ax.plot(dfMergeCompleto.dropna().loc['2020', 'W100(m/s)'].values, dfMergeCompleto.dropna().loc['2020', 'CF'].values, marker = 'o', markersize = 0.5, color = 'tab:pink', ls = '')
    ax.set_xlabel('Year', fontsize = 10)
    ax.set_ylabel('National Wind CF [-]', fontsize = 10)
    fig.savefig(f'{WorkDir.getDirectorioFiguras()}BoxplotNacionalAnual_{dictDfRegiones[NOMBRE_ESPANYA].index.year[0]}-{dictDfRegiones[NOMBRE_ESPANYA].index.year[-1]}.jpg', dpi = 600)
    plt.close(0)

    kk = dictDfRegiones[NOMBRE_ESPANYA].copy()
    kk['Year'] = [SumaYearDiciembres(date) for date in kk.index.to_pydatetime()]
    kk['Year_Estacion'] = [f'{year}_{DICCIONARIO_ESTACIONES[str(date.month).zfill(2)]}' for year, date in zip(kk['Year'].values, kk.index.to_pydatetime())]
    # kkAgrupado = kk.groupby('Year_Estacion').mean()
    fig = plt.figure(0, (19.0 / 2.54, 24.0 / 2.54), dpi = 600, clear = True)
    ax = {}
    iterator = 0
    for estacion in ['Primavera', 'Verano', 'Otonyo', 'Invierno']:
        ax[estacion] = fig.add_axes([0.12, 0.05 + iterator * 0.24, 0.85, 0.2])
        listaArraysParaBoxplotEstacionales = []
        for year in range(YEAR_INI, YEAR_FIN + 1):
            listaArraysParaBoxplotEstacionales.append(kk.loc[kk['Year_Estacion'] == f'{year}_{estacion}', NOMBRE_COL_CF.format(NOMBRE_ESPANYA)].values.copy())
        ax[estacion].boxplot(listaArraysParaBoxplotEstacionales, whis = [5, 95], patch_artist = True, positions = dictDfRegiones[NOMBRE_ESPANYA].index.year.unique().values, manage_ticks = False, flierprops = dict(markersize = 0.5, markerfacecolor = 'tab:red', markeredgecolor = 'tab:red'), zorder = 5)
        if iterator == 0:
            ax[estacion].set_xlabel('Year', fontsize = 8)
        ax[estacion].set_ylabel('National Wind CF [-]', fontsize = 8)
        ax[estacion].set_title(estacion, fontsize = 8)
        iterator += 1
    fig.savefig(f'{WorkDir.getDirectorioFiguras()}BoxplotNacionalAnualEstacional_{dictDfRegiones[NOMBRE_ESPANYA].index.year[0]}-{dictDfRegiones[NOMBRE_ESPANYA].index.year[-1]}.jpg', dpi = 600)
    plt.close(0)

    for nombreRegion in dictDfRegiones.keys():
        dfMerge = Merge2DataFrames(dictDfRegiones[nombreRegion].resample('MS').mean(), dfNAO, 'inner')
        dfMerge['Year'] = [SumaYearDiciembres(date) for date in dfMerge.index.to_pydatetime()]
        dfMerge['Year_Estacion'] = [f'{year}_{DICCIONARIO_ESTACIONES[str(date.month).zfill(2)]}' for year, date in zip(dfMerge['Year'].values, dfMerge.index.to_pydatetime())]
        dfInviernos = dfMerge[[f'{nombreRegion}_CF', 'idxNAO', 'Year_Estacion']].groupby('Year_Estacion').mean()
        dfAnual = dfMerge[[f'{nombreRegion}_CF', 'idxNAO', 'Year']].groupby('Year').mean()
        print(f'Correlacion {nombreRegion} --> {np.round(Correlacion(dfInviernos["idxNAO"].values, dfInviernos[f"{nombreRegion}_CF"].values), 2)} (Inviernos) \t {np.round(Correlacion(dfAnual["idxNAO"].values, dfAnual[f"{nombreRegion}_CF"].values), 2)} (Anual)')
    # for etiqueta, df in zip(('diario', 'mensual', 'mensual invierno', 'anual'), (dfMerge, dfMerge.resample('1M').mean(), dfMerge.loc[dfMerge.index.month.isin([1, 2, 12])].resample('1M').mean().dropna(), dfMerge.resample('1Y').mean())):
    #     print(f'Correlacion generacion-NAO {etiqueta}: {np.round(Correlacion(df[f"{NOMBRE_ESPANYA}_CF"].values, df["idxNAO"].values), 3)}')

    fig = plt.figure(0, figsize = (14.0 / 2.54, 7.0 / 2.54), dpi = 600, clear = True)
    ax = Artist.AxisElsevier(
        figura = fig,
        x0 = 0.08,
        y0 = 0.2,
        anchoRel = 0.9,
        altoRel = 0.78,
        xlabel = 'Date',
        ylabel = 'Capacity Factor [-]',
        xticksPos = pd.date_range(datetime(1993, 12, 23), datetime(1994, 1, 1), freq = 'D').to_pydatetime(),
        xticksLabels = [datetime.strftime(date, '%Y-%m-%d') for date in pd.date_range(datetime(1993, 12, 23), datetime(1994, 1, 1), freq = 'D').to_pydatetime()],
        xticksRotacion = 20.0,
        yticksPos = np.arange(0.0, 1.1, 0.1)
    )
    printaLabel = True
    for region in dictDfRegiones.keys():
        if region == 'Spain':
            anchoLinea, color, label = (1.5, 'black', 'NUTS 0')
            printaLabel = True
        else:
            anchoLinea, color, label = (0.5, 'tab:gray', 'NUTS 3')
        if printaLabel == True:
            ax.plot_date(dictDfRegiones[region].loc['1993-12-23':'1993-12-31 23:00:00'].index.to_pydatetime(), dictDfRegiones[region].loc['1993-12-23':'1993-12-31 23:00:00', f'{region}_CF'].values, lw = anchoLinea, color = color, fmt = 'None', label = label)
            printaLabel = False
        else:
            ax.plot_date(dictDfRegiones[region].loc['1993-12-23':'1993-12-31 23:00:00'].index.to_pydatetime(), dictDfRegiones[region].loc['1993-12-23':'1993-12-31 23:00:00', f'{region}_CF'].values, lw = anchoLinea, color = color, fmt = 'None')
    Artist.SetLeyendaEnAxis(ax, 'upper left', )
    fig.savefig(f'{WorkDir.getDirectorioFiguras()}CaseStudy.jpg', dpi = 600)
    plt.close(0)
    '''
    return 0

if __name__ == '__main__':
    main()
