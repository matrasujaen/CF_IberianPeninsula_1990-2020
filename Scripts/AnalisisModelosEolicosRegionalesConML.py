#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
#  AnalisisModelosEolicosRegionalesConML.py
#
#  2023 Antonio Jiménez-Garrote <agarrote@ujaen.es>
#
#
'''
Se me han ocurrido una serie de graficas y tablas que poner en la seccion de
resultados como analisis de los modelos regionales de ML devueltos por Madrid.

Nota: Es necesario tener montada la NAS-Eolica
    sshfs -o allow_other,default_permissions,uid=1000,gid=1000 agarrote@150.214.97.175:/datasets -p 27 /home/agarrote/NAS_EOLICA/
'''

import sys
from WorkingDirectory_py3 import WorkingDirectory
WorkDir = WorkingDirectory('archivoConfiguracion.ini')
sys.path.append(WorkDir.getDirectorioLibrerias())

#-------------------------------------------------------------------LIBRARIES---
import numpy as np
import pandas as pd
import geopandas as gpd
from glob import glob
from datetime import datetime
from netCDF4 import Dataset
from matplotlib import pyplot as plt
import ArtistElsevier_py3 as Artist
from WindUtils_py3 import Correlacion, Bias, RMSE
from WindCapacityModels_py3 import IdxMasCercanosLatLon2D
from SOWISP_py3 import SOWISP_HighResolution
#-------------------------------------------------------------------------------

#-------------------------------------------------------------------CONSTANTS---
INICIO_STRING_DIRECTORIOS_NIVELES_AGG = 'Wind_'
INICIO_STRING_DATOS_MADRID = 'Results_Wind_'
MODELO_METEO = 'ERA5'
DICT_REGIONES_IGN = {
    'ACoruna': 'A Coruña', 'Almeria': 'Almería', 'Andalucia': 'Andalucía',
    'Araba': 'Araba/Álava', 'Aragon': 'Aragón', 'Asturias': 'Principado de Asturias', 'Avila': 'Ávila',
    'Cadiz': 'Cádiz', 'Castellon': 'Castelló/Castellón',
    'CastillaLaMancha': 'Castilla-La Mancha', 'CastillaYLeon': 'Castilla y León',
    'Cataluna': 'Cataluña/Catalunya', 'CiudadReal': 'Ciudad Real',
    'CValenciana': 'Comunitat Valenciana', 'Jaen': 'Jaén', 'LaRioja': 'La Rioja',
    'Leon': 'León', 'Malaga': 'Málaga', 'Murcia': 'Región de Murcia', 'Navarra': 'Comunidad Foral de Navarra', 'PaisVasco': 'País Vasco/Euskadi',
    'Valencia': 'València/Valencia'
}
EXTENSION_MAINLAND_SPAIN_KM2 = 494011.0
DATETIME_EVALUACION = datetime(2020, 12, 31)
NOMBRE_COLUMNA_PROVINCIA_SOWISP = 'Provincia'
NOMBRE_COLUMNA_CA_SOWISP = 'CA'
NOMBRE_COLUMNA_PAIS_SOWISP = 'Pais'
NOMBRE_COLUMNA_N_INSTALACIONES = 'N_instalaciones'
NOMBRE_ALTURA_RETRODB = 'HGT'
NOMBRE_ERA5_DATEFORMAT = 'ERA5_75N60W25N40E_SnglPrLvl_hourly_%Y-%m-%d.nc'
NOMBRE_LAT_ERA5 = 'latitude'
NOMBRE_LON_ERA5 = 'longitude'
LISTA_REGIONES_FIGURA_BARRAS = [
    'Spain', 'Galicia', 'ACoruna', 'Lugo', 'Ourense', 'Pontevedra', 'Asturias',
    'Cantabria', 'PaisVasco', 'Araba', 'Bizkaia', 'Navarra', 'LaRioja',
    'Aragon', 'Huesca', 'Teruel', 'Zaragoza', 'CastillaYLeon', 'Avila', 'Burgos',
    'Leon', 'Palencia', 'Salamanca', 'Segovia', 'Soria', 'Valladolid', 'Zamora',
    'CastillaLaMancha', 'Albacete', 'CiudadReal', 'Cuenca', 'Guadalajara',
    'Toledo', 'Cataluna', 'Barcelona', 'Lleida', 'Tarragona', 'CValenciana',
    'Castellon', 'Valencia', 'Andalucia', 'Almeria', 'Cadiz', 'Granada', 'Huelva',
    'Jaen', 'Malaga', 'Sevilla', 'Murcia'
]
LISTA_MODELOS_FIGURA_BARRAS = [
    'nacional', 'autonomico', 'provincial', 'provincial', 'provincial',
    'provincial', 'autonomico', 'autonomico', 'autonomico', 'provincial',
    'provincial', 'autonomico', 'autonomico', 'autonomico', 'provincial',
    'provincial', 'provincial', 'autonomico', 'provincial', 'provincial',
    'provincial', 'provincial', 'provincial', 'provincial', 'provincial',
    'provincial', 'provincial', 'autonomico', 'provincial', 'provincial',
    'provincial', 'provincial', 'provincial', 'autonomico', 'provincial',
    'provincial', 'provincial', 'autonomico', 'provincial', 'provincial',
    'autonomico', 'provincial', 'provincial', 'provincial', 'provincial',
    'provincial', 'provincial', 'provincial', 'autonomico'
]
DICT_REGIONES_NUTS = {
    'Spain': 'ES0', 'Galicia': 'ES11', 'ACoruna': 'ES111', 'Lugo': 'ES112',
    'Ourense': 'ES113', 'Pontevedra': 'ES114', 'Asturias': 'ES12',
    'Cantabria': 'ES13', 'PaisVasco': 'ES21', 'Araba': 'ES211',
    'Bizkaia': 'ES213', 'Navarra': 'ES22', 'LaRioja': 'ES23', 'Aragon': 'ES24',
    'Huesca': 'ES241', 'Teruel': 'ES242', 'Zaragoza': 'ES243',
    'CastillaYLeon': 'ES41', 'Avila': 'ES411', 'Burgos': 'ES412', 'Leon': 'ES413',
    'Palencia': 'ES414', 'Salamanca': 'ES415', 'Segovia': 'ES416',
    'Soria': 'ES417', 'Valladolid': 'ES418', 'Zamora': 'ES419',
    'CastillaLaMancha': 'ES42', 'Albacete': 'ES421', 'CiudadReal': 'ES422',
    'Cuenca': 'ES423', 'Guadalajara': 'ES424', 'Toledo': 'ES425',
    'Cataluna': 'ES51', 'Barcelona': 'ES511', 'Lleida': 'ES513',
    'Tarragona': 'ES514', 'CValenciana': 'ES52', 'Castellon': 'ES522', 'Valencia': 'ES523', 'Andalucia': 'ES61',
    'Almeria': 'ES611', 'Cadiz': 'ES612', 'Granada': 'ES614', 'Huelva': 'ES615',
    'Jaen': 'ES616', 'Malaga': 'ES617', 'Sevilla': 'ES618', 'Murcia': 'ES62'
}
# LISTA_COLORES_FIGURA_BARRAS = [
#     'tab:gray', 'black', 'limegreen', 'limegreen', 'limegreen', 'limegreen', 'black',
#     'black', 'black', 'khaki', 'khaki', 'black', 'black',
#     'black', 'chocolate', 'chocolate', 'chocolate', 'black', 'forestgreen', 'forestgreen',
#     'forestgreen', 'forestgreen', 'forestgreen', 'forestgreen', 'forestgreen', 'forestgreen', 'forestgreen',
#     'black', 'tab:orange', 'tab:orange', 'tab:orange', 'tab:orange',
#     'tab:orange', 'black', 'tab:pink', 'tab:pink', 'tab:pink', 'black',
#     'tab:green', 'tab:green', 'black', 'gold', 'gold', 'gold', 'gold',
#     'gold', 'gold', 'gold', 'black'
# ]
LISTA_COLORES_FIGURA_BARRAS = [
    'tab:green', 'tab:orange', 'tab:blue', 'tab:blue', 'tab:blue', 'tab:blue', 'tab:orange',
    'tab:orange', 'tab:orange', 'tab:blue', 'tab:blue', 'tab:orange', 'tab:orange',
    'tab:orange', 'tab:blue', 'tab:blue', 'tab:blue', 'tab:orange', 'tab:blue', 'tab:blue',
    'tab:blue', 'tab:blue', 'tab:blue', 'tab:blue', 'tab:blue', 'tab:blue', 'tab:blue',
    'tab:orange', 'tab:blue', 'tab:blue', 'tab:blue', 'tab:blue',
    'tab:blue', 'tab:orange', 'tab:blue', 'tab:blue', 'tab:blue', 'tab:orange',
    'tab:blue', 'tab:blue', 'tab:orange', 'tab:blue', 'tab:blue', 'tab:blue', 'tab:blue',
    'tab:blue', 'tab:blue', 'tab:blue', 'tab:orange'
]
#-------------------------------------------------------------------------------

#-------------------------------------------------------------------FUNCTIONS---
def ListaArchivosOrdenados(cadenaStr):
    listaArchivos = glob(cadenaStr)
    listaArchivos.sort()
    return listaArchivos

def CheckValorEnLista(valor, listaValoresPosibles):
    if np.isin(valor, listaValoresPosibles).item() == False:
        raise ValueError(f'El argumento {valor} no se encuentra disponible. Los argumentos disponibles son: {listaValoresPosibles}')

def CalculaAreaRegionShapefile(geoDf, nombreRegion):
    areaKm2 = np.round(geoDf[geoDf['NAMEUNIT'] == nombreRegion].to_crs('epsg:3035').area.item() / 10**6, 1)
    return areaKm2

def ExtraeMascaraProvincias(archivoNc, nombreVar = 'provincias'):
    dictArrayProvincias = ExtraeDictFromNetCDF(archivoNc, [nombreVar])
    ncFile = Dataset(archivoNc, 'r')
    dictCodigos = ncFile.variables[nombreVar].datatype.enum_dict
    ncFile.close()
    return dictArrayProvincias[nombreVar].copy(), dictCodigos

def ExtraeDictFromNetCDF(archivoNc, listaVariables, dtUnico = False):
    dictDatos = {}
    ncFile = Dataset(archivoNc, 'r')
    for nombreVar in listaVariables:
        if dtUnico == True:
            dictDatos[nombreVar] = ncFile.variables[nombreVar][0, :].data.copy()
        else:
            dictDatos[nombreVar] = ncFile.variables[nombreVar][:].data.copy()
    ncFile.close()
    return dictDatos

def ExtraeLatLon2DfromERA5(archivoERA5, nombreLat = NOMBRE_LAT_ERA5, nombreLon = NOMBRE_LON_ERA5):
    dictLatLon2D = {}
    dictLatLon1D = ExtraeDictFromNetCDF(archivoERA5, [nombreLat, nombreLon])
    lon2D, lat2D = np.meshgrid(dictLatLon1D[nombreLon], dictLatLon1D[nombreLat])
    dictLatLon2D[nombreLat] = lat2D.copy()
    dictLatLon2D[nombreLon] = lon2D.copy()
    return dictLatLon2D

def MergeInner2DataFrames(df1, df2, etiqueta_suffixes = 'modelo'):
    return pd.merge(df1, df2, left_index = True, right_index = True, suffixes = ('', f'_{etiqueta_suffixes}'))

def CalculaEstadistico2DataFrames(funcionEstadistico, dfObservaciones, dfModelo):
    dfMerge = MergeInner2DataFrames(dfObservaciones, dfModelo)
    nombreColObs, nombreColMod = dfMerge.columns.values
    # if index == False:
    estadistico = funcionEstadistico(dfMerge[nombreColObs].values, dfMerge[nombreColMod].values)
    # else:
    #     estadistico = funcionEstadistico(dfMerge[nombreColObs].values, dfMerge[nombreColMod].values, dfMerge.index.to_pydatetime())
    return estadistico
#-------------------------------------------------------------------------------

#-------------------------------------------------------------------CLASSES-----
class MLdatabase(object):
    def __init__(self, directorioDatabase, modelo = MODELO_METEO, cabeceraDirectorios = INICIO_STRING_DIRECTORIOS_NIVELES_AGG, cabeceraArchivos = INICIO_STRING_DATOS_MADRID):
        self._directorioDatabase = directorioDatabase
        self._cabeceraDirectorios = cabeceraDirectorios
        self._cabeceraArchivos = cabeceraArchivos

        listaDirectoriosAgregaciones = ListaArchivosOrdenados(f'{directorioDatabase}{cabeceraDirectorios}*/')
        self._nivelesAgregacion = [nombreDir.split('/')[-2].split('_')[-1] for nombreDir in listaDirectoriosAgregaciones]
        self._modelo = modelo
        listaDirectoriosAnyo = ListaArchivosOrdenados(f'{directorioDatabase}{cabeceraDirectorios}{self._nivelesAgregacion[0]}/{self._modelo}/*/')
        self._anyosValidacion = [int(nombreDir.split('/')[-2]) for nombreDir in listaDirectoriosAnyo]
        self._nombreRegiones = {}
        for nivelAgg in self._nivelesAgregacion:
            listaArchivos = ListaArchivosOrdenados(f'{directorioDatabase}{cabeceraDirectorios}{nivelAgg}/{self._modelo}/{self._anyosValidacion[0]}/*.csv')
            region_name = [nombreArchivo.split('/')[-1].split(cabeceraArchivos)[-1].split('_')[0] for nombreArchivo in listaArchivos]
            self._nombreRegiones[nivelAgg] = list(np.unique(region_name))

        self._nombreColRegion = dict(zip(self._nivelesAgregacion, ('Autonomia', 'Nacional', 'Provincia')))
        self._nombreColDate = 'Fecha'
        self._nombreColGeneracionReal = 'target'
        self._nombreColGeneracionModelada = 'prediccion'

    def getSerieRegionalCF(self, nivelAgregacion, nombreRegion, modelada = True, year = None):
        if modelada == True:
            nombreCol = self._nombreColGeneracionModelada
        else:
            nombreCol = self._nombreColGeneracionReal
        for input, lista in zip((nivelAgregacion, nombreRegion), (self._nivelesAgregacion, self._nombreRegiones[nivelAgregacion])):
            CheckValorEnLista(input, lista)
        if year is not None:
            CheckValorEnLista(year, self._anyosValidacion)
            archivoCsv = f'{self._directorioDatabase}{self._cabeceraDirectorios}{nivelAgregacion}/{self._modelo}/{year}/{self._cabeceraArchivos}{nombreRegion}_{year}_{self._modelo}_L0.csv'
            df = self._ExtraeSerieTemporalDatosMadrid(archivoCsv, nivelAgregacion, nombreRegion, nombreCol)
        else:
            df = pd.DataFrame()
            for year in self._anyosValidacion:
                archivoCsv = f'{self._directorioDatabase}{self._cabeceraDirectorios}{nivelAgregacion}/{self._modelo}/{year}/{self._cabeceraArchivos}{nombreRegion}_{year}_{self._modelo}_L0.csv'
                dfAnyo = self._ExtraeSerieTemporalDatosMadrid(archivoCsv, nivelAgregacion, nombreRegion, nombreCol)
                df = pd.concat([df.copy(), dfAnyo.copy()])
        return df.copy()

    def _ExtraeSerieTemporalDatosMadrid(self, archivoCsv, nivelAgregacion, nombreRegion, nombreColumna):
        df = self._LeeDataFrameMadrid(archivoCsv)
        self._CheckRegionEnDataFrame(nivelAgregacion, nombreRegion, df)
        dfRecortado = pd.DataFrame(df[nombreColumna], columns = [nombreColumna]).copy()
        dfRecortado.rename(columns = {nombreColumna: nombreRegion}, inplace = True)
        return dfRecortado.copy()

    def _LeeDataFrameMadrid(self, archivoCsv):
        df = pd.read_csv(archivoCsv, index_col = self._nombreColDate, parse_dates = True)
        return df.copy()

    def _CheckRegionEnDataFrame(self, nivelAgregacion, nombreRegion, df):
        if df[self._nombreColRegion[nivelAgregacion]].unique().item() != nombreRegion:
            raise ValueError(f'El dataframe de la region {nombreRegion} en el nivel {nivelAgregacion} contiene datos de otra region')

    def getNivelesAgregacionEspacial(self):
        return self._nivelesAgregacion

    def getRegionesDisponibles(self, nivelAgregacion):
        return self._nombreRegiones[nivelAgregacion]
#-------------------------------------------------------------------------------

def main():
    dictgeoDataFrames = {}
    dictEstadisticos = {}
    dictAreasRegiones = {}
    listaX = []
    listaX2 = []
    primerEstadistico = True
    '''
    Cargamos las bases de datos: (i) los modelos de Madrid, (ii) los shapefiles
    con las regiones NUTS, (iii) SOWISP, (iv) las mascaras de provincias de ERA5
    y RetroDB y (v) el archivo estaticos.nc de RetroDB y un fichero de ERA5
    '''
    datosMadridML = MLdatabase(WorkDir.getFileFromClave('wind_database'))

    for nivelAgg, clave in zip (datosMadridML.getNivelesAgregacionEspacial()[::2], ('shapefile_autonomia', 'shapefile')):
        dictgeoDataFrames[nivelAgg] = gpd.read_file(WorkDir.getDirectorioAdicionalFromClave(clave))

    sowisp = SOWISP_HighResolution(WorkDir.getFileFromClave('sowisp'))
    sowisp.FiltraPorRangoDatetimes(DATETIME_EVALUACION, DATETIME_EVALUACION, inplace = True)
    sowisp.FiltraPorListaDeRegiones(list(np.unique(sowisp.getDataFrame()[[NOMBRE_COLUMNA_PROVINCIA_SOWISP, NOMBRE_COLUMNA_CA_SOWISP , NOMBRE_COLUMNA_PAIS_SOWISP]])), inplace = True)
    dfInstalacionesEolicas = sowisp.getDataFrame().copy()
    dfNumeroInstalaciones = pd.DataFrame(dfInstalacionesEolicas.groupby(NOMBRE_COLUMNA_PROVINCIA_SOWISP).apply(len), columns = [NOMBRE_COLUMNA_N_INSTALACIONES])
    dfPotIns = sowisp.SerieTemporalPotenciaInstaladaMW(DATETIME_EVALUACION, DATETIME_EVALUACION).copy() # este metodo resetea el filtro de fechas por lo que es necesario ponerlo lo ultimo

    arrayMaskProvinciasERA5, dictIdProvincias = ExtraeMascaraProvincias(WorkDir.getFileFromClave('mascara_provincias_era5')) # La mascara esta en la misma orientacion que los ERA5
    arrayMaskProvinciasRetroDB, dictIdProvincias = ExtraeMascaraProvincias(WorkDir.getFileFromClave('mascara_provincias_retrodb')) # No hay problema en que se machaque la variable dictIdProvincias porque es la misma para ambas mascaras

    dictAlturasRetroDB = ExtraeDictFromNetCDF(WorkDir.getDirectorioAdicionalFromClave('estaticos_retrodb'), [NOMBRE_ALTURA_RETRODB], dtUnico = True)
    # dictLatLonERA5 = ExtraeLatLon2DfromERA5(
    #     datetime.strftime(DATETIME_EVALUACION, f'{WorkDir.getDirectorioAdicionalFromClave("era5")}{DATETIME_EVALUACION.year}/{NOMBRE_ERA5_DATEFORMAT}')
    # )
    '''
    Calculo de un par de parametros que me he inventado: (i) ratio pixel con
    instalacion y totales y (ii) cociente desviacion estandar altura y media
    altura
    '''
    # for nombreProvincia in dfNumeroInstalaciones.index.values:
    #     listaX.append(np.std(dictAlturasRetroDB[NOMBRE_ALTURA_RETRODB][arrayMaskProvinciasRetroDB == dictIdProvincias[nombreProvincia]]) / np.mean(dictAlturasRetroDB[NOMBRE_ALTURA_RETRODB][arrayMaskProvinciasRetroDB == dictIdProvincias[nombreProvincia]]))
    #     dfLocalizacionesProvincia = dfInstalacionesEolicas[['Municipio', 'Latitud', 'Longitud']].loc[dfInstalacionesEolicas[NOMBRE_COLUMNA_PROVINCIA_SOWISP] == nombreProvincia].groupby('Municipio').apply(np.unique)
    #     listaIdx = []
    #     for municipio in dfLocalizacionesProvincia.index.values:
    #         lon0, lat0 = dfLocalizacionesProvincia.loc[municipio]
    #         listaIdx.append(IdxMasCercanosLatLon2D(dictLatLonERA5[NOMBRE_LAT_ERA5], dictLatLonERA5[NOMBRE_LON_ERA5], lat0, lon0))
    #     listaX2.append(len(np.unique(listaIdx)) / len(arrayMaskProvinciasERA5[arrayMaskProvinciasERA5 == dictIdProvincias[nombreProvincia]]))
    # df1 = pd.DataFrame(listaX, index = dfNumeroInstalaciones.index, columns = ['x'])
    # df2 = pd.DataFrame(listaX2, index = dfNumeroInstalaciones.index, columns = ['x2'])
    '''
    Elaboramos una tabla con los estadisticos en los distintos niveles de
    agregacion
    '''
    for estadistico, funcion in zip(('Correlation', 'Bias', 'RMSE'), (Correlacion, Bias, RMSE)):
        dictEstadisticos[estadistico] = {}
        for nivelAgg in datosMadridML.getNivelesAgregacionEspacial():
            dictEstadisticos[estadistico][nivelAgg] = []
            if primerEstadistico == True:
                dictAreasRegiones[nivelAgg] = []
            if len(datosMadridML.getRegionesDisponibles(nivelAgg)) == 1:
                tuplaRegiones = tuple(datosMadridML.getRegionesDisponibles(nivelAgg),)
            else:
                tuplaRegiones = tuple(datosMadridML.getRegionesDisponibles(nivelAgg))
            for nombreRegion in tuplaRegiones:
                dfREE = datosMadridML.getSerieRegionalCF(nivelAgg, nombreRegion, modelada = False)
                dfModelo = datosMadridML.getSerieRegionalCF(nivelAgg, nombreRegion)
                dictEstadisticos[estadistico][nivelAgg].append(CalculaEstadistico2DataFrames(funcion, dfREE, dfModelo))
                if ((primerEstadistico == True) & (nivelAgg != 'nacional')):
                    try:
                        dictAreasRegiones[nivelAgg].append(CalculaAreaRegionShapefile(dictgeoDataFrames[nivelAgg], DICT_REGIONES_IGN[nombreRegion]))
                    except KeyError:
                        dictAreasRegiones[nivelAgg].append(CalculaAreaRegionShapefile(dictgeoDataFrames[nivelAgg], nombreRegion))
        if primerEstadistico == True:
            dictAreasRegiones['nacional'].append(EXTENSION_MAINLAND_SPAIN_KM2)
        primerEstadistico = False

    dfTablaEstadisticos = pd.DataFrame([], index = pd.Index(['NUTS 3', 'NUTS 2', 'NUTS 0'], name = 'Region'), columns = list(dictEstadisticos.keys()))
    for idxName, nivelAgg in zip(dfTablaEstadisticos.index.values, ('provincial', 'autonomico', 'nacional')):
        for colName in list(dfTablaEstadisticos.keys()):
            if len(dictEstadisticos[colName][nivelAgg]) == 1:
                dfTablaEstadisticos.loc[idxName, colName] = f'{np.round(dictEstadisticos[colName][nivelAgg], 3).item()}'
            else:
                dfTablaEstadisticos.loc[idxName, colName] = f'{np.round(np.mean(dictEstadisticos[colName][nivelAgg]), 3)} +/- {np.round(np.std(dictEstadisticos[colName][nivelAgg]), 3)}'
    dfTablaEstadisticos.to_latex(
        f'{WorkDir.getDirectorioResultados()}TablaEstadisticosNUTS.tex',
    )
    '''
    Figura del error en funcion del tamanyo de la region
    '''
    # estadistico = 'RMSE'
    # fig = plt.figure(0, figsize = (9.0 / 2.54, 6.0 / 2.54), dpi = 600, clear = True)
    # ax = fig.add_axes([0.18, 0.2, 0.78, 0.78])
    # for nivelAgg in dictEstadisticos[estadistico].keys():
    #     ax.plot(np.mean(dictAreasRegiones[nivelAgg]) / 1e5, np.mean(dictEstadisticos[estadistico][nivelAgg]), color = 'black', marker = 'o', markersize = 1.5)
    #     ax.errorbar(np.mean(dictAreasRegiones[nivelAgg]) / 1e5, np.mean(dictEstadisticos[estadistico][nivelAgg]), xerr = np.std(dictAreasRegiones[nivelAgg]) / 1e5, yerr = np.std(dictEstadisticos[estadistico][nivelAgg]), color = 'tab:blue')
    # ax.set_xlabel(r'Area $\cdot 10^{-5}$ [km$^2$]')
    # ax.set_ylabel(f'{estadistico} [-]')
    # ax.grid(':')
    # fig.savefig(f'{WorkDir.getDirectorioFiguras()}{estadistico}VsTamanyoRegion.jpg', dpi = 600)
    # plt.close(0)

    # nivelAgg = 'provincial'
    metric = 'RMSE'
    fig, axs = plt.subplots(1, 2, figsize = (14.0 / 2.54, 9.0 / 2.54))
    for agg_level, marker in zip(dictEstadisticos[estadistico].keys(), ('s',)):
        if len(datosMadridML.getRegionesDisponibles(agg_level)) == 1:
            regions = tuple(datosMadridML.getRegionesDisponibles(agg_level),)
        else:
            regions = tuple(datosMadridML.getRegionesDisponibles(agg_level))
        for iterator in range(len(regions)):
            region_name = regions[iterator]
            rmse_value = dictEstadisticos[metric][agg_level][iterator]
            region_area = dictAreasRegiones[agg_level]
            for ax in axs:
                ax.plot(region_area, rmse_value, marker = marker)

    ax = {}
    for iterator, df, xlabel in zip(range(4), (dfPotIns.loc[DATETIME_EVALUACION], dfNumeroInstalaciones, df1['x'], df2['x2']), ('Ins Capacity [MW]', 'N facilities [-]', r'$\sigma_{\rm{HGT}}$ / $\overline{\rm{HGT}}$ [-]', r'$N_{\rm{px.fac.}}$ / $N_{\rm{px.NUTS3}}$ [-]')):
        ax[str(iterator)] = fig.add_axes([0.13 + 0.45 * np.mod(iterator, 2), 0.1 + 0.48 * [0, 0, 1, 1][iterator], 0.4, 0.38])
        for iteratorValor in range(len(dictEstadisticos[estadistico][nivelAgg])):
            nombreProvincia = datosMadridML.getRegionesDisponibles(nivelAgg)[iteratorValor]
            ax[str(iterator)].plot(df.loc[nombreProvincia], dictEstadisticos[estadistico][nivelAgg][iteratorValor], 'o', color = 'tab:blue', markersize = 2.0)
        ax[str(iterator)].set_xlabel(xlabel)
        if ((iterator == 0) | (iterator == 2)):
            ax[str(iterator)].set_ylabel('RMSE [-]')
        else:
            ax[str(iterator)].set_yticklabels([])
        ax[str(iterator)].grid(':')
    fig.savefig(f'{WorkDir.getDirectorioFiguras()}Estudio{estadistico}{nivelAgg}DiferentesParams.jpg', dpi = 600)
    plt.close(0)


    fig = plt.figure(0, figsize = (19.0 / 2.54, 9.0 / 2.54), dpi = 600, clear = True)
    ax = Artist.AxisElsevier(
        figura = fig,
        x0 = 0.08,
        y0 = 0.17,
        anchoRel = 0.86,
        altoRel = 0.80,
        xlabel = 'NUTS',
        ylabel = 'RMSE [-]'
    )
    listaCorrs = []
    for nombreRegion, nivelAgg in zip(DICT_REGIONES_NUTS.keys(), LISTA_MODELOS_FIGURA_BARRAS):
        if nivelAgg == 'nacional':
            color = 'tab:green'
        elif nivelAgg == 'autonomico':
            color = 'tab:orange'
        else:
            color = 'tab:blue'
        dfREE = datosMadridML.getSerieRegionalCF(nivelAgg, nombreRegion, modelada = False)
        dfModelo = datosMadridML.getSerieRegionalCF(nivelAgg, nombreRegion)
        ax.bar(nombreRegion, CalculaEstadistico2DataFrames(RMSE, dfREE, dfModelo), label = DICT_REGIONES_NUTS[nombreRegion], color = color, zorder = 3, edgecolor = 'black')
        listaCorrs.append(CalculaEstadistico2DataFrames(Correlacion, dfREE, dfModelo))
    for nivelAgg, color in zip(('nacional', 'autonomico', 'provincial'), ('tab:green', 'tab:orange', 'tab:blue')):
        ax.axhline(np.mean(dictEstadisticos['RMSE'][nivelAgg]), color = color, ls = '--', lw = 2, zorder = 5)
    ax2 = ax.twinx()
    ax2.plot(list(DICT_REGIONES_NUTS.values()), listaCorrs, marker = 'o', markersize = 4, color = 'tab:red')
    ax2.set_ylabel('Correlation [-]', fontsize = 10, family = 'Liberation Sans', color = 'tab:red')
    ax2.set_yticks(np.arange(0.0, 1.1, 0.1))
    ax2.tick_params(axis = 'y', labelsize = 8, direction = 'in', colors = 'tab:red')
    ax2.spines['right'].set_color('tab:red')
    ax.tick_params(axis = 'x', rotation = 90.0)
    fig.savefig(f'{WorkDir.getDirectorioFiguras()}BarrasRMSEyCorrelacion.jpg', dpi = 600)
    plt.close(0)
    return 0

if __name__ == '__main__':
    main()
