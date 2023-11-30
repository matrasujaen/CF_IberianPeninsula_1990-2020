#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
#  AnalisisWindCF30years.py
#
#  2023 Antonio Jiménez-Garrote <agarrote@ujaen.es>
#
#
'''
Este programa recopila las disferentes bases de datos de CF y las almacena en un
archivo comun.
'''

import sys
from WorkingDirectory_py3 import WorkingDirectory
WorkDir = WorkingDirectory('archivoConfiguracion.ini')
sys.path.append(WorkDir.getDirectorioLibrerias())

#-------------------------------------------------------------------LIBRARIES---
import numpy as np
import pandas as pd
from netCDF4 import Dataset
from datetime import datetime, timedelta
from matplotlib import pyplot as plt
#import ArtistElsevier_py3 as Artist
from REE_py3 import CargaArchivoGeneracionESIOS
from SOWISP_py3 import SOWISP_HighResolution
#from WindUtils_py3 import Correlacion
#-------------------------------------------------------------------------------

#-------------------------------------------------------------------CONSTANTS---
YEAR_INI = 2007
YEAR_FIN = 2015
#FECHA_CONGELAMIENTO = '2020-01-01 00:00:00' # date_format %Y-%m-%d %H:%M:%S
TUPLA_PROVINCIAS = ('ACoruna', 'Albacete', 'Almeria', 'Araba', 'Asturias',
    'Avila', 'Barcelona', 'Bizkaia', 'Burgos', 'Cadiz', 'Cantabria', 'Castellon',
    'CiudadReal', 'Cuenca', 'Granada', 'Guadalajara', 'Huelva', 'Huesca', 'Jaen',
    'LaRioja', 'Leon', 'Lleida', 'Lugo', 'Malaga', 'Murcia', 'Navarra', 'Ourense',
    'Palencia', 'Pontevedra', 'Salamanca', 'Segovia', 'Sevilla', 'Soria',
    'Tarragona', 'Teruel', 'Toledo', 'Valencia', 'Valladolid', 'Zamora',
    'Zaragoza'
)
NOMBRE_ESPANYA = 'Spain'
NOMBRE_COL_PESOS = 'Peso'
NOMBRE_COL_FECHA_ML = 'Fecha'
NOMBRE_COL_CF_ML = 'prediccion'
NOMBRE_COL_FECHA = 'Date_UTC'
NOMBRE_COL_CF = '{}_CF'
NOMBRE_COL_POT_INS = '{}_InsCap_[MW]'
NOMBRE_COL_MWH = '{}_WindGen_[MWh]'
#-------------------------------------------------------------------------------

#-------------------------------------------------------------------FUNCTIONS---
def LeeCsvMadrid(archivoCsv, columnaFechas = NOMBRE_COL_FECHA_ML, columnaCF = NOMBRE_COL_CF_ML):
    df = pd.read_csv(archivoCsv, usecols = [columnaFechas, columnaCF], index_col = columnaFechas, parse_dates = True)
    return df.copy()

def Merge2DataFrames(df1, df2, metodo):
    dfMerge = pd.merge(df1, df2, how = metodo, left_index = True, right_index = True)
    return dfMerge.copy()
#-------------------------------------------------------------------------------

def main():
    emhires = pd.read_excel(
        '/media/agarrote/DATOS/Doctorado/ComparativaModelosEolicosRegionales/Datos/EMHIRES_WIND_COUNTRY_June2019.xlsx',
        usecols = ['Year', 'Month', 'Day', 'Hour', 'ES'],
        parse_dates = [['Year', 'Month', 'Day', 'Hour']],
        decimal = ',')
    emhires[NOMBRE_COL_FECHA] = [datetime.strptime(date_str, '%Y %m %d %H') for date_str in emhires['Year_Month_Day_Hour'].values]
    emhires.set_index(NOMBRE_COL_FECHA, inplace = True)
    emhires.rename(columns = {'ES': NOMBRE_COL_CF.format('EMHIRES')}, inplace = True)
    emhires.drop(columns = 'Year_Month_Day_Hour', inplace = True)
    emhires[NOMBRE_COL_POT_INS.format('EMHIRES')] = np.repeat(np.nan, len(emhires))
    emhires.loc[str(YEAR_FIN), NOMBRE_COL_POT_INS.format('EMHIRES')] = np.repeat(23237.0, len(emhires.loc[str(YEAR_FIN)]))
    emhires[NOMBRE_COL_MWH.format('EMHIRES')] = emhires[NOMBRE_COL_CF.format('EMHIRES')].values * emhires[NOMBRE_COL_POT_INS.format('EMHIRES')].values

    ninja = pd.read_csv('/media/agarrote/DATOS/Doctorado/ComparativaModelosEolicosRegionales/Datos/ninja_wind_country_ES_current-merra-2_corrected.csv',
        skiprows = 2,
        index_col = 0,
        parse_dates = True)
    ninja.index.rename(NOMBRE_COL_FECHA, inplace = True)
    ninja.rename(columns = {'national': NOMBRE_COL_CF.format('Ninja')}, inplace = True)
    ninja[NOMBRE_COL_POT_INS.format('Ninja')] = np.repeat(np.nan, len(ninja))
    ninja.loc[str(YEAR_FIN), NOMBRE_COL_POT_INS.format('Ninja')] = np.repeat(23052.6, len(ninja.loc[str(YEAR_FIN)]))
    ninja[NOMBRE_COL_MWH.format('Ninja')] = ninja[NOMBRE_COL_CF.format('Ninja')].values * ninja[NOMBRE_COL_POT_INS.format('Ninja')].values

    reading_nc = Dataset('/media/agarrote/DATOS/Doctorado/ComparativaModelosEolicosRegionales/Datos/ERA5_data_1950-2020/wp_onshore/NUTS_0_wp_ons_sim_0_historical_loc_weighted.nc', 'r')
    reading_nuts = reading_nc.variables['NUTS_keys'][:]
    reading_timeseries = reading_nc.variables['timeseries_data'][reading_nuts == 'ES', :].data
    reading_datetime = [datetime(1950, 1, 1) + timedelta(hours = int(hour)) for hour in reading_nc.variables['time_in_hours_from_first_jan_1950'][:].data]
    reading_nc.close()
    reading = pd.DataFrame(reading_timeseries.reshape(len(reading_datetime), 1), index = pd.Index(reading_datetime, name = NOMBRE_COL_FECHA), columns = [NOMBRE_COL_CF.format('Reading')])
    reading[NOMBRE_COL_POT_INS.format('Reading')] = np.repeat(np.nan, len(reading))
    reading.loc[str(YEAR_FIN), NOMBRE_COL_POT_INS.format('Reading')] = dfPotIns.loc[str(YEAR_FIN), NOMBRE_ESPANYA].values.copy()
    reading[NOMBRE_COL_MWH.format('Reading')] = reading[NOMBRE_COL_CF.format('Reading')].values * reading[NOMBRE_COL_POT_INS.format('Reading')].values

    c3s = pd.read_csv('/media/agarrote/DATOS/Doctorado/ComparativaModelosEolicosRegionales/Datos/H_ERA5_ECMW_T639_WON_0100m_Euro_NUT0_S197901010000_E202309302300_CFR_TIM_01h_NA-_noc_org_NA_NA---_NA---_PhM01.csv',
        skiprows = 52,
        usecols = ['Date', 'ES'],
        parse_dates = True,
        index_col = 0)
    c3s.index.rename(NOMBRE_COL_FECHA, inplace = True)
    c3s.rename(columns = {'ES': NOMBRE_COL_CF.format('C3S')}, inplace = True)
    c3s[NOMBRE_COL_POT_INS.format('C3S')] = np.repeat(np.nan, len(c3s))
    c3s.loc[str(YEAR_FIN), NOMBRE_COL_POT_INS.format('C3S')] = dfPotIns.loc[str(YEAR_FIN), NOMBRE_ESPANYA].values.copy()
    c3s[NOMBRE_COL_MWH.format('C3S')] = c3s[NOMBRE_COL_CF.format('C3S')].values * c3s[NOMBRE_COL_POT_INS.format('C3S')].values

    dictDfRegiones = {}
    dfNacional = pd.DataFrame()

    sowisp = SOWISP_HighResolution(WorkDir.getFileFromClave('sowisp'))
    listRegiones = list(TUPLA_PROVINCIAS)
    listRegiones.append(NOMBRE_ESPANYA)
    sowisp.FiltraPorListaDeRegiones(listRegiones, inplace = True)
    dfPotIns = sowisp.SerieTemporalPotenciaInstaladaMW(datetime(YEAR_INI, 1, 1), datetime(YEAR_FIN, 12, 31, 23)) # por si se quiere modelizar el comportamiento historico de la generacion

    dfREE_PreESIOS = CargaArchivoGeneracionESIOS(WorkDir.getFileFromClave('generacion_ree_preesios'))
    dfREE_ESIOS = CargaArchivoGeneracionESIOS(WorkDir.getDirectorioAdicionalFromClave('generacion_ree_esios').format('Nacional'))
    dfREE = pd.concat([dfREE_PreESIOS[1:].copy(), dfREE_ESIOS.loc[:str(YEAR_FIN)].copy()])
    dfREE[NOMBRE_COL_CF.format('REE')] = dfREE['value'].values / dfPotIns.loc[dfREE.index, NOMBRE_ESPANYA].values
    dfREE.rename(columns = {'value': NOMBRE_COL_MWH.format('REE')}, inplace = True)
    dfREE.index.rename(NOMBRE_COL_FECHA, inplace = True)

    for nombreProvincia in TUPLA_PROVINCIAS:
        dictDfRegiones[nombreProvincia] = pd.DataFrame()
        for year in range(YEAR_INI, YEAR_FIN + 1):
            dfModeloCFanyo = LeeCsvMadrid(WorkDir.getFileFromClave('wind_cf30').format(year, nombreProvincia, year))
            dfModeloCFanyo.rename(columns = {NOMBRE_COL_FECHA_ML: NOMBRE_COL_FECHA, NOMBRE_COL_CF_ML: NOMBRE_COL_CF.format(nombreProvincia)}, inplace = True)
            dfModeloCFanyo[NOMBRE_COL_POT_INS.format(nombreProvincia)] = dfPotIns[nombreProvincia].loc[str(year)].values.copy()
            dfModeloCFanyo[NOMBRE_COL_MWH.format(nombreProvincia)] = dfModeloCFanyo[NOMBRE_COL_POT_INS.format(nombreProvincia)].values * dfModeloCFanyo[NOMBRE_COL_CF.format(nombreProvincia)]
            dictDfRegiones[nombreProvincia] = pd.concat([dictDfRegiones[nombreProvincia].copy(), dfModeloCFanyo.copy()])
        if nombreProvincia == TUPLA_PROVINCIAS[0]:
            metodo = 'outer'
        else:
            metodo = 'inner'
        dfNacional = Merge2DataFrames(
            dfNacional.copy(),
            pd.DataFrame(
                dictDfRegiones[nombreProvincia][NOMBRE_COL_MWH.format(nombreProvincia)].values,
                columns = [NOMBRE_COL_MWH.format(nombreProvincia)],
                index = dictDfRegiones[nombreProvincia].index
            ),
            metodo
        )
    dictDfRegiones[NOMBRE_ESPANYA] = pd.DataFrame(dfNacional.sum(axis = 1), columns = [NOMBRE_COL_MWH.format(NOMBRE_ESPANYA)])
    dictDfRegiones[NOMBRE_ESPANYA][NOMBRE_COL_POT_INS.format(NOMBRE_ESPANYA)] = dfPotIns[NOMBRE_ESPANYA].values.copy()
    dictDfRegiones[NOMBRE_ESPANYA][NOMBRE_COL_CF.format(NOMBRE_ESPANYA)] = dictDfRegiones[NOMBRE_ESPANYA][NOMBRE_COL_MWH.format(NOMBRE_ESPANYA)].values / dictDfRegiones[NOMBRE_ESPANYA][NOMBRE_COL_POT_INS.format(NOMBRE_ESPANYA)].values

    all_databases = dfREE.copy()
    for df in (c3s, emhires, ninja, reading, dictDfRegiones[NOMBRE_ESPANYA]):
        all_databases = Merge2DataFrames(
            all_databases.copy(),
            df.copy(),
            'inner'
        )

    fig, ax = plt.subplots(figsize = (9 / 2.54, 9 / 2.54), clear = True)
    for label in ('C3S', 'EMHIRES', 'Ninja', 'Reading', 'Spain'):
        values = all_databases.dropna().loc[str(YEAR_FIN), NOMBRE_COL_MWH.format(label)].values - all_databases.dropna().loc[str(YEAR_FIN), NOMBRE_COL_MWH.format('REE')].values
        ax.hist(values, bins = np.arange(-20000, 20000, 1000), histtype = 'step', label = label)
    ax.legend()
    fig.tight_layout()
    fig.savefig('/home/agarrote/Escritorio/kkHists.png', dpi = 600)
    return 0

if __name__ == '__main__':
    main()
