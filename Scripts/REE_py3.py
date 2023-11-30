#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 14 09:38 2023

@author: Antonio Jiménez-Garrote <agarrote@ujaen.es>

Libreria de REE en la que ire guardando funciones propias de archivos de REE,
sobre todo de lectura de archivos y transformacion de fechas locales a fechas
UTC. Esta libreria surge del beneficio de crear una clase api de REE.
"""
#-------------------------------------------------------------------LIBRARIES---
import requests
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
#-------------------------------------------------------------------------------

#-------------------------------------------------------------------CONSTANTS---
URL_CABECERA = 'https://apidatos.ree.es/es/datos/generacion/potencia-instalada?'
DICT_CODIGO_REGION = {'Andalucia': 4, 'Aragon': 5, 'Asturias': 11,
    'CValenciana': 15, 'Cantabria': 6, 'CastillaYLeon': 8, 'CastillaLaMancha': 7,
    'Cataluna': 9, 'Extremadura': 16, 'Galicia': 17, 'LaRioja': 20, 'Madrid': 13,
    'Murcia': 21, 'Navarra': 14, 'PaisVasco': 10, 'Spain': 8741} # Spain hace referencia a la peninsula
DICT_GEOLIMIT = {'Andalucia': 'ccaa', 'Aragon': 'ccaa', 'Asturias': 'ccaa',
    'CValenciana': 'ccaa', 'Cantabria': 'ccaa', 'CastillaYLeon': 'ccaa', 'CastillaLaMancha': 'ccaa',
    'Cataluna': 'ccaa', 'Extremadura': 'ccaa', 'Galicia': 'ccaa', 'LaRioja': 'ccaa', 'Madrid': 'ccaa',
    'Murcia': 'ccaa', 'Navarra': 'ccaa', 'PaisVasco': 'ccaa', 'Spain': 'peninsular'}
LISTA_TECNOLOGIAS = ['Hidráulica', 'Turbinación bombeo', 'Nuclear', 'Carbón',
    'Fuel + Gas', 'Ciclo combinado', 'Eólica', 'Solar fotovoltaica',
    'Solar térmica', 'Otras renovables', 'Cogeneración', 'Residuos no renovables',
    'Residuos renovables']
DT_FORMAT = '%Y-%m-%dT%H:%M:%S.000'
NOMBRE_CLAVE_DICCIONARIOS = 'included'
NOMBRE_CLAVE_TIPOTEC = 'type'
NOMBRE_CLAVE_ATRIBUTOS = 'attributes'
NOMBRE_CLAVE_VALORES = 'values'
NOMBRE_CLAVE_VALOR = 'value'
NOMBRE_CLAVE_DATE = 'datetime'
NOMBRE_INDEX_DATAFRAME = 'Date_UTC'
NOMBRE_COLUMNA_DATAFRAME = 'PotIns_MW'
#-------------------------------------------------------------------------------

#-------------------------------------------------------------------FUNCTIONS---
def CheckNombreEnLista(nombre, lista):
    return np.isin(nombre, lista).item()

def DatetimeUTCfromISO8601(fechaStr, dtFormat = DT_FORMAT):
    strUTC, strLocal = fechaStr.split('+')
    dateUTC = datetime.strptime(strUTC, dtFormat) - timedelta(hours = int(strLocal.split(':')[0]))
    return dateUTC

def CargaArchivoGeneracionESIOS(archivoCsv, columname = NOMBRE_CLAVE_VALOR, indexName = NOMBRE_CLAVE_DATE):
    return pd.read_csv(archivoCsv, sep = ';', index_col = indexName, usecols = [NOMBRE_CLAVE_DATE, NOMBRE_CLAVE_VALOR], parse_dates = True, date_parser = lambda x: pd.to_datetime(x, utc = True).tz_localize(None))
#-------------------------------------------------------------------------------

#---------------------------------------------------------------------CLASSES---
class REEapi(object):
    def __init__(self, year, region):
        self._year = year
        if CheckNombreEnLista(region, list(DICT_CODIGO_REGION.keys())) == True:
            self._region = region
        else:
            raise AttributeError(f'El nombre de region {region} no esta incluido en la base de datos de REE. Los valores posibles son:\n {list(DICT_CODIGO_REGION.keys())}')
        self._jSON = self._QueryREE()
        self._dictTipoTecID = self._DictTecnologias()

    def _QueryREE(self):
        peticionURL = requests.get(f'{URL_CABECERA}start_date={datetime.strftime(datetime(self._year, 1, 1), DT_FORMAT)}&end_date={datetime.strftime(datetime(self._year, 12, 31, 23, 59), DT_FORMAT)}&time_trunc=month&geo_trunc=electric_system&geo_limit={DICT_GEOLIMIT[self._region]}&geo_ids={DICT_CODIGO_REGION[self._region]}')
        jSON = peticionURL.json()
        return jSON

    def _GetListaDictsFromTipoTec(self, tipoTec):
        return self._jSON[NOMBRE_CLAVE_DICCIONARIOS][self._dictTipoTecID[tipoTec]][NOMBRE_CLAVE_ATRIBUTOS][NOMBRE_CLAVE_VALORES]

    def GetSerieTemporalPotenciaInstalada(self, tipoTec):
        if CheckNombreEnLista(tipoTec, LISTA_TECNOLOGIAS) == True:
            listaDates = []
            listaValores = []
            if CheckNombreEnLista(tipoTec, list(self._dictTipoTecID.keys())) == True:
                for dictDatos in self._GetListaDictsFromTipoTec(tipoTec):
                    listaDates.append(DatetimeUTCfromISO8601(dictDatos[NOMBRE_CLAVE_DATE]))
                    listaValores.append(dictDatos[NOMBRE_CLAVE_VALOR])
            return pd.DataFrame(listaValores, pd.DatetimeIndex(listaDates, name = NOMBRE_INDEX_DATAFRAME), columns = [NOMBRE_COLUMNA_DATAFRAME])
        else:
            print(f'El tipo de tecnologia {tipoTec} no esta incluido en la base de datos de REE. Los valores posibles son:\n {LISTA_TECNOLOGIAS}')

    def _DictTecnologias(self):
        diccionarioTec = {}
        valor = 0
        for dictDatos in self._jSON[NOMBRE_CLAVE_DICCIONARIOS]:
            clave = dictDatos[NOMBRE_CLAVE_TIPOTEC]
            diccionarioTec[clave] = valor
            valor += 1
        return diccionarioTec
#-------------------------------------------------------------------------------
