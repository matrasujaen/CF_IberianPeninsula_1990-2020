#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 09 15:54 2023

@author: Antonio Jiménez-Garrote <agarrote@ujaen.es>

Esta libreria contiene un par de clases basadas en SOWISP que pretenden reunir
los metodos principales con los que se trabaja normalmente esta base de datos
"""
#-------------------------------------------------------------------LIBRARIES---
import numpy as np
import pandas as pd
from datetime import datetime
#-------------------------------------------------------------------------------

#-------------------------------------------------------------------CONSTANTS---
NOMBRE_COLUMNA_LATITUD = 'Latitud'
NOMBRE_COLUMNA_LONGITUD = 'Longitud'
NOMBRE_COLUMNA_FECHAALTA = 'FechaAlta'
NOMBRE_COLUMNA_FECHABAJA = 'FechaBaja'
NOMBRE_COLUMNA_MUNICIPIO = 'Municipio'
NOMBRE_COLUMNA_PROVINCIA = 'Provincia'
NOMBRE_COLUMNA_CA = 'CA'
NOMBRE_COLUMNA_PAIS = 'Pais'
NOMBRE_COLUMNA_POTINS = 'PotInsKW'
NOMBRE_INDICE_SERIE_TEMPORAL = 'Date_UTC'
NOMBRE_ESPANYA = 'Spain'
FRECUENCIA_HORARIA_STR = '1H'
FRECUENCIA_MENSUAL_STR = '1M'
FECHA_INI_SOWISP = '20150131'
FECHA_FIN_SOWISP = '20201231'
DATE_FORMAT = '%Y%m%d'
NOMBRE_DATEFORMAT_SOWISP = f'InsPowMW_{DATE_FORMAT}'
TUPLA_SUFIJOS = ('', '_2')
EQUIV_POTENCIA_MW = 0.001
#-------------------------------------------------------------------------------

#-------------------------------------------------------------------FUNCTIONS---
def ConcatenaDosDataFrames(df1, df2, tipoDeConcatenado, sufijos = TUPLA_SUFIJOS):
    return pd.merge(df1, df2, tipoDeConcatenado, left_index = True, right_index = True, suffixes = sufijos)

def ListaDatesToDateStr(listaDates, dtFormat = NOMBRE_DATEFORMAT_SOWISP):
    return [datetime.strftime(date, dtFormat) for date in listaDates]
#-------------------------------------------------------------------------------

#---------------------------------------------------------------------CLASSES---
class SOWISP_HighResolution(object):
    def __init__(self, SOWISPcsv):
        dfSOWISP = pd.read_csv(SOWISPcsv, sep = ';', parse_dates = [NOMBRE_COLUMNA_FECHAALTA, NOMBRE_COLUMNA_FECHABAJA])
        self._dataFrameOriginal = dfSOWISP.copy()
        self.ResetSOWISPHighRes()

    def ResetSOWISPHighRes(self):
        dfSOWISPconPais = self._dataFrameOriginal.copy()
        dfSOWISPconPais[NOMBRE_COLUMNA_PAIS] = np.repeat(NOMBRE_ESPANYA, len(dfSOWISPconPais))
        self._regiones = [NOMBRE_ESPANYA]
        self._dataFrame = dfSOWISPconPais.copy()

    def pseudoSOWISP(self):
        self.FiltraPorListaDeRegiones(list(self._dataFrameOriginal[NOMBRE_COLUMNA_MUNICIPIO].unique()))
        dfSOWISP = self.SerieTemporalPotenciaInstaladaMW(datetime.strptime(FECHA_INI_SOWISP, DATE_FORMAT), datetime.strptime(FECHA_FIN_SOWISP, DATE_FORMAT), FRECUENCIA_MENSUAL_STR)
        listaFechas = ListaDatesToDateStr(dfSOWISP.index.to_pydatetime())
        return pd.DataFrame(dfSOWISP.T.values, index = dfSOWISP.columns, columns = listaFechas)

    def FiltraPorListaDeRegiones(self, listaRegiones, inplace = False):
        if inplace == True:
            self._regiones = listaRegiones
            self._dataFrame = self._dataFrame[self._mascaraRegion(self._regiones)].copy()
        else:
            return self._dataFrame[self._mascaraRegion(listaRegiones)].copy()

    def _mascaraRegion(self, listaRegiones):
        mascara = ((np.isin(self._dataFrame[NOMBRE_COLUMNA_MUNICIPIO].values, listaRegiones)) | (np.isin(self._dataFrame[NOMBRE_COLUMNA_PROVINCIA].values, listaRegiones)) | (np.isin(self._dataFrame[NOMBRE_COLUMNA_CA].values, listaRegiones)) | (np.isin(self._dataFrame[NOMBRE_COLUMNA_PAIS].values, listaRegiones)))
        return mascara

    def SerieTemporalPotenciaInstaladaMW(self, dateIni, dateFin, freq = FRECUENCIA_HORARIA_STR):
        listaRegiones = list(np.unique(self._regiones))
        indexDf = pd.DatetimeIndex(pd.date_range(dateIni, dateFin, freq = freq).to_pydatetime(), name = NOMBRE_INDICE_SERIE_TEMPORAL)
        dfSerieTemporal = pd.DataFrame([], index = indexDf)
        for region in listaRegiones:
            self.FiltraPorListaDeRegiones([region], inplace = True)
            dfPotInsAcc = self._DataFramePotenciaInstaladaAcumulada(indexDf)
            dfPotInsAcc.rename(columns = {NOMBRE_COLUMNA_POTINS: region}, inplace = True)
            dfSerieTemporal = ConcatenaDosDataFrames(dfSerieTemporal, dfPotInsAcc, 'inner')
            self.ResetSOWISPHighRes()
        self.FiltraPorListaDeRegiones(listaRegiones, inplace = True)
        return dfSerieTemporal * EQUIV_POTENCIA_MW

    def _DataFramePotenciaInstaladaAcumulada(self, dateIndex):
        dfPotAltaAcc = self._PotenciaInstaladaAcumuladaAgrupadaPor(NOMBRE_COLUMNA_FECHAALTA)
        dfPotAltaAccReindexado = dfPotAltaAcc.reindex(index = dateIndex, method = 'ffill', fill_value = 0.0).copy()
        try:
            dfPotBajaAcc = self._PotenciaInstaladaAcumuladaAgrupadaPor(NOMBRE_COLUMNA_FECHABAJA)
            dfPotBajaAccReindexado = dfPotBajaAcc.reindex(index = dateIndex, method = 'ffill', fill_value = 0.0).copy()
        except:
            dfPotBajaAccReindexado = pd.DataFrame([], index = dateIndex, columns = [f'{NOMBRE_COLUMNA_POTINS}'])
            dfPotBajaAccReindexado.fillna(0.0, inplace = True)
        dfPotInsAcc = ConcatenaDosDataFrames(dfPotAltaAccReindexado, dfPotBajaAccReindexado, 'inner')
        return pd.DataFrame((dfPotInsAcc[f'{NOMBRE_COLUMNA_POTINS}{TUPLA_SUFIJOS[0]}'] - dfPotInsAcc[f'{NOMBRE_COLUMNA_POTINS}{TUPLA_SUFIJOS[-1]}']), columns = [NOMBRE_COLUMNA_POTINS])

    def _PotenciaInstaladaAcumuladaAgrupadaPor(self, nombreCol):
        return self._dataFrame[[nombreCol, NOMBRE_COLUMNA_POTINS]].groupby(nombreCol).sum().cumsum()

    def FiltraPorRangoDatetimes(self, dateIni, dateFin, inplace = False):
        if inplace == True:
            self._dataFrame = self._dataFrame[self._mascaraFechas(dateIni, dateFin)].copy()
        else:
            return self._dataFrame[self._mascaraFechas(dateIni, dateFin)].copy()

    def _mascaraFechas(self, dateIni, dateFin):
        mascara = ((self._dataFrame[NOMBRE_COLUMNA_FECHAALTA] <= dateFin) & ((pd.isna(self._dataFrame[NOMBRE_COLUMNA_FECHABAJA]) == True) | (self._dataFrame[NOMBRE_COLUMNA_FECHABAJA] > dateIni)))
        return mascara

    def FiltraPorLatLon(self, latMin, latMax, lonMin, lonMax, inplace = False):
        extension = [latMin, latMax, lonMin, lonMax]
        if inplace == True:
            self._dataFrame = self._dataFrame[self._mascaraLatLon(extension)].copy()
        else:
            return self._dataFrame[self._mascaraLatLon(extension)].copy()

    def _mascaraLatLon(self, extension):
        latMin, latMax, lonMin, lonMax = extension
        mascara = ((self._dataFrame[NOMBRE_COLUMNA_LATITUD] >= latMin) & (self._dataFrame[NOMBRE_COLUMNA_LATITUD] <= latMax) & (self._dataFrame[NOMBRE_COLUMNA_LONGITUD] >= lonMin) & (self._dataFrame[NOMBRE_COLUMNA_LONGITUD] <= lonMax))
        return mascara

    def getLatLonMunicipio(self, municipio):
        lat = np.unique(self.FiltraPorListaDeRegiones([municipio])['Latitud']).item()
        lon = np.unique(self.FiltraPorListaDeRegiones([municipio])['Longitud']).item()
        return lat, lon

    def getDataFrame(self):
        return self._dataFrame.copy()

    def getSOWISPHightResolution(self):
        return self._dataFrameOriginal.copy()
#-------------------------------------------------------------------------------
