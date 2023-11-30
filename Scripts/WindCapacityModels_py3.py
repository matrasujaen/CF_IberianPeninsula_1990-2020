# -*- coding: utf-8 -*-
"""
Created on mar 04 may 2021 10:29:07 CET

@author: Antonio Jiménez-Garrote <agarrote@ujaen.es>

La finalidad de esta libreria es la de recopilar los diferentes modelos
de transformación a energia eolica que se han ido proponiendo, partiendo
de los mas simples a los mas elaborados.
"""
#-------------------------------------------------------------------LIBRARIES---
import sys
import numpy as np
import pandas as pd
import WindUtils_py3 as WU
from geopy import distance as GeoDist
from datetime import datetime, timedelta
from scipy.optimize import curve_fit
from scipy.interpolate import interp1d
#sys.path.append('/home/ajgarrote/Documentos/Libraries_Python3/floris/')
#from floris import tools as wfct
#-------------------------------------------------------------------------------

#-------------------------------------------------------------------CONSTANTS---
TUPLA_CF_LOWLAND = (0,0,0,1,3,6,11,18,25,35,48,61,73,82,89,92,95,95,94,89,83,69,52,34,19,11,3,0,0,0,0,0) # Anyado un 0 al original de Paco
TUPLA_CF_UPLAND = (0,0,1,3,5.5,8.5,12.5,18.5,26.5,36.5,47,57.5,67.5,75.5,82.5,88,90.5,91,89.5,85,77.5,68,58,47.5,37,27,17.5,9.5,4,1,0,0) # Anyado un 0 al original de Paco
ALTURA_REF = 400.0
#-------------------------------------------------------------------------------

#-------------------------------------------------------------------FUNCTIONS---
def IdxMasCercanosLatLon2D(lat2D, lon2D, lat0, lon0, verbose = False):
    """
    Funcion que devuelve los indices de un array bidimensional que contiene al
    punto mas cercano a unas coordenadas latitud y longitud dados unos arrays
    bidimensionales de latitudes y longitudes.

    Parameters
    ----------
    lat2D: numpy.ndarray
        Array bidimensional de latitudes de los puntos grid.
    lon2D: numpy.ndarray
        Array bidimensional de longitudes de los puntos grid.
    lat0: float
        Valor de la latitud exacta del punto del cual se quieren saber los
        indices de los arrays bidimensionales lat2D y lon2D que mas se parecen.
    lon0: float
        Analogo a lat0 pero con el valor de longitud.
    verbose: bool, optional
        Valor True/False con el que se muestra en pantalla una pequeña
        comprobacion de que se ha establecido correctamente el grid mas cercano.

    Returns
    -------
    idxLat: int
        Fila que contiene el punto mas cercano a (lat0, lon0)
    idxLon: int
        Columna que contiene el punto mas cercano a (lat0, lon0)
    """
    dist2D = np.sqrt((lat2D - lat0)**2 + (lon2D - lon0)**2)
    idxLat = np.where(dist2D == dist2D.min())[0].item()
    idxLon = np.where(dist2D == dist2D.min())[1].item()
    if verbose:
        print(f'Coordenadas originales: ({lat0},{lon0})')
        print('Punto grid seleccionado (centro) junto a primeros vecinos:')
        print('({},{})\t({},{})\t({},{})\n({},{})\t({},{})\t({},{})\n({},{})\t({},{})\t({},{})\n'.format(
            lat2D[idxLat + 1, idxLon - 1], lon2D[idxLat + 1, idxLon - 1],
            lat2D[idxLat + 1, idxLon], lon2D[idxLat + 1, idxLon],
            lat2D[idxLat + 1, idxLon + 1], lon2D[idxLat + 1, idxLon + 1],
            lat2D[idxLat, idxLon - 1], lon2D[idxLat, idxLon - 1],
            lat2D[idxLat, idxLon], lon2D[idxLat, idxLon],
            lat2D[idxLat, idxLon + 1], lon2D[idxLat, idxLon + 1],
            lat2D[idxLat - 1, idxLon - 1], lon2D[idxLat - 1, idxLon - 1],
            lat2D[idxLat - 1, idxLon], lon2D[idxLat - 1, idxLon],
            lat2D[idxLat - 1, idxLon + 1], lon2D[idxLat - 1, idxLon + 1]
            )
        )
    return idxLat, idxLon

def CoordenadasRelativasGridAerogen(archivoLocations, lat0, lon0, delimitador = '\t', Id1aColumna = True):
    """
    Funcion que devuelve las coordenadas x,y en metros de la
    localizacion en coordenadas latitud, longitud de los aerogeneradores
    de un parque eolico respecto a un origen de coordenadas dado.

    Parameters
    ----------
    archivoLocations: str
        Nombre del archivo .tex que contiene las coordenadas latitud,
        longitud de los aerogeneradores del parque. La cabecera y, por
        tanto, formato de dicho archivo debe ser (ID, Latitud, Longitud)
        o simplemente (Latitud, Longitud).
    lat0: float
        Valor de la latitud del origen de coordenadas a partir del cual
        se calcularan las coordenadas relativas de los aerogeneradores.
    lon0: float
        Analogo a lat0 pero con el valor de longitud.
    delimitador: str
        String con el delimitador con el que se lee el archivo de
        localizaciones (default: \t).
    Id1aColumna: bool
        Valor True/False que determina si en la primera columna de
        archivoLocations se encuentra un valor identificativo de tipo
        entero del aerogenerador.

    Returns
    -------
    x: numpy.ndarray
        Array unidimensional con los valores en metros de las
        coordenadas horizontales relativas de los aerogeneradores.
    y: numpy.ndarray
        Analogo a x pero con las coordenadas verticales.
    """
    if Id1aColumna:
        datos = np.loadtxt(archivoLocations, dtype = np.dtype({
                'names': ('id', 'lat', 'lon'),
                'formats': (np.int32, np.float64, np.float64)}),
            delimiter = delimitador)
    else:
        datos = np.loadtxt(archivoLocations, dtype = np.dtype({
                'names': ('lat', 'lon'),
                'formats': (np.float64, np.float64)}),
            delimiter = delimitador)
    nMolinos = len(datos)
    x = np.zeros(nMolinos)
    y = np.zeros(nMolinos)
    for idx in range(nMolinos):
        cateto = GeoDist.distance((lat0, lon0),
            (lat0, datos['lon'][idx])).m
        hipotenusa = GeoDist.distance((lat0, lon0),
            (datos['lat'][idx], datos['lon'][idx])).m
        x[idx] = cateto
        y[idx] = np.sqrt(hipotenusa**2 - cateto**2)
    return x.copy(), y.copy()

def AlturaRotorEnFuncionYearInstalacion(yearIns, yearLimite = 2005, yearIni = 1990, alturaIni = 30.0, alturaLimite = 80.0): # Cambio yearLimite (antes 2015) y alturaLimite (antes 100.0)
    """
    Modelo lineal de la altura del rotor en funcion del anyo de instalacion
    basado en McKenna, R., Pfenninger, S., Heinrichs, H., Schmidt, J., Staffell,
    I., Gruber, K., ... & Wohland, J. (2021). Reviewing methods and assumptions
    for high-resolution large-scale onshore wind energy potential assessments.
    arXiv preprint arXiv:2103.09781. Como aproximación, voy a discretizar las
    alturas en decenas para tener 6 alturas disponibles en el intervalo [50, 100].


    """
    if yearIns > yearLimite:
        altura = alturaLimite
    else:
        alturaTramoLineal = alturaIni + 3.0 * (yearIns - yearIni)
        altura = round(alturaTramoLineal / 100.0, 1) * 100.0
        '''
        if altura > 50.0:
            altura = altura - 10.0  # Le resto 10 m de altura porque sigue sobreestimando (solo si es mayor de 50 m porque es mi altura minima de vientos interpolados)
        '''
    return int(altura)

def CorreccionDegradacionWF(yearEstudio, yearInstalacion):
    """
    Modelo lineal de degradacion de un parque eolico en funcion del numero de
    anyos que han pasado entre su instalacion y el anyo de estudio. Está basado
    en el estudio de Staffell, I., & Green, R. (2014). How does wind farm
    performance decline with age?. Renewable energy, 66, 775-786.

    Parameters
    ----------
    yearStudio: int
        Entero con el anyo en el cual se esta evaluando la degradacion del
        parque
    yearInstalacion: int
        Entero con el anyo de instalacion del parque.

    Returns
    -------
    CFcor: float
        Numero en tanto por 1 que representa el rendimiento del parque eolico
    """
    if yearEstudio < yearInstalacion:
        raise ValueError(f'El anyo de instalacion es anterior al de evaluacion: {yearInstalacion} < {yearEstudio}')
    CFcor = 1.0 - 0.002 * (yearEstudio - yearInstalacion) # anteriormente 0.016 (1.6%) y 0.004 (0.4%)
    return CFcor

def CFfromCurvaPotenciaGenerica(v, vCutIn = 4.0, vNominal = 12.0, vCutOff = 25.0):
    """

    """
    arrayCF = np.zeros_like(v)
    mascaraTramoLineal = ((v >= vCutIn) & (v <= vNominal))
    mascaraNominal = ((v > vNominal) & (v <= vCutOff))
    arrayCF[mascaraTramoLineal] = (1.0 / (vNominal - vCutIn)) * (v[mascaraTramoLineal] - vCutIn)
    arrayCF[mascaraNominal] = np.repeat(1.0, len(arrayCF[mascaraNominal]))
    return arrayCF.copy()

def FuncionPowerCurvePaco(xAxis = np.arange(32.0), yAxisUp = TUPLA_CF_LOWLAND, yAxisLow = TUPLA_CF_LOWLAND):
    """

    """
    cfAltos = np.array(yAxisUp, dtype = float) / 100.0
    cfBajos = np.array(yAxisLow, dtype = float) / 100.0
    curvaPotenciaAltos = interp1d(xAxis, cfAltos, kind = 'linear', bounds_error = False, fill_value = 'extrapolate')
    curvaPotenciaBajos = interp1d(xAxis, cfBajos, kind = 'linear', bounds_error = False, fill_value = 'extrapolate')
    return curvaPotenciaAltos, curvaPotenciaBajos

def Sigmoidal(v, A, beta, alpha):
    return A * (1 - np.exp(-np.power((v / beta), alpha)))

def DumbRegionalWindCapacityModel(v, genCF, curvaTeor = Sigmoidal):
    """
    Funcion que establece una serie de coeficientes a la funcion que mejor se
    ajusta a la curva empirica de los datos meteo y los datos de generacion
    regional.

    Parameters
    ----------
    v: numpy.ndarray
        Array unidimensional que contiene la serie temporal de modulos
        horizontales del viento
    genCF: numpy.ndarray
        Array unidimensional con la serie temporal de generacion en CF de la
        region a estudiar
    curvaTeor: function
        Funcion de ajuste a los datos

    Returns
    -------
    paramsOpt: ??
        ??
    """
    return 

def DumbRegionaWindPowerMW():
    return

def RegionalWindCapacityModelFromGenericPowerCurve(viento4D, lat2D, lon2D, listaDatetimes, dictIdxAlturas, dfPotIns, nombreRegion):
    """
    Funcion que calcula el capacity factor de una determinada region a partir de
    informacion meteorologica y un dataframe de potencia instalada, usando
    una curva de potencia generica y teniendo en cuenta la evolucion de las
    alturas de los rotores de los parques eolicos y la degradacion de los mismos
    en funcion de los anyos.

    Parameters
    ----------
    viento4D: numpy.ndarray
        Array 4D de modulos de la velocidad horizontal del viento en un grid de
        latitudes y longitudes a varias alturas dadas en el que las dimensiones
        del array multidimensional son las siguientes: (time, height, lat, lon).
    lat2D: numpy.ndarray
        Array bidimensional de latitudes de los puntos grid con el mismo shape
        'espacial' del array de informacion meteo viento4D.
    lon2D: numpy.ndarray
        Analogo a lat2D pero con las longitudes del grid.
    listaDatetimes: list
        Lista de datetimes con las fechas de los instantes temporales con los
        que se ha construido el array viento4D.
    dictIdxAlturas: dict
        Diccionario con clave alturas (en metros y en formato str, sin unidad)
        en el que se asocia, para cada altura, el indice en la dimension height
        del array viento4D.
    dfPotIns: pandas.DataFrame
        Dataframe con la potencia instalada desagregada por instalaciones. Este
        dataframe debe cumplir los siguientes requisitos:
        i) en filas, las diferentes instalaciones (index != codigo)
        ii) en columnas, el codigo de instalacion, el municipio junto con sus
        coordenadas, la fecha de alta y de baja de dicha la instalacion, y la
        provincia y comunidad autonoma a la que pertenece el municipio donde se
        encuentra:
            1) el codigo es un identificativo de la instalacion en formato str.
            No es importante puesto que se deshecha. Es simplemente para tener
            un control visual de las instalaciones
            2) el municipio en formato str (se permite codificacion utf-8) con
            nombre "Municipio"
            3) la latitud con el nombre "Latitud" y en formato float
            4) la longitud con el nombre "Longitud" y en formato float
            5) la fecha de alta de instalacion en un formato datetime. El nombre
            de la columna debe ser "FechaAlta"
            6) Analogo a FechaAlta pero con la fecha de baja de la instalacion,
            si la hubiese. El nombre debe ser "FechaBaja"
            7) La potencia instalada de la instalacion, en kW y en formato float.
            El nombre de dicha columna debe ser "PotInsKW"
            8) la provincia a la que pertenece la localidad, en str y con nombre
            de columna "Provincia"
            8) la comunidad autonoma a la que pertenece la localidad, en str. No
            es importante puesto que, en un principio, se deshecha al igual que
            el codigo. Posteriormente puede ser util para implementar modelos de
            CCAA en vez de provincias
    nombreRegion: str
        Nombre de la region de la que se va a calcular el capacity factor. Debe
        estar incluida en la columna "Provincia" del dataframe "dfPotIns".
    dictPowerCurves: dict
        Falta describir exactamente lo que se quiere. Diria que un diccionario
        de funciones ya interpoladas de curvas de potencia en los que la clave
        es el anyo tipico de instalacion

    Returns
    -------
    arrayCapacityMW: numpy.ndarray
        Array unidimensional con la serie temporal de generacion regional de
        "nombreRegion", en MW.
    """
    # Lo primero es recortar la informacion. Para ello, a partir del argumento
    # nombreRegion, seleccionamos las instalaciones que nos interesan
    dfRecortado = dfPotIns[dfPotIns['Provincia'] == nombreRegion][dfPotIns.columns.values[1:-1]].copy()
    dfRecortado.reset_index(drop = True, inplace = True)
    # Anyadimos el indice de las celdas grid que mas se aproximan al municipio
    # donde se encuentra la instalacion
    arrayIdxLat = np.zeros(len(dfRecortado), dtype = int) - 9999
    arrayIdxLon = np.zeros_like(arrayIdxLat, dtype = int) - 9999
    for municipio in np.unique(dfRecortado['Municipio'].values):
        dfRecortadoMunicipio = dfRecortado[dfRecortado['Municipio'] == municipio]
        indicesMunicipio = dfRecortadoMunicipio.index.values
        if ((len(np.unique(dfRecortadoMunicipio['Latitud'])) != 1) | (len(np.unique(dfRecortadoMunicipio['Longitud'])) != 1)):
            raise ValueError(f'Aparece mas de un valor de Latitud y/o Longitud para el municipio de {municipio}')
        else:
            idxLat, idxLon = IdxMasCercanosLatLon2D(
                lat2D,
                lon2D,
                dfRecortadoMunicipio['Latitud'].values[0],
                dfRecortadoMunicipio['Longitud'].values[0]
            )
        for idx in indicesMunicipio:
            arrayIdxLat[idx] = idxLat
            arrayIdxLon[idx] = idxLon
    dfRecortado['IdxLat'] = arrayIdxLat.copy()
    dfRecortado['IdxLon'] = arrayIdxLon.copy()
    # creacion del array bidimensional de series temporales de potencia
    # instalada. Haciendo uso de dataframes con ".iloc[]" tarda demasiado.
    # Anyado un segundo array bidimensional de correcion de degradaciones en la
    # instalacion en funcion de la fecha de evaluacion y la fecha de alta
    arrayPotIns2D = np.zeros((len(dfRecortado), len(listaDatetimes))) - 9999.999
    arrayDegradaciones2D = np.zeros_like(arrayPotIns2D) - 9999.999
    arrayValoresDF = dfRecortado[['FechaAlta', 'FechaBaja', 'PotInsKW']].values.copy()
    for fila in range(len(dfRecortado)):
        columna = 0
        for fecha in listaDatetimes:
            if ((fecha >= arrayValoresDF[fila, 0]) & ((pd.isna(arrayValoresDF[fila, 1])) | (fecha < arrayValoresDF[fila, 1]))):
                arrayPotIns2D[fila, columna] = arrayValoresDF[fila, 2]
                arrayDegradaciones2D[fila, columna] = CorreccionDegradacionWF(
                    yearEstudio = fecha.year,
                    yearInstalacion = arrayValoresDF[fila, 0].year
                )
            else:
                arrayPotIns2D[fila, columna] = 0.0
                arrayDegradaciones2D[fila, columna] = 0.0
            columna += 1
    # creacion del dataframe de serie temporal de generacion eolica
    dfCapacity = pd.DataFrame(
        data = None,
        index = dfRecortado.index,
        columns = [datetime.strftime(fecha, '%Y%m%d%H%M') for fecha in listaDatetimes],
        dtype = float
    )
    for idx in dfCapacity.index.values:
        alturaRotor = AlturaRotorEnFuncionYearInstalacion(dfRecortado.iloc[idx]['FechaAlta'].year)
        dfCapacity.iloc[idx] = arrayPotIns2D[idx, :] *\
            CFfromCurvaPotenciaGenerica(
                v = viento4D[:, dictIdxAlturas[str(alturaRotor)], dfRecortado.iloc[idx]['IdxLat'], dfRecortado.iloc[idx]['IdxLon']]
            ) *\
            arrayDegradaciones2D[idx, :] * \
            0.95 * 0.95 # Perdidas
    # Calculo final
    arrayCapacityMW = dfCapacity.sum().values / 1000.0
    return arrayCapacityMW.copy()

def RegionalWindCapacityModelFromGenericPowerCurve_v2(viento4D, lat2D, lon2D, listaDatetimes, dictIdxAlturas, sowisp_HR, nombreRegion, alpha = None, beta = 0.0):
    """
    Funcion que calcula el capacity factor de una determinada region a partir de
    informacion meteorologica y un dataframe de potencia instalada, usando
    una curva de potencia generica y teniendo en cuenta la evolucion de las
    alturas de los rotores de los parques eolicos y la degradacion de los mismos
    en funcion de los anyos.

    Parameters
    ----------
    viento4D: numpy.ndarray
        Array 4D de modulos de la velocidad horizontal del viento en un grid de
        latitudes y longitudes a varias alturas dadas en el que las dimensiones
        del array multidimensional son las siguientes: (time, height, lat, lon).
    lat2D: numpy.ndarray
        Array bidimensional de latitudes de los puntos grid con el mismo shape
        'espacial' del array de informacion meteo viento4D.
    lon2D: numpy.ndarray
        Analogo a lat2D pero con las longitudes del grid.
    listaDatetimes: list
        Lista de datetimes ordenados con las fechas de los instantes temporales 
        con los que se ha construido el array viento4D.
    dictIdxAlturas: dict
        Diccionario con clave alturas (en metros y en formato str, sin unidad)
        en el que se asocia, para cada altura, el indice en la dimension height
        del array viento4D.
    sowisp_HR: class
        Clase SOWISP_HighResolution (ver SOWISP_py3) que contiene la informacion 
        y metodos para trabajar con la base de datos de instalaciones eolicas 
        desagregadas.
    nombreRegion: str
        Nombre de la region de la que se va a calcular el capacity factor. Debe
        estar incluida en la columna "Provincia" del dataframe "dfPotIns".
    freqStr: str
        Cadena de string con la resolucion temporal de los datos introducidos 
        en viento4D. Los valores posibles mas comunes son: '1H' y '10T'.

    Returns
    -------
    arrayCapacityMW: numpy.ndarray
        Array unidimensional con la serie temporal de generacion regional de
        "nombreRegion", en MW.
    """
    # Lo primero es conocer las instalaciones que estan contenidas en la region 
    # declarada en el argumento nombreRegion y con periodo de generacion electrica 
    # dentro de listaDatetimes
    sowisp_HR.FiltraPorListaDeRegiones([nombreRegion], inplace = True)
    sowisp_HR.FiltraPorRangoDatetimes(listaDatetimes[0], listaDatetimes[-1], inplace = True)
    dfFiltrado = sowisp_HR.getDataFrame()
    sowisp_HR.ResetSOWISPHighRes()
    
    arrayGeneracionRegion = np.zeros((len(dfFiltrado), len(listaDatetimes))) - 9999.999
    for instalacion in range(len(dfFiltrado)):
        # lat0, lon0, fechaAlta, fechaBaja, potIns = dfFiltrado.iloc[instalacion][['Latitud', 'Longitud', 'FechaAlta', 'FechaBaja', 'PotInsKW']].values.copy()
        alturaRotor = AlturaRotorEnFuncionYearInstalacion(dfFiltrado.iloc[instalacion]['FechaAlta'].to_pydatetime().year)
        idxLat, idxLon = IdxMasCercanosLatLon2D(lat2D, lon2D, dfFiltrado.iloc[instalacion]['Latitud'].item(), dfFiltrado.iloc[instalacion]['Longitud'].item())
        arrayViento = viento4D[:, dictIdxAlturas[str(alturaRotor)], idxLat, idxLon].copy()
        if alpha is not None:
            arrayVientoCorregido = alpha * arrayViento + beta
        else:
            arrayVientoCorregido = arrayViento.copy()
        arrayCF = CFfromCurvaPotenciaGenerica(arrayVientoCorregido)
        listaPotIns = []
        listaDegradacion = []
        for date in listaDatetimes:
            if ((date >= dfFiltrado.iloc[instalacion]['FechaAlta'].to_pydatetime()) & ((pd.isna(dfFiltrado.iloc[instalacion]['FechaBaja'])) | (date < dfFiltrado.iloc[instalacion]['FechaBaja'].to_pydatetime()))):
                listaPotIns.append(dfFiltrado.iloc[instalacion]['PotInsKW'].item())
                listaDegradacion.append(CorreccionDegradacionWF(date.year, dfFiltrado.iloc[instalacion]['FechaAlta'].to_pydatetime().year))
            else:
                listaPotIns.append(0.0)
                listaDegradacion.append(0.0)
        arrayGeneracionRegion[instalacion, :] = np.array(listaPotIns) * arrayCF * np.array(listaDegradacion)
    arrayGeneracionRegionAgregadaMW = arrayGeneracionRegion.sum(axis = 0) * 0.001
    return arrayGeneracionRegionAgregadaMW.copy()

def RegionalWindCapacityModelFromGenericPowerCurve_v3(viento4D, lat2D, lon2D, listaDatetimes, dictIdxAlturas, sowisp_HR, nombreRegion, alpha = None, beta = 0.0):
    """
    Funcion que calcula el capacity factor de una determinada region a partir de
    informacion meteorologica y un dataframe de potencia instalada, usando
    una curva de potencia generica y teniendo en cuenta la evolucion de las
    alturas de los rotores de los parques eolicos y la degradacion de los mismos
    en funcion de los anyos.

    Parameters
    ----------
    viento4D: numpy.ndarray
        Array 4D de modulos de la velocidad horizontal del viento en un grid de
        latitudes y longitudes a varias alturas dadas en el que las dimensiones
        del array multidimensional son las siguientes: (time, height, lat, lon).
    lat2D: numpy.ndarray
        Array bidimensional de latitudes de los puntos grid con el mismo shape
        'espacial' del array de informacion meteo viento4D.
    lon2D: numpy.ndarray
        Analogo a lat2D pero con las longitudes del grid.
    listaDatetimes: list
        Lista de datetimes ordenados con las fechas de los instantes temporales 
        con los que se ha construido el array viento4D.
    dictIdxAlturas: dict
        Diccionario con clave alturas (en metros y en formato str, sin unidad)
        en el que se asocia, para cada altura, el indice en la dimension height
        del array viento4D.
    sowisp_HR: class
        Clase SOWISP_HighResolution (ver SOWISP_py3) que contiene la informacion 
        y metodos para trabajar con la base de datos de instalaciones eolicas 
        desagregadas.
    nombreRegion: str
        Nombre de la region de la que se va a calcular el capacity factor. Debe
        estar incluida en la columna "Provincia" del dataframe "dfPotIns".
    freqStr: str
        Cadena de string con la resolucion temporal de los datos introducidos 
        en viento4D. Los valores posibles mas comunes son: '1H' y '10T'.

    Returns
    -------
    arrayCapacityMW: numpy.ndarray
        Array unidimensional con la serie temporal de generacion regional de
        "nombreRegion", en MW.
    """
    # Lo primero es conocer las instalaciones que estan contenidas en la region 
    # declarada en el argumento nombreRegion y con periodo de generacion electrica 
    # dentro de listaDatetimes
    sowisp_HR.FiltraPorListaDeRegiones([nombreRegion], inplace = True)
    sowisp_HR.FiltraPorRangoDatetimes(listaDatetimes[0], listaDatetimes[-1], inplace = True)
    dfFiltrado = sowisp_HR.getDataFrame()
    sowisp_HR.ResetSOWISPHighRes()
    
    arrayGeneracionRegion = np.zeros((len(dfFiltrado), len(listaDatetimes))) - 9999.999
    for instalacion in range(len(dfFiltrado)):
        lat0, lon0, fechaAlta, fechaBaja, potIns = dfFiltrado.iloc[instalacion][['Latitud', 'Longitud', 'FechaAlta', 'FechaBaja', 'PotInsKW']].values.copy()
        alturaRotor = AlturaRotorEnFuncionYearInstalacion(fechaAlta.to_pydatetime().year)
        idxLat, idxLon = IdxMasCercanosLatLon2D(lat2D, lon2D, lat0.item(), lon0.item())
        arrayViento = viento4D[:, dictIdxAlturas[str(alturaRotor)], idxLat, idxLon].copy()
        if alpha is not None:
            arrayVientoCorregido = alpha * arrayViento + beta
        else:
            arrayVientoCorregido = arrayViento.copy()
        arrayCF = CFfromCurvaPotenciaGenerica(arrayVientoCorregido)
        listaPotIns = []
        listaDegradacion = []
        for date in listaDatetimes:
            if ((date >= fechaAlta.to_pydatetime()) & ((pd.isna(fechaBaja)) | (date < fechaBaja.to_pydatetime()))):
                listaPotIns.append(potIns.item())
                listaDegradacion.append(CorreccionDegradacionWF(date.year, fechaAlta.to_pydatetime().year))
            else:
                listaPotIns.append(0.0)
                listaDegradacion.append(0.0)
        arrayGeneracionRegion[instalacion, :] = np.array(listaPotIns) * arrayCF * np.array(listaDegradacion)
    arrayGeneracionRegionAgregadaMW = arrayGeneracionRegion.sum(axis = 0) * 0.001
    return arrayGeneracionRegionAgregadaMW.copy()

def SmartRegionalWindCapacityModel(viento3D, lat2D, lon2D, listaDatetimes, dictDfPotIns, archivoGeneracion, nombreProvincia = 'Narnia'):
    """
    Funcion que entrena el modelo regional Smart y devuelve unos parametros
    optimos.

    Parameters
    ----------
    viento3D: numpy.ndarray
        Array tridimensional de modulos de la velocidad horizontal del viento en
        un grid de latitudes y longitudes a una altura dada en el que la primera
        dimension es la temporal: (time, lat, lon).
    lat2D: numpy.ndarray
        Array bidimensional de latitudes de los puntos grid con el mismo shape
        'espacial' del array de informacion meteo viento3D.
    lon2D: numpy.ndarray
        Analogo a lat2D pero con las longitudes del grid.
    listaDatetimes: list
        Lista de datetimes con las fechas de los instantes temporales con los
        que se ha construido el array viento3D.
    dictDfPotIns: dict
        Diccionario de dataframes con la potencia instalada por municipios.
        Estos dataframes deben estar separados por years. Estos years, en string,
        deben ser las claves del diccionario. Preguntar a A. Jimenez-Garrote por
        esta base de datos de potencia instalada por year y municipio.
    archivoGeneracion: str
        Archivo .csv de generacion eolica provincial descargado desde la
        plataforma ESIOS de REE.
    nombreProvincia: str, optional
        Nombre de la provincia con la que se esta entrenando el modelo. Es
        simplemente para escribir el nombre en la cabecera del archivo de texto
        que se genera automaticamente.

    Returns
    -------
    popt: tuple
        Tupla de floats con los parametros optimos de la Sigmoidal de la
        provincia a estudiar. Adicionalmente tambien se genera un archivo .txt
        con estos mismos valores para no tener que entrenar el modelo mas de una
        vez.
    """
    # Comprobacion de los shapes de los datos de entrada
    if lat2D.shape != lon2D.shape:
        raise ValueError('Los arrays bidimensionales de latitud y longitud deben tener el mismo shape')
    if viento3D.shape[1:] != lat2D.shape:
        raise ValueError('La informacion meteo y la referente al grid debe tener el mismo shape')
    if viento3D.shape[0] != len(listaDatetimes):
        raise ValueError('La lista de datetimes no coincide con la primera dimension de los array meteo u y v')

    # Se calcula la potencia instalada en la provincia en funcion del year a partir del diccionario de dataframes de la potencia instalda
    dictPotInsProvincia = {}
    for year in dictDfPotIns.keys():
        dictPotInsProvincia[year] = dictDfPotIns[year]['PotenciaInstaladaKW'].sum()

    # Se extrae la informacion de generacion provincial y se anyade una columna con la generacion en CF
    dataframeGeneracion = pd.read_csv(archivoGeneracion, sep = ';',
        usecols = ['value', 'datetime'], index_col = 'datetime',
        parse_dates = True, dtype = {'value': float, 'datetime': object},
        date_parser = lambda x: datetime.strptime(x[:-6], '%Y-%m-%dT%H:%M:%S') - timedelta(hours = int(x[-4])))
    arrayPotIns = np.zeros(len(dataframeGeneracion))
    iterator = 0
    for date in dataframeGeneracion.index:
        if np.isin(tuple(dictPotInsProvincia.keys()), str(date.year))[0]:
            arrayPotIns[iterator] = dictPotInsProvincia[str(date.year)] / 1000.0
        else:
            arrayPotIns[iterator] = np.nan
        iterator += 1
    dataframeGeneracion['valueCF'] = (dataframeGeneracion['value'].values / arrayPotIns)

    # Itera en las fechas de entrenamiento (listaDatetimes)
    yearActual = 1900
    arrayVelocidadPonderada = np.zeros(len(listaDatetimes)) - 9999.999
    iterator = 0
    for date in listaDatetimes:
        # Si la fecha de entrenamiento cambia de anyo, entonces calculo un dataframe con los indices lat lon de los municipios que contienen potencia instalada para ese anyo
        if date.year != yearActual:
            idxMasCercanosLatLonMunicipio = np.array(([IdxMasCercanosLatLon2D(lat2D,
                lon2D, dictDfPotIns[str(date.year)].loc[municipio]['Latitud'],
                dictDfPotIns[str(date.year)].loc[municipio]['Longitud']) for municipio in tuple(dictDfPotIns[str(date.year)].index.values)]))
            dataframeIndicesMunicipios = pd.DataFrame(idxMasCercanosLatLonMunicipio,
                index = dictDfPotIns[str(date.year)].index,
                columns = ['idLatitud', 'idLongitud'], dtype = int)
            yearActual = date.year
        # Una vez que tengo localizadas las instalaciones, se recorren en bucle para caluclar la suma ponderada en funcion de la potencia instalada
        sumaPonderada = 0.0
        for municipio in tuple(dataframeIndicesMunicipios.index.values):
            idxLat0, idxLon0 = dataframeIndicesMunicipios.loc[municipio]
            sumaPonderada += ((dictDfPotIns[str(date.year)].loc[municipio]['PotenciaInstaladaKW'] / dictPotInsProvincia[str(date.year)]) * viento3D[iterator, idxLat0, idxLon0].copy())
        arrayVelocidadPonderada[iterator] = sumaPonderada
        iterator += 1

    # Combinacion de bases de datos de generacion y meteo
    dataframeMeteo = pd.DataFrame(arrayVelocidadPonderada.copy(),
        index = listaDatetimes, columns = ['vPonderada'], dtype = float)
    dataframeMeteoHorario = dataframeMeteo.resample('1H').mean().copy()
    datosComunes = dataframeGeneracion.merge(dataframeMeteoHorario, left_index = True, right_index = True, validate = 'one_to_one') # plantear añadir sort = True en un hipotético set aleatorio de fechas de entrenamiento

    # Ajuste y almacenamiento de parametros optimos
    popt, pcov = curve_fit(Sigmoidal, datosComunes['vPonderada'].values,
        datosComunes['valueCF'].values, p0 = (1, 10, 3))
    np.savetxt(f'ParametrosSigmoidalSmart{nombreProvincia}.txt',
        popt.reshape(1,3), delimiter = '\t',
        header = f'Modelo Smart para la provincia de {nombreProvincia}.\n El modelo se ha entrenado con fechas comprendidas en el intervalo {datetime.strftime(listaDatetimes[0], "%Y-%m-%d")} - {datetime.strftime(listaDatetimes[-1], "%Y-%m-%d")}.\n En total, se han empleado {len(listaDatetimes)} registros.\n Parametros optimos (A\t beta\t alpha): ')
    return popt

def SmartRegionalWindPowerCF(viento2D, lat2D, lon2D, archivoPotIns, archivoParametros):
    """
    Empleo del modelo Smart de transformacion a generacion eolica
    provincial. Importante: es necesario entrenar primero el modelo
    SmartRegionalWindCapacityModel de esta misma libreria con la misma
    informacion meteorologica que se va a usar como datos de entrada
    para obtener los parametros A, beta y alpha que mejor se ajustan a
    la funcion Sigmoidal y que se introducen a través del
    archivoParametros.

    Parameters
    ----------
    viento3D: numpy.ndarray
        Array bidimensional de componentes horizontal zonal del viento
        en un grid de latitudes y longitudes a una altura dada.
    lat2D: numpy.ndarray
        Array bidimensional de latitudes de los puntos grid con el mismo
        shape de los arrays de informacion meteo u y v.
    lon2D: numpy.ndarray
        Analogo a lat2D pero con las longitudes del grid.
    archivoPotIns: str
        Nombre del archivo .csv que contiene la informacion de la
        potencia eolica instalada provincial por municipios formateado
        tal que la cabecera es:
        Municipio;PotenciaInstaladaKW;Latitud;Longitud.
        (Preguntar a Antonio Jiménez-Garrote por el programa
        SetLatLonToInstalledPowerDatabase.py)
    archivoParametros: str
        Nombre del archivo .txt con los valores de los parámetros que
        mejor se ajustan a la funcion sigmoidal para la provincia de
        estudio. Este archivo es generado automaticamente al entrenar el
        modelo.

    Returns
    -------
    generacionMW: float
        Valor en MW de la producción del parque.
    """
    # Comprobacion de los shapes de los datos de entrada
    if lat2D.shape != lon2D.shape:
        raise ValueError('Los arrays bidimensionales de latitud y longitud deben tener el mismo shape')
    if viento2D.shape != lat2D.shape:
        raise ValueError('La informacion meteo y la referente al grid debe tener el mismo shape')

    # Se carga la informacion referente a la potencia instalada y a los parametros optimos del modelo
    dataframePotIns = pd.read_csv(archivoPotIns, sep = ';',
        dtype = {'Municipio': str, 'PotenciaInstaladaKW': float,
            'Latitud': float, 'Longitud': float}, index_col = 'Municipio')
    potInsProvincia = dataframePotIns['PotenciaInstaladaKW'].sum()
    A, beta, alpha = np.loadtxt(archivoParametros, delimiter = '\t')

    # Calcula los indices de los puntos grid de los arrays de entrada mas cercanos a las localizaciones del centroide de los municipios involucrados
    idxMasCercanosLatLonMunicipio = np.array(([IdxMasCercanosLatLon2D(lat2D,
            lon2D, dataframePotIns.loc[municipio]['Latitud'],
            dataframePotIns.loc[municipio]['Longitud']) for municipio in tuple(dataframePotIns.index.values)]))
    dataframePotIns['idLatitud'] = idxMasCercanosLatLonMunicipio[:, 0]
    dataframePotIns['idLongitud'] = idxMasCercanosLatLonMunicipio[:, 1]
    #nMunicipios = len(dataframePotIns.index)

    # Aplica el modelo Smart con los parametro optimos
    sumaPonderada = 0.0
    for municipio in tuple(dataframePotIns.index.values):
        idxLat0 = int(dataframePotIns.loc[municipio]['idLatitud']) # no se extrae el valor en entero aunque el dataframe tenga dtype int
        idxLon0 = int(dataframePotIns.loc[municipio]['idLongitud']) # no se extrae el valor en entero aunque el dataframe tenga dtype int
        sumaPonderada += (dataframePotIns.loc[municipio]['PotenciaInstaladaKW'] / potInsProvincia) * viento2D[idxLat0, idxLon0].copy()
    generacionCF = Sigmoidal(sumaPonderada, A, beta, alpha)
    return generacionCF

def FlorisModelFarmPowerMW(configJSON, archivoLocations, lat0, lon0, u, v):
    """
    Implementacion del modelo Floris como funcion para calcular la
    generacion eolica en MW del parque. Informacion disponible en:
    https://floris.readthedocs.io/en/master/

    Parameters
    ----------
    configJSON: str
        Nombre del archivo .json necesario para inicializar el modelo
        Floris.
    archivoLocations: str
        Nombre del archivo .tex que contiene las coordenadas latitud,
        longitud de los aerogeneradores del parque. La cabecera y, por
        tanto, formato de dicho archivo debe ser (ID, Latitud, Longitud)
        o simplemente (Latitud, Longitud).
    lat0: float
        Valor de la latitud del origen de coordenadas a partir del cual
        se calcularan las coordenadas relativas de los aerogeneradores.
    lon0: float
        Analogo a lat0 pero con el valor de longitud.
    u: numpy.ndarray
        Array bidimensional de componentes horizontal zonal del viento
        en un grid de latitudes y longitudes a una altura dada.
    v: numpy.ndarray
        Analogo a u pero con la componente horizontal meridional.

    Returns
    -------
    generacionMW: float
        Valor en MW de la producción del parque.
    """
    fi = wfct.floris_interface.FlorisInterface(configJSON)
    x, y = CoordenadasRelativasGridAerogen(archivoLocations, lat0, lon0)
    ws = WU.VelocidadViento(u.copy(), v.copy())
    wd = WU.DireccionViento(u.copy(), v.copy())
    wsPromedio = np.mean(ws.copy())
    wdPromedio = WU.MediaDireccionViento(wd.copy())
    fi.reinitialize_flow_field(layout_array = [x, y], wind_speed = wsPromedio,
        wind_direction = wdPromedio)
    fi.calculate_wake()
    generacionMW = fi.get_farm_power() / (1000.0 * 1000.0)
    return generacionMW
#-----------------------------------------------------------------------
