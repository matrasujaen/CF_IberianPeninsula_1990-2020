# -*- coding: utf-8 -*-
"""
Created on mar 06 may 2021 10:29:07 CET

@author: Antonio Jiménez-Garrote <agarrote@ujaen.es>

La finalidad de esta libreria es la de extraer aquellas funciones que
originalmente estaban en la libreria WindCapacityModels_py3.py y que
son utiles a la hora de trabajar con datos de viento.
"""
#-------------------------------------------------------------------LIBRARIES---
import numpy as np
import pandas as pd
import multiprocessing
import warnings
from datetime import datetime
from geopy import distance as GeoDist
from sklearn.metrics import mean_squared_error
from scipy.interpolate import interp1d
from matplotlib import pyplot as plt
#-------------------------------------------------------------------------------

#-------------------------------------------------------------------CONSTANTS---

#-------------------------------------------------------------------------------

#-------------------------------------------------------------------FUNCTIONS---
def Get1dIndex(row, col, numCols):
    """
    Funcion made in Javi que asigna un unico valor a una celda de un array
    bidimensional (numRows, numCols).

    Parameters
    ----------
    row: int
        Indice de la fila de un array bidimensional.
    col: int
        Indice de la columna de un array bidimensional.
    numCols: int
        Numero total de columnas del array bidimensional.

    Returns
    -------
    indice: int
        Valor del indice unidimensional del array bidimensional al que
        corresponden los indices fila y columna 'row' y 'col', respectivamente.
    """
    indice = (row * numCols) + col
    return indice

def FuncionInterpolacionEnAltura(args):
    """
    Funcion made in Javi que realiza una interpolacion (y extrapolacion si fuese
    necesario) de un array unidimensional con el modulo de la velocidad
    horizontal del viento, v, a unas alturas determinadas (alturas) utilizando
    para ello el array unidimensional de alturas a las que hacen referencia el
    array 'v', z. Adicionalmente, se requiere un entero con la informacion del
    indice que contiene informacion de la posicion dentro de un grid
    bidimensional (ver 'Get1dIndex') para posteriormente reordenar todas las
    celdas grid. El tipo de interpolacion tambien es necesario.

    Parameters
    ----------
    args: object
        Entero con el indice de una celda grid de un array bidimensional (idx),
        array unidimensional de modulo de la velocidad horizontal del viento a
        unas alturas determinadas (v), array unidimensional con dichas alturas
        (z), lista de alturas a las que interpolar (alturas) y string con el
        tipo de interpolacion a realizar (ver documentacion de scipy).

    Returns
    -------
    vInterpolado: tuple
        tupla con los valores del modulo de la velocidad horizontal del viento a
        las alturas especificadas donde el primer registro es el indice 'idx'.
    """
    idx, v, z, alturas, tipoInterp = args
    funcionInterpolacionV = interp1d(
        z,
        v,
        kind = tipoInterp,
        bounds_error = False,
        fill_value = 'extrapolate'
    )
    vInterpolado = [float(funcionInterpolacionV(h0)) for h0 in alturas]
    vInterpolado.insert(0, idx)
    return tuple(vInterpolado)

def PlausibleValuesTest(arrayValores, isDireccion, vmin = 0, vmax = 75, dirVmin = 0.0, dirVmax = 360.0):
    """
    Test de valores posibles. En velocidad del viento esta claro que el limite
    inferior es 0 pero el limite superior no esta del todo claro: (WMO, 2017)
    propone un valor de 75 m/s. Jimenez et al., 2010) proponen unos valores de
    50 o 60 m/s para estaciones en superficie. Para el caso de la direccion del
    viento no hay discusion: valor € [0, 360].

    Parameters
    ----------


    Returns
    -------

    """
    if isDireccion == True:
        vmin, vmax = (dirVmin, dirVmax)
    arrayFlags = np.array(np.repeat('Pass', len(arrayValores)), dtype = 'U7')
    arrayFlags[((arrayValores < vmin) & (arrayValores > vmax))] = 'Fail'
    #arrayFlags[((arrayValores >= vmin) & (arrayValores <= vmax))] = 'Suspect'
    arrayFlags[np.isnan(arrayValores)] = 'Missing'
    return arrayFlags

def ExtremeValuesTest(arrayValores):
    """
    Test de diferencias entre valores extremos en la distribucion del viento.
    SOLO valido para el modulo de la VELOCIDAD del viento. Consiste en calcular
    la diferencia entre el maximo y el segundo maximo. Si dicha diferencia es
    mayor que el propio segundo maximo, el primer maximo se marca como
    sospechoso. Este procedimiento se repite iterativamente hasta que no se
    cumpla la condicion.

    Parameters
    ----------


    Returns
    -------

    """
    arrayValoresIdx = np.array(
        [(value, index, 'Pass') for value, index in zip(arrayValores, range(len(arrayValores)))],
        dtype = np.dtype({'names': ('valor', 'idx', 'test'), 'formats': (float, int, 'U7')})
    )
    arrayValoresIdxOrderValor = np.flipud(
        np.sort(arrayValoresIdx, order = 'valor')
    )
    nNaNs = len(arrayValores[np.isnan(arrayValores)])
    iterator = nNaNs + 1 # para empezar desde los que no son NaN
    diff = arrayValoresIdxOrderValor[iterator - 1]['valor'] - arrayValoresIdxOrderValor[iterator]['valor']
    while diff > arrayValoresIdxOrderValor[iterator]['valor']:
        diff = arrayValoresIdxOrderValor[iterator]['valor'] - arrayValoresIdxOrderValor[iterator + 1]['valor']
        iterator += 1
    arrayValoresIdxOrderValor[:nNaNs]['test'] = 'Missing' # los nan se marcan como perdidos
    arrayValoresIdxOrderValor[nNaNs:(iterator - 1)]['test'] = 'Suspect'
    arrayValoresIdxOrderIdx = np.sort(arrayValoresIdxOrderValor, order = 'idx')
    arrayFlags = arrayValoresIdxOrderIdx['test'].copy()
    return arrayFlags

def EsPersistencia(dfRecorte, difMin, minNdata, isDir = False):
    """
    Funcion para determinar si un conjunto de datos ha variado lo suficiente.
    Para evaluar si un dato es sospechoso o no, se coge el conjunto de datos de
    la variable valores, se identifican aquellos registros que no son vientos en
    calma (valores superiores a vCalma). Si la cantidad de registros no en calma
    supera los minNdata datos, entonces se calcula la máxima diferencia entre
    dichos registros. Si la maxima diferencia es inferior o igual a difMin, se
    marca el ultimo valor del conjunto como sospechoso (codigoBool = 1.0).

    Parameters
    ----------


    Returns
    -------

    """
    arrayValores = dfRecorte.values.squeeze()
    if (len(arrayValores[np.isnan(arrayValores) == False]) >= minNdata): # si los valores de velocidad del viento que no son en calma es superior al minimo de datos requeridos por el test...
        if isDir == True: # si el dfRecorte contiene valores de direccion del viento hay que aplicar la distancia circular...
            diffValores = DistanciaCircular(arrayValores, np.repeat(np.nanmax(arrayValores), len(arrayValores))).max().item()
        else: # ... en cambio si son valores de velocidad del viento se hace la resta normal y corriente
            diffValores = abs(np.nanmax(arrayValores) - np.nanmin(arrayValores))
        # Finalmente se evalua el test en funcion de diffValores
        if diffValores < difMin:
            codigoBool = 1.0
        else:
            codigoBool = 0.0
    else: # ... si no hay suficientes valores no en calma o no NaN, no se puede aplicar el test
        codigoBool = 0.5
    return codigoBool

def EsConstante(dfRecorte, minValoresConstantes, nDatos):
    """
    Funcion para determinar si un conjunto de datos es constante en el tiempo.
    El conjunto de valores resultara sospechoso si en dicho conjunto hay minimo
    de minNdata datos y todos son iguales.

    Parameters
    ----------


    Returns
    -------

    """
    diffMin = 0.0
    valores = np.flipud(dfRecorte.values.squeeze()) # coge los numeros anteriores asi que voy a darle la vuelta al array y voy a ir comprobando que no sean iguales de atras a adelante
    valoresNoNaN = valores[np.isnan(valores) == False].copy() # le quito los NaN
    if len(valoresNoNaN) < minValoresConstantes: # si el conjunto de datos sin NaN es menor a minValoresConstantes no puedo evaluar nada
        codigoBool = 0.5
    else:
        iterator = 0 # inicializo el primer elemento del array dado la vuelta
        diff = 0.0 # inicializo la diferencia
        while ((iterator < (len(valoresNoNaN) - 1)) & (diff == diffMin)): # voy a iterar hasta que llegue a nDatos - 1 (porque si no, el elemento nDatos del array valores no existe, se sale de la dimension) o la diferencia entre elementos consecutivos sea distinta de diffMin (lo que quiere decir que ya no son constantes)
            diff = valoresNoNaN[iterator + 1] - valoresNoNaN[iterator]
            iterator += 1
        # una vez terminado el bucle, me fijo en donde se paro iterator
        if iterator == (nDatos - 1): # si es igual a nDatos - 1 es porque hay nDatos elementos consecutivos con el mismo valor que el inicial y por tanto debe fallar el test
            codigoBool = -1.0
        elif iterator >= minValoresConstantes: # si es menor de (nDatos - 1) pero mayor o igual a (minValoresConstantes - 1) es porque es sospechoso
            codigoBool = 1.0
        else:
            codigoBool = 0.0 # en cualquier otro caso es un dato bueno y pasa el test
    return codigoBool

def TransformaNumeroToFlag(valores):
    """
    Funcion que transforma el numero float que devuelven las funciones
    EsPersistencia y EsConstante en los flags Pass y Suspect.

    Parameters
    ----------


    Returns
    -------

    """
    arrayFlags = np.array(np.repeat('', len(valores)), dtype = 'U7')
    arrayFlags[valores == 0.0] = 'Pass'
    arrayFlags[valores == 1.0] = 'Suspect'
    arrayFlags[valores == -1.0] = 'Fail'
    return arrayFlags

def PersistenceTest(dfDatos, isDir, nDatosVentana = 60, minNdata = 30, vmin = 0.7, dirVmin = 5.0, vCalma = 0.5, dfVel = None):
    """
    Este test no introduce ningun tipo de etiqueta para los vientos en calma
    (valor inferior a 0.5 m/s (vease EsPersistencia)). De forma provisional, la
    funcion esta pensada para datos de resolucion temporal 10-minutal. Si en una
    ventana movil de 1 h (6 registros) hay minimo 3 datos distintos de NaN y la
    maxima diferencia entre todos ellos es menor o igual a 0.7 m/s o menor a 5º,
    se considereran los mismos como sospechosos.

    Parameters
    ----------


    Returns
    -------

    """
    namecolDatos = dfDatos.columns[0]
    if isDir == True: # si es direccion cambio el umbral de diferencias entre valores
        vmin = dirVmin
        if dfVel is not None: # y tengo la opcion de usar la informacion del modulo del viento uniendolo al dataframe de datos. Ver funcion EsPersistencia para saber como procede con el dataframe
            dfDatos = dfDatos.merge(dfVel, left_index = True, right_index = True).copy()
            namecolVel = dfDatos.columns[1]
            dfDatosNoCalma = pd.DataFrame(data = None, index = dfDatos.index, columns = list([dfDatos.columns[0]]), dtype = float) # dataframe auxiliar
            dfDatosNoCalma[dfDatos[namecolVel] > vCalma] = pd.DataFrame(dfDatos[dfDatos[namecolVel] > vCalma][namecolDatos]).copy() # pongo como NaN los vientos inferiores a vCalma
        else:
            dfDatosNoCalma = dfDatos.copy()
    else: # este else parece no necesario pero si que lo es puesto que se emplea la resta normal y no la distancia ciruclar como ocurre con la direccion
        dfDatosNoCalma = pd.DataFrame(data = None, index = dfDatos.index, columns = list([dfDatos.columns[0]]), dtype = float) # dataframe auxiliar
        dfDatosNoCalma[dfDatos[namecolDatos] > vCalma] = pd.DataFrame(dfDatos[dfDatos[namecolDatos] > vCalma][namecolDatos]).copy() # pongo como NaN los vientos inferiores a vCalma
    dfRolling = dfDatosNoCalma.rolling(
        window = nDatosVentana - 1,
        min_periods = minNdata,
        closed = 'both'
        ).apply(lambda x: EsPersistencia(x, vmin, minNdata, isDir = isDir))
    arrayFlags = TransformaNumeroToFlag(dfRolling[namecolDatos].values)
    return arrayFlags

def FlatLineTest(dfDatos, isDireccion, nDatosVentana = 7, minNdata = 3, dirNdatos = 40, dirMinDatos = 20):
    """
    Este test introduce la etiqueta de Fallo cuando aparecen, en una ventana
    movil de 7 registros (40 para la direccion del viento) con identico valor.
    Si en esa ventana aparecen de 3 a 6 registros y tienen el mismo valor, se
    etiqueta el valor como Sospechoso.

    Parameters
    ----------


    Returns
    -------

    """
    namecol = dfDatos.columns[0]
    if isDireccion == True:
        nDatosVentana = dirNdatos
        minNdata = dirMinDatos
    dfRolling = dfDatos.rolling(
        window = nDatosVentana - 1,
        min_periods = 2,
        closed = 'both'
        ).apply(lambda x: EsConstante(x, minNdata, nDatosVentana))
    arrayFlags = TransformaNumeroToFlag(dfRolling[namecol].values)
    return arrayFlags

def AbnormalVariations(dfDatos, nDays = 30):
    """
    Test que realiza periodos moviles de 30 dias y calcula la desviacion
    estandar de cada periodo asi como la desviacion estandar promedio de todos
    los periodos. Si la desviacion estandar de un periodo difiere mas de 4 veces
    la promediada para todos los periodos, los registros contenidos en dicho
    registros se clasifican como sospechosos. Solo para la velocidad del viento
    """
    dictDatos1H = {'10T': 6, 'H': 1}
    nDatosMin = int(0.5 * dictDatos1H[dfDatos.index.freq.freqstr] * 24 * nDays) # 50% * nDatos / 1H * 24 H/dia * nDias dia
    namecol = dfDatos.columns[0]
    dfRolling = dfDatos.rolling(
        window = f'{nDays}D', # coge el intervalo, por ejemplo: '2020-02-15 00:00':'2020-03-16 00:00', en el que han pasado 30 dias naturales y ambos en la misma hora
        min_periods = nDatosMin,
        closed = 'both'
        ).apply(np.nanstd)
    arrayFlags = np.array(np.repeat('', len(dfDatos)), dtype = 'U7')
    if len(dfRolling.dropna()) != 0: # es posible que ningun registro del rolling tenga un numero minimo de datos para hacer nanstd
        mediaDesv = np.nanmean(dfRolling[namecol].values)
        stdDesv = np.nanstd(dfRolling[namecol].values)
        # arrayFlags = np.array(np.repeat('', len(dfDatos)), dtype = 'U7')
        arrayFlags[((dfRolling[namecol].values >= (mediaDesv - 4 * stdDesv)) & (dfRolling[namecol].values <= (mediaDesv + 4 * stdDesv)))] = 'Pass'
        arrayFlags[((dfRolling[namecol].values < (mediaDesv - 4 * stdDesv)) | (dfRolling[namecol].values > (mediaDesv + 4 * stdDesv)))] = 'Suspect'
    return arrayFlags

def SystematicErrorsTest(dfDatos, nDays = 30):
    """
    Test que realiza periodos moviles de 30 dias y calcula la media de cada
    periodo asi como el promedio de todas las medias de los periodos. Si la
    media de un periodo di
    """
    dictDatos1H = {'10T': 6, 'H': 1}
    nDatosMin = int(0.5 * dictDatos1H[dfDatos.index.freq.freqstr] * 24 * nDays) # 50% * nDatos / 1H * 24 H/dia * nDias dia
    namecol = dfDatos.columns[0]
    dfRolling = dfDatos.rolling(
        window = f'{nDays}D', # coge el intervalo, por ejemplo: '2020-02-15 00:00':'2020-03-16 00:00', en el que han pasado 30 dias naturales y ambos en la misma hora
        min_periods = nDatosMin,
        closed = 'both'
        ).apply(np.nanmean)
    arrayFlags = np.array(np.repeat('', len(dfDatos)), dtype = 'U7')
    if len(dfRolling.dropna()) != 0: # es posible que ningun registro del rolling tenga un numero minimo de datos para hacer nanstd
        mediaPromedio = np.nanmean(dfRolling[namecol].values)
        stdPromedio = np.nanstd(dfRolling[namecol].values)
        # arrayFlags = np.array(np.repeat('', len(dfDatos)), dtype = 'U7')
        arrayFlags[((dfRolling[namecol].values >= (mediaPromedio - 4 * stdPromedio)) & (dfRolling[namecol].values <= (mediaPromedio + 4 * stdPromedio)))] = 'Pass'
        arrayFlags[((dfRolling[namecol].values < (mediaPromedio - 4 * stdPromedio)) | (dfRolling[namecol].values > (mediaPromedio + 4 * stdPromedio)))] = 'Suspect'
    return arrayFlags

def RateOfChange(valores):
    """
    """
    q3, q1 = np.percentile(valores[np.isnan(valores) == False], [75, 25])
    iqr = q3 - q1
    difValores = abs(valores[1:] - valores[:-1])
    difValoresCompleto = np.concatenate((difValores.copy(), np.array([np.nan]))) # con el ultimo registro no se puede hacer una diferencia. Prefiero poner el ultimo como NaN porque si lo pongo al principio, el primer dato siempre cumplira la condicion de no ser evaluado por 3 o mas tests
    arrayFlags = np.array(np.repeat('Pass', len(difValoresCompleto)), dtype = 'U7')
    arrayFlags[np.isnan(difValoresCompleto)] = '' # es distinto no poder ser evaluado por estar al lado de un NaN...
    arrayFlags[np.isnan(valores)] = 'Missing' # que ser un NaN
    arrayFlags[difValoresCompleto >= (3.0 * iqr)] = 'Fail'
    arrayFlags[((difValoresCompleto >= (2.0 * iqr)) & (difValoresCompleto < (3.0 * iqr)))] = 'Suspect'
    return arrayFlags

def StepTest(valores, difMax = 20):
    """
    """
    difValores = abs(valores[1:] - valores[:-1])
    difValoresCompleto = np.concatenate((difValores, np.array([np.nan]))) # con el ultimo registro no se puede hacer una diferencia. Prefiero poner el ultimo como NaN porque si lo pongo al principio, el primer dato siempre cumplira la condicion de no ser evaluado por 3 o mas tests
    arrayFlags = np.array(np.repeat('Pass', len(difValoresCompleto)), dtype = 'U7')
    arrayFlags[np.isnan(difValoresCompleto)] = '' # es distinto no poder ser evaluado por estar al lado de un NaN...
    arrayFlags[np.isnan(valores)] = 'Missing' # que ser un NaN
    arrayFlags[difValoresCompleto >= difMax] = 'Fail'
    return arrayFlags

def SetFlag(arrayStr):
    if np.isin('Missing', arrayStr) == True:
        intFlag = 9
    elif np.isin('Fail', arrayStr) == True:
        intFlag = 4
    elif len(arrayStr[arrayStr == '']) >= 3:
        intFlag = 0
    elif np.isin('Suspect', arrayStr) == True:
        intFlag = 2
    else:
        intFlag = 1
    return str(intFlag)

def IsolatedPass(arrayValoresOriginal):
    arrayValores = arrayValoresOriginal.copy()
    for iterator in range(25, len(arrayValores) - 30):
        if arrayValores[iterator] == 'Pass':
            arrayUnico = np.unique(np.concatenate((arrayValores[(iterator - 3):iterator].copy(), arrayValores[(iterator + 1):(iterator + 4)].copy())))
            valorUnico = arrayUnico[0]
            if ((len(arrayUnico) == 1) & ((valorUnico == 'Fail') | (valorUnico == 'Suspect'))):
                arrayValores[iterator] = valorUnico
            elif arrayValores[iterator + 1] == 'Pass':
                arrayUnico = np.unique(np.concatenate((arrayValores[(iterator - 5):iterator].copy(), arrayValores[(iterator + 2):(iterator + 7)].copy())))
                valorUnico = arrayUnico[0]
                if ((len(arrayUnico) == 1) & ((valorUnico == 'Fail') | (valorUnico == 'Suspect'))):
                    arrayValores[iterator:(iterator + 2)] = valorUnico
                elif arrayValores[iterator + 2] == 'Pass':
                    arrayUnico = np.unique(np.concatenate((arrayValores[(iterator - 10):iterator].copy(), arrayValores[(iterator + 3):(iterator + 13)].copy())))
                    valorUnico = arrayUnico[0]
                    if ((len(arrayUnico) == 1) & ((valorUnico == 'Fail') | (valorUnico == 'Suspect'))):
                        arrayValores[iterator:(iterator + 3)] = valorUnico
                    elif arrayValores[iterator + 3] == 'Pass':
                        arrayUnico = np.unique(np.concatenate((arrayValores[(iterator - 15):iterator].copy(), arrayValores[(iterator + 4):(iterator + 19)].copy())))
                        valorUnico = arrayUnico[0]
                        if  ((len(arrayUnico) == 1) & ((valorUnico == 'Fail') | (valorUnico == 'Suspect'))):
                            arrayValores[iterator:(iterator + 4)] = valorUnico
                        elif arrayValores[iterator + 4] == 'Pass':
                            arrayUnico = np.unique(np.concatenate((arrayValores[(iterator - 25):iterator].copy(), arrayValores[(iterator + 5):(iterator + 30)].copy())))
                            valorUnico = arrayUnico[0]
                            if ((len(arrayUnico) == 1) & ((valorUnico == 'Fail') | (valorUnico == 'Suspect'))):
                                arrayValores[iterator:(iterator + 5)] = valorUnico
    return arrayValores

def QualityAssurance(dfDatos, freq, dfVel4Dir = None, isDireccion = False, verbose = False):
    """
    Sacado de Ramon J, Lledó L, Pérez-Zanon N, Soret A, Doblas-Reyes FJ. The
    tall tower dataset: a unique initiative to boost wind energy research. Earth
    Syst. Sci. Data 2020;(12):429–39. y Jiménez, P. A., González-Rouco, J. F.,
    Navarro, J., Montávez, J. P., & García-Bustamante, E. (2010). Quality
    assurance of surface wind observations from automated weather stations.
    Journal of atmospheric and Oceanic Technology, 27(7), 1101-1122. El test de
    Icing no lo he implementado
    """
    dfCompleto = dfDatos.resample(freq).asfreq()
    colname = dfCompleto.columns[0]
    if isDireccion == True:
        if dfVel4Dir is None:
            warning.warn('El test de persistencia no tiene en cuenta los vientos en calma', UserWarning)
            dfVelocidadesCompleto = None
        else:
            dfVelocidadesCompleto = dfVel4Dir.resample(freq).asfreq()
    else:
        dfVelocidadesCompleto = None
    dfTests = pd.DataFrame(
    	data = None,
    	index = dfCompleto.index,
    	columns = ['PlT', 'ExT', 'PeT', 'FlT', 'AbV', 'SyT', 'RaC', 'StT', 'Flag', 'IsP'],
    	dtype = str
    )
    dfTests['PlT'] = PlausibleValuesTest(dfCompleto[colname].values, isDireccion)
    dfTests['PeT'] = PersistenceTest(dfCompleto, dfVel = dfVelocidadesCompleto, isDir = isDireccion)
    dfTests['FlT'] = FlatLineTest(dfCompleto, isDireccion)
    if isDireccion == False:
        dfTests['ExT'] = ExtremeValuesTest(dfCompleto[colname].values)
        dfTests['AbV'] = AbnormalVariations(dfCompleto)
        dfTests['SyT'] = SystematicErrorsTest(dfCompleto)
        dfTests['RaC'] = RateOfChange(dfCompleto[colname].values)
        dfTests['StT'] = StepTest(dfCompleto[colname].values)
    dfTests['Flag'] = dfTests.apply(lambda x: SetFlag(x), axis = 1)
    dfTests['IsP'] = IsolatedPass(dfTests['Flag'].values)
    if verbose == True:
        fig = plt.figure(1, figsize = (11.5, 8), dpi = 300, clear = True)
        ax = fig.add_axes([0.15, 0.15, 0.8, 0.8])
        dictName = {'1': 'Pass', '0': 'NE3T', '2': 'Suspect', '4': 'Fail', '9': 'Missing'}
        dictColor = {'1': 'tab:green', '0': 'tab:gray', '2': 'tab:orange', '4': 'tab:red', '9': 'white'}
        dictSize = {'1': 0.5, '0': 2, '2': 2, '4': 2, '9': 2}
        for flag in dictColor.keys():
            ax.plot_date(dfCompleto.index[dfTests['IsP'] == flag], dfCompleto[dfTests['IsP'] == flag].values, color = dictColor[flag], markersize = dictSize[flag], label = f'{dictName[flag]} {np.round(len(dfCompleto[dfTests["IsP"] == flag])/len(dfCompleto) * 100, 1)}%')
            ax.legend(prop = {'size': 8, 'family': 'Liberation Sans'})
            ax.xaxis.set_tick_params(labelsize = 7)
            ax.yaxis.set_tick_params(labelsize = 7)
            ax.set_xlabel(f'{dfCompleto.index.name}', fontsize = 10, family = 'Liberation Sans')
            ax.set_ylabel(f'{colname}', fontsize = 10, family = 'Liberation Sans')
            #plt.plot_date(dfCompleto.index[dfTests['Flag'] == flag], dfCompleto[dfTests['Flag'] == flag].values, color = dictColor[flag], markersize = dictSize[flag])
        fig.show()
    return dfTests

def VelocidadViento(u, v):
    """
    Funcion que calcula la velocidad horizontal del viento a partir de las
    componentes horizontales del mismo mediante el cálculo del módulo de un
    vector a partir de sus componentes.

    Parameters
    ----------
    u: numpy.ndarray
        Array de componentes horizontales zonales del viento.
    v: numpy.ndarray
        Array de componentes horizontales meridionales del viento.

    Returns
    -------
    velocidadViento: numpy.ndarray
        Array con el modulo de las velocidades horizontales del viento.
    """
    if u.shape != v.shape:
        raise ValueError('Los arrays deben tener el mismo shape')
    velocidadViento = np.sqrt((u.copy())**2 + (v.copy())**2)
    return velocidadViento.copy()

def DireccionViento(u, v):
    """
    Sacado de motionVectorBase_py3.py aunque se ha modificado para trabajar todo
    en vista mapa. Info: Obtiene la direccion del viento a partir de las
    componentes horizontales del viento.

    De acuerdo con el estandar de meteorologia, el origen se halla en el norte y
    el sentido de crecimiento es siguiendo las agujas del reloj. Ademas, esta
    direccion indica el origen de donde proviene el viento.

    Para proceder con este cambio se llevan a cabo los siguientes pasos.
        i) Calcular el angulo entre 'u' y 'v' como en trigonometria. Esta
        operacion no devuelve angulos mayores que np.pi, sino angulos negativos.
        Para que solo se devuelvan angulos positivos le sumamos una vuelta
        completa, esto es, sumamos (2 * np.pi). No hay que preocuparse por que
        se exceda de una vuelta completa, ya que posteriormemnte el resultado
        "se recortara" a una vuelta. Por tanto:
            np.arctan2(v, u) + (2 * np.pi)
        ii) Para hacer que el inicio se situe en el norte se le resta
        (np.pi / 2.) puesto que, por defecto, esta en el este (debido al
        estandar en trigonometria). Entonces:
            np.arctan2(v, u) + (2 * np.pi) - (np.pi / 2.)
        iii) Si a (2 * np.pi) se le resta todo lo anterior, se le cambia el
        sentido de crecimiento. Otra vez, esto es necesario por los estandares
        en trigonometria (crecimiento en sentido antihorario):
            (2 * np.pi) - (np.arctan2(v, u) + (2 * np.pi) - (np.pi / 2.))
        iv) Antes de terminar, hay que sumar np.pi para pasar de "hacia donde va
        el viento" a "de donde viene el viento".
            ((2 * np.pi) - ((np.arctan2(v, u) + (2 * np.pi)) - (np.pi / 2.)) + np.pi)
        v) Finalmente, se simplifica la expresion y se limita el resultado a una
        sola vuelta.
            np.mod(((3 * np.pi / 2.) - np.arctan2(v, u)), (2 * np.pi))

    Parameters
    ----------
    u: numpy.ndarray
        Array de componentes horizontales zonales del viento.
    v: numpy.ndarray
        Array de componentes horizontales meridionales del viento.

    Returns
    -------
    dirDeg: numpy.ndarray
        Array con las direcciones de las que proviene el viento, en grados.
    """
    if u.shape != v.shape:
        raise ValueError('Los arrays deben tener el mismo shape')
    dirDeg = np.rad2deg(
        np.mod(
            ((3 * np.pi / 2.) - np.arctan2(v.copy(), u.copy())),
            (2 * np.pi)
        )
    )
    return dirDeg.copy()

def MediaDireccionViento(alpha):
    """
    Sacado de S. R. Jammalamadaka & A. Sengupta (2001). "Topics In Circular
    Statistics". Vol 5 (Series On Multivariate Analysis). p13

    Parameters
    ----------
    alpha: numpy.ndarray
        Array de direcciones del viento en grados.

    Returns
    -------
    alpha0Fin: numpy.float64
        Valor de la direccion promedio en el estandar de direcciones de
        meteorologia.
    """
    alphaRad = np.deg2rad(alpha.copy())
    numeroComplejo = complex(np.sum(np.cos(alphaRad)), np.sum(np.sin(alphaRad)))
    alpha0 = np.angle(numeroComplejo, deg = True)
    alpha0Fin = np.mod(alpha0 + 360.0, 360.0)
    return alpha0Fin

def NanMediaDireccionViento(alpha):
    alphaNoNan = alpha[np.isnan(alpha) == False].copy()
    if len(alphaNoNan) == 0:
        alpha0Fin = np.nan
    else:
        alpha0Fin = MediaDireccionViento(alphaNoNan)
    return alpha0Fin

def DistanciaCircular(alphaTrue, alphaPred):
    """
    Sacado de S. R. Jammalamadaka & A. Sengupta (2001). "Topics In Circular
    Statistics". Vol 5 (Series On Multivariate Analysis). p15

    Parameters
    ----------
    alphaTrue: numpy.ndarray
        Array de direcciones del viento de las observaciones en grados.
    alphaPred: numpy.ndarray
        Array de direcciones del viento modeladas en grados.

    Returns
    -------
    d0: numpy.ndarray
        Array de diferencias en grados de los arrays empleados como inputs
    """
    d0 = np.rad2deg(np.pi - abs(np.pi - abs(np.deg2rad(alphaTrue) - np.deg2rad(alphaPred))))
    return d0

def DACC(alphaTrue, alphaPred, umbral = 30.0):
    """
    Estadistico empleado por Paco en Santos-Alamillos, F. J., Pozo-Vázquez, D.,
    Ruiz-Arias, J. A., Lara-Fanego, V., & Tovar-Pescador, J. (2013). Analysis of
    WRF model wind estimate sensitivity to physics parameterization choice and
    terrain representation in Andalusia (Southern Spain). Journal of Applied
    Meteorology and Climatology, 52(7), 1592-1609. que representa el numero de
    ocasiones en tanto por ciento que la diferencia entre direcciones es menor a
    un cierto umbral.

    Parameters
    ----------
    alphaTrue: numpy.ndarray
        Array de direcciones del viento de las observaciones en grados.
    alphaPred: numpy.ndarray
        Array de direcciones del viento modeladas en grados.
    umbral: float, optional
        Valor numerico que se emplea como distancia limite para contar una
        estimacion en direccion como acierto

    Returns
    -------
    dacc: float
        Valor en tanto por ciento del 'Direction accuracy'
    """
    d0 = DistanciaCircular(alphaTrue, alphaPred)
    dacc = len(d0[d0 <= umbral]) / len(alphaTrue) * 100.0
    return dacc

def Correlacion(y_true, y_pred):
    return np.corrcoef(y_true, y_pred)[0,1]

def Bias(y_true, y_pred):
    return np.mean(y_pred - y_true)

def RMSE(y_true, y_pred):
    return mean_squared_error(y_true, y_pred, squared = False)

def MediaHorariaNormalizada(valores, arrayDates):
    if len(valores) != len(arrayDates):
        raise ValueError('La longitud de los datos no coincide con el array de fechas')
    valorMedio = np.mean(valores)
    datos = pd.DataFrame(None, columns = ('Date', 'MesHora', 'values'))
    datos['Date'] = arrayDates.copy()
    datos['MesHora'] = np.array([datetime.strftime(fecha, '%m%H') for fecha in arrayDates]).copy()
    datos['values'] = valores.copy()
    datosAgrupados = datos.groupby(['MesHora']).mean()
    datosNormalizados = datosAgrupados / valorMedio
    return datosNormalizados.values.reshape(len(datosNormalizados),).copy()

def ErrorMediaHorariaNormalizada(y_true, y_pred, arrayDates):
    valorObs = MediaHorariaNormalizada(y_true, arrayDates)
    valorPred = MediaHorariaNormalizada(y_pred, arrayDates)
    return np.mean(valorPred - valorObs)

def FuncionDensidadAcumulada(valores, nBins = 25, vMin = 0., vMax = 25.0):
    hist, intervalos = np.histogram(valores, bins = nBins, range = (vMin, vMax), density = True)
    dx = np.mean(intervalos[1:] - intervalos[:-1])
    densidadAcumulada = np.cumsum(hist * dx)
    return densidadAcumulada

def ErrorFuncionDensidadAcumulada(y_true, y_pred):
    valorObs = FuncionDensidadAcumulada(y_true)
    valorPred = FuncionDensidadAcumulada(y_pred)
    return np.sum(abs(valorPred - valorObs))

def ArrayCorrelacionEspacial(dfSerieTemporal, dfLocalizaciones, latName = 'lat', lonName = 'lon'):
    """
    Funcion que, a partir de dos dataframes, construye un array
    bidimensional parecido al que devuelve np.corrcoef pero mas util. En la
    diagonal apareceran 1's de correlaciones de series temporales consigo mismas.
    En la mitad superior apareceran las correlaciones de Pearson de las series
    temporales, mientras que en la mitad inferior apareceran las distancias en
    km de las "fuentes" que producen esas series temporales.

    Parameters
    ----------
    dfSerieTemporal: pandas.core.frame.DataFrame
        Dataframe que contiene las series temporales de distintas fuentes. Estas
        fuentes pueden ser pixeles de un modelo, torres de medicion,
        generaciones provincias, etc. El index debe estar compuestos de
        datetimes y las distintas fuentes separadas en columnas con un nombre
        que las identifique.
    dfLocalizaciones: pandas.core.frame.DataFrame
        Este dataframe debe contener las coordenadas latitud y longitud de las
        distintas fuentes que aparecen en dfSerieTemporal. Es decir, el index de
        este objeto deber coincidir con el atributo columnas de dfSerieTemporal.

    Returns
    -------
    arrayCorr: numpy.ndarray
        Array bidimensional que contiene las correlaciones por pares de las
        distintas fuentes de series temporales en la esquina superior derecha.
        Asi, la diagonal de dicho array contiene 1's por la correlacion de las
        series temporales consigo mismas. La esquina inferior izquierda contiene
        las distancias espaciales en km de las distintas fuentes.
    """
    arrayCorrDist = np.ones((len(dfLocalizaciones), len(dfLocalizaciones)), dtype = float)
    iterator1 = 0
    while iterator1 < len(dfLocalizaciones):
        codigo1 = dfLocalizaciones.index.values[iterator1]
        iterator2 = iterator1 + 1
        while iterator2 < len(dfLocalizaciones):
            codigo2 = dfLocalizaciones.index.values[iterator2]
            array2D = dfSerieTemporal[[codigo1, codigo2]].dropna().values
            arrayCorrDist[iterator1, iterator2] = np.corrcoef(array2D, rowvar = False)[0, 1]
            arrayCorrDist[iterator2, iterator1] = GeoDist.distance((dfLocalizaciones.loc[codigo1][latName], dfLocalizaciones.loc[codigo1][lonName]), (dfLocalizaciones.loc[codigo2][latName], dfLocalizaciones.loc[codigo2][lonName])).km
            iterator2 += 1
        iterator1 += 1
    return arrayCorrDist

def InterpolaValores3dEnAltura(valores3D, altura3D, heights = [100], nProcesos = 1, nNivelesVert = 10, interp = 'linear'):
    """
    Pendiente de comentar...
    """
    if valores3D.shape != altura3D.shape:
        raise ValueError('El array de valores debe tener el mismo shape que el array de alturas')
    if altura3D.shape[0] < nNivelesVert:
        raise ValueError(f'El array de alturas no tiene suficientes niveles verticales: {altura3D.shape[0]}/{nNivelesVert}')
    listaNames = [f'{h0}m' for h0 in heights]
    listaNames.insert(0, 'idx')
    listaTypes = [float] * len(heights)
    listaTypes.insert(0, int)
    nFilas = valores3D.shape[1]
    nColumnas = valores3D.shape[2]
    reqCores = multiprocessing.Pool(nProcesos)
    interpArray = np.array(
        reqCores.map(
            FuncionInterpolacionEnAltura,
            [(
                Get1dIndex(idxLat, idxLon, nColumnas),
                valores3D[:nNivelesVert, idxLat, idxLon].copy(),
                altura3D[:nNivelesVert, idxLat, idxLon].copy(),
                heights, interp
            ) for idxLat in range(nFilas) for idxLon in range(nColumnas)]
        ),
        dtype = np.dtype({
            'names': tuple(listaNames),
            'formats': tuple(listaTypes)})
    )
    reqCores.close()
    np.sort(interpArray)
    interpArray = interpArray.reshape(nFilas, nColumnas)
    return interpArray.copy()

def GridLatLonEsquinasFromStaggeredGrid(latStagX, lonStagY):
    """
    Pendiente de comentar...
    """
    latEsquinas = np.zeros((lonStagY.shape[0], latStagX.shape[1])) - 9999.999
    lonEsquinas = np.zeros((lonStagY.shape[0], latStagX.shape[1])) - 9999.999
    '''
    Latitud
    '''
    latEsquinas[1:-1, :] = (latStagX[1:, :].copy() + latStagX[:-1, :].copy()) / 2.0
    arrayDifSup = latStagX[-1, :].copy() - latStagX[-2, :].copy()
    arrayDifInf = latStagX[1, :].copy() - latStagX[0, :].copy()
    latEsquinas[-1, :] = latEsquinas[-2, :].copy() + arrayDifSup.copy()
    latEsquinas[0, :] = latEsquinas[1, :].copy() - arrayDifInf.copy()
    '''
    Longitud
    '''
    lonEsquinas[:, 1:-1] = (lonStagY[:, 1:].copy() + lonStagY[:, :-1].copy()) / 2.0
    arrayDifIzq = lonStagY[:, 1].copy() - lonStagY[:, 0].copy()
    arrayDifDcha = lonStagY[:, -1].copy() - lonStagY[:, -2].copy()
    lonEsquinas[:, 0] = lonEsquinas[:, 1].copy() - arrayDifIzq.copy()
    lonEsquinas[:, -1] = lonEsquinas[:, -2].copy() + arrayDifDcha.copy()
    return latEsquinas.copy(), lonEsquinas.copy()
#-----------------------------------------------------------------------
