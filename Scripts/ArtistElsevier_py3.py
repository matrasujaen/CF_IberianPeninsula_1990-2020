#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 29 16:07 2023

@author: Antonio Jiménez-Garrote <agarrote@ujaen.es>

Se me acaba de ocurrir crear un axis con toda la informacion que es "constante"
en las figuras de Elsevier. En verdad casi que valen para todas las editoriales
porque los requisitos no son nada exoticos.
"""
#-------------------------------------------------------------------LIBRARIES---
# from matplotlib import pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cf
#-------------------------------------------------------------------------------

#-------------------------------------------------------------------CONSTANTS---
PROYECCION = ccrs.PlateCarree()
TAMANYO_LETRA_MAX = 10
TAMANYO_LETRA_MIN = 8
FUENTE = 'Liberation Sans'
#-------------------------------------------------------------------------------

#-------------------------------------------------------------------FUNCTIONS---
def AxisElsevier(figura, x0, y0, anchoRel, altoRel, titulo = '', xlabel = '', ylabel = '', panel = False, numeroAxis = 0, panelPosRelX = 0.0, panelPosRelY = 1.0, yLim = [], xticksPos = [], yticksPos = [], xticksLabels = [], yticksLabels = [], xticksRotacion = 0.0, fuente = FUENTE, sizeLetraMax = TAMANYO_LETRA_MAX, sizeLetraMin = TAMANYO_LETRA_MIN):
    axis = figura.add_axes([x0, y0, anchoRel, altoRel])
    axis.grid(ls = ':')
    axis.tick_params(labelsize = sizeLetraMin, direction = 'in')
    if len(yLim) != 0:
        axis.set_ylim(yLim)
    if titulo != '':
        axis.set_title(titulo, fontsize = sizeLetraMax, family = fuente)
    if xlabel != '':
        axis.set_xlabel(xlabel, fontsize = sizeLetraMax, family = fuente)
        if len(xticksPos) != 0:
            axis.set_xticks(xticksPos)
            if len(xticksLabels) != 0:
                axis.set_xticklabels(xticksLabels, rotation = xticksRotacion)
    if ylabel != '':
        axis.set_ylabel(ylabel, fontsize = sizeLetraMax, family = fuente)
        if len(yticksPos) != 0:
            axis.set_yticks(yticksPos)
            if len(yticksLabels) != 0:
                axis.set_yticklabels(yticksLabels)
    if panel == True:
        axis.text(panelPosRelX, panelPosRelY, f'{LetraAbecedario(numeroAxis)})', size = sizeLetraMax, family = fuente, bbox = dict(ec = 'black', fc = 'white'))
    return axis

def AxisConMapa(figura, x0, y0, anchoRel, altoRel, latMin, latMax, lonMin, lonMax, proj = PROYECCION, fondo = 'basemap', resolucion = 'high', panel = False, numeroAxis = 0, panelPosRelX = -1.0, panelPosRelY = 38.0, grid = True, tamanyoLetra = TAMANYO_LETRA_MAX, tamanyoTicks = TAMANYO_LETRA_MIN, fuente = FUENTE):
    axis = figura.add_axes([x0, y0, anchoRel, altoRel], projection = proj)
    axis.set_extent([lonMin, lonMax, latMin, latMax], proj)
    axis.background_img(name = fondo, resolution = resolucion)
    axis.coastlines(resolution = '50m')
    axis.add_feature(cf.BORDERS)
    if panel == True:
        axis.text(panelPosRelX, panelPosRelY, f'{LetraAbecedario(numeroAxis)})', size = tamanyoLetra, family = fuente, bbox = dict(ec = 'black', fc = 'white'))
    if grid == True:
        gl = axis.gridlines(crs = proj, draw_labels = True, color = 'black', linestyle = ':')
        gl.top_labels = False
        gl.right_labels = False
        gl.xlabel_style = {'size': tamanyoTicks, 'name': fuente}
        gl.ylabel_style = {'size': tamanyoTicks, 'name': fuente}
    return axis

def LetraAbecedario(numero):
    letra = chr(ord('a') + numero)
    return letra

def SetLeyendaEnAxis(axis, localizacion = 'upper right', ncolumnas = 1, tuplaBbox = (), fuente = FUENTE, sizeLetra = TAMANYO_LETRA_MAX):
    if tuplaBbox != ():
        axis.legend(loc = localizacion, ncol = ncolumnas, bbox_to_anchor = tuplaBbox, prop = {'size': sizeLetra, 'family': fuente})
    else:
        axis.legend(loc = localizacion, ncol = ncolumnas, prop = {'size': sizeLetra, 'family': fuente})
    # return axis
#-------------------------------------------------------------------------------
