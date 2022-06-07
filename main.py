import math

import numpy as np
import pyproj
import vtk
from vtkmodules.vtkCommonCore import vtkDoubleArray, vtkPoints
from vtkmodules.vtkCommonDataModel import vtkStructuredGrid
from vtkmodules.vtkInteractionStyle import vtkInteractorStyleTrackballCamera
from vtkmodules.vtkRenderingCore import (
    vtkRenderWindow,
    vtkRenderWindowInteractor,
    vtkRenderer
)
# intersection de segments
# https://stackoverflow.com/questions/3252194/numpy-and-line-intersections


def render_map(values, tr, tl, br, bl):

    nx, ny = values.shape

    point_values = vtkDoubleArray()
    point_values.SetNumberOfComponents(1)
    point_values.SetNumberOfTuples(nx * ny)
    for i in range(nx * ny):
        point_values.SetValue(i, 2)

    min_h = math.inf
    max_h = 0.0
    angle = 5
    r_terre = 6352800
    d_angle = angle / (len(values) - 1)
    points = vtkPoints()

    for y in range(ny):
        for x in range(nx):
            current = values[x][y]
            r = r_terre + current
            phi = math.radians(45 + d_angle * x)
            theta = math.radians(90 + d_angle * y)
            cart_x = -r * math.sin(phi) * math.cos(theta)
            cart_z = r * math.sin(phi) * math.sin(theta)
            cart_y = r * math.cos(phi)
            min_h = min(min_h, current)
            max_h = max(max_h, current)
            err = 1000
            if 0 < x < nx - 1 and 0 < y < ny - 1:
                err = 0
                for i in range(x - 1, x + 2):
                    for j in range(y - 1, y + 2):
                        err = max(err, abs(values[i][j] - values[x][y]))
            if err == 0:
                point_values.SetValue(y * nx + x, 0)
            else:
                point_values.SetValue(y * nx + x, current)
            points.InsertNextPoint(cart_x, cart_y, cart_z)

    struct_grid = vtkStructuredGrid()
    struct_grid.SetDimensions(nx, ny, 1)
    struct_grid.SetPoints(points)
    struct_grid.GetPointData().SetScalars(point_values)

    ren = vtkRenderer()

    return ren


def main(data, tr,tl,br,bl):
    ren_win = vtkRenderWindow()

    ren = render_map(data, tr,tl,br,bl)

    ren.SetBackground(1.0, 1.0, 1.0)

    ren_win.AddRenderer(ren)
    ren_win.SetSize(600, 600)

    iren = vtkRenderWindowInteractor()
    iren.SetRenderWindow(ren_win)

    style = vtkInteractorStyleTrackballCamera()
    iren.SetInteractorStyle(style)

    iren.Initialize()
    iren.Start()


if __name__ == '__main__':
    # import data

    gps_file = open('vtkgps.txt', 'r')
    count = 0
    Lines = gps_file.readlines()

    parcours_avion = []
    # Strips the newline character
    for line in Lines:
        infos = line.split()
        if len(infos) > 4:
            # RT90 coordinates
            coord = (infos[0], int(infos[1]), int(infos[2]))
            # altitude en mètres
            height = float(infos[3])
            parcours_avion.append((coord, height))
    gps_file.close()
    print(len(parcours_avion))

    # coord de l'image jpg :
    # Haut-gauche : 1349340 7022573
    # Haut-droite : 1371573 7022967
    # Bas-droite : 1371835 7006362
    # Bas-gauche : 1349602 7005969

    rt90 = 'epsg:3021'  # RT90
    wgs84 = 'epsg:4326'  # Global lat-lon coordinate system

    rt90_to_latlon = pyproj.Transformer.from_crs(rt90, wgs84)
    tl = rt90_to_latlon.transform(1349340, 7022573)
    tr = rt90_to_latlon.transform(1371573, 7022967)
    br = rt90_to_latlon.transform(1371835, 7006362)
    bl = rt90_to_latlon.transform(1349602, 7005969)
    print(tl, tr, br, bl)

    # hauteurs depuis fichier binaire
    h_data = np.fromfile("EarthEnv-DEM90_N60E010.bil", dtype=np.dtype(np.int16))
    h_data = h_data.reshape((6000, 6000))
    # data entre latitude 60 et 65 et longitude 5 et 10

    # rendu
    main(h_data, tr, tl, br, bl)
