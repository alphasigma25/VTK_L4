import math

import numpy as np
import pyproj
import vtk
from vtkmodules.vtkCommonColor import vtkNamedColors
from vtkmodules.vtkCommonCore import vtkDoubleArray, vtkPoints, vtkLookupTable
from vtkmodules.vtkCommonDataModel import vtkStructuredGrid
from vtkmodules.vtkIOImage import vtkImageReader
from vtkmodules.vtkInteractionStyle import vtkInteractorStyleTrackballCamera
from vtkmodules.vtkRenderingCore import (
    vtkRenderWindow,
    vtkRenderWindowInteractor,
    vtkRenderer, vtkDataSetMapper, vtkActor, vtkTexture
)


# intersection de segments
# https://stackoverflow.com/questions/3252194/numpy-and-line-intersections

colors = vtkNamedColors()


def render_map(values, conv, img):
    nx, ny = values.shape

    point_values = vtkDoubleArray()
    point_values.SetNumberOfComponents(1)
    point_values.SetNumberOfTuples(nx * ny)

    r_terre = 6352800
    points = vtkPoints()
    for y in range(ny):
        for x in range(nx):
            current = values[x][y]
            lon, lat = conv.XtoL(x/nx, y/ny)
            r = r_terre + current * 2
            phi = math.radians(lat)
            theta = math.radians(lon)
            cart_x = r * math.sin(phi) * math.cos(theta)
            cart_y = r * math.sin(phi) * math.sin(theta)
            cart_z = r * math.cos(phi)
            point_values.SetValue(y * nx + x, current)
            points.InsertNextPoint(cart_x, cart_y, cart_z)

    struct_grid = vtkStructuredGrid()
    struct_grid.SetDimensions(nx, ny, 1)
    struct_grid.SetPoints(points)
    struct_grid.GetPointData().SetScalars(point_values)


    txt = vtkTexture()
    txt.SetInputConnection(img.GetOutputPort())


    lut = vtkLookupTable()

    lut.SetBelowRangeColor(0.529, 0.478, 1.000, 1.0)
    lut.UseBelowRangeColorOn()
    lut.SetHueRange(0.33, 0)
    lut.SetValueRange(0.63, 1)
    lut.SetSaturationRange(0.48, 0)
    lut.Build()

    mapper = vtkDataSetMapper()
    mapper.SetInputData(struct_grid)
    mapper.SetLookupTable(lut)
    mapper.SetScalarRange(200, 900)
    mapper.ScalarVisibilityOn()

    actor = vtkActor()
    actor.SetMapper(mapper)
    actor.SetTexture(txt)

    ren = vtkRenderer()
    ren.AddActor(actor)

    return ren


def main(data, conv, img):
    ren_win = vtkRenderWindow()

    ren = render_map(data, conv, img)

    ren.SetBackground(0.9, 0.9, 0.9)

    ren_win.AddRenderer(ren)
    ren_win.SetSize(600, 600)

    iren = vtkRenderWindowInteractor()
    iren.SetRenderWindow(ren_win)

    style = vtkInteractorStyleTrackballCamera()
    iren.SetInteractorStyle(style)

    iren.Initialize()
    iren.Start()


class Converter:
    def __init__(self, tr, tl, br, bl):
        # mapping
        px = [bl[0], br[0], tr[0], tl[0]]
        py = [bl[1], br[1], tr[1], tl[1]]

        # compute coefficients
        A = [[1, 0, 0, 0], [1, 1, 0, 0], [1, 1, 1, 1], [1, 0, 1, 0]]
        AI = np.linalg.inv(A)
        self.a = np.matmul(AI, px)
        self.b = np.matmul(AI, py)

    def LtoX(self, x, y):
        # quadratic equation coeffs, aa*mm^2+bb*m+cc=0
        aa = self.a[3] * self.b[2] - self.a[2] * self.b[3]
        bb = self.a[3] * self.b[0] - self.a[0] * self.b[3] + self.a[1] * self.b[2] - self.a[2] * self.b[1] + x * self.b[
            3] - y * self.a[3]
        cc = self.a[1] * self.b[0] - self.a[0] * self.b[1] + x * self.b[1] - y * self.a[1]

        # compute m = (-b+sqrt(b^2-4ac))/(2a)
        det = math.sqrt(bb * bb - 4 * aa * cc)
        m = (-bb + det) / (2 * aa)

        # compute l
        l = (x - self.a[0] - self.a[2] * m) / (self.a[1] + self.a[3] * m)
        return l, m

    def XtoL(self, l, m):
        newx = self.a[0] + self.a[1] * l + self.a[2] * m + self.a[3] * l * m
        newy = self.b[0] + self.b[1] * l + self.b[2] * m + self.b[3] * l * m
        return newx, newy


def get_index(lon, lat):
    return round((lat-60)*6000/(65-60)), round((lon-5) * 6000/(10-5))

def index_to_coord(i,j):
    return 60+i*(65-60)/6000, 5+j*(10-5)/6000

if __name__ == '__main__':
    # import data

    gps_file = open('vtkgps.txt', 'r')
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

    # utiliser XtoL pour récupérer les coordonnées dans le maxi tableau
    # faire calcul pour chercher les points qui sont les plus près du truc qu'on cherche

    # hauteurs depuis fichier binaire
    h_data = np.fromfile("EarthEnv-DEM90_N60E010.bil", dtype=np.dtype(np.int16))
    h_data = h_data.reshape((6000, 6000))
    # data entre latitude 60 et 65 et longitude 5 et 10

    extracted_data = []

    conv = Converter(tr, tl, br, bl)
    # extract data
    # boucler sur n valeurs comprises dans le carré
    datas = []
    ni = 100
    nj = 100
    for i in range(ni):
        datas.append([])
        for j in range(nj):
            lon, lat = conv.XtoL(i/ni, j/nj)
            x, y = get_index(lon, lat)
            datas[i].append(h_data[x][y])

    # get image
    imageReader = vtkImageReader()
    imageReader.SetFileName('glider_map.jpg')

    # rendu
    main(np.array(datas), conv, imageReader)
