import math

from typing import Tuple, List

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


class Converter:
    """Classe pour convertir des coordonnées depuis un espace quadrilatère de départ vers un espace carré dont les coordonnées sont comprises entre 0 et 1"""

    def __init__(self, tr: Tuple[float, float], tl: Tuple[float, float], br: Tuple[float, float], bl: Tuple[float, float]) -> None:
        """Crée un convertisseur avec les cordonnée de coin l'espace de départ pour les mapper sur des coordonnées dont les valeurs sont comprises entre 0 et 1

        Args:
            tr (Tuple[float,float]): _description_
            tl (Tuple[float,float]): _description_
            br (Tuple[float,float]): _description_
            bl (Tuple[float,float]): _description_
        """
        # mapping
        px = [bl[0], br[0], tr[0], tl[0]]
        py = [bl[1], br[1], tr[1], tl[1]]

        # compute coefficients
        A = [[1, 0, 0, 0], [1, 1, 0, 0], [1, 1, 1, 1], [1, 0, 1, 0]]
        AI = np.linalg.inv(A)
        self.a = np.matmul(AI, py)
        self.b = np.matmul(AI, px)

    def l_to_x(self, y: float, x: float, ) -> Tuple[float, float]:
        """Convertit valeurs de l'espace de départ en valeurs entre 0 and 1

        Args:
            x (float): latitude
            y (float): longitude

        Returns:
            Tuple[float,float]: coordonnées dans l'espace de départ
        """
        # quadratic equation coeffs, aa*mm^2+bb*m+cc=0
        aa = self.a[3] * self.b[2] - self.a[2] * self.b[3]
        bb = self.a[3] * self.b[0] - self.a[0] * self.b[3] + self.a[1] * \
            self.b[2] - self.a[2] * self.b[1] + x * self.b[3] - y * self.a[3]
        cc = self.a[1] * self.b[0] - self.a[0] * \
            self.b[1] + x * self.b[1] - y * self.a[1]

        # compute m = (-b+sqrt(b^2-4ac))/(2a)
        det = math.sqrt(bb * bb - 4 * aa * cc)
        m = (-bb + det) / (2 * aa)

        # compute l
        l = (x - self.a[0] - self.a[2] * m) / (self.a[1] + self.a[3] * m)
        return l, m

    def x_to_l(self, l: float, m: float) -> Tuple[float, float]:
        """Convertit des coordonnées entre 0 et 1 vers l'espace de départ

        Args:
            l (float): valeur entre 0 et 1
            m (float): valeur entre 0 et 1
        Returns:
             Tuple[float, float] (latitude, longitude)
        """
        assert l >= 0
        assert l <= 1
        assert m >= 0
        assert m <= 1
        newx = self.a[0] + self.a[1] * l + self.a[2] * m + self.a[3] * l * m
        newy = self.b[0] + self.b[1] * l + self.b[2] * m + self.b[3] * l * m
        return newy, newx


def get_index(lat: float, lon: float) -> Tuple[float, float]:
    assert lat >= 60
    assert lat <= 65
    assert lon >= 10
    assert lon <= 15
    return (65-lat)*6000/(65-60), (lon-10) * 6000/(15-10)


def index_to_coord(i: int, j: int) -> Tuple[float, float]:
    assert i >= 0
    assert i <= 6000
    assert j >= 0
    assert j <= 6000
    return 60+(6000-i)/6000*5, 10+j/6000*5


def approximer(h_data, x: float, y: float) -> float:
    """va chercher les hauteurs et interpole la bonne valeur

    Args:
        h_data List[int]: tableau des valeurs
        x (float): indice dans la direction longitudinale
        y (float): indice dans la direction latitudinale

    Returns:
        float: hauteur
    """
    xf = math.floor(x)
    xc = math.ceil(x)
    yf = math.floor(y)
    yc = math.ceil(y)

    xi = x-xf
    xii = xc-x
    yi = y-yf
    yii = yc-y

    cf = h_data[xc][yf]
    fc = h_data[xf][yc]
    cc = h_data[xc][yc]
    ff = h_data[xf][yf]

    return ff * (xii*yii) + cf * (xi*yii) + fc * (xii*yi) + cc * (xi*yi)


def render_map(values: List[List[float]], conv: Converter, img):
    nx, ny = values.shape

    point_values = vtkDoubleArray()
    point_values.SetNumberOfComponents(1)
    point_values.SetNumberOfTuples(nx * ny)

    points = vtkPoints()

    for y in range(ny):
        for x in range(nx):
            current = values[x][y]
            point_values.SetValue(y * nx + x, current/100)
            points.InsertNextPoint(x, y, current/100)

    """
    r_terre = 6352800
    points = vtkPoints()
    for y in range(ny):
        for x in range(nx):
            current = values[x][y]
            lat, lon = conv.x_to_l(x/nx, y/ny)
            r = r_terre + current * 2
            phi = math.radians(lat)
            theta = math.radians(lon)
            cart_x = r * math.sin(phi) * math.cos(theta)
            cart_y = r * math.sin(phi) * math.sin(theta)
            cart_z = r * math.cos(phi)
            point_values.SetValue(y * nx + x, current)
            points.InsertNextPoint(cart_x, cart_y, cart_z)
    """

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
    mapper.SetScalarRange(6, 8)
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
    tl = rt90_to_latlon.transform(7022573, 1349340)
    tr = rt90_to_latlon.transform(7022967, 1371573)
    br = rt90_to_latlon.transform(7006362, 1371835)
    bl = rt90_to_latlon.transform(7005969, 1349602)

    conv = Converter(tr, tl, br, bl)

    # hauteurs depuis fichier binaire
    h_data = np.fromfile("EarthEnv-DEM90_N60E010.bil",
                         dtype=np.dtype(np.int16))
    h_data = h_data.reshape((6000, 6000))
    # data entre latitude 60 et 65 et longitude 5 et 10

    # utiliser XtoL pour récupérer les coordonnées dans le maxi tableau
    # faire calcul pour chercher les points qui sont les plus près du truc qu'on cherche
    datas = []
    ni = 100
    nj = 100
    for i in range(ni):
        datas.append([])
        for j in range(nj):
            lon, lat = conv.x_to_l(ni-i/ni, j/nj)
            x, y = get_index(lat, lon)
            datas[i].append(approximer(h_data, x, y))

    # get image
    imageReader = vtkImageReader()
    imageReader.SetFileName('glider_map.jpg')

    # rendu
    main(np.array(datas), conv, imageReader)
