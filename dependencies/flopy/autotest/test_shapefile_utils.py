"""
Test functions in flopy/export/shapefile_utils.py
"""

import numpy as np
from modflow_devtools.markers import requires_pkg

import flopy
from flopy.discretization import StructuredGrid, UnstructuredGrid, VertexGrid
from flopy.export.shapefile_utils import model_attributes_to_shapefile, shp2recarray
from flopy.utils.crs import get_shapefile_crs

from .test_export import disu_sim
from .test_grid import minimal_unstructured_grid_info, minimal_vertex_grid_info


@requires_pkg("geopandas")
def test_model_attributes_to_shapefile(example_data_path, function_tmpdir):
    # freyberg mf2005 model
    name = "freyberg"
    namfile = f"{name}.nam"
    ws = example_data_path / name
    m = flopy.modflow.Modflow.load(namfile, model_ws=ws, check=False, verbose=False)
    shpfile_path = function_tmpdir / f"{name}.shp"
    pkg_names = ["DIS", "BAS6", "LPF", "WEL", "RIV", "RCH", "OC", "PCG"]
    gdf = m.to_geodataframe(package_names=pkg_names)
    gdf.to_file(shpfile_path)
    assert shpfile_path.exists()

    # freyberg mf6 model
    name = "mf6-freyberg"
    sim = flopy.mf6.MFSimulation.load(sim_name=name, sim_ws=example_data_path / name)
    m = sim.get_model()
    shpfile_path = function_tmpdir / f"{name}.shp"
    pkg_names = ["dis", "bas6", "npf", "wel", "riv", "rch", "oc", "pcg"]
    gdf = m.to_geodataframe(package_names=pkg_names)
    gdf.to_file(shpfile_path)
    assert shpfile_path.exists()

    # model with a DISU grid with no angldegx arrays
    # (https://github.com/modflowpy/flopy/issues/1775)
    name = "mf6-disu"
    sim = disu_sim(name, function_tmpdir, missing_arrays=True)
    m = sim.get_model(name)
    shpfile_path = function_tmpdir / f"{name}.shp"
    pkg_names = ["dis"]
    gdf = m.to_geodataframe(package_names=pkg_names)
    gdf.to_file(shpfile_path)
    assert shpfile_path.exists()


@requires_pkg("geopandas")
def test_create_geodataframe(
    minimal_unstructured_grid_info, minimal_vertex_grid_info, function_tmpdir
):
    d = minimal_unstructured_grid_info
    delr = np.ones(10)
    delc = np.ones(10)
    crs = 26916
    shapefilename = function_tmpdir / "grid.shp"
    sg = StructuredGrid(delr=delr, delc=delc, nlay=1, crs=crs)
    gdf = sg.to_geodataframe()
    gdf.to_file(shapefilename)

    data = gdf.to_records()
    # check that row and column appear as integers in recarray
    assert np.issubdtype(data.dtype["row"], np.integer)
    assert np.issubdtype(data.dtype["col"], np.integer)
    assert len(data) == sg.nnodes
    written_crs = get_shapefile_crs(shapefilename)
    assert written_crs.to_epsg() == crs

    usg = UnstructuredGrid(**d, crs=crs)
    gdf = usg.to_geodataframe()
    gdf.to_file(shapefilename)

    data = gdf.to_records()
    assert len(data) == usg.nnodes
    written_crs = get_shapefile_crs(shapefilename)
    assert written_crs.to_epsg() == crs

    d = minimal_vertex_grid_info
    vg = VertexGrid(**d, crs=crs)
    gdf = vg.to_geodataframe()
    gdf.to_file(shapefilename)

    data = gdf.to_records()
    assert len(data) == vg.nnodes
    written_crs = get_shapefile_crs(shapefilename)
    assert written_crs.to_epsg() == crs
