import sys
import math
import numpy as np
import ROOT as root
import numpy as np
from ROOT import gROOT, gSystem, TH2F, TTree, TFile, AddressOf, TLine, TMultiGraph, TEllipse, TH1F
from scipy.signal import savgol_filter
from scipy.signal import find_peaks
from scipy.interpolate import CubicSpline
from shapely.geometry import Polygon, LineString, Point
from shapely.ops import split
import pandas as pd
from libraries import Reference, ConversionFactors, RunParameters

class Energy:

    def __init__(self, points, endpoints, calib):
        self.points = points
        self.endpoints = endpoints
        self.calib_table = calib

    def calculate_weights_polygon(self, polygon1_arr, line1_arr, line2_arr, point_arr, verbose):

        """
            Splits a polygon by two lines and calculates area proportions based on a given point.

            Parameters:
                polygon1_arr (np.ndarray): 2x4 array defining the 4 corners of a polygon.
                line1_arr (np.ndarray): 2x2 array defining the first cutting line.
                line2_arr (np.ndarray): 2x2 array defining the second cutting line.
                point_arr (np.ndarray): 1D array (2,) defining a single point.
                verbose (bool): Whether to print debug information.

            Returns:
                list: Areas of the three resulting sub-polygons.
        """

        polygon1 = Polygon(
            [(polygon1_arr[0, 0], polygon1_arr[1, 0]), (polygon1_arr[0, 1], polygon1_arr[1, 1]),
             (polygon1_arr[0, 2], polygon1_arr[1, 2]), (polygon1_arr[0, 3], polygon1_arr[1, 3]), ])
        line1 = LineString([(line1_arr[0, 0], line1_arr[1, 0]), (line1_arr[0, 1], line1_arr[1, 1])])
        line2 = LineString([(line2_arr[0, 0], line2_arr[1, 0]), (line2_arr[0, 1], line2_arr[1, 1])])
        point = Point(point_arr[0], point_arr[1])
        area_main_polygon = polygon1.area
        if verbose:
            x, y = polygon1.exterior.xy
        total_area = []
        areas = [0, 0, 0]

        # helper to iterate over shapely results which may be GeometryCollection
        def _iter_geoms(g):
            if hasattr(g, 'geoms'):
                try:
                    return list(g.geoms)
                except Exception:
                    pass
            # some collections are iterable directly
            if hasattr(g, '__iter__'):
                try:
                    return list(g)
                except Exception:
                    pass
            return [g]

        cut_poly1 = split(polygon1, line1)
        for i in _iter_geoms(cut_poly1):
            if i.contains(point):
                cut_poly2 = split(i, line2)
                for j in _iter_geoms(cut_poly2):
                    if j.contains(point):
                        total_area.append(j.area)
                        areas[1] = j.area
                        if verbose:
                            cox, coy = j.exterior.xy
                    else:
                        total_area.append(j.area)
                        areas[2] = j.area
            else:
                total_area.append(i.area)
                areas[0] = i.area
        tot = np.array(total_area)
        if verbose:
            print('NOT NEEDED')
        return areas

    def perp_line(self, x1, x2, y1, y2, cd_length):
        ab = LineString([(x1, y1), (x2, y2)])
        # create parallel offset lines on both sides
        left = ab.parallel_offset(cd_length / 2, 'left')
        right = ab.parallel_offset(cd_length / 2, 'right')

        # helper to extract a single point from a boundary geometry
        def _boundary_point(geom, idx):
            """Return one point from ``geom.boundary`` safely.

            The boundary may be a ``Point``, ``LineString``, ``MultiPoint`` or
            ``GeometryCollection``.  ``MultiPoint`` is iterable but not
            subscriptable, which used to trigger
            ``TypeError: 'MultiPoint' object is not subscriptable`` when
            attempting to index it.  Normalise the boundary into a list first
            and then pick the requested element (falling back to element 0).
            """
            b = geom.boundary
            # attempt to convert to a list of geometries
            geoms = []
            if hasattr(b, '__iter__'):
                try:
                    geoms = list(b)
                except Exception:
                    geoms = []
            if not geoms and hasattr(b, 'geoms'):
                try:
                    geoms = list(b.geoms)
                except Exception:
                    geoms = []
            if not geoms:
                geoms = [b]
            if idx < len(geoms):
                return geoms[idx]
            else:
                return geoms[0]

        c = _boundary_point(left, 1)
        d = _boundary_point(right, 0)
        cd = LineString([c, d])
        return c.x, c.y, d.x, d.y

    #Static Method
    def vector_length(endpts):
        return math.sqrt(((endpts[1, 0] - endpts[0, 0]) ** 2) + ((endpts[1, 1] - endpts[0, 1]) ** 2) + (
                (endpts[1, 2] - endpts[0, 2]) ** 2))

    # Static Method
    def find_nearest(array, value):
        array = np.asarray(array)
        idx = (np.abs(array - value)).argmin()
        return idx

    def calculate_profiles(self):
        # Initialization and Setup
        #
        # This routine relies on :meth:`perp_line` to compute perpendicular
        # offsets for bin edges.  ``perp_line`` was recently strengthened to
        # cope with ``shapely`` geometries whose ``boundary`` is a
        # ``MultiPoint``; see its docstring for details.
        calibration_table = self.calib_table
        calibration_table.columns = ["chno", "xx", "yy", "par0", "par1", "chi"]
        data_array = self.points
        data_array_X = data_array[:, 0]
        data_array_Y = data_array[:, 1]
        data_array_Z = data_array[:, 2]
        data_array_Q = data_array[:, 3]

        # Line Vector Calculation
        line_vector_start_3d = [self.endpoints[0, 0], self.endpoints[0, 1], self.endpoints[0, 2]]
        line_vector_end_3d = [self.endpoints[1, 0], self.endpoints[1, 1], self.endpoints[1, 2]]
        line_vector_3d = np.subtract(line_vector_end_3d, line_vector_start_3d)
        unit_vector_3d = line_vector_3d / np.linalg.norm(line_vector_3d)
        line_length_3d = np.linalg.norm(line_vector_3d)
        line_vector_start_2d = [self.endpoints[0, 0], self.endpoints[0, 1]]
        line_vector_end_2d = [self.endpoints[1, 0], self.endpoints[1, 1]]
        line_vector_2d = np.subtract(line_vector_end_2d, line_vector_start_2d)
        line_length_2d = np.linalg.norm(line_vector_2d)
        unit_vector_2d = line_vector_2d / np.linalg.norm(line_vector_2d)

        # Bin Edge Calculation
        num_bins = int(line_length_2d / Reference.RANGE_BIN_SIZE.value)
        arr_bin_edges = np.zeros([num_bins + 2, 7])
        for i in range(0, num_bins + 2):
            arr_bin_edges[i, 0] = i
            next_vector = line_vector_start_2d + i * Reference.RANGE_BIN_SIZE.value * unit_vector_2d
            arr_bin_edges[i, 1] = next_vector[0]
            arr_bin_edges[i, 2] = next_vector[1]
            if i > 0:
                x3, y3, x4, y4 = self.perp_line(line_vector_start_2d[0], next_vector[0], line_vector_start_2d[1], next_vector[1], Reference.RANGE_BIN_PER.value)
                arr_bin_edges[i, 3] = x3
                arr_bin_edges[i, 4] = y3
                arr_bin_edges[i, 5] = x4
                arr_bin_edges[i, 6] = y4
            else:
                x3, y3, x4, y4 = self.perp_line(line_vector_end_2d[0], line_vector_start_2d[0], line_vector_end_2d[1], line_vector_start_2d[1], Reference.RANGE_BIN_PER.value)
                arr_bin_edges[i, 3] = x3
                arr_bin_edges[i, 4] = y3
                arr_bin_edges[i, 5] = x4
                arr_bin_edges[i, 6] = y4

        # Energy Weighting
        histogram_array_list = []
        for j in range(0, len(data_array_X)):
            p = Point(data_array_X[j], data_array_Y[j])
            ab = LineString([(line_vector_start_2d[0], line_vector_start_2d[1]), (line_vector_end_2d[0], line_vector_end_2d[1])])
            dist = ab.project(p)
            (prx, pry) = ab.interpolate(dist).coords.xy
            projection_next = np.subtract([prx[0], pry[0]], line_vector_start_2d)
            mag_projection_next = np.linalg.norm(projection_next)
            possible_id = int(mag_projection_next / Reference.RANGE_BIN_SIZE.value)
            polygon1_arr = np.array(
                [[data_array_X[j] - 1, data_array_X[j] + 1, data_array_X[j] + 1, data_array_X[j] - 1],
                 [data_array_Y[j] - 1, data_array_Y[j] - 1, data_array_Y[j] + 1, data_array_Y[j] + 1]])
            line1_arr = np.array([[arr_bin_edges[possible_id, 3], arr_bin_edges[possible_id, 5]],
                                  [arr_bin_edges[possible_id, 4], arr_bin_edges[possible_id, 6]]])
            line2_arr = np.array([[arr_bin_edges[possible_id + 1, 3], arr_bin_edges[possible_id + 1, 5]],
                                  [arr_bin_edges[possible_id + 1, 4], arr_bin_edges[possible_id + 1, 6]]])
            point_arr = [data_array_X[j], data_array_Y[j]]
            areas = self.calculate_weights_polygon(polygon1_arr, line1_arr, line2_arr, point_arr, False)
            pad_info_single = calibration_table.loc[(calibration_table['xx'] == int(data_array_X[j]/ConversionFactors.X_CONVERSION_FACTOR.value)) & (calibration_table['yy'] == int(data_array_Y[j]/ConversionFactors.Y_CONVERSION_FACTOR.value))]
            try:
                if RunParameters.sim.value:
                    Qvox_value = data_array_Q[j]
                else:
                    if pad_info_single['chi'].values[0] < 8000:
                        Qvox_value = (data_array_Q[j] - pad_info_single['par0'].values[0]) / pad_info_single['par1'].values[0]
                    else:
                        Qvox_value = 0
                Qvox = round(float(Qvox_value), 2)
            except:
                Qvox = data_array_Q[j]
            charge_info = Qvox
            histogram_array_list.append([possible_id - 1, (areas[0] / Reference.AREA_TOTAL_PAD.value) * charge_info])
            histogram_array_list.append([possible_id, (areas[1] / Reference.AREA_TOTAL_PAD.value) * charge_info])
            histogram_array_list.append([possible_id + 1, (areas[2] / Reference.AREA_TOTAL_PAD.value) * charge_info])

        #Histogram Construction
        histogram_array = np.array(histogram_array_list)
        histogram_array_new = np.zeros([len(np.unique(arr_bin_edges[:, 0])), 6])
        counter = 0
        ratio_3d = round(line_length_3d / line_length_2d, 2)
        for r in np.unique(arr_bin_edges[:, 0]):
            histogram_array_new_id = histogram_array[:, 0]
            result = np.where(histogram_array_new_id == r)
            result_charge = histogram_array[result, 1]
            ve = np.where(arr_bin_edges[:, 0] == r)
            histogram_array_new[counter, 0] = r + (Reference.RANGE_BIN_SIZE.value / 2)
            vector_id_2d = line_vector_start_2d + ((r * Reference.RANGE_BIN_SIZE.value) + (Reference.RANGE_BIN_SIZE.value / 2)) * unit_vector_2d
            vector_id_3d = line_vector_start_3d + ((r * Reference.RANGE_BIN_SIZE.value) + (Reference.RANGE_BIN_SIZE.value / 2)) * ratio_3d * unit_vector_3d
            point_bin_2d = np.linalg.norm(np.array(vector_id_2d)-np.array(line_vector_start_2d))
            point_bin_3d = ratio_3d*point_bin_2d
            histogram_array_new[counter, 1] = arr_bin_edges[ve, 1]
            histogram_array_new[counter, 2] = arr_bin_edges[ve, 2]
            histogram_array_new[counter, 3] = arr_bin_edges[ve, 2]
            histogram_array_new[counter, 4] = np.sum(result_charge)
            histogram_array_new[counter, 5] = round(point_bin_3d, 2)
            counter += 1

        # Interpolation and Peak Fiting
        new_energy = histogram_array_new[:, 4]
        new_position = histogram_array_new[:, 5]
        new_energy = np.append(new_energy, 0)
        new_position = np.append(new_position, new_position[-1] + (new_position[1] - new_position[0]))
        if line_length_3d > Reference.LINE_LENGTH_THRESHOLD.value:
            golay_window = Reference.SAVITZKY_GOLAY_WINDOW_LARGE.value
        else:
            golay_window = Reference.SAVITZKY_GOLAY_WINDOW_SMALL.value
        try:
            if len(new_energy) > golay_window:
                fit_energy_ = savgol_filter(new_energy, golay_window, 3, mode='interp')
            else:
                if len(new_energy) % 2 == 0:
                    fit_energy_ = savgol_filter(new_energy, len(new_energy) - 1, 1,
                                                mode='interp')
                else:
                    fit_energy_ = savgol_filter(new_energy, len(new_energy) - 2, 1,
                                                mode='interp')
        except:
            fit_energy_ = new_energy

        return new_position, fit_energy_, line_vector_start_3d, unit_vector_3d, line_length_2d, line_vector_end_3d, histogram_array_new

    def energy_weighted(self, alpha, new_position, fit_energy_, line_vector_start_3d, unit_vector_3d, line_length_2d, line_vector_end_3d, histogram_array_new):
        #Final Calculations and Return
        try:
            cs_position = np.linspace(new_position.min(), new_position.max(), 5000)
            cs_function = CubicSpline(new_position, fit_energy_)
            cs_energy = cs_function(cs_position)
            threshold_peak_height = Reference.THRESHOLD_PEAKS.value * np.average(cs_energy)
            peaks, _ = find_peaks(cs_energy, height=threshold_peak_height)
            if len(peaks) != 0:
                peak_lines = cs_energy[peaks]
            else:
                peak_lines = cs_energy[-1]
            range_max = np.max(peak_lines)
            idx_ = Energy.find_nearest(cs_energy, range_max)
            max_energy = cs_energy[idx_]
            max_position = cs_position[idx_]
            fit_energy_new = cs_energy[idx_:]
            positions_ = cs_position
            positions_new = positions_[idx_:]
            arr_min_ = np.abs(fit_energy_new - (alpha * max_energy))
            idx_low = -1
            idx_low_new = np.argmin(arr_min_)
            idx_low = idx_low_new

             #Introduced to find the intersection points
            f = cs_energy
            g = alpha * max_energy*np.ones(cs_position.shape)
            idx = np.argwhere(np.diff(np.sign(f - g))).flatten()
            idx_max = max(idx)


            vector_id_3d_final = line_vector_start_3d + positions_new[idx_low] * unit_vector_3d
            delta_z = vector_id_3d_final[2] - line_vector_start_3d[2]
            r2d = np.linalg.norm(np.array(vector_id_3d_final[0:2]) - np.array(line_vector_start_3d[0:2]))
        except:
            return line_length_2d, line_vector_end_3d[2]-line_vector_start_3d[2], np.mean(histogram_array_new[:, 5]), np.mean(histogram_array_new[:, 4]), np.mean(
                histogram_array_new[:, 5]), np.mean(histogram_array_new[:, 4]), histogram_array_new[:,
                                                                                5], histogram_array_new[:,
                                                                                    4], histogram_array_new[:,
                                                                                        5], histogram_array_new[:, 4]
        else:
            return r2d, delta_z, cs_position[idx_max], cs_energy[idx_max], positions_[idx_], max_energy, histogram_array_new[:,
                                                                                                  5], histogram_array_new[
                                                                                                      :,
                                                                                                      4], cs_position, cs_energy

