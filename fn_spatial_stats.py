import numpy as np
import math
import scipy
from collections import defaultdict


class RectangleM_new:
    """
    Rectangle grid structure for quadrat-based method.

    Parameters
    ----------
    pp                : :class:`.PointPattern`
                        Point Pattern instance.
    count_column      : integer
                        Number of rectangles in the horizontal
                        direction. Use in pair with count_row to
                        fully specify a rectangle. Incompatible with
                        rectangle_width and rectangle_height.
    count_row         : integer
                        Number of rectangles in the vertical
                        direction. Use in pair with count_column to
                        fully specify a rectangle. Incompatible with
                        rectangle_width and rectangle_height.
    rectangle_width   : float
                        Rectangle width. Use in pair with
                        rectangle_height to fully specify a rectangle.
                        Incompatible with count_column & count_row.
    rectangle_height  : float
                        Rectangle height. Use in pair with
                        rectangle_width to fully specify a rectangle.
                        Incompatible with count_column & count_row.

    Attributes
    ----------
    pp                : :class:`.PointPattern`
                        Point Pattern instance.
    mbb               : array
                        Minimum bounding box for the point pattern.
    points            : array
                        x,y coordinates of the point points.
    count_column      : integer
                        Number of columns.
    count_row         : integer
                        Number of rows.
    num               : integer
                        Number of rectangular quadrats.

    """

    def __init__(self, pp, labels, count_column = 3, count_row = 3,
                 rectangle_width = 0, rectangle_height = 0):
        self.mbb = pp.mbb
        self.pp = pp
        self.points = np.asarray(pp.points)
        self.labels = np.array(labels)
        x_range = self.mbb[2]-self.mbb[0]
        y_range = self.mbb[3]-self.mbb[1]
        if rectangle_width & rectangle_height:
            self.rectangle_width = rectangle_width
            self.rectangle_height = rectangle_height

            # calculate column count and row count
            self.count_column = int(math.ceil(x_range / rectangle_width))
            self.count_row = int(math.ceil(y_range / rectangle_height))
        else:
            self.count_column = count_column
            self.count_row = count_row

            # calculate the actual width and height of cell
            self.rectangle_width = x_range/float(count_column)
            self.rectangle_height = y_range/float(count_row)
        self.num = self.count_column * self.count_row

    def point_location_sta(self):
        """
        Count the point events in each cell.

        Returns
        -------
        dict_id_count : dict
                        keys: rectangle id, values: number of point
                        events in each cell.
        """

        dict_id_count = {}
        for i in range(self.count_row):
            for j in range(self.count_column):
                dict_id_count[j+i*self.count_column] = 0

        for point in self.points:
            index_x = (point[0]-self.mbb[0]) // self.rectangle_width
            index_y = (point[1]-self.mbb[1]) // self.rectangle_height
            if index_x == self.count_column:
                index_x -= 1
            if index_y == self.count_row:
                index_y -= 1
            id = index_y * self.count_column + index_x
            dict_id_count[id] += 1
        return dict_id_count

    def _get_dict_id_labels(self):
        """
        Get a list of labels in each cell.

        Returns
        -------
        dict_id_count : dict
                        keys: rectangle id, values: number of point
                        events in each cell.
        """

        # dict_id_count = {}
        # for i in range(self.count_row):
        #     for j in range(self.count_column):
        #         dict_id_points[j+i*self.count_column] = []
        dict_id_labels = defaultdict(list)
        for point, lab in zip(self.points, self.labels):
            index_x = (point[0]-self.mbb[0]) // self.rectangle_width
            index_y = (point[1]-self.mbb[1]) // self.rectangle_height
            if index_x == self.count_column:
                index_x -= 1
            if index_y == self.count_row:
                index_y -= 1
            id = index_y * self.count_column + index_x
            dict_id_labels[id].append(lab)
        self.dict_id_labels = dict_id_labels

    def get_shannon_diversities(self):
        self._get_dict_id_labels()
        dict_idx_shannon = {}
        for idx, labels in self.dict_id_labels.items():
            labels = np.array(labels)
            nlab = len(labels)
            Hs = []
            for l in np.unique(labels):
                nl = sum(labels==l)
                p = nl / nlab
                Hs.append(p * np.log(p))
            H = -np.sum(Hs)
            dict_idx_shannon[idx] = H
        return dict_idx_shannon

    def get_multispecies_tiles(self):
        self._get_dict_id_labels()
        dict_idx_multi = {}
        for idx, labels in self.dict_id_labels.items():
            c = 0
            if len(np.unique(labels)) > 1:
                c = 1
            dict_idx_multi[idx] = c
        return dict_idx_multi

    def get_simpson_diversities(self):
        self._get_dict_id_labels()
        dict_idx_simpson = {}
        for idx, labels in self.dict_id_labels.items():
            labels = np.array(labels)
            nlab = len(labels)
            if nlab > 1:
                etas = []
                for l in np.unique(labels):
                    nl = sum(labels==l)
                    etas.append(nl*(nl-1))
                D = 1 - np.sum(etas) / (nlab*(nlab-1))
            else:
                D = 0
            dict_idx_simpson[idx] = D
        return dict_idx_simpson

    def get_partition_values(self, q):
        self._get_dict_id_labels()
        dict_idx = {}
        for idx, labels in self.dict_id_labels.items():
            labels = np.array(labels)
            nlab = len(labels)
            Xs = []
            for l in np.unique(labels):
                nl = sum(labels==l)
                p = nl / nlab
                if q == 1:
                    Xs.append(-p * np.log(p))
                else:
                    Xs.append(p**q)
            Xa = np.sum(Xs)
            dict_idx[idx] = Xa
        return dict_idx

    def get_partition_values_noreplace(self, q):
        self._get_dict_id_labels()
        dict_idx = {}
        for idx, labels in self.dict_id_labels.items():
            labels = np.array(labels)
            nlab = len(labels)
            Xs = []
            if nlab >= abs(q):
                for l in np.unique(labels):
                    nl = sum(labels==l)
                    p = nl / nlab
                    if q == 1:
                        Xs.append(-p * np.log(p))
                    elif q == 0:
                        Xs.append(1)
                    else:
                        if nl >= abs(q):
                            minus = 0
                            ns = []
                            Ns = []
                            while minus < abs(q):
                                ns.append(nl - minus)
                                Ns.append(nlab - minus)
                                minus += 1
                            p_ = (np.prod(ns)/np.prod(Ns)) ** np.sign(q)
                            Xs.append(p_)
                        else:
                            Xs.append(0)
            else:
                Xs.append(0)
            Xa = np.sum(Xs)
            dict_idx[idx] = Xa
        return dict_idx

    def get_partition_values_noreplace_nocell_nan(self, q):
        self._get_dict_id_labels()
        dict_idx = {}
        for idx, labels in self.dict_id_labels.items():
            labels = np.array(labels)
            nlab = len(labels)
            if nlab > 0:
                Xs = []
                if nlab >= abs(q):
                    for l in np.unique(labels):
                        nl = sum(labels==l)
                        p = nl / nlab
                        if q == 1:
                            Xs.append(-p * np.log(p))
                        elif q == 0:
                            Xs.append(1)
                        else:
                            if nl >= abs(q):
                                minus = 0
                                ns = []
                                Ns = []
                                while minus < abs(q):
                                    ns.append(nl - minus)
                                    Ns.append(nlab - minus)
                                    minus += 1
                                p_ = (np.prod(ns)/np.prod(Ns)) ** np.sign(q)
                                Xs.append(p_)
                            else:
                                Xs.append(0)
                else:
                    Xs.append(0)
                Xa = np.sum(Xs)
                dict_idx[idx] = Xa
            else:
                dict_idx[idx] = np.nan
        return dict_idx

def get_density_arr(coords_scn, step, radius, radius_um, xlim, ylim):
    xs = np.arange(xlim[0], xlim[1], step)
    ys = np.arange(ylim[0], ylim[1], step)

    density_arr = np.zeros((len(ys), len(xs)))
    for j, x in enumerate(xs):
        eboolx = (x > (xlim[0] + radius)) * (x < (xlim[1] - radius))
        if eboolx:
            for i, y in enumerate(ys):
                ebooly = (y > (ylim[0] + radius)) * (y < (ylim[1] - radius))
                if ebooly:
                    bools = (
                        (coords_scn[:, 1] > (x - radius))
                        * (coords_scn[:, 1] < (x + radius))
                        * (coords_scn[:, 0] > (y - radius))
                        * (coords_scn[:, 0] < (y + radius))
                    )
                    density_arr[i, j] = sum(bools) / (radius_um**2)
    return density_arr
