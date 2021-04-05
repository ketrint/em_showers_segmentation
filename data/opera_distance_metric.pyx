
# User specific aliases and functions
import cython
import numpy as np
cimport numpy as np
from libcpp.vector cimport vector
from cython.parallel import prange
from libcpp cimport bool
from libc.math cimport sqrt, fabs, log, abs, copysign
np.import_array()
BT_Z_LAYERS = np.array([     0.,   1293.,   2586.,   3879.,   5172.,   6465.,   7758.,
                          9051.,  10344.,  11637.,  12930.,  14223.,  15516.,  16809.,
                         18102.,  19395.,  20688.,  21981.,  23274.,  24567.,  25860.,
                         27153.,  28446.,  29739.,  31032.,  32325.,  33618.,  34911.,
                         36204.,  37497.,  38790.,  40083.,  41376.,  42669.,  43962.,
                         45255.,  46548.,  47841.,  49134.,  50427.,  51720.,  53013.,
                         54306.,  55599.,  56892.,  58185.,  59478.,  60771.,  62064.,
                         63357.,  64650.,  65943.,  67236.,  68529.,  69822.,  71115.,
                         72408.,  73701.])
cdef double DISTANCE = 1293.
cdef double EPS = 1e-6

@cython.linetrace(True)
@cython.binding(True)
@cython.wraparound(False)
@cython.boundscheck(False)
@cython.cdivision(True)
cdef double opera_distance_metric(double[:] basetrack_left, double[:] basetrack_right, bool symmetric=False, bool directed=False) nogil:
    cdef double dx, dy, dz, dtx, dty
    dz = basetrack_right[3] - basetrack_left[3]
    
    # symmetric metric 
    if dz <= EPS:
        if directed:
            return 1e10
        if symmetric:
            basetrack_left, basetrack_right = basetrack_right, basetrack_left
            dz = -dz

    dx = basetrack_left[1] - (basetrack_right[1] - basetrack_right[4] * dz)
    dy = basetrack_left[2] - (basetrack_right[2] - basetrack_right[5] * dz)
    
    #dtx = basetrack_left[4] # * copysign(1.0, dz)
    dtx = (basetrack_left[4] - basetrack_right[4]) # * copysign(1.0, dz)
    
    #dty = basetrack_left[5] # * copysign(1.0, dz)
    dty = (basetrack_left[5] - basetrack_right[5]) # * copysign(1.0, dz)

    # dz = DISTANCE
    cdef double a = (dtx * dz) ** 2 + (dty * dz) ** 2
    cdef double b = 2 * (dtx * dz * dx +  dty * dz * dy)
    cdef double c = dx ** 2 + dy ** 2
    if a == 0.:
        return fabs(sqrt(c))
    cdef double discriminant = (b ** 2 - 4 * a * c)
    cdef double log_denominator = 2 * sqrt(a) * sqrt(a + b + c) + 2 * a + b
    cdef double log_numerator = 2 * sqrt(a) * sqrt(c) + b
    cdef double first_part = ( (2 * a + b) * sqrt(a + b + c) - b * sqrt(c) ) / (4 * a)
    return fabs((discriminant * log(fabs(log_numerator / log_denominator)) / (8 * sqrt(a * a * a)) + first_part))

@cython.linetrace(True)
@cython.binding(True)
@cython.wraparound(False)
@cython.boundscheck(False)
@cython.cdivision(True)
cpdef opera_distance_metric_py(double[:] basetrack_left, double[:] basetrack_right, bool symmetric=False, bool directed=False):
    return opera_distance_metric(basetrack_left, basetrack_right, symmetric, directed=directed)

@cython.linetrace(True)
@cython.binding(True)
@cython.wraparound(False)
@cython.boundscheck(False)
@cython.cdivision(True)
cpdef generate_radius_graph(double[:, :] X, double threshold, float e, int skip_layers=5, bool symmetric=False, bool directed=False):
    # cdef double[:, :] X_cython = X
    cdef int N = len(X)
    cdef int i, j
    cdef vector[int] edge_from
    cdef vector[int] edge_to
    cdef vector[double] distances
    cdef double d
    
    # for i in prange(N, nogil=True, num_threads=8, schedule='static'):
    for i in range(N):
        for j in range(N):
            if i==j:
                continue
            if abs(X[i][3] - X[j][3]) < e:
                continue
            if abs(X[i][3] - X[j][3]) > skip_layers * DISTANCE:
                continue

            d = opera_distance_metric(X[i], X[j], symmetric, directed)
            if d < threshold:
                #with gil:
                if True:
                    edge_from.push_back(i)
                    edge_to.push_back(j)
                    distances.push_back(d)
    return edge_from, edge_to, distances

@cython.linetrace(True)
@cython.binding(True)
@cython.wraparound(False)
@cython.boundscheck(False)
@cython.cdivision(True)
cpdef generate_k_nearest_graph(double[:, :] X, int k, float e, int skip_layers=5, bool symmetric=False, bool directed=False):
    cdef int N = len(X)
    cdef int i, j
    cdef vector[double] distances
    cdef vector[double] distances_per_row
    cdef vector[int] argsorted_per_row
    cdef vector[int] edge_from
    cdef vector[int] edge_to
    
    # for i in prange(N, nogil=True, num_threads=8, schedule='static'):
    for i in range(N):
        for j in range(N):
            if i==j:
                distances_per_row.push_back(1e10)
                continue
            if abs(X[i][3] - X[j][3]) < e:
                distances_per_row.push_back(1e10)
                continue
            if abs(X[i][3] - X[j][3]) > skip_layers * DISTANCE:
                distances_per_row.push_back(1e10)
                continue
            distances_per_row.push_back(opera_distance_metric(X[i], X[j], symmetric, directed))
        argsorted_per_row = <vector[int]>np.argpartition(distances_per_row, k)
        distances_per_row.clear()
        for j in range(k):
            if distances_per_row[argsorted_per_row[j]] != 1e10:
                edge_from.push_back(i)
                edge_to.push_back(argsorted_per_row[j])
                distances.push_back(distances_per_row[argsorted_per_row[j]])
            
            
    return edge_from, edge_to, distances

