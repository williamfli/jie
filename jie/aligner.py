#!/usr/bin/env python

#-----------------------------------------------------------------------------
# Copyright (c) 2022--, Bojing Jia.
#
# Distributed under the terms of the GPLv3 license.
#
# The full license is in the file LICENSE.txt, distributed with this software.
#-----------------------------------------------------------------------------

from importlib import reload
from . import utilities
reload(utilities)


import numpy as np
import pandas as pd
import igraph
import copy
import warnings
import random
from scipy.spatial.distance import cdist

from collections.abc import Iterable

from .utilities import (cartesian_esqsum, 
                        cartesian_sqdiff, 
                        cartesian_diff, 
                        calculate_s0_sqr,
                        find_loci_dist)

scale_factor_global = 1.1

# this should be smaller for the factor for threshold and edge penalty, allowing truncated fiber to be picked
boundary_scale_factor = 1.5

def log_bond(genomic_interval, l_p_bp, corr_factor, stretch_factor=1.0):
    s0_sqr = 2* genomic_interval * l_p_bp * corr_factor**2
    
    # calculate relative penalty
    expected_edge_weights =  3/2 * (np.log(2/3*np.pi*s0_sqr) + stretch_factor)
    
    return expected_edge_weights


def cdf_thresh(gene_dist, l_p_bp=150., corr_factor=0.3/108.):
    '''
    Input:
        gene_dist: [list]
            list of genomic distances relative to starting loci
            eg. [0, 5kb, 10kb, 15kb, ...]
        l_p_bp: float
            persistence length (bp) - here 150bp nucleosome DNA with 30bp linker
        corr_factor: float
            nm per persistence length (20nm) / pixel_dist = 108
    Output:
        total_score: float
            conformational distribution function (CDF) threshold score for for calling chr
    '''
    if not isinstance(gene_dist, Iterable):
        raise TypeError('gene_dist needs to be an iterable list of numerical values.')
        
    if not isinstance(gene_dist[0], (int, float, np.int64, np.float64)):
        raise TypeError('gene_dist needs to be an iterable list of numerical values.')

    if not np.all(np.diff(gene_dist) > 0):
        raise ValueError('gene_dist must be an ascending sorted list of numerical values.')

    if not np.all(np.array(gene_dist) >= 0):
        raise ValueError('gene_dist must be positive numerical values.')
        
    if not isinstance(l_p_bp, (int, float, np.int64, np.float64)):
        raise TypeError('l_p_bp needs to be a numerical value')
        
    if l_p_bp < 0:
        raise ValueError('l_p_bp needs to be greater than 0.')
    
    s0_sqr = 2 * np.array([(gene_dist[i]-gene_dist[i-1]) * l_p_bp * corr_factor**2 for i in range(1, len(gene_dist))])
       
    # calculate relative penalty
    expected_edge_weights = [3/2 * (np.log(2/3*np.pi*s0_sqr) + 1)]
    
    return scale_factor_global*np.sum(expected_edge_weights)

    
def edge_penalty(current_hyb, skips, l_p_bp, corr_fac, gene_dist_matrix, scale_factor=1.0):
    '''
    Input:
        current_hyb:
            current hybridization value
        skips: [curr_hyb x next_hyb]
            2D array of degrees-of-separation between nodes
        l_p_bp: float 
            persistence length (bp)
        corr_fac: float 
            scale genomic dist (bp) into pixels (e.g nm_per_unit / pixel_dist)
        gene_dist_matrix: float 
             genomic distance interval (bp) matrix
        scale_factor: float
            scale the penalty
    Output:
        penalty: [curr_hyb x next_hyb] 
            2D array of penalty per transition edge weight
    '''
    if not isinstance(skips, np.ndarray) or skips.dtype != int:
        raise TypeError('skips needs to be a 2D numpy array of integers.')       
    
    if not isinstance(l_p_bp, (int, float, np.int64, np.float64)):
        raise TypeError('l_p_bp needs to be a numerical value.')
    
    if l_p_bp < 0:
        raise ValueError('l_p_bp needs to be a positive numerical value.')
        
    if not isinstance(corr_fac, (int, float, np.int64, np.float64)):
        raise TypeError('corr_fac needs to be a numerical value.')
    
    if corr_fac < 0:
        raise ValueError('corr_fac needs to be a positive numerical value.')

    # calculate relative penalty
    next_hyb_values = current_hyb + np.ravel(skips) + 1
    penalty_linear_const = []
    for next_hyb in next_hyb_values:
        _dist = 0
        for i in range(current_hyb, next_hyb, 1):
            _dist += log_bond(gene_dist_matrix[i, i+1],  l_p_bp, corr_fac, 1.0)
        total_skip_dist = log_bond(gene_dist_matrix[current_hyb, next_hyb],  l_p_bp, corr_fac, 1.0)
        penalty_linear_const.append(_dist/total_skip_dist)
        

    penalty_const = np.array(penalty_linear_const).reshape(skips.shape)
    # multiply the penalty score based on how many skips
    # penalize more if skipping too many nodes
    penalty_exp = 2*skips - 2
    penalty_exp[skips==0] = 1
    penalty_exp[skips==1] = 2

    return penalty_const, penalty_exp

def single_edge_penalty(real_distance_sqr, penalty_value_base, penalty_threshold, pixel_dist):
    # this function penalize single edge that are too long
    
    penalty_threshold_pixel = (penalty_threshold/pixel_dist)*(penalty_threshold/pixel_dist)
    _real = real_distance_sqr*(pixel_dist/1000)*(pixel_dist/1000)
    penalty = _real * penalty_value_base
    penalty [real_distance_sqr<=penalty_threshold_pixel] = 0
    return penalty
    

def edge_weights(pts_clr_curr, 
                 pts_clr_next, 
                 bin_size, 
                 l_p_bp,
                 nm_per_bp, 
                 pixel_dist, 
                 theta, 
                 gene_dist,skips, cur_hyb, num_skip, iteration,
                 lim_min_dist = True):
    '''
    Input:
        pts_clr_curr: [DataFrame]
            table of spatial coordinates + metadata of current nodes
        pts_clr_next: [DataFrame]
            table of spatial coordinates + metadata of reachable nodes
        bin_size: float
            median genomic distance interval (bp)
        l_p_bp: float
            persistence length (bp)
        nm_per_bp: float
            0.3nm per bp
        pixel_dist: float
            pixel size (nm)
        theta: float
            bond angle
        gene_dist: [num_hyb x num_hyb] 
            2D array of pairwise expected spatial distance given genomic distance
        lim_min_dist : boolean
            penalize successively choosing most proximal spot 
    Output:
        trans_prob: [curr_hyb x next_hyb] 
            2D array of transition edge weights
    '''
    if not isinstance(pts_clr_curr, pd.core.frame.DataFrame) or not isinstance(pts_clr_next, pd.core.frame.DataFrame):
        raise TypeError('Both pts_clr_curr and pts_clr_next must be pandas DataFrames')
        
    if not set(['x_hat', 'y_hat', 'z_hat', 'hyb', 'sig_x', 'sig_y', 'sig_z']).issubset(pts_clr_curr.columns) or \
       not set(['x_hat', 'y_hat', 'z_hat', 'hyb', 'sig_x', 'sig_y', 'sig_z']).issubset(pts_clr_next.columns):
        raise KeyError('pts_clr_curr and pts_clr_next must have the following columns: [x_hat, y_hat, z_hat, hyb, sig_x, sig_y, sig_z]')
                       
    if pts_clr_next['hyb'].size > 0:
        if max(pts_clr_next['hyb']) > gene_dist.shape[0]:
            raise IndexError('hyb index out of bounds with respect to gene_dist. Check if the correct reference genome is being used.')

    if not pts_clr_curr.hyb.is_monotonic_increasing or not pts_clr_next.hyb.is_monotonic_increasing:
        raise ValueError('hyb in both pts_clr_curr and pts_clr_next must be sorted in ascending order.')
                
    if not all(isinstance(x, (int, float, np.int64, np.float64)) for x in [bin_size, l_p_bp, nm_per_bp, pixel_dist, theta]) or \
       not all(x >= 0 for x in [bin_size, l_p_bp, nm_per_bp, pixel_dist, theta]):
        raise ValueError('bin_size, l_p_bp, nm_per_unit, pixel_dist, theta must be all positive numerical values.')
        
    # grab output shape
    shape = (pts_clr_curr.shape[0], pts_clr_next.shape[0])
    
    ##### OBSERVED ######
    # Calculate observed sq distance
    real_distance_sqr = cdist(pts_clr_curr[['z_hat', 'y_hat', 'x_hat']], 
                               pts_clr_next[['z_hat', 'y_hat', 'x_hat']],
                               'sqeuclidean')
    #####################

    # calculate edge penalties
    penalty_const, penalty_exp = edge_penalty(cur_hyb, skips, l_p_bp, nm_per_bp/pixel_dist, gene_dist, 1)
    penalty_value_base = 3.
    penalty_real_distance = single_edge_penalty(real_distance_sqr, penalty_value_base, penalty_threshold=1500,  pixel_dist=pixel_dist)
    

    #### UNCERTAINTY #####
    # Calculate uncertainty d.t. contour
    s_sqr = calculate_s0_sqr(pts_clr_curr['hyb'], 
                            pts_clr_next['hyb'], 
                            l_p_bp, 
                            (nm_per_bp / pixel_dist),
                            gene_dist)
    s_sqr = np.reshape(s_sqr, shape)     
    
    ### TRANSITION EDGE WEIGHTS ###
    # Calculate constant term
    const = 2/3*np.pi*s_sqr
    
    # Calculate exp term
    exp = real_distance_sqr/s_sqr
    
    # Calculate negative log prob of Gaussian chain link
    trans_prob = 3/2*np.add(np.log(const)*penalty_const, exp*penalty_exp) + penalty_real_distance
    
    assert trans_prob.shape == shape 
    
    return trans_prob


def boundary_init(trans_mat, 
                  gene_dist_matrix, 
                  l_p_bp,
                  corr_fac,
                  n_colours, 
                  cell_pts, 
                  exp_stretch, 
                  stretch_factor, 
                  lim_init_skip, 
                  init_skip,
                  end_skip):
    '''
    Input:
        trans_mat: [ndarray]
            2D adjacency matrix
        gene_dist_matrix: [num_hyb x num_hyb ndarray] 
            2D array of pairwise genomic distance
        l_p_bp: float
            persistence length (bp)
        corr_fac: float
            scale genomic dist (bp) into pixels (e.g nm_per_unit / pixel_dist)
        n_colours: float
            number of loci imaged for given chr
        cell_pts: [DataFrame]
            table of spatial coordinates of all nodes + metadata in cell
        exp_stretch: float
            expected bond extension
        stretch_factor: float
            allowable bond extension
        lim_init_skip: boolean
            limit number of skips at graph source and sink
        init_skip: int
            index of hyb to be skipped to from graph source
        end_skip: int
            index of hyb to allow skips to graph sink
    Output:
        trans_mat_pad: [(num_hyb + 2) x (num_hyb + 2)] 
            2D array of transition edge weights, padded with initial and terminal gap penalties
    '''
    if not isinstance(cell_pts, pd.core.frame.DataFrame):
        raise TypeError('cell_pts needs to be a pandas DataFrame.')
        
    if not set(['hyb', 'CurrIndex']).issubset(cell_pts.columns):
        raise KeyError('cell_pts must have the following columns: [hyb, CurrIndex]')
                       
    if not cell_pts['hyb'].is_monotonic_increasing:
        raise IndexError('cell_pts[hyb] not sorted.')
        
    if not isinstance(trans_mat, np.ndarray) or not isinstance(gene_dist_matrix, np.ndarray):
        raise TypeError('Adjacency matrix (trans_mat) and genomic distance (gene_dist_matrix) must both be numpy arrays.')
                
    if not gene_dist_matrix.shape[0] == gene_dist_matrix.shape[1] == n_colours:
        raise ValueError('Dimension mismatch: gene_dist_matrix, trans_mat must both be n_colours x n_colours arrays.')
        
    if not all(isinstance(x, (int, float, np.int64, np.float64)) for x in [l_p_bp, corr_fac, exp_stretch, stretch_factor]):
        raise TypeError('l_p_bp, corr_fac, exp_stretch, stretch_factor must be all positive numerical values.')
        
    if not all(x >= 0 for x in [l_p_bp, corr_fac, exp_stretch, stretch_factor]):
        raise ValueError('l_p_bp, corr_fac, exp_stretch, stretch_factor must be all positive numerical values.')
        
    if not isinstance(init_skip, (int, np.int64)) or not isinstance(end_skip, (int, np.int64)):
        raise TypeError('init_skip, end_skip must be integer indeces.')
        
    if not 0 <= init_skip < end_skip:
        raise ValueError('The following must be satisfied: 0 <= init_skip < end_skip .')      
        
    # Pad transition matrix with source and sink
    trans_mat_pad = np.zeros((trans_mat.shape[0]+2, trans_mat.shape[1]+2))

    # dummy probability
    small_num = 1e-26

    # get ideal genomic distance intervals
    bp_intervals = [gene_dist_matrix[0][i] - gene_dist_matrix[0][i-1] for i in range(1, len(gene_dist_matrix[0]))]

    # add "border" transition probabilities
    for h in range(n_colours):

        # use subset index to subset dataframe
        pts_clr_curr = cell_pts.loc[cell_pts['hyb']==h]

        # parse row, col indeces 
        row_idx = pts_clr_curr['CurrIndex'].values.astype(int) + 1 # +1 to shift over from source
        col_idx = [(elem+1)*trans_mat_pad.shape[0]-1 for elem in row_idx] #+1 to shift over from sink

        # calculate ideal bond prob
        # NB: imaginary linkages based on genomic dist (better behaviour)
        ideal_l_arr_row = [log_bond(exp_stretch*interval, l_p_bp, corr_fac, stretch_factor) for interval in bp_intervals[:h]]
        stretch_bond_row = np.sum(ideal_l_arr_row)        
        ideal_l_arr_col = [log_bond(exp_stretch*interval, l_p_bp, corr_fac, stretch_factor) for interval in bp_intervals[h:]]
        stretch_bond_col = np.sum(ideal_l_arr_col)
        
        # calculate values of transition prob
        if h == 0:
            prob_row = small_num
        elif h != 0:
            prob_row = stretch_bond_row
        else:
            prob_row = None

        if h == n_colours - 1:
            prob_col = small_num
        elif h != n_colours -1:
            prob_col = stretch_bond_col
        else:
            prob_col = None

        # update transition matrix
        if prob_row:
            if lim_init_skip == True:
                if h <= init_skip:
                    np.put(trans_mat_pad, row_idx, prob_row)
                else:
                    np.put(trans_mat_pad, row_idx, [0, ] * len(row_idx))
            else:
                np.put(trans_mat_pad, row_idx, prob_row)

        if prob_col:
            if lim_init_skip == True:
                if h >= end_skip:
                    np.put(trans_mat_pad, col_idx, prob_col)
                else:
                    np.put(trans_mat_pad, col_idx, [0, ] * len(col_idx))
            else:
                np.put(trans_mat_pad, col_idx, prob_col)

    # fill in transition matrix "center"
    trans_mat_pad[1:-1, 1:-1] = trans_mat

    # replace nan's as inaccessible edges
    trans_mat_pad[np.isnan(trans_mat_pad)] = 0
    
    return trans_mat_pad


def find_chr(cell_pts_input,
             gene_dist,
             bin_size,
             iteration, 
             nm_per_unit = 0.3,
             pixel_dist = 108.,
             l_p_bp = 150., 
             stretch_factor = boundary_scale_factor, 
             exp_stretch = 1., 
             num_skip = 7, 
             total_num_skip_frac = 0.7,
             init_skip_frac = 0.15,
             theta = np.pi/20, 
             norm_skip_penalty = True,
             lim_init_skip = True,
             lim_min_dist = True,
            ):
    
    '''
    From a DataFrame cell_pts_input, builds a graph where transition probs based on freely jointed chain model
    of DNA and returns the most likely polymer path.

    NB: prob of emission over one path P(X,Π) = conformational distribution function (product of N bonds)
    NB2: prob of emission over all paths P(X) = partition function

    Input:
        cell_pts_input: [DataFrame] 
            spatial coordinates + metadata of locis detected in one nucleus
        gene_dist : [list]
            reference genomic distances between locis imaged on given chr
        bin_size : float
            median base pair interval between genomic loci
        nm_per_unit : float
            length scale of chromosome 
        pixel_dist : float
            nm of one pixel
        l_p_bp: float
            persistence length of DNA (bp)
        stretch_factor: float
            fraction of max allowable ideal bond length to determine skip
        exp_stretch: float
            fraction of ideal bond length expected to span two loci
        num_skip: int
            number of locis allowed to skip for one step
        total_num_skip_frac: float
            fraction of total locis allowed to skip for entire path
        init_skip_frac: float
            fraction of total locis allowed to skip at graph source and sink
        theta: float
            bond angle
        lim_init_skip: boolean
            limit number of skips at graph source and sink
        lim_min_dist : boolean
            penalize successively skipping
    Output:
        trans_mat_pad: [ndarray] 
            adjacency matrix of polymer model
        shortest_path: [list] 
            shortest path (most likely chromosome)
        shortest_path_length: float 
            conformational distribution function (CDF) of most likely polymer
    '''
    
    #if not stretch_factor > exp_stretch >= 1:
    #    raise ValueError('stretch_factor must be greater than exp_stretch, which must be greater or equal to 1.')
        
    #if not bin_size/l_p_bp >= 10:
    #    raise ValueError('bin_size (countour length) must be >> l_p_bp. Double check the input persistence length and bin size.')
        
    ## Define constants ##
    n_colours = len(gene_dist)
    total_num_skip = int(total_num_skip_frac*len(gene_dist)) 
    corr_fac = nm_per_unit / pixel_dist
    init_skip = int(init_skip_frac*n_colours)
    end_skip = int((1-init_skip_frac)*n_colours)
    gene_dist_matrix = np.abs( np.array([gene_dist]* len(gene_dist)) - np.array([gene_dist]*len(gene_dist)).transpose() )
    gdintervals = [gene_dist_matrix[0][i] - gene_dist_matrix[0][i-1] for i in range(1, len(gene_dist_matrix[0]))]
    
    #if not np.all([elem/l_p_bp >= 10 for elem in gdintervals]):
    #    raise ValueError('Spatial distance estimated from genomic intervals separating loci (contour length) must be >> l_p (pixel dist). Double check the input persistence length (l_p_bp) and reference genome intervals (gene_dist).')

    # make copy of input dataframe
    cell_pts = copy.deepcopy(cell_pts_input)
    
    # check if sorted
    try:
        assert cell_pts['hyb'].is_monotonic_increasing   
    except AssertionError:
        cell_pts = cell_pts.sort_values(by='hyb')  

    # add current index
    cell_pts.reset_index(inplace=True, drop = True)
    cell_pts['CurrIndex'] = cell_pts.index

    # create transition matrix
    n_states =  cell_pts.shape[0]  
    trans_mat = np.zeros((n_states, n_states))

    for i in set(cell_pts['hyb']):
        # grab nodes of curr hyb and reachable hyb
        pts_clr_curr = cell_pts.loc[cell_pts['hyb']==i]
        pts_clr_next = cell_pts.loc[cell_pts['hyb'].between(i, i+num_skip, inclusive = 'neither')]

        # grab node indeces in adjacency matrix

        ## NB: rows --> starting vertices | cols --> ending vertices
        rows = pts_clr_curr['CurrIndex'].values
        cols = pts_clr_next['CurrIndex'].values

        # convert node indeces to position in transition matrix
        trans_idx = (np.array([n_states*rows, ] * len(cols)).T + np.array(cols)).flatten().astype(int)

        # find degrees of separation between nodes
        next_hyb = i+1
        skips = np.array([pts_clr_next['hyb'].values,] * pts_clr_curr.shape[0] ) - np.min([next_hyb, n_colours-1])

        

        # calculate edge weights
        trans_prob = edge_weights(pts_clr_curr, pts_clr_next, 
                                  bin_size, l_p_bp, 
                                  nm_per_unit, pixel_dist, 
                                  theta, gene_dist_matrix, 
                                  skips,i, num_skip, iteration, lim_min_dist)

        # update transition matrix
        np.put(trans_mat, trans_idx, trans_prob)

    # calculate initial and terminal gap penalties
    trans_mat_pad = boundary_init(trans_mat=trans_mat,
                                  gene_dist_matrix=gene_dist_matrix, 
                                  l_p_bp=l_p_bp,
                                  corr_fac=corr_fac,
                                  n_colours=n_colours,
                                  cell_pts=cell_pts,
                                  exp_stretch=exp_stretch,
                                  stretch_factor=stretch_factor,
                                  lim_init_skip=lim_init_skip,
                                  init_skip=init_skip,
                                  end_skip=end_skip)
    
    
    if not np.all(trans_mat_pad >= 0):
        raise ValueError('Edge weights cannot be negative. Double check persistence length (l_p_bp), bin size (bin_size), distance parameter (nm_per_unit) and pixel distance (pixel_dist).')
    
    # create discrete state space model
    G = igraph.Graph.Adjacency((trans_mat_pad > 0).tolist())


    # add edge weights
    G.es['weight'] = trans_mat_pad[trans_mat_pad.nonzero()]
    
    # check boundary conditions
    if len(np.unique(cell_pts_input.hyb)) < len(gene_dist) - total_num_skip:
        print('skipeed?')
        return trans_mat_pad, [], -1
        
    else:
        try:
            # find shortest path (Dijkstra)
            shortest_path_length = G.shortest_paths(source = 0, 
                                                    target = trans_mat_pad.shape[0]-1, 
                                                    weights = 'weight')[0][0]
            
            ## NB: path is written in index of FUTURE cell_pts dataframe (subsetted after iterative subtraction)
            shortest_path = [elem-1 for elem in G.get_shortest_paths(0, to = trans_mat_pad.shape[0]-1, 
                                                                     weights = 'weight')[0][1:-1]]
            
            
            if len(shortest_path) < len(gene_dist) - total_num_skip:
                return trans_mat_pad, [], -1
        except:
            return trans_mat_pad, [], -1
            

    return trans_mat_pad, shortest_path, shortest_path_length


def find_all_chr(cell_pts_input,
                 gene_dist,
                 bin_size,
                 nm_per_unit = 0.3,
                 num_skip = 5,
                 total_num_skip_frac = 0.7,
                 init_skip_frac = 0.15,
                 pixel_dist = 108.,
                 l_p_bp = 150., 
                 stretch_factor = boundary_scale_factor, 
                 exp_stretch = 1., 
                 theta = np.pi/20, 
                 norm_skip_penalty = True,
                 lim_init_skip = True,
                 max_iter = 5):
    '''
    From a DataFrame cell_pts_input, iteratively builds graphs where transition probs based on 
    freely jointed chain model of DNA. Each iteration finds a most likely path, upon which
    nodes on the path are subtracted from the graph and a new graph is built for the next
    iteration.

    Input:
        cell_pts_input: [DataFrame] 
            spatial coordinates + metadata of locis detected in one nucleus
        gene_dist : [list]
            reference genomic distances between locis imaged on given chr
        bin_size : float
            median base pair interval between genomic loci
        nm_per_unit : float
            np per bp
        num_skip: int
            number of locis allowed to skip for one step
        total_num_skip_frac: float
            fraction of total locis allowed to skip for entire path
        init_skip_frac: float
            fraction of total locis allowed to skip at graph source and sink
        pixel_dist : float
            nm of one pixel
        l_p_bp: float
            persistence length of DNA (bp)
        stretch_factor: float
            fraction of max allowable ideal bond length to determine skip
        exp_stretch: float
            fraction of ideal bond length expected to span two loci
        theta: float
            bond angle
        lim_init_skip: boolean
            limit number of skips at graph source and sink
        max_iter : int
            max iteration for iterative path finding
    Output:
        all_put_chr: [list]
            list of DataFrames, each an orthogonal set of coords belonging to a chromatin fiber
    '''
    # define constants
    n_colours = len(gene_dist)
    cdf_threshold_for_path = cdf_thresh(gene_dist, l_p_bp, nm_per_unit/pixel_dist)
    total_num_skip = int(total_num_skip_frac * len(gene_dist))
    
    # copy dataframe
    cell_pts = copy.deepcopy(cell_pts_input)
    
    # check hyb rnd is int
    try:
        assert cell_pts['hyb'].dtype == int

    except AssertionError:
        warnings.warn("Hybridization rounds should be integers.")
        cell_pts['hyb'] = cell_pts['hyb'].astype(int)
        
    # check if sorted
    try:
        assert cell_pts['hyb'].is_monotonic_increasing
    
    except AssertionError:
        warnings.warn("Input DataFrame needs to be sorted in order of imaged loci ('hyb').")
        cell_pts = cell_pts.sort_values(by='hyb')    

    # drop index
    cell_pts.reset_index(inplace=True, drop = True)
    
    all_put_chr = []   
    last_len = cell_pts.shape[0]
    last_cdf = 0
    last_path = None
    all_put_chr_parameters = []
    short_path_encountered = False

    for iteration in range(max_iter):
        
        if iteration == 0:
            if last_len <= 0:
                break
        else:
            if (len(cell_pts) <= 0) or \
               (last_cdf >= cdf_threshold_for_path):
                break

        _, path, cd_func = find_chr(cell_pts, 
                                    gene_dist = gene_dist,
                                    bin_size = bin_size, 
                                    iteration = iteration,
                                    nm_per_unit = nm_per_unit, 
                                    stretch_factor = stretch_factor, 
                                    num_skip = num_skip, 
                                    theta=theta,
                                    total_num_skip_frac = total_num_skip_frac,
                                    init_skip_frac = init_skip_frac,
                                    lim_init_skip = True)

        # grab fiber pts
        put_chr = cell_pts.iloc[[elem for elem in path]]
        
        # threshold for visitation length and physical likelihood
        if (len(path) >= n_colours - total_num_skip) and (cd_func < cdf_threshold_for_path):
            # save chr
            all_put_chr.append(put_chr)

        # measure length
        last_cdf = cd_func
        last_len = len(cell_pts)
        last_path = path
        
        # prune nodes
        cell_pts = cell_pts.iloc[[i for i in range(cell_pts.shape[0]) if i not in path]]

    return all_put_chr


# Function to sort picked spots based on (local) fiber density, this would help resolve less probable trans-homolog transversing (like "fiber swapping during tracing")
def sort_spots_by_fiber_density (picked_chr_pts_cell, _iter_num_th = 30, _local_reg_num = 30):
    #import scipy
    from scipy.spatial.distance import cdist
    from itertools import permutations

    picked_chr_pts_cell_new =  picked_chr_pts_cell.copy(deep=True)
    chosen_chrom = picked_chr_pts_cell_new['chr'].values[0]
    cell_id = picked_chr_pts_cell_new['orig_cellID'].values[0]

    # process if more than one fiber exists for a cell
    if len(np.unique(picked_chr_pts_cell_new['fiberidx'])) >1:
        sel_pts_zxyhfa = picked_chr_pts_cell[['z_hat','x_hat','y_hat','hyb','fiberidx']].copy(deep=True)
        sel_pts_zxyhfa['fiberidx_jie']=sel_pts_zxyhfa['fiberidx']
        _iter = 0
        while _iter <_iter_num_th:
            _iter +=1
            curr_fiberidx = sel_pts_zxyhfa['fiberidx'] # flag the current fiberidx
            picked_hyb_list = sorted(np.unique(sel_pts_zxyhfa['hyb']))
            # 10% of the longest picked fiber as local ref 
            _neighbor_len = round(np.max(picked_hyb_list)/5)
            for _hyb in picked_hyb_list:
                # sel_pts_zxyhfa below will be updated after processing each hyb
                _cand_spots = sel_pts_zxyhfa[sel_pts_zxyhfa['hyb']==_hyb]
                # local neighboring region
                _start, _end = max(_hyb - _neighbor_len, np.min(picked_hyb_list)), min(_hyb + _neighbor_len, np.max(picked_hyb_list)) 
                ref_ct_list = []
                # assign spots to closest fiber
                for _fiber_id in np.unique(sel_pts_zxyhfa['fiberidx']): # order of fibers to use
                    sel_pts_cluster = sel_pts_zxyhfa[sel_pts_zxyhfa['fiberidx']==_fiber_id]
                    _local_hyb_list = np.intersect1d(np.arange(_start, _end+1), np.unique(sel_pts_cluster['hyb']))
                    # use local center if there are enough nearby spots
                    if len(_local_hyb_list) >= _local_reg_num:
                        ref_spots = sel_pts_cluster[sel_pts_cluster['hyb'].isin(_local_hyb_list)]
                    # use all the decoded chr fiber center
                    else:
                        ref_spots = sel_pts_cluster.copy(deep=True)
                    ref_ct = np.nanmean(ref_spots[['z_hat','x_hat','y_hat']],axis=0) # local center coords
                    ref_ct_list.append(ref_ct)
                # dist matrix for cand_spots and ref_centers
                ref_cts = np.array(ref_ct_list)
                _dist_mtx =  cdist(_cand_spots[['z_hat','x_hat','y_hat']],ref_cts)
                # re-assign for fibers based on the min summed distance
                _cand_spots_new = _cand_spots.copy(deep=True)
                _l = list(permutations(range(0, len(ref_cts)),len(_dist_mtx)))
                _sum_dist_list = []
                for _i in _l:
                    # distance sum for this permutation
                    _dist_sum = 0
                    for _spot_index in range(len(_dist_mtx)):
                        _dist_sum += _dist_mtx [_spot_index,_i[_spot_index]] 
                    _sum_dist_list.append(_dist_sum)
                ########################################## 
                # pick the permutation with lowest summed distance
                _exchanged_index = _l[np.argmin(_sum_dist_list)]
                _cand_spots_new['fiberidx']= np.unique(sel_pts_zxyhfa['fiberidx'])[list(_exchanged_index)]
                # assign back to the sel_pts_zxyhfa for the next hyb
                sel_pts_zxyhfa_new = sel_pts_zxyhfa.copy(deep=True)
                sel_pts_zxyhfa_new[sel_pts_zxyhfa_new['hyb']==_hyb] = _cand_spots_new
                sel_pts_zxyhfa = sel_pts_zxyhfa_new.copy(deep=True)

            # check reshuffling rate
            num_change = np.sum(sel_pts_zxyhfa['fiberidx']!=curr_fiberidx)
            rate_change = num_change/len(sel_pts_zxyhfa)
            if rate_change<=0.001:
                break
                
        print(f'Complete sorting for {chosen_chrom} in {cell_id} within {_iter} iteration(s).')
        picked_chr_pts_cell_new[['z_hat','x_hat','y_hat','hyb','fiberidx','fiberidx_jie']]=sel_pts_zxyhfa[['z_hat','x_hat','y_hat','hyb','fiberidx','fiberidx_jie']]
        
    # skip if only one fiber
    else:
        picked_chr_pts_cell_new['fiberidx_jie']=picked_chr_pts_cell_new['fiberidx']
        print(f'Skip sorting for {chosen_chrom} in {cell_id} because there is only one fiber.')
        
    return picked_chr_pts_cell_new