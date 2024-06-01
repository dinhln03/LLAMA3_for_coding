# External Dependencies
from __future__ import division
from numpy import isclose
from svgpathtools import Path

# Internal Dependencies
from misc4rings import isNear


class ClosedRingsOverlapError(Exception):
    def __init__(self,mes):
        self.mes = mes
    def __str__(self):
       return repr(self.mes)


def findAppropriateTstep(path, T, stepInPositiveDirection):
# Often the overlapping part of two paths is so small that when removed, pathXpathIntersections, will still consider the two curves as intersecting.  This function is to find the smallest (signed) Tstep such that isNear(path(T),path(T+Tstep))==False.
# note: stepInPositiveDirection should be True if Tstep should be positve

    # set initial guess as max possible step distance (and set sign of Tstep)
    # T = float(T)
    if stepInPositiveDirection:
        Tstep = 1 - T
    else:
        Tstep = 0 - T

    #check that what we're asking for is possible
    if isNear(path.point(T + Tstep), path.point(T)):
        raise Exception("An impossible Tstep was asked for.")

    #Find a lower bound for Tstep by bisection
    maxIts = 200  # limits Tstep to be > (1/2)**200
    its = 0
    while not isNear(path.point(T + Tstep), path.point(T)) and its < maxIts:
        Tstep /= 2
        its += 1
    if its >= maxIts:
        raise Exception("Max iterations reached in bisection to find "
                        "appropriate Tstep.  This could theoretically be ok "
                        "if you have a curve with a huge number of "
                        "segments... just increase the maxIts in "
                        "findAppropriateTstep if you have a such a curve "
                        "(but I doubt that's the case - so tell Andy).")
    return 2 * Tstep


def shortPart(path,T):
    if isclose(T, 0) or isclose(T, 1):
        return Path()
    if T < 1-T:  # T is closer to 0
        # return cropPath(path,0,T)
        return path.cropped(0, T)
    else:  # T is closer to 1
        # return cropPath(path,T,1)
        return path.cropped(T, 1)


def longPart(path, T, remove_a_little_extra=True):
    if remove_a_little_extra:
        if T < 1 - T:  # T is closer to 0 than 1
            extra = T
            if isNear(path.point(T + extra), path.point(T)):
                extra = findAppropriateTstep(path, T, True)
        else:  # T is closer to 1 than 0
            extra = 1-T
            if isNear(path.point(T+extra), path.point(T)):
                extra = -1 * findAppropriateTstep(path, T, False)
    else:
        extra = 0
    if T < 1 - T: #T is closer to 0 than 1
        # return cropPath(path,T+extra,1)
        return path.cropped(T + extra, 1)
    else: #T is closer to 1 than 0
        # return cropPath(path,0,T-extra)
        return path.cropped(0, T - extra)


def remove_intersections(ipath, jpath, iclosed, jclosed, iringupdated=False, jringupdated=False): #removes one intersection at a time until all are gone
    new_ipath = ipath
    new_jpath = jpath

    #find all intersections
    res = ipath.intersect(jpath, justonemode=True)
    # res = pathXpathIntersections(ipath, jpath, justonemode=True)
    if res:
        iT, iseg, i_t  = res[0]
        jT, jseg, j_t = res[1]
        # iT = ipath.t2T(iseg, i_t)
        # jT = jpath.t2T(jseg, j_t)
    else:
        run_again = False
        return new_ipath, new_jpath, iringupdated, jringupdated, run_again


    #Now find crop the path (if one ring is closed, crop the other ring)
    if iclosed and jclosed: #then crop jpath
        raise ClosedRingsOverlapError("")

    elif jclosed: #jring closed so crop iring
        new_ipath = longPart(ipath, iT)
        new_jpath = jpath
        iringupdated = True

    elif iclosed: #iring closed so crop jring
        new_jpath = longPart(jpath, jT)
        new_ipath = ipath
        jringupdated = True

    else: #both rings are incomplete
        if iT in [0, 1]:
            new_ipath = longPart(ipath, iT)
            new_jpath = jpath
            iringupdated = True
        elif jT in [0, 1]:
            new_jpath = longPart(jpath, jT)
            new_ipath = ipath
            jringupdated = True
        elif shortPart(ipath, iT).length() < shortPart(jpath, jT).length():
            new_ipath = longPart(ipath, iT)
            new_jpath = jpath
            iringupdated = True
        else:
            new_jpath = longPart(jpath, jT)
            new_ipath = ipath
            jringupdated = True
    run_again = True  # might be more intersections to remove, so run again
    return new_ipath, new_jpath, iringupdated, jringupdated, run_again


def remove_intersections_from_rings(rings):
    from options4rings import intersection_removal_progress_output_on
    from time import time as current_time
    from andysmod import n_choose_k, format_time

    [r.record_wasClosed() for r in rings]  # record the current closure status

    #for output
    num_segments_in_ring_list = sum(len(r.path) for r in rings)
    num_seg_pairs2check = n_choose_k(num_segments_in_ring_list, 2)
    num_seg_pairs_checked = 0
    current_percent_complete = 0
    start_time = current_time()

    count = 0
    overlappingClosedRingPairs = []
    for i in range(len(rings)):
        iring = rings[i]
        ipath = iring.path
        new_ipath = ipath
        iclosed = iring.wasClosed
        iringupdated = False
        num_segs_in_ipath = len(ipath) # for progress output

        for j in range(i+1, len(rings)):
            if rings[j].maxR < rings[i].minR or rings[i].maxR < rings[j].minR:
                continue
            jring = rings[j]
            jpath = jring.path
            new_jpath = jpath
            jclosed = jring.wasClosed
            jringupdated = False
            num_segs_in_jpath = len(jpath) #for progress output

            # while loop to remove intersections between iring and jring (if any exist)
            run_again = True
            maxits = 20
            its = 0
            while run_again and its < maxits:
                try:
                    args = (new_ipath, new_jpath, iclosed, jclosed)
                    res = remove_intersections(*args, iringupdated=iringupdated, jringupdated=jringupdated)
                    new_ipath, new_jpath, iringupdated, jringupdated, run_again = res
                except ClosedRingsOverlapError:
                    overlappingClosedRingPairs.append((i, j))
                    run_again = False
                    pass
                its += 1

            # raise Exception if while loop terminateded due to reaching max allowed iteratations
            if its >= maxits:
                # remove_intersections_from_rings([iring, jring])
                # print(iring.xml)
                # print(jring.xml)
                raise Exception("Max iterations reached while removing intersections.  Either the above two rings have over 20 intersections or this is a bug.")

            # Output progess
            if intersection_removal_progress_output_on.b:
                num_seg_pairs_checked += num_segs_in_jpath*num_segs_in_ipath
                if 100 * num_seg_pairs_checked / num_seg_pairs2check > int(100 * current_percent_complete):
                    current_percent_complete = num_seg_pairs_checked / num_seg_pairs2check
                    time_elapsed = current_time() - start_time
                    estimated_time_remaining = (1-current_percent_complete) * time_elapsed / current_percent_complete
                    stuff = (int(100 * current_percent_complete),
                             format_time(estimated_time_remaining),
                             format_time(time_elapsed))
                    mes = ("[%s%% complete || Est. Remaining Time = %s || "
                           "Elapsed Time = %s]\r" % stuff)
                    intersection_removal_progress_output_on.dprint(mes)

            # update jring if jpath was trimmed
            if jringupdated:
                jring.updatePath(new_jpath)
                count += 1
        # update iring if ipath was trimmed
        if iringupdated:
            iring.updatePath(new_ipath)
            count += 1
    return rings, count, overlappingClosedRingPairs