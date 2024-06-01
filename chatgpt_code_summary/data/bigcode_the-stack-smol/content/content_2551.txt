"""
    Functions computing the signal shapes
"""

import numpy as np
from time import time

import src.constants as const


def subtract_signal(t, signal, fit_params=3):
    """

    Returns the subtracted signal

    """

    # fit dphi(t) to polynomials and subtract the contribution from n=0, 1 and 2
    coef = np.polynomial.polynomial.polyfit(t, signal, fit_params - 1)  # (3)
    delta_signal = np.einsum(
        "n,nj->j", coef, np.asarray([np.power(t, n) for n in range(fit_params)])
    )  # (Nt)

    # compute the subtracted signal
    ht = signal - delta_signal  # (Nt), unit = s

    return ht


def dphi_dop_chunked(
    t,
    profile,
    r0_vec,
    v_vec,
    d_hat,
    use_form=False,
    use_chunk=False,
    chunk_size=10000,
    verbose=False,
    form_fun=None,
    interp_table=None,
    time_end=np.inf,
):
    """

    Compute dphi but in chunks over the subhalos, use when Nt x N is too large an array to
    store in memory

    """

    num_objects = len(list(profile.items())[0][1]) # number of elements of 1st dict entry

    dphi = np.zeros(len(t))

    if use_chunk == True:

        if num_objects % chunk_size == 0:
            num_chunks = num_objects // chunk_size
        else:
            num_chunks = num_objects // chunk_size + 1

        if verbose:
            print("   Chunking data (%d chunks) ... "%num_chunks)
            print()

        for i in range(num_chunks):
            
            if time() > time_end: raise TimeoutError

            r0_c = r0_vec[i * chunk_size : (i + 1) * chunk_size]
            v_c = v_vec[i * chunk_size : (i + 1) * chunk_size]

            profile_c = {}
            for key in list(profile):
                profile_c[key] = profile[key][i * chunk_size : (i + 1) * chunk_size]

            dphi += dphi_dop(
                t, profile_c, r0_c, v_c, d_hat, use_form=use_form, form_fun=form_fun, interp_table=interp_table
            )
    else:

        dphi += dphi_dop(t, profile, r0_vec, v_vec, d_hat, use_form=use_form, form_fun=form_fun, interp_table=interp_table)

    return dphi


def dphi_dop_chunked_vec(
    t,
    profile,
    r0_vec,
    v_vec,
    use_form=False,
    use_chunk=False,
    chunk_size=10000,
    verbose=False,
    form_fun=None,
    interp_table=None,
    time_end=np.inf,
):
    """

    Compute dphi but in chunks over the subhalos, use when Nt x N is too large an array to
    store in memory

    """

    num_objects = len(list(profile.items())[0][1]) # number of elements of 1st dict entry

    dphi_vec = np.zeros((len(t), 3))

    if use_chunk == True:

        if verbose:
            print("   Chunking data ... ")
            print()

        if num_objects % chunk_size == 0:
            num_chunks = num_objects // chunk_size
        else:
            num_chunks = num_objects // chunk_size + 1

        for i in range(num_chunks):
            
            if time() > time_end: raise TimeoutError

            r0_c = r0_vec[i * chunk_size : (i + 1) * chunk_size]
            v_c = v_vec[i * chunk_size : (i + 1) * chunk_size]

            profile_c = {}
            for key in list(profile):
                profile_c[key] = profile[key][i * chunk_size : (i + 1) * chunk_size]

            dphi_vec += dphi_dop_vec(
                t, profile_c, r0_c, v_c, use_form=use_form, form_fun=form_fun, interp_table=interp_table
            )
    else:

        dphi_vec += dphi_dop_vec(t, profile, r0_vec, v_vec, use_form=use_form, form_fun=form_fun, interp_table=interp_table)

    return dphi_vec


def dphi_dop_vec(t, profile, r0_vec, v_vec, use_form=False, form_fun=None,
             interp_table=None):
    """

    Returns the vector phase shift due to the Doppler delay for subhalos of mass, mass.
    Dot with d_hat to get dphi_I

    TODO: add use_closest option

    """

    v_mag = np.linalg.norm(v_vec, axis=1)

    r0_v = np.einsum("ij, ij -> i", r0_vec, v_vec)
    t0 = -r0_v / np.square(v_mag)  # year

    b_vec = r0_vec + v_vec * t0[:, np.newaxis]  # (N, 3)
    b_mag = np.linalg.norm(b_vec, axis=1)  # (N)
    tau = b_mag / v_mag

    b_hat = b_vec / b_mag[:, np.newaxis]  # (N, 3)
    v_hat = v_vec / v_mag[:, np.newaxis]

    x = np.subtract.outer(t, t0) / tau
    x0 = -t0 / tau

    prefactor = (
        const.yr_to_s
        * const.GN
        / (const.km_s_to_kpc_yr * const.c_light * np.square(v_mag))
    )

    if interp_table is None:
        
        bd_term = (np.sqrt(1 + x ** 2) + x) - (np.sqrt(1 + x0 ** 2) + x0)  # (Nt, N)
        vd_term = np.arcsinh(x) - np.arcsinh(x0)
        
        if 'M' in list(profile):
            prefactor *= profile['M']
    
            if use_form:
    
                t_cl = np.maximum(np.minimum(t0, t[-1]), 0)
                x_cl = (t_cl - t0) / tau
                r_cl = tau * v_mag * np.sqrt(1 + x_cl ** 2)
    
                rv = ((3 * profile['M'] / (4 * np.pi)) * (1 / 200) * (1 / const.rho_crit)) ** (1 / 3)
    
                form_func = np.where(r_cl<rv, form(r_cl / rv, profile['c']), 1)  # (N)
    
                bd_term *= prefactor * form_func
                vd_term *= prefactor * form_func
    
            else:
    
                bd_term = prefactor * bd_term
                vd_term = prefactor * vd_term
        else:
            if form_fun is not None:
                t_cl = np.maximum(np.minimum(t0, t[-1]), 0)
                x_cl = (t_cl - t0) / tau
                r_cl = tau * v_mag * np.sqrt(1 + x_cl ** 2)
                
                form_func = form_fun(r_cl, profile['rs'], profile['rhos'])
    
                bd_term *= prefactor * form_func
                vd_term *= prefactor * form_func
    
            else:
                raise ValueError('rho_s, r_s halo description currently requires custom density profile ("USE_FORMTAB")')
                
    else:
        
        y = b_mag / profile['rs']
        
        bd_term0, vd_term0 = interp_table.bd_vd_terms(x0, y)
        
        y.shape = (1,-1)
        y = np.broadcast_to(y,x.shape)
        
        bd_term, vd_term = interp_table.bd_vd_terms(x, y)
        
        bd_term -= bd_term0
        vd_term -= vd_term0
    
        bd_term *= prefactor * profile['rhos'] * profile['rs']**3
        vd_term *= prefactor * profile['rhos'] * profile['rs']**3

    # sum the signal over all the events
    sig = np.einsum("to, oi -> ti", bd_term, b_hat) - np.einsum(
        "to, oi -> ti", vd_term, v_hat
    )

    return sig


def dphi_dop(t, profile, r0_vec, v_vec, d_hat, use_form=False, form_fun=None,
             interp_table=None):
    """

    Returns the phase shift due to the Doppler delay for subhalos of mass, mass

    TODO: add use_closest option

    """

    v_mag = np.linalg.norm(v_vec, axis=1)

    r0_v = np.einsum("ij, ij -> i", r0_vec, v_vec)  # kpc^2/yr
    t0 = -r0_v / np.square(v_mag)  # year

    b_vec = r0_vec + v_vec * t0[:, np.newaxis]  # (N, 3), kpc
    b_mag = np.linalg.norm(b_vec, axis=1)  # (N)
    tau = b_mag / v_mag  # year

    b_hat = b_vec / b_mag[:, np.newaxis]
    v_hat = v_vec / v_mag[:, np.newaxis]

    b_d = np.dot(b_hat, d_hat)
    v_d = np.dot(v_hat, d_hat)

    x = np.subtract.outer(t, t0) / tau
    x0 = -t0 / tau

    prefactor = (
        const.yr_to_s
        * const.GN
        / (const.km_s_to_kpc_yr * const.c_light * np.square(v_mag))
    )

    if interp_table is None:
        
        bd_term = (np.sqrt(1 + x ** 2) + x) - (np.sqrt(1 + x0 ** 2) + x0)
        vd_term = np.arcsinh(x) - np.arcsinh(x0)
    
        sig = bd_term * b_d - vd_term * v_d
    
        if 'M' in list(profile):
            prefactor *= profile['M']
    
            if use_form:
    
                t_cl = np.maximum(np.minimum(t0, t[-1]), 0)
                x_cl = (t_cl - t0) / tau
                r_cl = tau * v_mag * np.sqrt(1 + x_cl ** 2)
    
                rv = ((3 * profile['M'] / (4 * np.pi)) * (1 / 200) * (1 / const.rho_crit)) ** (1 / 3)
    
                form_func = np.where(r_cl<rv, form(r_cl / rv, profile['c']), 1)  # (N)
    
                sig = form_func * sig
        else:
            if form_fun is not None:
                t_cl = np.maximum(np.minimum(t0, t[-1]), 0)
                x_cl = (t_cl - t0) / tau
                r_cl = tau * v_mag * np.sqrt(1 + x_cl ** 2)
                
                form_func = form_fun(r_cl, profile['rs'], profile['rhos'])
    
                sig = form_func * sig
    
            else:
                raise ValueError('rho_s, r_s halo description currently requires custom density profile ("USE_FORMTAB")')
                
    else:
        
        y = b_mag / profile['rs']
        
        bd_term0, vd_term0 = interp_table.bd_vd_terms(x0, y)
        
        y.shape = (1,-1)
        y = np.broadcast_to(y,x.shape)
        
        bd_term, vd_term = interp_table.bd_vd_terms(x, y)
        
        bd_term -= bd_term0
        vd_term -= vd_term0
    
        sig = profile['rhos'] * profile['rs']**3 * (bd_term * b_d + vd_term * v_d)

    sig = prefactor * sig

    # sum the signal over all the events
    return np.sum(sig, axis=-1)


def form(s, c):

    return (np.log(1 + c * s) - c * s / (1 + c * s)) / (np.log(1 + c) - c / (1 + c))

