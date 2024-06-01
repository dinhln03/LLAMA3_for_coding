from __future__ import print_function
from __future__ import absolute_import
from __future__ import division
import numpy as np

__all__ = [
    "wigner3j",
    "get_camb_cl",
    "scale_dust",
]


def blackbody(nu, ref_freq=353.0):
    """
    The ratio of the blackbody function for dust at frequency nu
    over the value for reference frequency ref_freq

    Arguments
    ---------
    nu : float
        Frequency in GHz.
    ref_freq : float
        Reference frequency in GHz.

    Returns
    -------
    blackbody_ratio : float
        B(nu, T_dust) / B(nu_ref, T_dust)
    """
    k = 1.38064852e-23  # Boltzmann constant
    h = 6.626070040e-34  # Planck constant
    T = 19.6
    nu_ref = ref_freq * 1.0e9
    nu *= 1.0e9  # GHz -> Hz
    x = h * nu / k / T
    x_ref = h * nu_ref / k / T
    return x ** 3 / x_ref ** 3 * (np.exp(x_ref) - 1) / (np.exp(x) - 1)


def rj2cmb(nu_in):
    """
    Conversion from Rayleigh-Jeans units to CMB temperature units

    Arguments
    ---------
    nu_in : float
        Frequency in GHz.

    Returns
    -------
    cal_fac : float
        Number by which to multiply a RJ temperature to get a CMB temp
    """
    k = 1.38064852e-23  # Boltzmann constant
    h = 6.626070040e-34  # Planck constant
    T = 2.72548  # Cmb BB temp in K
    nu = nu_in * 1.0e9  # GHz -> Hz
    x = h * nu / k / T
    return (np.exp(x) - 1.0) ** 2 / (x ** 2 * np.exp(x))


def scale_dust(freq0, freq1, ref_freq, beta, delta_beta=None, deriv=False):
    """
    Get the factor by which you must multiply the cross spectrum from maps of
    frequencies freq0 and freq1 to match the dust power at ref_freq given
    spectra index beta.

    If deriv is True, return the frequency scaling at the reference beta,
    and the first derivative w.r.t. beta.

    Otherwise if delta_beta is given, return the scale factor adjusted
    for a linearized offset delta_beta from the reference beta.

    Arguments
    ---------
    freq0 : float
        Frequency of map0 in GHz.
    freq1 : float
        Frequency of map1 in GHz.
    ref_freq : float
        Reference frequency from which to compute relative scaling in GHz.
    beta : float
        Dust spectral index.
    delta_beta : float
        Difference from beta-- scaling computed as a first order Taylor
        expansion from original beta-scaling.
    deriv : bool
        If true, return the frequency scaling at the reference beta, along with
        the first derivative w.r.t. beta at the reference beta.

    Returns
    -------
    freq_scale : float
        The relative scaling factor for the dust cross spectrum-- multiply by
        this number to get the dust spectrum at the reference frequency
    -- or --
    freq_scale, deriv : floats
        The relative scaling factor and its derivative
    """
    freq_scale = (
        rj2cmb(freq0)
        * rj2cmb(freq1)
        / rj2cmb(ref_freq) ** 2.0
        * blackbody(freq0, ref_freq=ref_freq)
        * blackbody(freq1, ref_freq=ref_freq)
        * (freq0 * freq1 / ref_freq ** 2) ** (beta - 2.0)
    )

    if deriv or delta_beta is not None:
        delta = np.log(freq0 * freq1 / ref_freq ** 2)
        if deriv:
            return (freq_scale, freq_scale * delta)
        return freq_scale * (1 + delta * delta_beta)

    return freq_scale


def wigner3j(l2, m2, l3, m3):
    r"""
    Wigner 3j symbols computed for all valid values of ``L``, as in:

    .. math::

        \begin{pmatrix}
         \ell_2 & \ell_3 & L \\
         m_2 & m_3 & 0 \\
        \end{pmatrix}

    Arguments
    ---------
    l2, m2, l3, m3 : int
        The ell and m values for which to compute the symbols.

    Returns
    -------
    fj : array_like
        Array of size ``l2 + l3 + 2``, indexed by ``L``
    lmin : int
        The minimum value of ``L`` for which ``fj`` is non-zero.
    lmax : int
        The maximum value of ``L`` for which ``fj`` is non-zero.
    """
    import camb

    try:
        from camb.mathutils import threej
    except ImportError:
        from camb.bispectrum import threej
    arr = threej(l2, l3, m2, m3)

    lmin = np.max([np.abs(l2 - l3), np.abs(m2 + m3)])
    lmax = l2 + l3
    fj = np.zeros(lmax + 2, dtype=arr.dtype)
    fj[lmin : lmax + 1] = arr
    return fj, lmin, lmax


def get_camb_cl(r, lmax, nt=None, spec="total", lfac=True):
    """
    Compute camb spectrum with tensors and lensing.

    Parameter values are from arXiv:1807.06209 Table 1 Plik best fit

    Arguments
    ---------
    r : float
        Tensor-to-scalar ratio
    lmax : int
        Maximum ell for which to compute spectra
    nt : scalar, optional
        Tensor spectral index.  If not supplied, assumes
        slow-roll consistency relation.
    spec : string, optional
        Spectrum component to return.  Can be 'total', 'unlensed_total',
        'unlensed_scalar', 'lensed_scalar', 'tensor', 'lens_potential'.
    lfac: bool, optional
        If True, multiply Cls by ell*(ell+1)/2/pi

    Returns
    -------
    cls : array_like
        Array of spectra of shape (lmax + 1, nspec).
        Diagonal ordering (TT, EE, BB, TE).
    """
    # Set up a new set of parameters for CAMB
    import camb

    pars = camb.CAMBparams()

    # This function sets up CosmoMC-like settings, with one massive neutrino and
    # helium set using BBN consistency
    pars.set_cosmology(
        H0=67.32,
        ombh2=0.022383,
        omch2=0.12011,
        mnu=0.06,
        omk=0,
        tau=0.0543,
    )

    ln1010As = 3.0448

    pars.InitPower.set_params(As=np.exp(ln1010As) / 1.0e10, ns=0.96605, r=r, nt=nt)
    if lmax < 2500:
        # This results in unacceptable bias. Use higher lmax, then cut it down
        lmax0 = 2500
    else:
        lmax0 = lmax
    pars.set_for_lmax(lmax0, lens_potential_accuracy=2)
    pars.WantTensors = True
    pars.do_lensing = True

    # calculate results for these parameters
    results = camb.get_results(pars)
    powers = results.get_cmb_power_spectra(pars, CMB_unit="muK", raw_cl=not lfac)

    totCL = powers[spec][: lmax + 1, :4].T

    return totCL
