import numpy as np
from PIL import Image, ImageDraw
from scipy import interpolate, ndimage, stats, signal, integrate, misc
from astropy.io import ascii, fits
from astropy.wcs import WCS
from astropy.coordinates import SkyCoord
import astropy.units as u
import astropy.constants as c
import corner as triangle  # formerly dfm/triangle
# from astropy.modeling import models, fitting
from astropy.modeling.models import custom_model
from astropy.modeling.fitting import LevMarLSQFitter # , SimplexLSQFitter
import matplotlib.pyplot as plt
import matplotlib as mpl
import emcee
#import ipdb;
import pdb

# # # # # # # # # # # # # # # # # # # # # #
# make iPython print immediately
import sys
oldsysstdout = sys.stdout


class flushfile():

    def __init__(self, f):
        self.f = f

    def __getattr__(self, name):
        return object.__getattribute__(self.f, name)

    def write(self, x):
        self.f.write(x)
        self.f.flush()

    def flush(self):
        self.f.flush()
# sys.stdout = flushfile(sys.stdout)
# sys.stdout = oldsysstdout


def rot_matrix(theta):
    '''
    rot_matrix(theta)
    2D rotation matrix for theta in radians
    returns numpy matrix
    '''
    c, s = np.cos(theta), np.sin(theta)
    return np.matrix([[c, -s], [s, c]])


def rectangle(c, w, h, angle=0, center=True):
    '''
    create rotated rectangle
    for input into PIL ImageDraw.polygon
    to make a rectangle polygon mask

    Rectagle is created and rotated with center
    at zero, and then translated to center position

    accepters centers
    Default : center
    tl, tr, bl, br
    '''
    cx, cy = c
    # define initial polygon irrespective of center
    x = -w / 2., +w / 2., +w / 2., -w / 2.
    y = +h / 2., +h / 2., -h / 2., -h / 2.
    # correct center if starting from corner
    if center is not True:
        if center[0] == 'b':
            # y = tuple([i + h/2. for i in y])
            cy = cy + h / 2.
        else:
            # y = tuple([i - h/2. for i in y])
            cy = cy - h / 2.
        if center[1] == 'l':
            # x = tuple([i + w/2 for i in x])
            cx = cx + w / 2.
        else:
            # x = tuple([i - w/2 for i in x])
            cx = cx - w / 2.

    R = rot_matrix(angle * np.pi / 180.)
    c = []

    for i in range(4):
        xr, yr = np.dot(R, np.asarray([x[i], y[i]])).A.ravel()
        # coord switch to match ordering of FITs dimensions
        c.append((cx + xr, cy + yr))
    # print (cx,cy)
    return c


def comp(arr):
    '''
    returns the compressed version
    of the input array if it is a
    numpy MaskedArray
    '''
    try:
        return arr.compressed()
    except:
        return arr


def mavg(arr, n=2, mode='valid'):
    '''
    returns the moving average of an array.
    returned array is shorter by (n-1)
    '''
    if len(arr) > 400:
        return signal.fftconvolve(arr, [1. / float(n)] * n, mode=mode)
    else:
        return signal.convolve(arr, [1. / float(n)] * n, mode=mode)


def mgeo(arr, n=2):
    '''
    Returns array of lenth len(arr) - (n-1)

    # # written by me
    # # slower for short loops
    # # faster for n ~ len(arr) and large arr
    a = []
    for i in xrange(len(arr)-(n-1)):
        a.append(stats.gmean(arr[i:n+i]))

    # # Original method# #
    # # written by me ... ~10x faster for short arrays
    b = np.array([np.roll(np.pad(arr,(0,n),mode='constant',constant_values=1),i)
              for i in xrange(n)])
    return np.product(b,axis=0)[n-1:-n]**(1./float(n))
    '''
    a = []
    for i in range(len(arr) - (n - 1)):
        a.append(stats.gmean(arr[i:n + i]))

    return np.asarray(a)


def avg(arr, n=2):
    '''
    NOT a general averaging function
    return bin centers (lin and log)
    '''
    diff = np.diff(arr)
    # 2nd derivative of linear bin is 0
    if np.allclose(diff, diff[::-1]):
        return mavg(arr, n=n)
    else:
        return np.power(10., mavg(np.log10(arr), n=n))
        # return mgeo(arr, n=n) # equivalent methods, only easier

def shift_bins(arr,phase=0,nonneg=False):
    # assume original bins are nonneg
    if phase != 0:
        diff = np.diff(arr)
        if np.allclose(diff,diff[::-1]):
            diff = diff[0]
            arr = arr + phase*diff
            #pre = arr[0] + phase*diff
            return arr
        else:
            arr = np.log10(arr)
            diff = np.diff(arr)[0]
            arr = arr + phase * diff
            return np.power(10.,arr)
    else:
        return arr

def llspace(xmin, xmax, n=None, log=False, dx=None, dex=None):
    '''
    llspace(xmin, xmax, n = None, log = False, dx = None, dex = None)
    get values evenly spaced in linear or log spaced
    n [10] -- Optional -- number of steps
    log [false] : switch for log spacing
    dx : spacing for linear bins
    dex : spacing for log bins (in base 10)
    dx and dex override n
    '''
    xmin, xmax = float(xmin), float(xmax)
    nisNone = n is None
    dxisNone = dx is None
    dexisNone = dex is None
    if nisNone & dxisNone & dexisNone:
        print('Error: Defaulting to 10 linears steps')
        n = 10.
        nisNone = False

    # either user specifies log or gives dex and not dx
    log = log or (dxisNone and (not dexisNone))
    if log:
        if xmin == 0:
            print("log(0) is -inf. xmin must be > 0 for log spacing")
        xmin, xmax = np.log10(xmin), np.log10(xmax)
    # print nisNone, dxisNone, dexisNone, log # for debugging logic
    if not nisNone:  # this will make dex or dx if they are not specified
        if log and dexisNone:  # if want log but dex not given
            dex = (xmax - xmin) / n
            # print dex
        elif (not log) and dxisNone:  # else if want lin but dx not given
            dx = (xmax - xmin) / n  # takes floor
            #print dx

    if log:
        #return np.power(10, np.linspace(xmin, xmax , (xmax - xmin)/dex + 1))
        return np.power(10, np.arange(xmin, xmax + dex, dex))
    else:
        #return np.linspace(xmin, xmax, (xmax-xmin)/dx + 1)
        return np.arange(xmin, xmax + dx, dx)


def nametoradec(name):
    '''
    Get names formatted as
    hhmmss.ss+ddmmss to Decimal Degree
    only works for dec > 0 (splits on +, not -)
    Will fix this eventually...
    '''
    if 'string' not in str(type(name)):
        rightascen = []
        declinatio = []
        for n in name:
            ra, de = n.split('+')
            ra = ra[0:2] + ':' + ra[2:4] + ':' + ra[4:6] + '.' + ra[6:8]
            de = de[0:2] + ':' + de[2:4] + ':' + de[4:6]
            coord = SkyCoord(ra, de, frame='icrs',
                             unit=('hourangle', 'degree'))
            rightascen.append(coord.ra.value)
            declinatio.append(coord.dec.value)
        return np.array(rightascen), np.array(declinatio)
    else:
        ra, de = name.split('+')
        ra = ra[0:2] + ':' + ra[2:4] + ':' + ra[4:6] + '.' + ra[6:8]
        de = de[0:2] + ':' + de[2:4] + ':' + de[4:6]
        coord = SkyCoord(ra, de, frame='icrs', unit=('hourangle', 'degree'))
        return np.array(coord.ra.value), np.array(coord.dec.value)


def get_ext(extmap, errmap, extwcs, ra, de):
    '''
    Get the extinction (errors) for a particular position or
    list of positions
    More generally get the value (error) for a particular
    position given a wcs and world coordinates
    '''
    try:
        xp, yp = extwcs.all_world2pix(
            np.array([ra]).flatten(), np.array([de]).flatten(), 0)
    except:
        xp, yp = WCS(extwcs).all_world2pix(
            np.array([ra]).flatten(), np.array([de]).flatten(), 0)
    ext = []
    err = []
    for i in range(len(np.array(xp))):
        try:
            ext.append(extmap[yp[int(round(i))], xp[int(round(i))]])
            if errmap is not None:
                err.append(errmap[yp[int(round(i))], xp[int(round(i))]])
        except IndexError:
            ext.append(np.nan)
            if errmap is not None:
                err.append(np.nan)
    if errmap is not None:
        return np.array(ext), np.array(err)
    else:
        return np.array(ext), None


def pdf(values, bins):
    '''
    ** Normalized differential area function. **
    (statistical) probability denisty function
    normalized so that the integral is 1
    and. The integral over a range is the
    probability of the value is within
    that range.

    Returns array of size len(bins)-1
    Plot versus bins[:-1]
    '''
    if hasattr(bins,'__getitem__'):
        range=(np.nanmin(bins),np.nanmax(bins))
    else:
        range = None

    h, x = np.histogram(values, bins=bins, range=range, density=False)
    # From the definition of Pr(x) = dF(x)/dx this
    # is the correct form. It returns the correct
    # probabilities when tested
    pdf = h / (np.sum(h, dtype=float) * np.diff(x))
    return pdf, avg(x)


def pdf2(values, bins):
    '''
    The ~ PDF normalized so that
    the integral is equal to the
    total amount of a quantity.
    The integral over a range is the
    total amount within that range.

    Returns array of size len(bins)-1
    Plot versus bins[:-1]
    '''
    if hasattr(bins,'__getitem__'):
        range=(np.nanmin(bins),np.nanmax(bins))
    else:
        range = None

    pdf, x = np.histogram(values, bins=bins, range=range, density=False)
    pdf = pdf.astype(float) / np.diff(x)
    return pdf, avg(x)


def edf(data, pdf=False):
    y = np.arange(len(data), dtype=float)
    x = np.sort(data).astype(float)
    return y, x


def cdf(values, bins):
    '''
    (statistical) cumulative distribution function
    Integral on [-inf, b] is the fraction below b.
    CDF is invariant to binning.
    This assumes you are using the entire range in the binning.
    Returns array of size len(bins)
    Plot versus bins[:-1]
    '''
    if hasattr(bins,'__getitem__'):
        range = (np.nanmin(bins),np.nanmax(bins))
    else:
        range = None

    h, bins = np.histogram(values, bins=bins, range=range, density=False)  # returns int

    c = np.cumsum(h / np.sum(h, dtype=float)) # cumulative fraction below bin_k
    # append 0 to beginning because P( X < min(x)) = 0
    return np.append(0, c), bins


def cdf2(values, bins):
    '''
    # # Exclusively for area_function which needs to be unnormalized
    (statistical) cumulative distribution function
    Value at b is total amount below b.
    CDF is invariante to binning

    Plot versus bins[:-1]
    Not normalized to 1
    '''
    if hasattr(bins,'__getitem__'):
        range=(np.nanmin(bins),np.nanmax(bins))
    else:
        range = None

    h, bins = np.histogram(values, bins=bins, range=range, density=False)
    c = np.cumsum(h).astype(float)
    return np.append(0., c), bins


def area_function(extmap, bins):
    '''
    Complimentary CDF for cdf2 (not normalized to 1)
    Value at b is total amount above b.
    '''
    c, bins = cdf2(extmap, bins)
    return c.max() - c, bins


def diff_area_function(extmap, bins,scale=1):
    '''
    See pdf2
    '''
    s, bins = area_function(extmap, bins)
    dsdx = -np.diff(s) / np.diff(bins)
    return dsdx*scale, avg(bins)

def log_diff_area_function(extmap, bins):
    '''
    See pdf2
    '''
    s, bins = diff_area_function(extmap, bins)
    g=s>0
    dlnsdlnx = np.diff(np.log(s[g])) / np.diff(np.log(bins[g]))
    return dlnsdlnx, avg(bins[g])

def mass_function(values, bins, scale=1, aktomassd=183):
    '''
    M(>Ak), mass weighted complimentary cdf
    '''
    if hasattr(bins,'__getitem__'):
        range=(np.nanmin(bins),np.nanmax(bins))
    else:
        range = None

    h, bins = np.histogram(values, bins=bins, range=range, density=False, weights=values*aktomassd*scale)
    c = np.cumsum(h).astype(float)
    return c.max() - c, bins


def hist(values, bins, err=False, density=False, **kwargs):
    '''
    really just a wrapper for numpy.histogram
    '''
    if hasattr(bins,'__getitem__'):
        range=(np.nanmin(bins),np.nanmax(bins))
    else:
        range = None

    hist, x = np.histogram(values, bins=bins, range=range, density=density, **kwargs)
    if (err is None) or (err is False):
        return hist.astype(np.float), avg(x)
    else:
        return hist.astype(np.float), avg(x), np.sqrt(hist)


def bootstrap(X, X_err=None, n=None, smooth=False):
    '''
    (smooth) bootstrap
    bootstrap(X,Xerr,n,smooth=True)
    X : array to be resampled
    X_err [optional]: errors to perturb data for smooth bootstrap
                      only provide is doing smooth bootstrapping
    n : number of samples. Default - len(X)
    smooth: optionally use smooth bootstrapping.
            will be set to False if no X_err is provided
    '''

    if X_err is None:
        smooth = False

    if n is None:  # default n
        n = len(X)

    resample_i = np.random.randint(0,len(X),size=(n,))
    X_resample = np.asarray(X)[resample_i]

    if smooth:
        X_resample = np.random.normal(X_resample, \
                                        np.asarray(X_err)[resample_i])

    return X_resample


def num_above(values, level):
    return np.sum((values >= level) & np.isfinite(values), dtype=np.float)


def num_below(values, level):
    return np.sum((values < level) & np.isfinite(values), dtype=np.float)


def alpha_ML(data, xmin,xmax):
    '''
    uses maximum likelihood to estimation
    to determine power-law and error
    From Clauset et al. 2010
    '''
    data = data[np.isfinite(data)]
    data = data[(data >= xmin) & (data <= xmax)]
    alpha = 1 + len(data) * (np.sum(np.log(data / xmin))**(-1))
    error = (alpha -1 )/np.sqrt(len(data))
    #loglike = np.sum((-1+alpha)*np.log(xmin)-alpha*np.log(data)+np.log(-1+alpha))
    N = len(data)
    loglike = N*np.log(alpha-1) - N*np.log(xmin) - alpha * np.sum(np.log(data/xmin))
    return alpha  , error, loglike, xmin, xmax

def sigconf1d(n):
    cdf = (1/2.)*(1+special.erf(n/np.sqrt(2)))
    return (1-cdf)*100,100* cdf,100*special.erf(n/np.sqrt(2))



def surfd(X, Xmap, bins, Xerr = None, Xmaperr = None, boot=False, scale=1., return_err=False, smooth=False):
    '''
    call: surfd(X, map, bins,
                    xerr = None, merr = None, scale = 1.)
    calculates H(X)/H(M) = Nx pdf(x) dx / Nm pdf(m) dm ; dm = dx
          so it is independent of whether dx or dlog(x)
    '''
    # get dn/dx
    if boot:
        n = np.histogram(bootstrap(X,Xerr,smooth=True), bins = bins, range=(bins.min(),bins.max()))[0]
        s = np.histogram(bootstrap(Xmap,Xmaperr,smooth=True), bins = bins, range=(bins.min(),bins.max()))[0] * scale
    else:
        n = np.histogram(X, bins = bins, range=(bins.min(),bins.max()))[0]
        s = np.histogram(Xmap, bins = bins, range=(bins.min(),bins.max()))[0] * scale


    if not return_err:
        return n / s
    else:
        return n / s, n / s * np.sqrt(1. / n - scale / s)


def alpha(y, x, err=None, return_kappa=False, cov=False):
    '''
    this returns -1*alpha, and optionally kappa and errors
    '''
    a1 = set(np.nonzero(np.multiply(x, y))[0])
    a2 = set(np.where(np.isfinite(np.add(x, y, err)))[0])
    a = np.asarray(list(a1 & a2))
    y = np.log(y[a])
    x = np.log(x[a])
    if err is None:
        p, covar = np.polyfit(x, y, 1, cov=True)
        m, b = p
        me, be = np.sqrt(np.sum(covar * [[1, 0], [0, 1]], axis=1))
        me, be
    else:
        err = err[a]
        err = err / y
        p, covar = np.polyfit(x, y, 1, w=1. / err**2, cov=True)
        m, b = p
        me, be = np.sqrt(np.sum(covar * [[1, 0], [0, 1]], axis=1))
        me, be
    if return_kappa:
        if cov:
            return m, np.exp(b), me, be
        else:
            return m, np.exp(b)
    else:
        if cov:
            return m, me
        else:
            return m


def Heaviside(x):
    return 0.5 * (np.sign(x) + 1.)


def schmidt_law(Ak, theta):
    '''
    schmidt_law(Ak,(beta,kappa))
    beta is the power law index (same as alpha)
    '''
    if len(theta) == 2:
        beta, kappa = theta
        return kappa * (Ak ** beta)
    elif len(theta) == 3:
        beta, kappa, Ak0 = theta
        sfr = Heaviside(Ak - Ak0) * kappa * (Ak ** beta)
        sfr[Ak < Ak0] = 0#np.nan # kappa * (Ak0 ** beta)
        return sfr


def lmfit_powerlaw(x, y, yerr=None, xmin=-np.inf, xmax=np.inf, init=None, maxiter=1000000):
    @custom_model
    def model(x, beta=init[0], kappa=init[1]):
        return np.log(kappa * (np.exp(x) ** beta))
    keep = np.isfinite(1. / y) & (x >= xmin) & (x <= xmax)
    if yerr is not None:
        keep = keep & np.isfinite(1. / yerr)
    m_init = model()
    fit = LevMarLSQFitter()
    #weights = (yerr / y)[keep]**(-2.)
    m = fit(m_init, np.log(x[keep]), np.log(y[keep]), maxiter=maxiter)
    return m, fit


def fit_lmfit_schmidt(x, y, yerr, init=None):
    m, _ = lmfit_powerlaw(x,y,yerr,init=init)
    return m.parameters


def emcee_schmidt(x, y, yerr, pos=None, pose=None,
                  nwalkers=None, nsteps=None, burnin=200,verbose=True):
    '''
    emcee_schmidt provides a convenient wrapper for fitting the schimdt law
    to binned x,log(y) data. Generally, it fits a normalization and a slope
    '''
    def model(x, theta):
        '''
        theta = (beta, kappa)
        '''
        return np.log(schmidt_law(x, theta))

    def lnlike(theta, x, y, yerr):
        mod = model(x, theta)

        inv_sigma2 = 1 / yerr**2
        # Poisson statistics -- not using this
        #mu = (yerr)**2  # often called lambda = poisson variance for bin x_i
        #resid = np.abs(y - mod) # where w calculate the poisson probability
        #return np.sum(resid * np.log(mu) - mu) - np.sum(np.log(misc.factorial(resid)))
        #######################################################
        ########## CHI^2 log-likelihood #######################
        return -0.5 * (np.sum((y - mod)**2 * inv_sigma2))#  - 0.5 * 3 * np.log(np.sum(k))

    def lnprior(theta):
        # different priors for different version of
        # the schmidt law
        if len(theta) == 3:
            beta, kappa, Ak0 = theta
            c3 = 0. < Ak0 <= 5.
            c4 = True
        else:
            beta, kappa = theta
            c3 = True
            c4 = True
        c1 = 0 <= beta <= 6# Never run's into this region
        c2 = 0 <= kappa  # Never run's into this region
        if c1 and c2 and c3 and c4:
            return 0.0
        return -np.inf

    def lnprob(theta, x, y, yerr):
        ## update likelihood
        lp = lnprior(theta)
        if not np.isfinite(lp):
            return -np.inf
        return lp + lnlike(theta, x, y, yerr)

    ndim, nwalkers = len(pos), nwalkers

    pos = [np.array(pos) + np.array(pose) * 0.5 *
           (0.5 - np.random.rand(ndim)) for i in range(nwalkers)]

    sampler = emcee.EnsembleSampler(
        nwalkers, ndim, lnprob, args=(x, y, yerr))

    sampler.run_mcmc(pos, nsteps)

    # Get input values
    # x, y, yerr = sampler.args
    samples = sampler.chain[:, burnin:, :].reshape((-1, sampler.ndim))

    # # Print out final values # #
    theta_mcmc = np.percentile(samples, [16, 50, 84], axis=0).T

    if verbose: print(sampler.acor)

    if verbose:
        for i, item in enumerate(theta_mcmc):
            j = ['beta', 'kappa', 'A_{K,0}', 'A_{K,f}']
            inserts = (j[i], item[1], item[2] - item[1], item[1] - item[0])
            print('%s = %0.2f (+%0.2f,-%0.2f)' % inserts)

    return sampler, np.median(samples, axis=0), np.std(samples, axis=0)



def fit(bins, samp, samperr, maps, mapserr, scale=1., sampler=None, log=False,
        pos=None, pose=None, nwalkers=100, nsteps=1e4, boot=1000, burnin=200,
         threshold=False, threshold2=False,verbose=True):
    '''
    # # # A Schmidt Law fitting Function using EMCEE by D.F.M.
    fit(bins, samp, samperr, maps, mapserr, scale=1.,
            pos=None, pose=None, nwalkers=100, nsteps=1e4)
    bins: bin edges for binning data (I know it's bad to bin)
    samp : values for your sample
    samperr : errors on values for you sample
    maps: map of values from which you drew your sample
    mapserr: error on maps...
    pos : initial location of ball of walkers
    pose : initial spread of walkers
    '''

    #print 'Hi!. It\'s hammer time...'

    # x values are bin midpoints
    x = avg(bins)  # assume if log=True, then bins are already log
    # x = bins[:-1]
    # y = np.asarray([surfd(samp,maps,bins,boot=True,scale=scale) for i in xrange(boot)])
    # yerr = np.nanstd(y,axis=0)
    #if log:
    #    samp = np.log10(samp)
    #    maps = np.log10(maps)
    #    bins = np.log10(bins)  # because bins doesn't get used again after surfd
    y, yerr = surfd(samp, maps, bins, scale=scale, return_err=True)
    ###########################################+
    ###### ADDED FOR SHIFTING EXPERIMENT ######+
    ###########################################+

    bins2 = shift_bins(bins,0.5)
    bin
    x2 = avg(bins2)
    y2, yerr2 = surfd(samp, maps, bins2, scale=scale, return_err=True)
    concatx = np.concatenate((x,x2))
    concaty = np.concatenate((y,y2))
    concatyerr = np.concatenate((yerr,yerr2))
    srt = np.argsort(concatx)
    x = concatx[srt]
    y = concaty[srt]
    yerr = concatyerr[srt]

    nonzero = np.isfinite(1. / y) & np.isfinite(yerr) & np.isfinite(1./yerr)

    y = y[nonzero]
    yerr = yerr[nonzero]
    x = x[nonzero]
    # initialize walker positions and walker bundle size
    init = alpha(y, x, return_kappa=True, cov=True)
    if pos is None:
        pos = init[:2]
    if pose is None:
        if np.isnan(init[2] + init[3]):
            pose = (1, 1)
        else:
            pose = (init[2], init[3])
    if threshold | threshold2:
        pos = pos + (0.4,)
        pose = pose + (0.2,)
    if threshold2:
        pos = pos + (8.,)
        pose = pose + (.5,)
    #print pos
    #print pose
    pos = np.asarray(pos)
    pose = .1*pos#np.asarray(pose)

    # This function only fits sources, it doesn't plot, so don't pass
    # and emcee sampler type. it will spit it back out
    # # # # # # # RUN EMCEE # # # # # # #
    # pdb.set_trace()
    if sampler is None:
        if verbose: print('Sampler autocorrelation times . . .')
        sampler, theta, theta_std = emcee_schmidt(x, np.log(y), yerr/y,
                                                  pos=pos, pose=pose,
                                                  nwalkers=nwalkers,
                                                  nsteps=nsteps, burnin=burnin,verbose=verbose)
    else:
        print('Next time don\'t give me a ' + str(type(sampler)) + '.')

    #
    try:
        return sampler, x, y, yerr, theta, theta_std
    except:
        return sampler, x, y, yerr

def schmidt_results_plots(sampler, model, x, y, yerr, burnin=200, akmap=None,
                          bins=None, scale=None, triangle_plot=True):
    '''
    model: should pass schmidt_law()
    '''
    try:
        mpl.style.use('john')
    except:
        None
    # Get input values
    # x, y, yerr = sampler.args
    if hasattr(sampler,'__getitem__'):
        chain = sampler
        dim = chain.shape[-1]
    else:
        chain = sampler.chain
        dim = sampler.dim

    samples = chain[:, burnin:, :].reshape((-1, dim))
    # # Print out final values # #
    theta_mcmc = np.percentile(samples, [16, 50, 84], axis=0).T  # Get percentiles for each parameter
    n_params = len(theta_mcmc[:,1])
    #print n_params
    for i, item in enumerate(theta_mcmc):
        j = ['beta', 'kappa', 'A_{K,0}','A_{K,f}']
        inserts = (j[i], item[1], item[2] - item[1], item[1] - item[0])
        print('%s = %0.2f (+%0.2f,-%0.2f)' % inserts)
    # Plot corner plot
    if triangle_plot:
        if n_params == 3:
            labels = ['beta', 'kappa', 'A_{K,0}']
        elif n_params == 4:
            labels = ['beta', 'kappa', 'A_{K,0}', 'A_{K,f}']
        else:
            labels = ['beta', 'kappa']
        #print labels
        _ = triangle.corner(samples, labels=labels,
                              truths=theta_mcmc[:, 1], quantiles=[.16, .84],
                              verbose=False)
    # generate schmidt laws from parameter samples
    xln = np.logspace(np.log10(x.min()*.5),np.log10(x.max()*2.),100)
    smlaw_samps = np.asarray([schmidt_law(xln, samp) for samp in samples])
    # get percentile bands
    percent = lambda x: np.nanpercentile(smlaw_samps, x, interpolation='linear', axis=0)
    # Plot fits
    fig = plt.figure()
    # Plot data with errorbars
    plt.plot(xln, percent(50), 'k')  # 3 sigma band
    # yperr = np.abs(np.exp(np.log(y)+yerr/y) - y)
    # ynerr = np.abs(np.exp(np.log(y)-yerr/y) - y)
    plt.errorbar(x, y, yerr, fmt='rs', alpha=0.7, mec='none')
    plt.legend(['Median', 'Data'],
               loc='upper left', fontsize=12)
    # draw 1,2,3 sigma bands
    plt.fill_between(xln, percent(1), percent(99), color='0.9')  # 1 sigma band
    plt.fill_between(xln, percent(2), percent(98), color='0.75')  # 2 sigma band
    plt.fill_between(xln, percent(16), percent(84), color='0.5')  # 3 sigma band

    plt.loglog(nonposy='clip')

    return plt.gca()

def flatchain(chain):
    return chain.reshape((-1,chain.shape[-1]))

def norm_chain(chain, axis=0):
    std = np.std(flatchain(chain), axis=axis)
    med = np.median(flatchain(chain), axis=axis)
    return (chain-med)/std

def plot_walkers(sampler,limits = None, bad = None):
    '''
    sampler :  emcee Sampler class
    '''

    if hasattr(sampler,'__getitem__'):
        chain = sampler
        ndim = chain.shape[-1]
    else:
        chain = sampler.chain
        ndim = sampler.ndim

    fig = plt.figure(figsize=(8 * ndim, 4 * ndim))

    if hasattr(limits,'__getitem__'):
        limits += [None] * (3-len(limits))
        slices = slice(limits[0],limits[1],limits[2])
    else:
        slices = slice(None,limits,None)


    for w,walk in enumerate(chain[:,slices,:]):
        if bad is None:
            color = 'k'
        elif bad[w]:
            color = 'r'
        else:
            color = 'k'
        for p, param in enumerate(walk.T):
            ax = plt.subplot(ndim, 1, p + 1)
            ax.plot(param, color, alpha=.75, lw=0.75)
            # ax.set_ylim(param.min()*0.5,param.max()*1.5)
            # ax.semilogy()
    plt.tight_layout()
    return fig


def tester():
    print('hi ya\'ll')


