import numpy
# FIXME: copy the functions here

from sklearn.mixture.gmm import log_multivariate_normal_density, logsumexp

def sample_gaussian2(means, cv, size, random_state, mins, maxes):
    def once(size1):
        g = random_state.multivariate_normal(means, cv, size1).T
        g = g.reshape(len(means), -1)
        mask = (g >= mins[:, None]).all(axis=0)
        mask &= (g <= maxes[:, None]).all(axis=0)
        return g[:, mask]

    g = once(size)
    generated = size

    while g.shape[1] < size:
        fac = 1.0 * g.shape[1] / size
        togen = (size - g.shape[1]) * generated // g.shape[1]
        g1 = once(togen)
        generated = generated + togen
        g = numpy.append(g, g1, axis=1)
    return g[:, :size]

class GMM(object):
    def __init__(self, weights, means, covs, lims):
        self.weights = numpy.array(weights)
        self.means = numpy.array(means)
        self.covs = numpy.array(covs)
        self.lims = numpy.array(lims)

        [nc] = self.weights.shape

        assert self.means.shape[0] == nc
        [nc, nf] = self.means.shape

        assert self.covs.shape[0] == nc
        assert self.covs.shape[1] == nf
        assert self.covs.shape[2] == nf
        [nc, nf, nf] = self.covs.shape

        assert self.lims.shape[0] == nf
        assert self.lims.shape[1] == 2

    def score(self, X, return_responsibilities=False):
        nc = len(self.weights)
        X = numpy.array(X)
        if X.ndim == 1:
            X = X[:, None]

        if X.shape[1] != self.means.shape[1]:
            raise ValueError('The shape of X  is not compatible with self')

        mins = self.lims[:, 0]
        maxes = self.lims[:, 1]

        lpr = numpy.log(self.weights) + \
              log_multivariate_normal_density(X,
                    self.means,
                    self.covs, 'full')
        mask = (X >= mins[None, :]).all(axis=-1)
        mask &= (X <= maxes[None, :]).all(axis=-1)
        logprob = logsumexp(lpr, axis=1)
        logprob[~mask] = -numpy.inf
        if return_responsibilities:
            responsibilities = numpy.exp(lpr - logprob[:, None])
            responsibilities[~mask] = 0
            return logprob, responsibilities
        return logprob

    def marginalize(self, axes):
        return GMM(self.weights, self.means[..., axes], self.covs[..., axes][..., axes, :], self.lims[axes])

    def sample(self, size, random_state=None):
        """Generate random samples from the model.
        Returns
        -------
        X : array_like, shape (n_samples, n_features)
            List of samples
        """

        if random_state is None:
            random_state = numpy.random

        mins = self.lims[:, 0]
        maxes = self.lims[:, 1]

        X = numpy.empty(size, ('f8', (self.means.shape[1],)))
        # decide which component to use for each sample
        comps = random_state.choice(len(self.weights), p=self.weights, size=size)
        # for each component, generate all needed samples
        for comp in range(len(self.weights)):
            # occurrences of current component in X
            comp_in_X = (comp == comps)
            # number of those occurrences
            num_comp_in_X = comp_in_X.sum()
            if num_comp_in_X > 0:
                cv = self.covs[comp]
                g = sample_gaussian2(
                    self.means[comp], cv,
                    num_comp_in_X, random_state, mins, maxes).T
                X[comp_in_X] = g
        return X

    @classmethod
    def fit(kls, nc, X, lims):
        # FIXME: get rid of this and add weights support
        from sklearn import mixture

        # XXX: Do not use DPGMM because the normalization is buggy
        # https://github.com/scikit-learn/scikit-learn/issues/7371

        model = mixture.GMM(nc, covariance_type='full', n_iter=1000)
        model.fit(X)

        if not model.converged_:
            raise ValueError("Your data is strange. Gaussian mixture failed to converge")

        return kls(model.weights_, model.means_, model.covars_, lims)

class Confidence(object):
    def __init__(self, model, confidence_table):
        self.model = model
        self.confidence_table = confidence_table
        
    def score(self, sc):
        x, y = self.confidence_table
        return numpy.interp(sc, x, y, left=1., right=0.)

    @classmethod
    def fit(kls, model, nsample=4*1024, vmin=-5, vmax=0, nb=100):
        X = model.sample(nsample)
        sc = model.score(X)
        confidence_levels = 1 - numpy.logspace(vmin, vmax, num=nb)
        # FIXME: add weight support here
        sc_cl = numpy.percentile(sc, 100. - confidence_levels * 100.)
        confidence_table = numpy.array([sc_cl, confidence_levels])
        return kls(model, confidence_table)

class CombinedModel(object):
    def __init__(self, models):
        self.models = models

    def score(self, X):
        return sum([model.score(X) for model in self.models])

    def marginalize(self, axes):
        return CombinedModel([
            model.marginalize(axes) for model in self.models])

    def sample(self, nsample, random_state=None):
        if random_state is None:
            random_state = numpy.random

        def once(size):
            X = self.models[0].sample(size, random_state)
            nf = X.shape[-1]
            lnprob = sum([model.score(X) for model in self.models[1:]])
            prob = numpy.exp(lnprob)
            prob /= prob.max()
            keep = random_state.rand(len(X)) < prob
            return X[keep].reshape(-1, nf)
        g = once(nsample)
        ng = nsample
        while len(g) < nsample:
            togen = (nsample - len(g)) * ng // len(g)
            g1 = once(togen)
            ng = ng + togen
            g = numpy.append(g, g1, axis=0)
        return g[:nsample]
