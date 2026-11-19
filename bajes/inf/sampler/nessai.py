from __future__ import division, unicode_literals, absolute_import
import os
import numpy as np

import logging
logger = logging.getLogger(__name__)

from nessai.flowsampler import FlowSampler
from nessai.model import Model
from nessai.livepoint import numpy_array_to_live_points

from . import SamplerBody

class ParameterTransform(object):
    """
        Monotonic one-dimensional change of variable between a bajes parameter
        (native) and the parameter seen by nessai (sampled).

        Arguments:
        - native        : str, name of the bajes parameter
        - sampled       : str, name of the parameter seen by nessai
        - to_native     : callable, sampled -> native (vectorised)
        - to_sampled    : callable, native -> sampled (vectorised)
        - log_jacobian  : callable, log |d native / d sampled| as a function of sampled, or None if zero
        - increasing    : bool, whether the map is increasing
        - periodic      : bool or None, periodicity in sampled space (None keeps the native flag)

        For maps that are not one-to-one (e.g. a k-to-1 cover of a periodic parameter),
        the sampled-space prior can be given explicitly with
        - bounds        : [min, max] in sampled space
        - log_prior     : callable, sampled-space log-density
        - from_unit     : callable, unit interval -> sampled
        - to_unit       : callable, sampled -> unit interval
        in which case to_sampled is only used for reporting and the native prior is not used.
    """

    def __init__(self, native, sampled, to_native=None, to_sampled=None, log_jacobian=None, increasing=True, periodic=None,
                 bounds=None, log_prior=None, from_unit=None, to_unit=None):
        self.native         = native
        self.sampled        = sampled
        self.to_native      = to_native if to_native is not None else (lambda x: x)
        self.to_sampled     = to_sampled if to_sampled is not None else (lambda x: x)
        self.log_jacobian   = log_jacobian
        self.increasing     = increasing
        self.periodic       = periodic
        self.bounds         = bounds
        self.log_prior      = log_prior
        self.from_unit      = from_unit
        self.to_unit        = to_unit
        if (log_prior is None) != (from_unit is None) or (from_unit is None) != (to_unit is None) or (bounds is None) != (to_unit is None):
            raise ValueError("Transform {} -> {}: bounds, log_prior, from_unit and to_unit must be given together.".format(native, sampled))

    @property
    def explicit(self):
        return self.log_prior is not None

class ParameterMap(object):
    """
        Collection of one-dimensional transforms, one per prior parameter.
        Parameters without a transform are passed to nessai unchanged.
        Since every transform acts on a single parameter, the prior stays separable
        and the unit-hypercube maps are exact. The sampled-space prior includes the
        Jacobians, so the evidence is unchanged.
    """

    def __init__(self, prior, transforms=[]):

        tr_dict = {ti.native: ti for ti in transforms}
        unknown = [n for n in tr_dict.keys() if n not in prior.names]
        if unknown:
            logger.warning("Ignoring transforms for parameters not in the prior: {}".format(unknown))

        self.parameters = prior.parameters
        self.transforms = [tr_dict.get(pi.name, ParameterTransform(pi.name, pi.name)) for pi in prior.parameters]

        # sampled names and bounds
        self.native_names   = [pi.name for pi in self.parameters]
        self.names          = [ti.sampled for ti in self.transforms]
        if len(set(self.names)) != len(self.names):
            logger.error("Parameter map produced repeated names: {}".format(self.names))
            raise ValueError("Parameter map produced repeated names: {}".format(self.names))

        self.bounds = {}
        for pi, ti in zip(self.parameters, self.transforms):
            if ti.explicit:
                b = np.sort(np.array(ti.bounds, dtype=float))
            else:
                b = np.sort(ti.to_sampled(np.array(pi.bound, dtype=float)))
            self.bounds[ti.sampled] = [float(b[0]), float(b[1])]

        self.periodics = {ti.sampled: bool(pi.periodic) if ti.periodic is None else bool(ti.periodic)
                          for pi, ti in zip(self.parameters, self.transforms)}

    def to_native(self, x):
        """
            Map a structured array (or dict) of sampled values to a dict of native values
        """
        return {ti.native: ti.to_native(x[ti.sampled]) for ti in self.transforms}

    def to_sampled(self, p):
        """
            Map a dict (or structured array) of native values to a dict of sampled values
        """
        return {ti.sampled: ti.to_sampled(p[ti.native]) for ti in self.transforms}

    def log_prior(self, x):
        """
            Log-prior density in sampled space (native density times Jacobians),
            -inf outside the bounds
        """
        x       = np.atleast_1d(x)
        logp    = np.zeros(x.size)
        inside  = np.ones(x.size, dtype=bool)
        for pi, ti in zip(self.parameters, self.transforms):
            xs      = np.asarray(x[ti.sampled], dtype=float)
            lo, hi  = self.bounds[ti.sampled]
            inside &= (xs >= lo) & (xs <= hi)
        if not np.any(inside):
            return np.full(x.size, -np.inf)
        xi = x[inside]
        for pi, ti in zip(self.parameters, self.transforms):
            xs  = np.asarray(xi[ti.sampled], dtype=float)
            if ti.explicit:
                lp = ti.log_prior(xs)
            else:
                lp = pi.log_density(ti.to_native(xs))
                if ti.log_jacobian is not None:
                    lp = lp + ti.log_jacobian(xs)
            logp[inside] += np.broadcast_to(lp, xs.shape)
        logp[~inside] = -np.inf
        return logp

    def from_unit_hypercube(self, u):
        out = {}
        for pi, ti in zip(self.parameters, self.transforms):
            ui = np.asarray(u[ti.sampled], dtype=float)
            if ti.explicit:
                out[ti.sampled] = ti.from_unit(ui)
                continue
            if not ti.increasing:
                ui = 1. - ui
            out[ti.sampled] = ti.to_sampled(pi.quantile(ui))
        return out

    def to_unit_hypercube(self, x):
        out = {}
        for pi, ti in zip(self.parameters, self.transforms):
            xs = np.asarray(x[ti.sampled], dtype=float)
            if ti.explicit:
                out[ti.sampled] = ti.to_unit(xs)
                continue
            ui = pi.cumulative(ti.to_native(xs))
            out[ti.sampled] = ui if ti.increasing else 1. - ui
        return out

class BajesNessaiModel(Model):
    """
        nessai model wrapping a bajes Posterior, optionally through a ParameterMap
    """

    def __init__(self, posterior, parameter_map=None):

        self.posterior      = posterior
        self.parameter_map  = parameter_map if parameter_map is not None else ParameterMap(posterior.prior)
        self.names          = list(self.parameter_map.names)
        self.bounds         = dict(self.parameter_map.bounds)

        # bajes likelihoods take one point at a time
        self.allow_vectorised       = False
        self.vectorised_likelihood  = False

    @property
    def periodic_parameters(self):
        return [n for n in self.names if self.parameter_map.periodics[n]]

    def _to_structured(self, d, like):
        out = np.empty(like.shape, dtype=like.dtype)
        for n in like.dtype.names:
            out[n] = like[n]
        for n in self.names:
            out[n] = d[n]
        return out

    def log_prior(self, x):
        return self.parameter_map.log_prior(x)

    def log_likelihood(self, x):
        native  = self.parameter_map.to_native(x)
        p       = {k: float(np.squeeze(v)) for k, v in native.items()}
        return self.posterior.like.log_like(self.posterior.prior.this_sample(p))

    def new_point(self, N=1):
        return self.from_unit_hypercube(self.sample_unit_hypercube(N))

    def new_point_log_prob(self, x):
        return self.log_prior(x)

    def from_unit_hypercube(self, x):
        return self._to_structured(self.parameter_map.from_unit_hypercube(x), x)

    def to_unit_hypercube(self, x):
        return self._to_structured(self.parameter_map.to_unit_hypercube(x), x)

    def native_samples(self, x):
        """
            Return an (N, ndim) array of native bajes parameters, ordered as prior.names
        """
        native = self.parameter_map.to_native(x)
        return np.array([np.asarray(native[n], dtype=float) for n in self.posterior.prior.names]).T

def load_nessai_config(path):
    """
        Import the nessai configuration module.
        The module may define:
        - parameter_map(prior, likelihood)              -> ParameterMap, default identity
        - nessai_kwargs(model, likelihood, prior)       -> dict of FlowSampler / FlowSampler.run keyword arguments,
                                                           applied on top of the bajes defaults
    """
    if path is None:
        return None
    import importlib.util
    path = os.path.abspath(path)
    spec = importlib.util.spec_from_file_location('bajes_nessai_config', path)
    mod  = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

class SamplerNessai(SamplerBody):

    # FlowSampler.run arguments
    _run_keys = ['plot', 'save', 'plot_indices', 'plot_posterior', 'plot_logXlogL']

    def __initialize__(self, posterior,
                       nlive=1000,
                       nprocs=None,
                       nessai_config=None,
                       importance_nested_sampler=False,
                       pool=None,
                       **kwargs):

        if pool is not None:
            logger.warning("nessai creates its own multiprocessing pool (n_pool = nprocs); the given pool is ignored.")

        self.nlive          = nlive
        self.nprocs         = nprocs
        self.nessai_config  = os.path.abspath(nessai_config) if nessai_config is not None else None
        self.importance_nested_sampler = importance_nested_sampler

        self._setup(posterior)

    @staticmethod
    def _share_bajes_handlers():
        # print nessai and nessai-gw messages through the bajes handlers (stream and bajes.log)
        bajes_logger = logging.getLogger('bajes')
        for name in ['nessai', 'nessai_gw']:
            lg = logging.getLogger(name)
            lg.setLevel(bajes_logger.level)
            lg.propagate = False
            for h in bajes_logger.handlers:
                if h not in lg.handlers:
                    lg.addHandler(h)

    def _setup(self, posterior, resume=True):

        self._share_bajes_handlers()
        config = load_nessai_config(self.nessai_config)

        # parameter map
        if config is not None and hasattr(config, 'parameter_map'):
            parameter_map = config.parameter_map(posterior.prior, posterior.like)
        else:
            parameter_map = ParameterMap(posterior.prior)
        self.model = BajesNessaiModel(posterior, parameter_map)
        logger.info("nessai parameters: {}".format(', '.join(self.model.names)))

        # sampler keyword arguments
        fs_kwargs = {'nlive': self.nlive}
        if self.seed is not None:
            fs_kwargs['seed'] = self.seed
        if self.nprocs is not None and self.nprocs > 1:
            fs_kwargs['n_pool'] = int(self.nprocs)
        if self.model.periodic_parameters and not self.importance_nested_sampler:
            # keep the periodic flags of the bajes prior (as the other samplers do)
            def _angle(n):
                width = np.diff(self.model.bounds[n])[0]
                if np.isclose(width, 2.*np.pi): return 'angle-2pi'
                if np.isclose(width, np.pi):    return 'angle-pi'
                return 'periodic'
            fs_kwargs['reparameterisations'] = {n: _angle(n) for n in self.model.periodic_parameters}
        if config is not None and hasattr(config, 'nessai_kwargs'):
            fs_kwargs.update(config.nessai_kwargs(self.model, posterior.like, posterior.prior))

        self.run_kwargs = {k: fs_kwargs.pop(k) for k in self._run_keys if k in fs_kwargs}
        fs_kwargs.setdefault('importance_nested_sampler', self.importance_nested_sampler)

        # nessai checkpoints on signals itself: drop the bajes handlers (set by SamplerBody),
        # which would otherwise be inherited by the pool workers and write resume.pkl from each of them
        import signal
        for sig in [signal.SIGTERM, signal.SIGINT, signal.SIGALRM]:
            signal.signal(sig, signal.SIG_DFL)

        logger.info("Initializing nessai FlowSampler ...")
        logger.debug("FlowSampler arguments: {}".format(fs_kwargs))
        self.sampler = FlowSampler(self.model,
                                   output=os.path.join(self.outdir, 'nessai'),
                                   resume=resume,
                                   **fs_kwargs)

    def __getstate__(self):
        # nessai checkpoints itself in outdir/nessai; keep only what rebuilds the sampler
        state = self.__dict__.copy()
        state.pop('sampler', None)
        state.pop('model', None)
        return state

    def __restore__(self, posterior, **kwargs):
        self._setup(posterior, resume=True)

    def __run__(self):
        self.sampler.run(**self.run_kwargs)

    def get_posterior(self):

        samples = self.sampler.posterior_samples
        native  = self.model.native_samples(samples)
        logl    = np.asarray(samples['logL'], dtype=float)
        logpr   = np.array([self.model.posterior.log_prior(pi) for pi in native])

        self.posterior_samples  = np.column_stack([native, logl, logpr])
        self.real_nout          = len(self.posterior_samples)
        logger.info(" - number of posterior samples : {}".format(self.real_nout))

        names = list(self.names) + ['logL', 'logPrior']
        np.savetxt(os.path.join(self.outdir, 'posterior.dat'), self.posterior_samples,
                   header='\t'.join(names), delimiter='\t', comments='#')

        with open(os.path.join(self.outdir, 'evidence.dat'), 'w') as f:
            f.write('logZ = {} +- {}\n'.format(self.sampler.log_evidence, self.sampler.log_evidence_error))

        logger.info(" - logZ = {:.3f} +- {:.3f}".format(self.sampler.log_evidence, self.sampler.log_evidence_error))

    def make_plots(self):
        # nessai writes its own diagnostics in outdir/nessai
        pass
