#!/usr/bin/env python
"""
    Glue between bajes GW models and nessai-gw.

    nessai-gw expects bilby parameter names and conventions, and a
    DetectorNetworkGeometry built from the detector network. This module provides
    - bilby_parameter_map : a ParameterMap from the bajes GW parameters to the bilby ones
    - network_geometry    : a DetectorNetworkGeometry from the bajes detectors
    to be used in a nessai configuration file (see `python -m bajes --nessai-config`).
"""
from __future__ import division, unicode_literals, absolute_import
import numpy as np

import logging
logger = logging.getLogger(__name__)

from ...inf.sampler.nessai import ParameterMap, ParameterTransform

_TWO_PI = 2. * np.pi

# bajes -> bilby names for parameters that only need to be renamed
__bilby_names__ = { 'mchirp':   'chirp_mass',
                    'mtot':     'total_mass',
                    'distance': 'luminosity_distance',
                    's1z':      'chi_1',
                    's2z':      'chi_2',
                    's1':       'a_1',
                    's2':       'a_2',
                    'tilt1':    'tilt_1',
                    'tilt2':    'tilt_2',
                    'lambda1':  'lambda_1',
                    'lambda2':  'lambda_2'}

def find_gw_likelihood(likelihood):
    """
        Return the GW component of a (joint) bajes likelihood, None if there is none
    """
    if hasattr(likelihood, 'dets') and hasattr(likelihood, 'ifos'):
        return likelihood
    for li in getattr(likelihood, 'likes', []):
        gw = find_gw_likelihood(li)
        if gw is not None:
            return gw
    return None

def default_phase_convention(approx):
    """
        Meaning of the bajes `phi_ref` for a given approximant, compared with bilby's `phase`:
        - 'orbital' : phi_ref is the orbital reference phase, as bilby's phase (LAL approximants)
        - 'gw'      : phi_ref is the phase of the (2,2) mode, i.e. -2 times bilby's phase (TaylorF2)
        Returns None if the convention is not known.
    """
    if approx is None:
        return None
    if approx.startswith('TaylorF2'):
        return 'gw'
    if approx.startswith('LALSim'):
        return 'orbital'
    return None

def _phase_transform(prior, convention):

    pr = [pi for pi in prior.parameters if pi.name == 'phi_ref'][0]

    if convention == 'orbital':
        return ParameterTransform('phi_ref', 'phase', periodic=True)

    elif convention == 'gw':
        # phi_ref = -2 phase (mod 2pi): with phase uniform in [0, 2pi) this is a 2-to-1 cover
        # whose push-forward is the uniform phi_ref prior, so evidence and phi_ref posterior are unchanged.
        if pr._kind != 'uniform' or not np.allclose(pr.bound, [0., _TWO_PI]):
            logger.error("The 'gw' phase convention needs a uniform phi_ref prior in [0, 2pi].")
            raise ValueError("The 'gw' phase convention needs a uniform phi_ref prior in [0, 2pi].")
        return ParameterTransform('phi_ref', 'phase',
                                  to_native  = lambda x: np.mod(-2. * x, _TWO_PI),
                                  to_sampled = lambda x: np.mod(-0.5 * x, np.pi),
                                  periodic   = True,
                                  bounds     = [0., _TWO_PI],
                                  log_prior  = lambda x: np.full(np.shape(x), -np.log(_TWO_PI)),
                                  from_unit  = lambda u: _TWO_PI * u,
                                  to_unit    = lambda x: x / _TWO_PI)
    else:
        raise ValueError("Unknown phase convention {}. Please use 'orbital' or 'gw'.".format(convention))

def bilby_parameter_map(prior, likelihood=None, phase_convention='auto'):
    """
        ParameterMap from the bajes GW parameters to the bilby ones used by nessai-gw:

        mchirp -> chirp_mass, mtot -> total_mass, q -> mass_ratio = 1/q,
        s1z, s2z -> chi_1, chi_2, s1, s2, tilt1, tilt2 -> a_1, a_2, tilt_1, tilt_2,
        lambda1, lambda2 -> lambda_1, lambda_2, distance -> luminosity_distance,
        cos_iota -> theta_jn = arccos(cos_iota), time_shift -> geocent_time = t_gps + time_shift,
        phi_ref -> phase (see default_phase_convention).
        ra, dec, psi share names and conventions. Other parameters keep their bajes names.

        Arguments:
        - prior             : bajes.inf.Prior
        - likelihood        : bajes likelihood, used to read the approximant for phase_convention='auto'
        - phase_convention  : 'auto', 'orbital' or 'gw'
    """

    names       = prior.names
    transforms  = []

    for ni, si in __bilby_names__.items():
        if ni in names:
            transforms.append(ParameterTransform(ni, si))

    if 'q' in names:
        # bajes q = m1/m2 >= 1, bilby mass_ratio = m2/m1 <= 1
        transforms.append(ParameterTransform('q', 'mass_ratio',
                                             to_native      = lambda x: 1. / x,
                                             to_sampled     = lambda x: 1. / x,
                                             log_jacobian   = lambda x: -2. * np.log(x),
                                             increasing     = False))

    if 'cos_iota' in names:
        transforms.append(ParameterTransform('cos_iota', 'theta_jn',
                                             to_native      = np.cos,
                                             to_sampled     = lambda x: np.arccos(np.clip(x, -1., 1.)),
                                             log_jacobian   = lambda x: np.log(np.abs(np.sin(x))),
                                             increasing     = False))

    if 'time_shift' in names:
        t_gps = float(prior.const['t_gps'])
        transforms.append(ParameterTransform('time_shift', 'geocent_time',
                                             to_native      = lambda x: x - t_gps,
                                             to_sampled     = lambda x: x + t_gps))

    if 'phi_ref' in names:
        if phase_convention == 'auto':
            gw = find_gw_likelihood(likelihood) if likelihood is not None else None
            approx = getattr(getattr(gw, 'wave', None), 'approx', None) if gw is not None else None
            phase_convention = default_phase_convention(approx)
            if phase_convention is None:
                logger.warning("Unknown phi_ref convention for approximant {}, assuming the orbital (LAL) one. Set phase_convention explicitly to silence this warning.".format(approx))
                phase_convention = 'orbital'
            logger.info("Using the '{}' phase convention for approximant {}".format(phase_convention, approx))
        transforms.append(_phase_transform(prior, phase_convention))

    unmapped = [ni for ni in names if ni not in [ti.native for ti in transforms] + ['ra', 'dec', 'psi']]
    if unmapped:
        logger.info("Parameters passed to nessai with their bajes names: {}".format(unmapped))

    return ParameterMap(prior, transforms)

def detector_sensitivities(likelihood):
    """
        Relative SNR of a face-on, overhead inspiral in each detector,
        sqrt(sum f^(-7/3) / PSD), from the PSDs stored in the bajes detectors
    """
    gw  = find_gw_likelihood(likelihood)
    out = {}
    for ifo in gw.ifos:
        det = gw.dets[ifo]
        if hasattr(gw, 'noises') and hasattr(gw, 'datas'):
            # binning likelihood: detector frequencies are the bins
            series  = gw.datas[ifo]
            freqs   = series.freqs[series.mask]
            psd     = gw.noises[ifo].interp_psd_pad(freqs)
        else:
            freqs, psd = det.freqs, det.psd
        ok = (freqs > 0) & np.isfinite(psd) & (psd > 0)
        out[ifo] = float(np.sqrt(np.sum(freqs[ok]**(-7./3.) / psd[ok])))
    return out

def network_geometry(likelihood, snrs=None, **kwargs):
    """
        nessai_gw.network_group.DetectorNetworkGeometry from the detectors of a bajes GW likelihood

        Arguments:
        - likelihood    : bajes likelihood (GW or joint)
        - snrs          : dict of per-detector SNRs, e.g. from bajes_inject. If None,
                          the relative sensitivities of the PSDs are used (no antenna patterns).
        - kwargs        : further arguments for DetectorNetworkGeometry
    """
    from nessai_gw.network_group import DetectorNetworkGeometry

    gw = find_gw_likelihood(likelihood)
    if gw is None:
        logger.error("Unable to build the network geometry, no GW likelihood found.")
        raise ValueError("Unable to build the network geometry, no GW likelihood found.")

    ifos = list(gw.ifos)
    dets = [gw.dets[ifo] for ifo in ifos]

    if snrs is None:
        snrs = detector_sensitivities(gw)
        logger.warning("No SNRs given for the network geometry, using the PSD sensitivities: {}".format(
                       ', '.join('{}={:.3g}'.format(k, v) for k, v in snrs.items())))

    return DetectorNetworkGeometry(vertices         = [di.location for di in dets],
                                   snrs             = [float(snrs[ifo]) for ifo in ifos],
                                   reference_time   = float(dets[0].reference_time),
                                   names            = ifos,
                                   tensors          = [di.response for di in dets],
                                   **kwargs)
