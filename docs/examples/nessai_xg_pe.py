"""
nessai configuration for bajes: the nessai-gw network group-mixture proposal
in the xg_pe setup (jacopok/nessai and jacopok/nessai-gw, branch xg_pe).

    pip install -e .[nessai]                    # bajes + the xg_pe nessai, nessai-gw
                                                # (torch CPU first: pip install torch --index-url https://download.pytorch.org/whl/cpu)
    bajes_setup -o ./run --tag gw ...           # as usual, without --marg-phi-ref
    python -m bajes -I ./run/inf.pkl -o ./run -e nessai --nlive 1000 -n 4 \
                    --nessai-config nessai_xg_pe.py

bajes calls
- parameter_map(prior, likelihood): the bajes GW parameters are passed to nessai
  with bilby names and conventions (mass_ratio = 1/q, theta_jn = arccos(cos_iota),
  geocent_time = t_gps + time_shift, phase from phi_ref), as nessai-gw expects;
- nessai_kwargs(model, likelihood, prior): keyword arguments for nessai's FlowSampler
  (and FlowSampler.run), on top of nlive / seed / n_pool from the bajes options.
The output (posterior.dat, evidence.dat) is written with the bajes names.
"""

from bajes.pipe.utils.nessai_gw import bilby_parameter_map, network_geometry

#: per-detector SNRs for the network geometry, e.g. {'H1': 16.1, 'L1': 21.8, 'V1': 2.6}
#: (bajes_inject.log for an injection). None uses the relative PSD sensitivities.
SNRS = None

#: 'network' (group proposal, latent ball) or 'network-full' (circular angles,
#: sampler options, per-expert latent distortion)
LEVEL = 'network-full'

#: stop after this many iterations (quick checks of the setup), None to run to convergence
MAX_ITERATION = None

#: xg_pe's tuned flow and training for the group proposals ("v19")
V19_FLOW = dict(ftype="realnvp", n_blocks=8, n_layers=2, n_neurons=41,
                batch_norm_between_layers=True, linear_transform="lu",
                activation="silu", dropout_probability=0.0894)
V19_TRAINING = dict(lr=2.62e-3, batch_size=256, max_epochs=500, patience=28,
                    optimiser="adamw", optimiser_kwargs={"weight_decay": 1.07e-5},
                    val_size=0.1, clip_grad_norm=5.0, noise_type="adaptive",
                    noise_scale=6.33e-3, annealing=False)

#: the fork's nessai sampler options
SAMPLER_OPTIONS = dict(
    truncation_methods=["log_proposal_threshold"],
    truncation_kwargs={"log_proposal_threshold": {
        "quantile": 0.005, "prior_relative": True, "mode": "clip"}},
    threshold_from_live_points=True,
    sum_mirror_density=True,
    drawsize=10000,
    latent_temperature=1.0,
    retrain_decision=True,
    retrain_costs="benchmark",
)

def parameter_map(prior, likelihood):
    return bilby_parameter_map(prior, likelihood)

def nessai_kwargs(model, likelihood, prior):

    from nessai_gw.network_group import (make_network_group_flow_proposal,
                                         network_group_reparameterisations)

    names    = model.names
    geometry = network_geometry(likelihood, snrs=SNRS)

    coordinates = dict(chirp_distance=True, chirp_distance_inversion=True,
                       log_mass_ratio=True,
                       effective_spin=('chi_1' in names and 'chi_2' in names),
                       precessing_spin=('phi_12' in names and 'phi_jl' in names),
                       time_reference_frequency="adaptive")

    full     = LEVEL == 'network-full'
    proposal = make_network_group_flow_proposal(
        names, geometry, adaptive_domain=True, gaussianise_sky=True,
        mass_ratio_shear=True, circular_psi_phase=full, circular_sky_u=full,
        circular_standardisation=full, phase_recanon=full, **coordinates)

    kwargs = dict(flow_proposal_class=proposal,
                  reparameterisations=network_group_reparameterisations(names, geometry, **coordinates),
                  fallback_reparameterisation="zscore",
                  training_config=V19_TRAINING)

    if LEVEL == 'network':
        kwargs.update(flow_config=V19_FLOW,
                      latent_temperature=1.42435181480039,
                      truncation_methods=["latent_radius"],
                      truncation_kwargs={"latent_radius": {"constant_volume_mode": True,
                                                           "fixed_radius": False,
                                                           "volume_fraction": 0.97}})
    else:
        kwargs.update(SAMPLER_OPTIONS)
        kwargs.update(flow_config=dict(V19_FLOW, ftype="circular", real_transform="affine",
                                       num_bins=4, circular_shift="angle"),
                      latent_distortion="expert", reset_flow=2)

    if MAX_ITERATION is not None:
        kwargs.update(max_iteration=MAX_ITERATION, maximum_uninformed=50)

    return kwargs
