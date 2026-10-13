#!/usr/bin/env python3
"""
formation_nmpc.py
=================

Hierarchical Nonlinear Model Predictive Control (NMPC) for fixed-wing
leader-follower close formation flight with ArduPlane and pymavlink.

Architecture
------------
::

    leader  (ArduPlane) --LOCAL_POSITION_NED, ATTITUDE, VFR_HUD, WIND-->+
    follower(ArduPlane) --LOCAL_POSITION_NED, ATTITUDE, VFR_HUD, WIND-->+
                                                                        |
        [estimation]  wind, EKF-origin alignment, prediction through the |
                      inner-loop dead time, disturbance observer        |
        [reference ]  leader constant-turn-rate prediction (air-mass    |
                      frame) + formation slot kinematics (omega x r)    |
        [NMPC      ]  10-20 Hz, 5-DoF point-mass model, bounded         |
                      Gauss-Newton (or SLSQP), warm-started shift       |
                      register, soft state limits                       |
        [safety    ]  breakaway ellipsoid (closest approach), link loss v
                                                                        |
            SET_ATTITUDE_TARGET (roll, pitch, thrust) in GUIDED <--+
                          |
            ArduPlane roll/pitch controllers -> surfaces, throttle -> motor

ArduPlane interface notes (checked against ArduPlane master source)
-------------------------------------------------------------------
* SET_ATTITUDE_TARGET is accepted only in GUIDED.
* IMPORTANT: with a full-attitude type_mask (e.g. 0b00000111) ArduPlane copies
  the quaternion's yaw into ``forced_rpy_cd.z`` and ``calc_nav_yaw_coordinated()``
  then uses that number directly as the RUDDER command (not a heading). Sending
  a heading there gives large, arbitrary rudder and crossed-control flight.
  We therefore use ArduPlane's partial-attitude convention:
  ``type_mask = 0b10000100`` (attitude-ignore bit set, roll-rate and pitch-rate
  bits clear -> use roll and pitch from q, yaw-rate bit set -> leave yaw/rudder
  to ArduPlane's turn coordination, throttle bit clear -> use thrust).
* ``thrust`` (0..1) becomes a direct throttle override, but only when
  ``TRIM_THROTTLE > 1``. While pitch and throttle are forced, TECS demands
  are not used for those axes: the NMPC owns energy management and ArduPlane
  owns the attitude loops and all actuators.
* Forced roll/pitch/throttle expire after ``GUID_TIMEOUT`` seconds
  (default 3 s), after which ArduPlane reverts to its own GUIDED navigation.
  That is the last-resort failsafe if this process dies.
* ArduPlane clips commands to ``ROLL_LIMIT_DEG`` and ``PTCH_LIM_MIN/MAX_DEG``,
  and slews throttle at ``THR_SLEWRATE``. Keep the NMPC limits inside those.
* ``ATTITUDE.pitch`` is reported with ``PTCH_TRIM_DEG`` removed and the nav
  pitch demand has it added back, so commanded and measured pitch share a
  frame. Any residual offset is absorbed by the disturbance observer.
* ``LOCAL_POSITION_NED`` is relative to each vehicle's own EKF origin. The two
  origins are aligned using ``GPS_GLOBAL_ORIGIN`` from both vehicles.
* Recommended on the follower: ``TRIM_THROTTLE`` > 1, ``GUID_TIMEOUT`` 1-3 s,
  ``ROLL_LIMIT_DEG`` / ``PTCH_LIM_*`` wider than the NMPC limits, and no stick
  input (or ``STICK_MIXING`` 0) while the NMPC flies.

Validation (October 2026)
-------------------------
* ``--selftest``: offline closed loop against a separate truth model with 25 %
  drag error, 8 % thrust error, 0.4 s inner-loop dead time, 0.72 turn
  efficiency, 1 deg pitch-tracking offset, 5 m/s gusting wind, sensor noise:
  about 0.3 / 0.2 / 0.2 m RMS slot error (along / cross / vertical).
* Two ArduPlane SITL instances (current master, built-in ``plane`` model,
  different home locations), driven through this file's MAVLink runner:
  0.08 / 0.01 / 0.03 m RMS on straight legs, 0.27 / 0.05 / 0.15 m in a 180 m
  loiter turn, 1.4 m peak during the leader's roll-out with a 15 m climb.
  Solves averaged ~14 ms (single core, numpy).

Not modelled (by design): aerodynamic interaction between aircraft
(upwash, downwash, wake, induced-drag benefit).

Usage
-----
::

    # offline closed-loop self-test (no autopilot needed)
    python formation_nmpc.py --selftest --plot selftest.png

    # two SITL instances, e.g.
    #   sim_vehicle.py -v ArduPlane -I0 --out=udp:127.0.0.1:14550
    #   sim_vehicle.py -v ArduPlane -I1 --out=udp:127.0.0.1:14560
    python formation_nmpc.py --leader udpin:0.0.0.0:14550 \\
                             --follower udpin:0.0.0.0:14560 \\
                             --offset -6 4 0 --rate 10 --viz

Dependencies: numpy, scipy, pymavlink; matplotlib only for --viz / --plot.
"""

from __future__ import annotations

import argparse
import csv
import logging
import math
import os
import queue as _queue
import threading
import time
from collections import Counter, deque
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

# Small dense problems run faster and with less timing jitter on one BLAS
# thread; must be set before numpy is imported (an explicit env var wins).
for _var in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_var, "1")

import numpy as np
from scipy.linalg import cho_factor, cho_solve
from scipy.optimize import Bounds, minimize

os.environ.setdefault("MAVLINK20", "1")
try:  # pymavlink is only needed for the live MAVLink runner
    from pymavlink import mavutil
except ImportError:  # pragma: no cover
    mavutil = None

LOG = logging.getLogger("formation_nmpc")
G0 = 9.80665


# =============================================================================
# Small utilities
# =============================================================================

def wrap_pi(a):
    """Wrap an angle (scalar or array) to [-pi, pi)."""
    return (a + math.pi) % (2.0 * math.pi) - math.pi


def euler_to_quaternion(roll: float, pitch: float, yaw: float) -> List[float]:
    """ZYX Euler angles [rad] -> unit quaternion [w, x, y, z] (MAVLink order)."""
    cr, sr = math.cos(0.5 * roll), math.sin(0.5 * roll)
    cp, sp = math.cos(0.5 * pitch), math.sin(0.5 * pitch)
    cy, sy = math.cos(0.5 * yaw), math.sin(0.5 * yaw)
    return [
        cr * cp * cy + sr * sp * sy,
        sr * cp * cy - cr * sp * sy,
        cr * sp * cy + sr * cp * sy,
        cr * cp * sy - sr * sp * cy,
    ]


def euler_yaw_rate(roll: float, pitch: float, q: float, r: float) -> float:
    """Euler yaw rate from body rates: psi_dot = (q sin(phi) + r cos(phi)) / cos(theta)."""
    return (q * math.sin(roll) + r * math.cos(roll)) / max(math.cos(pitch), 0.2)


def _ema_alpha(dt: float, tau: float) -> float:
    return 1.0 - math.exp(-max(dt, 0.0) / max(tau, 1e-6))


EARTH_RADIUS_M = 6378137.0


def origin_offset_ned(origin_from: Tuple[float, float, float],
                      origin_to: Tuple[float, float, float]) -> np.ndarray:
    """NED position of ``origin_from`` expressed in the local frame of ``origin_to``.

    Origins are (lat_deg, lon_deg, alt_m AMSL). Flat-earth approximation, fine
    for separations of a few km.
    """
    lat_f, lon_f, alt_f = origin_from
    lat_t, lon_t, alt_t = origin_to
    dn = math.radians(lat_f - lat_t) * EARTH_RADIUS_M
    de = math.radians(lon_f - lon_t) * EARTH_RADIUS_M * math.cos(math.radians(lat_t))
    dd = -(alt_f - alt_t)
    return np.array([dn, de, dd])


def box_qp(H: np.ndarray, g: np.ndarray, lb: np.ndarray, ub: np.ndarray,
           max_iter: int = 50, tol: float = 1e-10) -> np.ndarray:
    """Solve min 0.5 x'Hx + g'x  s.t. lb <= x <= ub  (H symmetric positive definite).

    Projected Newton with an epsilon-active set (Bertsekas, 1982): Newton step
    on the free variables (Cholesky), projected Armijo line search. For the
    ~60-variable GN subproblem it is about 7x faster than scipy's BVLS and
    agrees with it to ~1e-5 relative objective.
    """
    n = g.size
    x = np.clip(np.zeros(n), lb, ub)
    qx = 0.5 * x @ (H @ x) + g @ x
    for _ in range(max_iter):
        grad = H @ x + g
        eps = min(1e-8 + float(np.linalg.norm(x - np.clip(x - grad, lb, ub))), 1e-3)
        active = ((x <= lb + eps) & (grad > 0.0)) | ((x >= ub - eps) & (grad < 0.0))
        free = ~active
        d = np.zeros(n)
        if free.any():
            Hf = H[np.ix_(free, free)]
            d[free] = -cho_solve(cho_factor(Hf, check_finite=False), grad[free], check_finite=False)
        a = 1.0
        while True:
            xn = np.clip(x + a * d, lb, ub)
            qn = 0.5 * xn @ (H @ xn) + g @ xn
            if qn <= qx + 1e-4 * (grad @ (xn - x)) or a < 1e-8:
                break
            a *= 0.5
        step = float(np.max(np.abs(xn - x)))
        x, qx = xn, qn
        if step < tol:
            break
    return x


# =============================================================================
# Configuration (all physical and tuning parameters live here)
# =============================================================================

@dataclass
class AircraftParams:
    """Follower parameters used inside the NMPC prediction model.

    Defaults approximate ArduPilot's built-in SITL ``plane`` model
    (libraries/SITL/SIM_Plane.*): m = 2 kg, S = 0.45 m^2, b = 1.88 m,
    CL0 = 0.56, CL_alpha = 6.9 /rad, CD_p = 0.10, Oswald e = 0.9,
    full-throttle thrust = m*g/0.7. The inner-loop numbers (delay, lags) were
    identified from ArduPlane SITL step responses through SET_ATTITUDE_TARGET
    (roll: 0.21 s lag + 0.41 s dead time, pitch: 0.16 s + 0.38 s). Identify your
    own airframe the same way; the disturbance observer absorbs moderate errors.
    """
    mass: float = 5.9                # [kg]
    wing_area: float = 0.982          # S [m^2]
    rho: float = 1.225               # air density [kg/m^3]
    CL0: float = 0.25                 # lift coefficient at alpha = 0
    CL_alpha: float = 5            # lift-curve slope [1/rad]
    CL_max: float = 1.2              # usable max lift coefficient (stall margin)
    CD0: float = 0.028                # parasite drag coefficient
    k_induced: float = 0.04         # induced drag factor 1/(pi e AR)
    thrust_max: float = 38.9         # full-throttle static thrust [N]
    thrust_lapse: float = 0.0        # T_avail = T_max (1 - lapse V) [s/m]
    input_delay: float = 0.35        # command -> attitude dead time [s] (incl. link)
    tau_roll: float = 0.22           # closed-loop ArduPlane roll lag after the delay [s]
    tau_pitch: float = 0.18          # closed-loop ArduPlane pitch lag after the delay [s]
    tau_throttle: float = 0.30       # ESC/motor/prop + THR_SLEWRATE lag [s]
    turn_efficiency: float = 1.0     # chi_dot = k g tan(phi)/V; 1 = perfectly coordinated.
                                     # Adapted online (ArduPlane SITL plane: ~0.72).
    g: float = G0                    # [m/s^2]


@dataclass
class LimitParams:
    airspeed_min: float = 12.0       # soft [m/s]
    airspeed_max: float = 28.0       # soft [m/s]
    roll_max_deg: float = 40.0       # hard, keep below ROLL_LIMIT_DEG
    pitch_min_deg: float = -15.0     # hard, keep inside PTCH_LIM_MIN_DEG
    pitch_max_deg: float = 18.0      # hard, keep inside PTCH_LIM_MAX_DEG
    load_factor_max: float = 2.0     # n = cos(gamma)/cos(phi): tightens roll limit
    gamma_max_deg: float = 15.0      # soft flight-path-angle limit
    closure_speed_max: float = 6.0   # soft |V - V_slot| limit [m/s]: gentle join-ups,
                                     # no high-speed overtakes next to the leader
    throttle_min: float = 0.0        # hard
    throttle_max: float = 1.0        # hard
    roll_rate_max_deg: float = 35.0  # hard, on the command sequence [deg/s]; faster
                                     # reversals excite roll-pitch coupling in ArduPlane
    pitch_rate_max_deg: float = 15.0 # hard [deg/s]
    throttle_rate_max: float = 0.8   # hard [1/s] (ArduPlane THR_SLEWRATE=100 %/s)

    @property
    def roll_max(self) -> float:
        """Bank limit [rad] = min(roll_max, acos(1/n_max)) (level-turn load factor)."""
        n_lim = math.acos(1.0 / max(self.load_factor_max, 1.0001))
        return min(math.radians(self.roll_max_deg), n_lim)

    @property
    def pitch_min(self) -> float:
        return math.radians(self.pitch_min_deg)

    @property
    def pitch_max(self) -> float:
        return math.radians(self.pitch_max_deg)

    @property
    def gamma_max(self) -> float:
        return math.radians(self.gamma_max_deg)

    @property
    def rate_max(self) -> np.ndarray:
        return np.array([math.radians(self.roll_rate_max_deg),
                         math.radians(self.pitch_rate_max_deg),
                         self.throttle_rate_max])


@dataclass
class FormationParams:
    # Desired slot relative to the leader, in the leader's heading frame:
    # x forward, y right, z down [m]. Leader roll/pitch are deliberately NOT
    # applied, otherwise the slot would heave vertically whenever the leader banks.
    offset: Tuple[float, float, float] = (-6.0, 4.0, 0.0)
    # Soft keep-out ellipsoid (NMPC penalty) semi-axes [m], leader heading frame.
    separation_axes: Tuple[float, float, float] = (3.0, 2.5, 1.5)
    # Hard breakaway ellipsoid (safety supervisor) semi-axes [m].
    breakaway_axes: Tuple[float, float, float] = (2.0, 1.6, 1.0)
    breakaway_lookahead_s: float = 0.8     # also trigger on predicted closest approach
    breakaway_roll_deg: float = 30.0
    breakaway_pitch_deg: float = 6.0       # added to / subtracted from trim pitch
    breakaway_hold_s: float = 4.0
    rejoin_scale: float = 1.5              # rejoin once outside scale * separation ellipsoid
    auto_rejoin: bool = True


@dataclass
class MPCWeights:
    # Tracking terms, errors expressed in the slot frame (along, cross, vertical).
    pos: Tuple[float, float, float] = (1.0, 2.5, 4.0)   # [1/m^2]
    # Pseudo-Huber scale for the position errors [m]: quadratic below, linear
    # above, so a 50 m join-up error gives bounded urgency, not saturation.
    pos_huber: Tuple[float, float, float] = (5.0, 5.0, 3.0)
    vel: Tuple[float, float, float] = (1.0, 1.0, 2.0)   # [1/(m/s)^2]
    heading: float = 20.0          # on 2(1 - cos(chi - chi_ref)) ~ dchi^2
    terminal_scale: float = 4.0    # multiplies the tracking terms at k = N
    # Inputs, in normalised units z = (u - centre)/half_range (roll, pitch, throttle).
    effort: Tuple[float, float, float] = (1.0, 2.0, 1.0)   # on (z - z_ff)^2
    rate: Tuple[float, float, float] = (1.0, 2.0, 0.5)     # on (dz/dt)^2 [s^2]
    power: float = 0.0             # optional soft energy term on T*V / P_ref
    # Soft constraints (squared hinge on the violation).
    airspeed: float = 2e4          # [1/(m/s)^2] effectively hard: never trade
                                   # airspeed margin for position
    closure: float = 200.0         # on (|V - V_slot| - closure_speed_max)+^2
    stall: float = 4000.0          # on (CL - CL_max)^2
    gamma: float = 2000.0          # on (|gamma| - gamma_max)^2 [1/rad^2]
    separation: float = 3000.0     # on (1 - ellipsoid_metric)^2


@dataclass
class MPCParams:
    horizon: int = 20              # N (15-25)
    dt: float = 0.2                # prediction step [s] (0.15-0.25)
    integrator: str = "rk2"        # "rk2" (midpoint) or "rk4" for the slow states
    solver: str = "gn"             # "gn" (bounded Gauss-Newton) or "slsqp"
    max_iter: int = 10             # iterations per solve (GN converges in 2-5 warm)
    ftol: float = 1e-3             # GN: relative cost decrease; SLSQP: absolute
    xtol: float = 1e-3             # GN only: max |dz| step size (normalised units)
    gn_damping: float = 1e-3       # GN only: Levenberg-Marquardt damping on |dz|^2
    fd_step: float = 1e-6          # finite-difference step in normalised inputs
    time_budget_frac: float = 0.70 # fraction of the control period for the solver
    max_overrun_evals: int = 6     # evaluations allowed past the budget to get one improving step


@dataclass
class ObserverParams:
    """Disturbance observer / online adaptation that makes tracking offset-free
    despite model error (wrong CD0, thrust, CL0, pitch trim, sideslip ...)."""
    enabled: bool = True
    tau_gamma: float = 1.5         # flight-path-angle bias filter [s]
    tau_turn: float = 2.0          # turn-efficiency / turn-rate bias filter [s]
    speed_gain: float = 0.10       # integrator gain on one-step airspeed error
    pitch_gain: float = 0.05       # integrator gain on one-step pitch error
    max_gamma_bias_deg: float = 6.0
    max_turn_bias_dps: float = 3.0
    turn_efficiency_range: Tuple[float, float] = (0.5, 1.3)
    max_accel_bias: float = 3.0    # [m/s^2]
    max_pitch_offset_deg: float = 5.0


@dataclass
class LinkParams:
    leader_url: str = "udpin:0.0.0.0:14550"
    follower_url: str = "udpin:0.0.0.0:14560"
    leader_sysid: Optional[int] = None    # required if both share one link
    follower_sysid: Optional[int] = None
    compid: int = 1                       # autopilot component
    baud: int = 57600                     # serial links only
    source_system: int = 250              # this companion's MAVLink sysid
    stream_hz: float = 25.0               # LOCAL_POSITION_NED + ATTITUDE rate
    state_timeout_s: float = 0.5          # position/attitude considered stale
    leader_coast_s: float = 1.0           # dead-reckon leader this long after stale
    heartbeat_timeout_s: float = 3.0
    origin_timeout_s: float = 90.0        # wait for GPS lock / EKF origin at startup
    assume_common_origin: bool = False    # skip GPS_GLOBAL_ORIGIN alignment
    link_latency_s: float = 0.03          # added to solve time for latency comp.


@dataclass
class ControllerConfig:
    aircraft: AircraftParams = field(default_factory=AircraftParams)
    limits: LimitParams = field(default_factory=LimitParams)
    formation: FormationParams = field(default_factory=FormationParams)
    weights: MPCWeights = field(default_factory=MPCWeights)
    mpc: MPCParams = field(default_factory=MPCParams)
    observer: ObserverParams = field(default_factory=ObserverParams)
    link: LinkParams = field(default_factory=LinkParams)
    control_rate_hz: float = 10.0
    auto_guided: bool = True              # put follower into GUIDED once when ready
    lost_link_hold_s: float = 2.0         # wings-level hold before failsafe mode
    failsafe_mode: Optional[str] = "LOITER"
    exit_mode: Optional[str] = "LOITER"   # mode requested on clean shutdown
    visualize: bool = False
    log_csv: Optional[str] = None

    @property
    def control_period(self) -> float:
        return 1.0 / self.control_rate_hz

    def validate(self):
        """Raise on unsafe/inconsistent settings, warn on questionable ones."""
        lim, f, m = self.limits, self.formation, self.mpc
        if not (lim.airspeed_min < lim.airspeed_max and lim.pitch_min < lim.pitch_max
                and lim.throttle_min < lim.throttle_max):
            raise ValueError("limit ranges must satisfy min < max")
        if m.horizon < 2 or m.dt <= 0.0 or self.control_rate_hz <= 0.0:
            raise ValueError("horizon >= 2, dt > 0 and control_rate_hz > 0 required")
        if m.solver not in ("gn", "slsqp") or m.integrator not in ("rk2", "rk4"):
            raise ValueError("mpc.solver must be 'gn'|'slsqp', mpc.integrator 'rk2'|'rk4'")
        if ellipsoid_metric(np.array(f.offset), f.breakaway_axes) <= 1.0:
            raise ValueError(f"formation slot {f.offset} lies inside the breakaway ellipsoid")
        if any(b > s for b, s in zip(f.breakaway_axes, f.separation_axes)):
            LOG.warning("breakaway ellipsoid is larger than the keep-out ellipsoid on some axis")
        if ellipsoid_metric(np.array(f.offset), f.separation_axes) <= 1.0:
            LOG.warning("slot %s is inside the keep-out ellipsoid: the NMPC will hold at its boundary",
                        f.offset)
        if not (15 <= m.horizon <= 25) or not (0.15 <= m.dt <= 0.25):
            LOG.warning("horizon/dt outside the recommended 15-25 steps / 0.15-0.25 s")
        if self.control_period > m.dt + 1e-9:
            LOG.warning("control period (%.3f s) longer than the MPC step dt (%.3f s)",
                        self.control_period, m.dt)
        if not (5.0 <= self.control_rate_hz <= 25.0):
            LOG.warning("control rate %.1f Hz outside the intended 10-20 Hz range", self.control_rate_hz)


# =============================================================================
# Prediction model: 5-DoF point mass with attitude/throttle lags
# =============================================================================

class FixedWingPointMassModel:
    r"""Smooth 5-DoF point-mass model in NED, wind-relative (air-mass) formulation.

    State  x = [n, e, d, V, chi, phi, theta, delta]   (8)
        n, e, d  position NED [m]           V      airspeed [m/s]
        chi      air-relative heading [rad] phi    bank [rad]
        theta    pitch [rad]                delta  throttle state [0..1]
    Input  u = [phi_c, theta_c, delta_c]               (3)
    Bias   b = [b_Vdot, b_gamma, b_chidot, k_turn, b_theta]   (observer; see BIAS_*)
    Wind   w = [w_n, w_e, w_d]  air-mass velocity in NED

    Inner loops (ArduPlane attitude controllers, ESC/slew): a dead time
    ``input_delay`` (handled outside the model by predicting through the
    commands already in flight, see ``CommandHistory``) followed by
    first-order lags:
        phi_dot   = (phi_c             - phi)   / tau_phi
        theta_dot = (theta_c + b_theta - theta) / tau_theta   (pitch-tracking offset)
        delta_dot = (delta_c           - delta) / tau_delta
    These are linear with zero-order-hold inputs, so they are integrated
    exactly: x+ = u + (x - u) exp(-dt/tau). No stiffness, any dt.

    Aerodynamics (standard quadratic polar):
        qbar S = 0.5 rho V^2 S
        L = qbar S CL,   D = qbar S (CD0 + k CL^2),   T = delta T_max (1 - lapse V)

    Flight-path angle (quasi-steady, singular perturbation). The flight path
    responds to pitch with time constant 2m/(rho V S CL_alpha), about 0.05-0.2 s,
    much faster than dt. With ArduPlane holding pitch, gamma therefore sits on
    its slow manifold where lift balances weight across the path:
        L cos(phi) = m g cos(gamma)   ->   CL = m g cos(theta) / (qbar S cos(phi))
        alpha = (CL - CL0) / CL_alpha
        gamma = theta - alpha cos(phi) + b_gamma
    (small-angle body-to-wind geometry for a banked aircraft). This keeps the
    pitch -> lift -> climb physics, captures the bank-dependent alpha/stall
    margin and the induced-drag increase in turns, and stays smooth.

    Point-mass dynamics (wind axes):
        V_dot   = (T cos(alpha) - D)/m - g sin(gamma)            + b_Vdot
        chi_dot = k_turn g tan(phi) / V                          + b_chidot
                  (coordinated turn, k_turn = 1, equals L sin(phi)/(m V cos(gamma))
                   when L cos(phi) = m g cos(gamma); k_turn < 1 captures the
                   sideslip of an imperfectly coordinated turn)
        n_dot   = V cos(gamma) cos(chi) + w_n
        e_dot   = V cos(gamma) sin(chi) + w_e
        d_dot   = -V sin(gamma)         + w_d

    Every function takes a ``lib`` argument (numpy by default). Only
    sin/cos/tan/fmax and arithmetic are used, so passing the ``casadi`` module
    with SX/MX symbols yields the same model symbolically for acados/CasADi.
    """

    NX, NU = 8, 3
    IN, IE, ID, IV, ICHI, IPHI, ITH, IDL = range(8)
    B_VDOT, B_GAMMA, B_CHIDOT, K_TURN, B_THETA = range(5)

    def __init__(self, p: AircraftParams, v_floor: float = 4.0):
        self.p = p
        self.v_floor = v_floor
        self._weight = p.mass * p.g
        self._half_rho_s = 0.5 * p.rho * p.wing_area
        self._tau = np.array([p.tau_roll, p.tau_pitch, p.tau_throttle])

    def default_bias(self) -> np.ndarray:
        return np.array([0.0, 0.0, 0.0, self.p.turn_efficiency, 0.0])

    @staticmethod
    def lag_offset(bias) -> np.ndarray:
        """Offset added to the commands to get the inner-loop lag targets."""
        return np.array([0.0, bias[4], 0.0])

    # ---------------------------------------------------------------- aero
    def aero(self, V, phi, theta, b_gamma=0.0, lib=np):
        """Return (gamma, alpha, CL, drag, V_safe) for the quasi-steady path."""
        p = self.p
        Vs = lib.fmax(V, self.v_floor)
        qS = self._half_rho_s * Vs * Vs
        cphi = lib.cos(phi)
        CL = self._weight * lib.cos(theta) / (qS * cphi)
        alpha = (CL - p.CL0) / p.CL_alpha
        gamma = theta - alpha * cphi + b_gamma
        drag = qS * (p.CD0 + p.k_induced * CL * CL)
        return gamma, alpha, CL, drag, Vs

    def thrust(self, delta, V_safe):
        p = self.p
        return delta * (p.thrust_max * (1.0 - p.thrust_lapse * V_safe))

    def derivatives(self, V, chi, phi, theta, delta, wind, bias, lib=np):
        """Time derivatives of the slow states (n, e, d, V, chi) plus outputs (gamma, CL)."""
        p = self.p
        gamma, alpha, CL, drag, Vs = self.aero(V, phi, theta, bias[1], lib)
        T = self.thrust(delta, Vs)
        V_dot = (T * lib.cos(alpha) - drag) / p.mass - p.g * lib.sin(gamma) + bias[0]
        chi_dot = bias[3] * p.g * lib.tan(phi) / Vs + bias[2]
        Vh = V * lib.cos(gamma)
        n_dot = Vh * lib.cos(chi) + wind[0]
        e_dot = Vh * lib.sin(chi) + wind[1]
        d_dot = -V * lib.sin(gamma) + wind[2]
        return (n_dot, e_dot, d_dot, V_dot, chi_dot), (gamma, CL)

    # ------------------------------------------------------- integration
    def lag_factors(self, dt: float) -> Tuple[np.ndarray, np.ndarray]:
        """Exact decay factors of the first-order lags at dt/2 and dt."""
        return np.exp(-0.5 * dt / self._tau), np.exp(-dt / self._tau)

    def propagate(self, x: Sequence[float], u: Sequence[float], dt: float,
                  wind: Sequence[float], bias: Sequence[float]) -> np.ndarray:
        """Single-state propagation over dt (RK4 slow states + exact lags)."""
        x = np.asarray(x, dtype=float)
        if dt <= 1e-6:
            return x.copy()
        n, e, d, V, chi = x[:5]
        lag0 = x[5:8]
        u = np.asarray(u, dtype=float) + self.lag_offset(bias)
        hm, he = self.lag_factors(dt)
        lagm = u + (lag0 - u) * hm
        lage = u + (lag0 - u) * he
        f1, _ = self.derivatives(V, chi, *lag0, wind, bias)
        f2, _ = self.derivatives(V + 0.5 * dt * f1[3], chi + 0.5 * dt * f1[4], *lagm, wind, bias)
        f3, _ = self.derivatives(V + 0.5 * dt * f2[3], chi + 0.5 * dt * f2[4], *lagm, wind, bias)
        f4, _ = self.derivatives(V + dt * f3[3], chi + dt * f3[4], *lage, wind, bias)
        c = dt / 6.0
        s = [n, e, d, V, chi]
        out = [s[i] + c * (f1[i] + 2.0 * (f2[i] + f3[i]) + f4[i]) for i in range(5)]
        return np.array(out + list(lage))

    # --------------------------------------------------------------- trim
    def trim(self, V, gamma, phi, bias=None, iters: int = 3):
        """Pitch and throttle COMMANDS that hold airspeed V on flight path gamma
        at bank phi. Vectorised. Used for feedforward and warm starts. Includes
        the observer biases so the feedforward matches the corrected model.
        """
        p = self.p
        bias = self.default_bias() if bias is None else bias
        V = np.asarray(V, dtype=float)
        gamma = np.asarray(gamma, dtype=float)
        phi = np.asarray(phi, dtype=float)
        theta = gamma.copy()
        for _ in range(iters):  # theta = gamma - b_gamma + alpha(theta) cos(phi)
            _, alpha, _, _, _ = self.aero(V, phi, theta)
            theta = gamma - bias[1] + alpha * np.cos(phi)
        _, alpha, CL, drag, Vs = self.aero(V, phi, theta)
        T_req = drag + p.mass * p.g * np.sin(gamma) - p.mass * bias[0]
        delta = T_req / (np.cos(alpha) * p.thrust_max * (1.0 - p.thrust_lapse * Vs))
        return theta - bias[4], delta


# =============================================================================
# Leader prediction and formation reference
# =============================================================================

@dataclass
class LeaderEstimate:
    t: float               # time the position/velocity refer to [s, monotonic]
    pos: np.ndarray        # NED in the controller (follower) frame [m]
    vel: np.ndarray        # ground velocity NED [m/s]
    heading: float         # air-relative heading psi [rad]
    airspeed_h: float      # horizontal airspeed [m/s]
    turn_rate: float       # psi_dot [rad/s]
    vd: float              # filtered down-velocity [m/s]


@dataclass
class FormationReference:
    """Reference trajectory over k = 0..N (index 0 = time of x0)."""
    tau: np.ndarray        # (N+1,) seconds after the leader measurement
    leader_pos: np.ndarray # (N+1, 3)
    psi: np.ndarray        # (N+1,) leader heading (defines the slot frame)
    cpsi: np.ndarray
    spsi: np.ndarray
    slot_pos: np.ndarray   # (N+1, 3)
    slot_vel: np.ndarray   # (N+1, 3) ground velocity incl. omega x r
    chi_ref: np.ndarray    # (N+1,) desired air-relative heading
    V_ref: np.ndarray      # (N+1,) desired airspeed
    gamma_ref: np.ndarray  # (N+1,)
    u_ff: np.ndarray       # (N, 3) feedforward (roll, pitch, throttle)


class ReferenceGenerator:
    """Leader constant-turn-rate prediction and formation-slot kinematics.

    The leader is assumed to keep horizontal airspeed V_h, turn rate omega and
    vertical speed over the horizon. Propagation is done in the air mass (a
    constant-rate turn in wind is a trochoid over the ground):

        psi(t)  = psi0 + omega t
        p_h(t)  = p_h0 + w_h t + V_h t [cos(psi0 + omega t/2), sin(psi0 + omega t/2)] sinc(omega t/2)
        d(t)    = d0 + vd t

    (the sinc form is exact and well conditioned as omega -> 0).

    The slot is a rigid offset r = R(psi) [x, y]^T + [0, 0, z] that rotates with
    the leader heading, so its ground velocity contains the transport term

        v_slot = v_leader + omega_vec x r = v_leader + [-omega r_e, omega r_n, 0].

    Inside the turn the slot is slower, outside it is faster, and its velocity
    direction leads/lags the leader course for fore/aft offsets. The desired
    follower airspeed, air heading and flight path follow from v_slot - w, and
    the feedforward bank is phi_ff = atan(V_ref omega / (k_turn g)).
    """

    def __init__(self, model: FixedWingPointMassModel, limits: LimitParams,
                 formation: FormationParams):
        self.model = model
        self.lim = limits
        self.form = formation

    def propagate_leader(self, L: LeaderEstimate, tau: np.ndarray, wind: np.ndarray):
        tau = np.asarray(tau, dtype=float)
        w = L.turn_rate
        half = 0.5 * w * tau
        sinc = np.sinc(half / math.pi)                     # sin(x)/x
        mid = L.heading + half
        dn = L.airspeed_h * tau * np.cos(mid) * sinc + wind[0] * tau
        de = L.airspeed_h * tau * np.sin(mid) * sinc + wind[1] * tau
        dd = L.vd * tau
        pos = L.pos[None, :] + np.stack([dn, de, dd], axis=-1)
        psi = L.heading + w * tau
        vel = np.stack([L.airspeed_h * np.cos(psi) + wind[0],
                        L.airspeed_h * np.sin(psi) + wind[1],
                        np.full_like(tau, L.vd)], axis=-1)
        return pos, vel, psi

    def slot(self, L: LeaderEstimate, tau: np.ndarray, wind: np.ndarray):
        pos, vel, psi = self.propagate_leader(L, tau, wind)
        x, y, z = self.form.offset
        c, s = np.cos(psi), np.sin(psi)
        rn = x * c - y * s
        re = x * s + y * c
        slot_pos = pos + np.stack([rn, re, np.full_like(rn, z)], axis=-1)
        w = L.turn_rate
        slot_vel = vel + np.stack([-w * re, w * rn, np.zeros_like(rn)], axis=-1)
        return pos, psi, slot_pos, slot_vel

    def generate(self, L: LeaderEstimate, t0: float, N: int, dt: float,
                 wind: np.ndarray, bias: np.ndarray) -> FormationReference:
        tau = (t0 - L.t) + dt * np.arange(N + 1)
        lpos, psi, spos, svel = self.slot(L, tau, wind)
        air = svel - wind[None, :]
        V_ref = np.maximum(np.linalg.norm(air, axis=1), 1.0)
        chi_ref = np.arctan2(air[:, 1], air[:, 0])
        gamma_ref = np.arcsin(np.clip(-air[:, 2] / V_ref, -1.0, 1.0))

        lim = self.lim
        V_ff = np.clip(V_ref[:-1], lim.airspeed_min, lim.airspeed_max)
        g_ff = np.clip(gamma_ref[:-1], -lim.gamma_max, lim.gamma_max)
        phi_ff = np.clip(np.arctan(V_ff * L.turn_rate / (self.model.p.g * bias[3])),
                         -lim.roll_max, lim.roll_max)
        th_ff, dl_ff = self.model.trim(V_ff, g_ff, phi_ff, bias)
        u_ff = np.stack([phi_ff,
                         np.clip(th_ff, lim.pitch_min, lim.pitch_max),
                         np.clip(dl_ff, lim.throttle_min, lim.throttle_max)], axis=-1)
        return FormationReference(tau=tau, leader_pos=lpos, psi=psi, cpsi=np.cos(psi),
                                  spsi=np.sin(psi), slot_pos=spos, slot_vel=svel,
                                  chi_ref=chi_ref, V_ref=V_ref, gamma_ref=gamma_ref, u_ff=u_ff)


# =============================================================================
# NMPC (optimal control problem + solver backends)
# =============================================================================

class _DeadlineReached(Exception):
    pass


@dataclass
class NMPCSolution:
    U: np.ndarray                      # (N, 3) commands [rad, rad, 0..1]
    traj: Dict[str, np.ndarray]        # predicted outputs at k = 1..N, each (N,)
    cost: float
    status: int
    message: str
    nit: int
    nfev: int
    solve_time: float
    deadline_hit: bool
    success: bool


@dataclass
class _EvalContext:
    x0: np.ndarray
    ref: FormationReference
    wind: np.ndarray
    bias: np.ndarray
    z_prev: np.ndarray                 # (3,) last sent command, normalised
    z_ff: np.ndarray                   # (N, 3) feedforward, normalised
    deadline: float
    best_cost: float = math.inf
    best_z: Optional[np.ndarray] = None
    first_cost: Optional[float] = None
    nfev: int = 0
    njev: int = 0
    n_batches: int = 0


class FormationNMPC:
    r"""Formation NMPC, single shooting, nonlinear least-squares cost.

    Decision variables
        z = [z_0, ..., z_{N-1}],  z_k in R^3, with u_k = c + s * z_k
        (roll, pitch, throttle; normalised so every channel spans [-1, 1]).
        Yaw is not a free variable: in a turn the heading rate is fixed by bank
        (chi_dot = k_turn g tan(phi)/V), and in ArduPlane GUIDED a forced yaw is
        applied as a raw rudder command, which would fight turn coordination.
        So yaw/heading is controlled through bank; ArduPlane keeps the rudder.

    Cost J = ||r(z)||^2 / N  (predicted state at k = 1..N, w_N = terminal_scale)
        tracking   w_k [ e_p^T Q_p e_p + e_v^T Q_v e_v + q_chi 4 sin^2((chi - chi_ref)/2) ]
                   (4 sin^2(x/2) = 2(1 - cos x): smooth, wrap-free heading error)
        effort     (z_k - z_ff,k)^T R (z_k - z_ff,k)          about the feedforward trim
        rate       |(z_k - z_{k-1})/dt|^2_Rd                  z_{-1} = last sent command
        energy     w_P T_k V_k / P_ref                        optional
        soft constraints (squared hinges, always feasible):
            w_V [(V_min - V)+^2 + (V - V_max)+^2]     airspeed envelope
            w_c (|V - V_slot| - dV_max)+^2            closure-speed limit
            w_CL (CL - CL_max)+^2                     stall margin (bank dependent)
            w_g (|gamma| - gamma_max)+^2              flight-path limit
            w_s (1 - s)+^2,  s = (dx/a)^2 + (dy/b)^2 + (dz/c)^2   keep-out ellipsoid
        Position and velocity errors are rotated into the slot frame
        (along-track, cross-track, vertical) so each axis has its own weight.
        Position errors pass through a pseudo-Huber map rho(e) = e (1 + (e/c)^2)^(-1/4)
        (rho^2 ~ e^2 for |e| << c, ~ c |e| for |e| >> c): large join-up errors
        get a bounded pull instead of driving every input into saturation.

    Hard constraints
        bounds        |phi| <= phi_max = min(roll limit, acos(1/n_max)),
                      theta in [theta_min, theta_max], delta in [0, 1]
        first step    |u_0 - u_last| <= rate_max * T_ctrl     (as tightened bounds)
        later steps   |u_k - u_{k-1}| <= rate_max * dt
                      SLSQP: linear inequality constraints.
                      GN:    enforced on the returned plan by projection (only
                             u_0 is applied, and it is always hard-limited).

    State constraints are soft on purpose: the problem is always feasible,
    which matters more in a real-time loop than an exact constraint boundary.
    The hard separation guarantee comes from ``SafetySupervisor``.

    Solver backends (MPCParams.solver)
        "gn"     Bounded Gauss-Newton / Levenberg-Marquardt (``_solve_gn``):
                 bounded step from a projected-Newton box QP and a batched
                 line search. Uses the residual Jacobian, so it converges in
                 2-5 iterations from a warm start; the real-time default.
        "slsqp"  scipy.optimize.minimize(method="SLSQP") on J with exact linear
                 rate constraints. More iterations (quasi-Newton from scratch
                 each solve), so give it a lower rate or a shorter horizon.

    Derivatives: batched finite differences. All 1 + 3N perturbed input
    sequences go through one vectorised rollout, which in numpy costs about the
    same as a single rollout. The residual form maps directly onto acados'
    NONLINEAR_LS cost; to port, rebuild ``FixedWingPointMassModel.derivatives``
    with lib=casadi and transcribe ``_residuals``.

    Real time: every evaluation checks the wall-clock deadline. On timeout the
    best evaluated iterate is used (all iterates satisfy the hard constraints).
    """

    NU = 3

    def __init__(self, model: FixedWingPointMassModel, limits: LimitParams,
                 weights: MPCWeights, params: MPCParams, formation: FormationParams,
                 control_period: float):
        self.model = model
        self.lim = limits
        self.w = weights
        self.p = params
        self.form = formation
        self.N = params.horizon
        self.dt = params.dt
        self.Tc = control_period

        self.u_lb = np.array([-limits.roll_max, limits.pitch_min, limits.throttle_min])
        self.u_ub = np.array([limits.roll_max, limits.pitch_max, limits.throttle_max])
        self.u_c = 0.5 * (self.u_lb + self.u_ub)
        self.u_s = 0.5 * (self.u_ub - self.u_lb)
        self.rate_max = limits.rate_max
        self._dz_max = self.rate_max * self.dt / self.u_s

        self._lag_mid, self._lag_end = model.lag_factors(self.dt)
        wk = np.ones(self.N)
        wk[-1] = weights.terminal_scale
        self._sw = np.sqrt(wk)                                  # (N,)
        self._sq_pos = np.sqrt(np.asarray(weights.pos, dtype=float))
        self._sq_vel = np.sqrt(np.asarray(weights.vel, dtype=float))
        self._sq_head = math.sqrt(weights.heading)
        self._huber = tuple(float(x) for x in weights.pos_huber)
        self._sq_R = np.sqrt(np.asarray(weights.effort, dtype=float))
        self._sq_Rd = np.sqrt(np.asarray(weights.rate, dtype=float))
        self._sq_V = math.sqrt(weights.airspeed)
        self._sq_close = math.sqrt(weights.closure)
        self._sq_CL = math.sqrt(weights.stall)
        self._sq_g = math.sqrt(weights.gamma)
        self._sq_sep = math.sqrt(weights.separation)
        self._inv_sqrt_n = 1.0 / math.sqrt(self.N)
        self._sep_axes = np.asarray(formation.separation_axes, dtype=float)
        v_mid = 0.5 * (limits.airspeed_min + limits.airspeed_max)
        self._p_ref = model.p.thrust_max * v_mid
        self._A, self._b = self._rate_constraint_matrices()

        self._plan_U: Optional[np.ndarray] = None
        self._plan_t0: Optional[float] = None
        self._eval_time = 0.003                                  # EMA of one evaluation [s]
        self._gn_iter_time = 0.008                               # EMA of one GN iteration [s]

    # -------------------------------------------------------------- helpers
    def to_z(self, U: np.ndarray) -> np.ndarray:
        return (U - self.u_c) / self.u_s

    def to_u(self, Z: np.ndarray) -> np.ndarray:
        return self.u_c + self.u_s * Z

    def reset(self):
        self._plan_U = None
        self._plan_t0 = None

    def _rate_constraint_matrices(self):
        """Linear inequalities b - A z >= 0 for |z_k - z_{k-1}| <= dz_max, k >= 1."""
        N, nu = self.N, self.NU
        rows = 2 * (N - 1) * nu
        A = np.zeros((rows, N * nu))
        b = np.zeros(rows)
        r = 0
        for k in range(1, N):
            for j in range(nu):
                ik, ip = k * nu + j, (k - 1) * nu + j
                A[r, ik], A[r, ip], b[r] = 1.0, -1.0, self._dz_max[j]
                A[r + 1, ik], A[r + 1, ip], b[r + 1] = -1.0, 1.0, self._dz_max[j]
                r += 2
        return A, b

    def _project(self, U: np.ndarray, lb: np.ndarray, ub: np.ndarray) -> np.ndarray:
        """Make an input sequence feasible: box, then forward rate limits."""
        U = np.clip(U, lb, ub)
        dmax = self.rate_max * self.dt
        for k in range(1, self.N):
            U[k] = np.clip(np.clip(U[k], U[k - 1] - dmax, U[k - 1] + dmax), lb[k], ub[k])
        return U

    def warm_start(self, t0: float, u_ff: np.ndarray) -> np.ndarray:
        """Shift register: previous plan resampled at t0 + k dt (piecewise constant),
        tail extended by repeating the last input. Cold start uses the feedforward."""
        if self._plan_U is None or self._plan_t0 is None:
            return u_ff.copy()
        elapsed = t0 - self._plan_t0
        if elapsed < 0.0 or elapsed > 0.5 * self.N * self.dt:
            return u_ff.copy()
        idx = np.floor((np.arange(self.N) * self.dt + elapsed) / self.dt + 1e-9).astype(int)
        return self._plan_U[np.minimum(idx, self.N - 1)].copy()

    # ------------------------------------------------------------ rollout
    def _rollout(self, x0: np.ndarray, U: np.ndarray, wind: np.ndarray,
                 bias: np.ndarray) -> Dict[str, np.ndarray]:
        """Simulate B input sequences U (B, N, 3) in parallel.

        Returns arrays (B, N) for the predicted state/outputs at k = 1..N.
        Lags are integrated exactly; the slow states (n, e, d, V, chi) with RK2
        (midpoint) or RK4, using the exact lag values at the stage times.
        """
        B, N, _ = U.shape
        f = self.model.derivatives
        dt, half = self.dt, 0.5 * self.dt
        hm, he = self._lag_mid, self._lag_end
        rk4 = self.p.integrator.lower() == "rk4"

        n = np.full(B, x0[0]); e = np.full(B, x0[1]); d = np.full(B, x0[2])
        V = np.full(B, x0[3]); chi = np.full(B, x0[4])
        lag = np.empty((B, 3)); lag[:] = x0[5:8]

        keys = ("n", "e", "d", "V", "chi", "phi", "theta", "delta",
                "gamma", "CL", "vn", "ve", "vd")
        out = {k: np.empty((B, N)) for k in keys}

        off = self.model.lag_offset(bias)
        for k in range(N):
            uk = U[:, k, :] + off
            dl = lag - uk
            lag_m = uk + dl * hm
            lag_e = uk + dl * he
            f1, aux = f(V, chi, lag[:, 0], lag[:, 1], lag[:, 2], wind, bias)
            if k > 0:  # outputs at the start of step k = state index k
                j = k - 1
                out["gamma"][:, j], out["CL"][:, j] = aux
                out["vn"][:, j], out["ve"][:, j], out["vd"][:, j] = f1[0], f1[1], f1[2]
            if rk4:
                f2, _ = f(V + half * f1[3], chi + half * f1[4],
                          lag_m[:, 0], lag_m[:, 1], lag_m[:, 2], wind, bias)
                f3, _ = f(V + half * f2[3], chi + half * f2[4],
                          lag_m[:, 0], lag_m[:, 1], lag_m[:, 2], wind, bias)
                f4, _ = f(V + dt * f3[3], chi + dt * f3[4],
                          lag_e[:, 0], lag_e[:, 1], lag_e[:, 2], wind, bias)
                c = dt / 6.0
                n = n + c * (f1[0] + 2.0 * (f2[0] + f3[0]) + f4[0])
                e = e + c * (f1[1] + 2.0 * (f2[1] + f3[1]) + f4[1])
                d = d + c * (f1[2] + 2.0 * (f2[2] + f3[2]) + f4[2])
                V = V + c * (f1[3] + 2.0 * (f2[3] + f3[3]) + f4[3])
                chi = chi + c * (f1[4] + 2.0 * (f2[4] + f3[4]) + f4[4])
            else:
                f2, _ = f(V + half * f1[3], chi + half * f1[4],
                          lag_m[:, 0], lag_m[:, 1], lag_m[:, 2], wind, bias)
                n = n + dt * f2[0]; e = e + dt * f2[1]; d = d + dt * f2[2]
                V = V + dt * f2[3]; chi = chi + dt * f2[4]
            lag = lag_e
            out["n"][:, k], out["e"][:, k], out["d"][:, k] = n, e, d
            out["V"][:, k], out["chi"][:, k] = V, chi
            out["phi"][:, k], out["theta"][:, k], out["delta"][:, k] = lag[:, 0], lag[:, 1], lag[:, 2]

        f1, aux = f(V, chi, lag[:, 0], lag[:, 1], lag[:, 2], wind, bias)
        out["gamma"][:, -1], out["CL"][:, -1] = aux
        out["vn"][:, -1], out["ve"][:, -1], out["vd"][:, -1] = f1[0], f1[1], f1[2]
        return out

    # ---------------------------------------------------------- residuals
    def _residuals(self, tr: Dict[str, np.ndarray], Zr: np.ndarray,
                   ctx: _EvalContext) -> np.ndarray:
        """Weighted residual vector r (B, n_r) with J = ||r||^2 (vectorised over B)."""
        r, w, lim = ctx.ref, self.w, self.lim
        c, s = r.cpsi[1:], r.spsi[1:]
        sw = self._sw
        B = Zr.shape[0]
        hx, hy, hz = self._huber

        # --- tracking in the slot frame
        en = tr["n"] - r.slot_pos[1:, 0]
        ee = tr["e"] - r.slot_pos[1:, 1]
        ed = tr["d"] - r.slot_pos[1:, 2]
        vn = tr["vn"] - r.slot_vel[1:, 0]
        ve = tr["ve"] - r.slot_vel[1:, 1]
        vd = tr["vd"] - r.slot_vel[1:, 2]
        ex = en * c + ee * s
        ey = -en * s + ee * c
        parts = [
            (self._sq_pos[0] * sw) * ex * (1.0 + (ex / hx) ** 2) ** -0.25,
            (self._sq_pos[1] * sw) * ey * (1.0 + (ey / hy) ** 2) ** -0.25,
            (self._sq_pos[2] * sw) * ed * (1.0 + (ed / hz) ** 2) ** -0.25,
            (self._sq_vel[0] * sw) * (vn * c + ve * s),
            (self._sq_vel[1] * sw) * (-vn * s + ve * c),
            (self._sq_vel[2] * sw) * vd,
            (self._sq_head * sw) * (2.0 * np.sin(0.5 * (tr["chi"] - r.chi_ref[1:]))),
        ]

        # --- soft state constraints (hinges)
        V = tr["V"]
        dn = tr["n"] - r.leader_pos[1:, 0]
        de = tr["e"] - r.leader_pos[1:, 1]
        dd = tr["d"] - r.leader_pos[1:, 2]
        a, b, cz = self._sep_axes
        sx = (dn * c + de * s) / a
        sy = (-dn * s + de * c) / b
        sz = dd / cz
        parts += [
            self._sq_V * np.maximum(lim.airspeed_min - V, 0.0),
            self._sq_V * np.maximum(V - lim.airspeed_max, 0.0),
            self._sq_close * np.maximum(np.abs(V - r.V_ref[1:]) - lim.closure_speed_max, 0.0),
            self._sq_CL * np.maximum(tr["CL"] - self.model.p.CL_max, 0.0),
            self._sq_g * np.maximum(np.abs(tr["gamma"]) - lim.gamma_max, 0.0),
            self._sq_sep * np.maximum(1.0 - (sx * sx + sy * sy + sz * sz), 0.0),
        ]
        if w.power > 0.0:                                   # optional energy term
            Vs = np.maximum(V, self.model.v_floor)
            P = np.maximum(self.model.thrust(tr["delta"], Vs) * Vs, 0.0)
            parts.append(np.sqrt(w.power * P / self._p_ref + 1e-12))

        # --- inputs: effort about trim, and input rates
        dz = (Zr - ctx.z_ff[None]) * self._sq_R
        prev = np.broadcast_to(ctx.z_prev, (B, 1, 3))
        dZ = np.diff(Zr, axis=1, prepend=prev)
        blocks = [np.concatenate(parts, axis=1), dz.reshape(B, -1),
                  (dZ * (self._sq_Rd / self.dt)).reshape(B, -1)]
        return np.concatenate(blocks, axis=1) * self._inv_sqrt_n

    def _check_deadline(self, ctx: _EvalContext):
        """Real-time-iteration policy: once the budget (minus the expected cost of
        one more evaluation) is spent, stop, but only after at least one step has
        improved on the warm start. Returning the warm start unchanged would
        replay an old plan open loop. ``max_overrun_evals`` caps the overrun."""
        if ctx.best_z is None:
            return
        if time.monotonic() + self._eval_time > ctx.deadline:
            improved = ctx.first_cost is not None and ctx.best_cost < ctx.first_cost
            if improved or ctx.n_batches >= self.p.max_overrun_evals:
                raise _DeadlineReached()

    def _batch_residuals(self, Zb: np.ndarray, ctx: _EvalContext,
                         check_deadline: bool = True) -> np.ndarray:
        if check_deadline:
            self._check_deadline(ctx)
        t_eval = time.monotonic()
        B = Zb.shape[0]
        Zr = Zb.reshape(B, self.N, self.NU)
        tr = self._rollout(ctx.x0, self.to_u(Zr), ctx.wind, ctx.bias)
        R = self._residuals(tr, Zr, ctx)
        ctx.n_batches += 1
        self._eval_time += 0.2 * ((time.monotonic() - t_eval) - self._eval_time)
        J0 = float(R[0] @ R[0])
        if ctx.first_cost is None:
            ctx.first_cost = J0
        # Every evaluated point satisfies the hard constraints (bounds, and for
        # SLSQP the linear rate constraints), so keep the best as a fallback.
        if math.isfinite(J0) and J0 < ctx.best_cost:
            ctx.best_cost, ctx.best_z = J0, Zb[0].copy()
        return R

    def _perturbed(self, z: np.ndarray) -> np.ndarray:
        nz = z.size
        Zb = np.repeat(z[None], nz + 1, axis=0)
        Zb[1 + np.arange(nz), np.arange(nz)] += self.p.fd_step
        return Zb

    # ---- SLSQP interface: J and dJ/dz
    def _cost_and_grad(self, z: np.ndarray, ctx: _EvalContext):
        R = self._batch_residuals(self._perturbed(z), ctx)
        J = np.einsum("ij,ij->i", R, R)
        ctx.nfev += 1
        return float(J[0]), (J[1:] - J[0]) / self.p.fd_step

    # ---- bounded Gauss-Newton (Levenberg-Marquardt damped)
    def _solve_gn(self, z0: np.ndarray, z_lb: np.ndarray, z_ub: np.ndarray,
                  ctx: _EvalContext) -> Tuple[np.ndarray, int, int, str, bool]:
        """Bounded Gauss-Newton with a speculative batched line search.

        Each iteration solves the bounded linear least-squares subproblem
            min_dz ||J dz + r||^2 + mu ||dz||^2   s.t.  z_lb <= z + dz <= z_ub
        as a box QP (``box_qp``), so every iterate respects the hard bounds.
        The next evaluation batch holds the full-step point, the 3N finite-
        difference perturbations around it and two shorter steps (alpha = 0.5,
        0.25). If the full step decreases the cost (the usual case) its Jacobian
        is already available, so one vectorised rollout per iteration suffices.

        Real-time-iteration policy: the first iteration always completes; later
        ones only start if the expected iteration time fits the budget.
        """
        p, h, nz = self.p, self.p.fd_step, z0.size
        eye = np.eye(nz)
        alphas = np.array([0.5, 0.25])
        mu_scale = 1.0

        z = z0.copy()
        R = self._batch_residuals(self._perturbed(z), ctx, check_deadline=False)
        r, J = R[0], ((R[1:] - R[0]) / h).T
        cost = float(r @ r)
        it, status, message, deadline_hit = 0, 0, "max iterations", False
        t_iter = self._gn_iter_time
        while it < p.max_iter:
            if it > 0 and time.monotonic() + t_iter > ctx.deadline:
                deadline_hit, message = True, "time budget reached"
                break
            t_start = time.monotonic()
            it += 1
            H = J.T @ J + (p.gn_damping * mu_scale) * eye
            dz = box_qp(H, J.T @ r, z_lb - z, z_ub - z)
            if np.max(np.abs(dz)) < p.xtol:
                status, message = 1, "step below xtol"
                break
            z1 = z + dz
            Zb = np.vstack([self._perturbed(z1), z[None] + alphas[:, None] * dz[None]])
            Rb = self._batch_residuals(Zb, ctx, check_deadline=False)
            c_full = float(Rb[0] @ Rb[0])
            c_alpha = np.einsum("ij,ij->i", Rb[nz + 1:], Rb[nz + 1:])
            if c_full < cost:
                z, r, J = z1, Rb[0], ((Rb[1:nz + 1] - Rb[0]) / h).T
                new_cost = c_full
                mu_scale = max(mu_scale / 3.0, 1.0)
            elif np.min(c_alpha) < cost:
                z = z + alphas[int(np.argmin(c_alpha))] * dz
                R = self._batch_residuals(self._perturbed(z), ctx, check_deadline=False)
                r, J = R[0], ((R[1:] - R[0]) / h).T
                new_cost = float(r @ r)
                mu_scale *= 3.0
            else:
                mu_scale *= 10.0                 # no descent: damp harder and retry
                if mu_scale > 1e4:
                    status, message = 2, "no descent direction"
                    break
                continue
            rel = (cost - new_cost) / max(cost, 1e-12)
            cost = new_cost
            self._gn_iter_time += 0.3 * ((time.monotonic() - t_start) - self._gn_iter_time)
            t_iter = self._gn_iter_time
            if rel < p.ftol:
                status, message = 0, "ftol reached"
                break
        return z, it, status, message, deadline_hit

    # -------------------------------------------------------------- solve
    def solve(self, t0: float, x0: np.ndarray, ref: FormationReference,
              u_last: np.ndarray, wind: np.ndarray, bias: np.ndarray,
              deadline: float) -> NMPCSolution:
        t_start = time.monotonic()
        N, nu = self.N, self.NU
        u_last = np.clip(np.asarray(u_last, dtype=float), self.u_lb, self.u_ub)

        # Bounds; the first step is also limited by the rate over one control period.
        lb = np.tile(self.u_lb, (N, 1))
        ub = np.tile(self.u_ub, (N, 1))
        step0 = self.rate_max * self.Tc
        lb[0] = np.maximum(self.u_lb, u_last - step0)
        ub[0] = np.minimum(self.u_ub, u_last + step0)
        z_lb, z_ub = self.to_z(lb).ravel(), self.to_z(ub).ravel()

        U0 = self._project(self.warm_start(t0, ref.u_ff), lb, ub)
        z0 = self.to_z(U0).ravel()
        ctx = _EvalContext(x0=np.asarray(x0, dtype=float), ref=ref,
                           wind=np.asarray(wind, dtype=float), bias=np.asarray(bias, dtype=float),
                           z_prev=self.to_z(u_last), z_ff=self.to_z(ref.u_ff), deadline=deadline)

        status, message, nit, deadline_hit = -1, "", 0, False
        try:
            if self.p.solver == "gn":
                z_gn, nit, status, message, deadline_hit = self._solve_gn(z0, z_lb, z_ub, ctx)
                ctx.best_z = z_gn
            else:
                A, b = self._A, self._b
                cons = [{"type": "ineq", "fun": lambda z: b - A @ z, "jac": lambda z: -A}]
                res = minimize(self._cost_and_grad, z0, args=(ctx,), jac=True, method="SLSQP",
                               bounds=Bounds(z_lb, z_ub), constraints=cons,
                               options={"maxiter": self.p.max_iter, "ftol": self.p.ftol})
                status, message, nit = int(res.status), str(res.message), int(res.nit)
        except _DeadlineReached:
            deadline_hit, message = True, "time budget reached"
        except (ValueError, FloatingPointError, np.linalg.LinAlgError) as ex:
            message = f"solver error: {ex}"

        if ctx.best_z is None or not np.all(np.isfinite(ctx.best_z)):
            z_sol, success = z0, False
        else:
            z_sol, success = ctx.best_z, True
        U = self._project(self.to_u(z_sol.reshape(N, nu)), lb, ub)

        tr = self._rollout(ctx.x0, U[None], ctx.wind, ctx.bias)
        R = self._residuals(tr, self.to_z(U)[None], ctx)[0]
        traj = {k: v[0] for k, v in tr.items()}

        self._plan_U, self._plan_t0 = U.copy(), t0
        return NMPCSolution(U=U, traj=traj, cost=float(R @ R), status=status, message=message,
                            nit=nit, nfev=ctx.n_batches, solve_time=time.monotonic() - t_start,
                            deadline_hit=deadline_hit, success=success)

# =============================================================================
# Estimation: wind, leader tracking, disturbance observer
# =============================================================================

@dataclass
class VehicleSnapshot:
    """Thread-safe copy of the latest telemetry of one vehicle (times: monotonic)."""
    t_pos: Optional[float] = None
    pos: Optional[np.ndarray] = None        # NED [m] (already shifted to the follower frame)
    vel: Optional[np.ndarray] = None        # NED [m/s]
    t_att: Optional[float] = None
    roll: float = 0.0
    pitch: float = 0.0
    yaw: float = 0.0
    p: float = 0.0
    q: float = 0.0
    r: float = 0.0
    t_vfr: Optional[float] = None
    airspeed: Optional[float] = None        # VFR_HUD airspeed [m/s]
    throttle: Optional[float] = None        # VFR_HUD throttle [0..1]
    t_wind: Optional[float] = None
    wind: Optional[np.ndarray] = None       # NED air-mass velocity [m/s]
    t_hb: Optional[float] = None
    mode: Optional[int] = None
    armed: bool = False

    def age(self, now: float, field_time: Optional[float]) -> float:
        return math.inf if field_time is None else now - field_time


class WindEstimator:
    """Wind priority: follower WIND msg > leader WIND msg > crude estimate from
    VFR_HUD airspeed and heading (zero sideslip) > zero. Low-pass filtered."""

    def __init__(self, tau: float = 3.0, max_age: float = 3.0):
        self.tau, self.max_age = tau, max_age
        self.w = np.zeros(3)
        self._t: Optional[float] = None
        self.source = "none"

    def update(self, now: float, leader: Optional[VehicleSnapshot],
               follower: VehicleSnapshot) -> np.ndarray:
        raw, src = None, "none"
        for name, snap in (("follower", follower), ("leader", leader)):
            if snap is not None and snap.wind is not None and snap.age(now, snap.t_wind) < self.max_age:
                raw, src = snap.wind, f"WIND({name})"
                break
        if raw is None and follower.airspeed is not None and follower.vel is not None \
                and follower.age(now, follower.t_vfr) < self.max_age and follower.airspeed > 5.0:
            va = follower.airspeed
            raw = np.array([follower.vel[0] - va * math.cos(follower.yaw) * math.cos(follower.pitch),
                            follower.vel[1] - va * math.sin(follower.yaw) * math.cos(follower.pitch),
                            0.0])
            src = "airspeed/heading"
        if raw is None:
            raw = np.zeros(3)
        if self._t is None:
            self.w = np.array(raw, dtype=float)
        else:
            self.w += _ema_alpha(now - self._t, self.tau) * (raw - self.w)
        self._t, self.source = now, src
        return self.w.copy()


class LeaderTracker:
    """Builds the LeaderEstimate (heading, horizontal airspeed, filtered turn rate
    from gyro-derived Euler yaw rate, filtered vertical speed)."""

    def __init__(self, tau_turn: float = 0.4, tau_vd: float = 0.3, max_bank_deg: float = 60.0):
        self.tau_turn, self.tau_vd = tau_turn, tau_vd
        self.max_bank = math.radians(max_bank_deg)
        self.omega = 0.0
        self.vd = 0.0
        self._t_att: Optional[float] = None
        self._t_pos: Optional[float] = None

    def reset(self):
        self._t_att = self._t_pos = None

    def update(self, snap: VehicleSnapshot, wind: np.ndarray) -> LeaderEstimate:
        v_air = snap.vel - wind
        Vh = math.hypot(v_air[0], v_air[1])
        heading = math.atan2(v_air[1], v_air[0]) if Vh > 3.0 else snap.yaw
        psi_dot = euler_yaw_rate(snap.roll, snap.pitch, snap.q, snap.r)
        lim = G0 * math.tan(self.max_bank) / max(Vh, 5.0)
        psi_dot = float(np.clip(psi_dot, -lim, lim))
        if self._t_att is None:
            self.omega = psi_dot
        elif snap.t_att != self._t_att:
            self.omega += _ema_alpha(snap.t_att - self._t_att, self.tau_turn) * (psi_dot - self.omega)
        self._t_att = snap.t_att
        if self._t_pos is None:
            self.vd = float(snap.vel[2])
        elif snap.t_pos != self._t_pos:
            self.vd += _ema_alpha(snap.t_pos - self._t_pos, self.tau_vd) * (snap.vel[2] - self.vd)
        self._t_pos = snap.t_pos
        return LeaderEstimate(t=snap.t_pos, pos=snap.pos.copy(), vel=snap.vel.copy(),
                              heading=heading, airspeed_h=Vh, turn_rate=self.omega, vd=self.vd)


class CommandHistory:
    """Commands sent to the autopilot, used to predict through the input dead time.

    The plant input at time tau is the command that was sent at tau - delay.
    """

    def __init__(self, maxlen: int = 400):
        self._t: deque = deque(maxlen=maxlen)
        self._u: deque = deque(maxlen=maxlen)

    def clear(self):
        self._t.clear()
        self._u.clear()

    def push(self, t: float, u: Sequence[float]):
        if self._t and t < self._t[-1]:
            t = self._t[-1]
        self._t.append(float(t))
        self._u.append(np.array(u, dtype=float))

    def _at(self, t: float) -> np.ndarray:
        i = int(np.searchsorted(np.asarray(self._t), t, side="right")) - 1
        return self._u[max(i, 0)]

    def segments(self, t_from: float, t_to: float, delay: float) -> List[Tuple[float, np.ndarray]]:
        """Piecewise-constant plant input over [t_from, t_to] as (duration, u) pairs."""
        if not self._t or t_to <= t_from:
            return []
        brk = [t + delay for t in self._t if t_from < t + delay < t_to]
        edges = [t_from] + brk + [t_to]
        return [(b - a, self._at(a - delay)) for a, b in zip(edges[:-1], edges[1:]) if b - a > 1e-6]


def predict_through_history(model: FixedWingPointMassModel, x: np.ndarray, t_from: float,
                            t_to: float, history: CommandHistory, delay: float,
                            wind: np.ndarray, bias: np.ndarray, max_step: float = 0.1) -> np.ndarray:
    """Propagate x from t_from to t_to with the delayed command history as input."""
    for h, u in history.segments(t_from, t_to, delay):
        n = max(1, int(math.ceil(h / max_step)))
        for _ in range(n):
            x = model.propagate(x, u, h / n, wind, bias)
    return x


class DisturbanceObserver:
    """Online estimates b = [b_Vdot, b_gamma, b_chidot, k_turn, b_theta] that make
    the prediction model offset-free under parameter error.

    * b_gamma : measured flight-path angle minus the model's quasi-steady gamma
                at the measured (V, phi, theta). CL0 / CL_alpha / mass errors.
    * k_turn  : turn efficiency, regressed as psi_dot / (g tan(phi)/V) in steady
                banked flight (|phi| > 10 deg, small roll rate). Multiplicative,
                because an uncoordinated turn loses a fraction of the turn rate,
                not a constant amount.
    * b_chidot: residual heading drift in wings-level flight.
    * b_Vdot  : integrator on the one-step-ahead airspeed prediction error
                (CD0, thrust, lapse errors).
    * b_theta : integrator on the one-step-ahead pitch prediction error, i.e.
                the steady pitch-tracking offset of ArduPlane's pitch loop.
    One-step predictions run through the delayed command history.
    """

    def __init__(self, p: ObserverParams, model: FixedWingPointMassModel):
        self.p = p
        self.model = model
        self.bias = model.default_bias()
        self._t: Optional[float] = None

    def reset(self):
        self.bias = self.model.default_bias()
        self._t = None

    def update(self, snap: VehicleSnapshot, x_meas: np.ndarray, wind: np.ndarray,
               prev: Optional[Tuple[float, np.ndarray]], history: CommandHistory, delay: float):
        if not self.p.enabled:
            return
        t = snap.t_pos
        if self._t is not None and t - self._t < 1e-3:
            return  # no new measurement
        dt = 0.1 if self._t is None else t - self._t
        self._t = t
        m, b, p = self.model, self.bias, self.p
        V, phi, th = x_meas[m.IV], x_meas[m.IPHI], x_meas[m.ITH]
        g_tan = m.p.g * math.tan(phi) / max(V, m.v_floor)

        # flight-path angle bias
        v_air = snap.vel - wind
        gamma_meas = math.atan2(-v_air[2], math.hypot(v_air[0], v_air[1]))
        gamma_mod = float(m.aero(V, phi, th)[0])
        gmax = math.radians(p.max_gamma_bias_deg)
        b[1] += _ema_alpha(dt, p.tau_gamma) * ((gamma_meas - gamma_mod) - b[1])
        b[1] = float(np.clip(b[1], -gmax, gmax))

        # turn efficiency (banked, steady) or wings-level heading drift; only in
        # quasi-steady flight, since pitch manoeuvres at bank leak q sin(phi)
        # into the Euler yaw rate
        psi_dot = euler_yaw_rate(snap.roll, snap.pitch, snap.q, snap.r)
        phi_dot = snap.p + psi_dot * math.sin(snap.pitch)
        theta_dot = snap.q * math.cos(snap.roll) - snap.r * math.sin(snap.roll)
        steady = abs(phi_dot) < math.radians(8.0) and abs(theta_dot) < math.radians(4.0)
        a_turn = _ema_alpha(dt, p.tau_turn)
        if not steady:
            pass
        elif abs(phi) > math.radians(10.0):
            ratio = (psi_dot - b[2]) / g_tan
            lo, hi = p.turn_efficiency_range
            b[3] += a_turn * (float(np.clip(ratio, lo, hi)) - b[3])
        elif abs(phi) < math.radians(5.0):
            cmax = math.radians(p.max_turn_bias_dps)
            b[2] += a_turn * ((psi_dot - b[3] * g_tan) - b[2])
            b[2] = float(np.clip(b[2], -cmax, cmax))

        # one-step-ahead prediction through the command history
        if prev is not None:
            t_prev, x_prev = prev
            h = t - t_prev
            if 1e-3 < h < 0.5:
                x_pred = predict_through_history(m, x_prev, t_prev, t, history, delay, wind, b)
                b[0] += p.speed_gain * (V - x_pred[m.IV]) / h
                b[0] = float(np.clip(b[0], -p.max_accel_bias, p.max_accel_bias))
                tmax = math.radians(p.max_pitch_offset_deg)
                b[4] += p.pitch_gain * (th - x_pred[m.ITH])
                b[4] = float(np.clip(b[4], -tmax, tmax))


# =============================================================================
# Safety supervisor
# =============================================================================

@dataclass
class ControlOutput:
    roll: float = 0.0
    pitch: float = 0.0
    yaw: float = 0.0
    throttle: float = 0.0
    send: bool = False
    source: str = "none"               # nmpc | fallback | breakaway | lost_leader | passive | ...
    request_mode: Optional[str] = None
    diag: Dict[str, float] = field(default_factory=dict)


def ellipsoid_metric(rel: np.ndarray, axes: Sequence[float]) -> float:
    """s = sum((rel_i/axis_i)^2); s < 1 means inside the ellipsoid."""
    return float(np.sum((np.asarray(rel) / np.asarray(axes)) ** 2))


def ellipsoid_metric_cpa(rel: np.ndarray, rel_vel: np.ndarray, axes: Sequence[float],
                         horizon: float) -> Tuple[float, float]:
    """Minimum ellipsoid metric along rel + rel_vel * tau, tau in [0, horizon]
    (closest point of approach in the ellipsoid metric). Returns (s_min, tau)."""
    a = np.asarray(axes, dtype=float)
    r, v = np.asarray(rel) / a, np.asarray(rel_vel) / a
    vv = float(v @ v)
    tau = 0.0 if vv < 1e-12 else float(np.clip(-(r @ v) / vv, 0.0, horizon))
    p = r + v * tau
    return float(p @ p), tau


class SafetySupervisor:
    """Independent of the optimiser. Overrides the NMPC when

    * the relative position is inside the breakaway ellipsoid now, or its
      closest approach along the current relative velocity within the
      look-ahead time is -> latched breakaway: roll away from the leader,
      pitch to open vertical separation, throttle to open longitudinal
      separation; rejoin only after hold time and once well clear;
    * leader telemetry is lost beyond the coast time -> wings-level hold,
      then request the failsafe flight mode and stop commanding.
    """

    NOMINAL, BREAKAWAY, LOST_LEADER = "nominal", "breakaway", "lost_leader"

    def __init__(self, cfg: ControllerConfig):
        self.cfg = cfg
        self.state = self.NOMINAL
        self._t_enter = 0.0
        self._signs = (1.0, 1.0, 1.0)

    def evaluate(self, now: float, rel: Optional[np.ndarray], rel_vel: Optional[np.ndarray],
                 leader_age: float, trim: Tuple[float, float]) -> Optional[ControlOutput]:
        cfg, f = self.cfg, self.cfg.formation
        theta_trim, delta_trim = trim
        max_age = cfg.link.state_timeout_s + cfg.link.leader_coast_s

        if leader_age > max_age or rel is None:
            if self.state != self.LOST_LEADER:
                LOG.error("Leader telemetry lost (%.1f s): wings-level hold", leader_age)
                self.state, self._t_enter = self.LOST_LEADER, now
            if now - self._t_enter < cfg.lost_link_hold_s:
                return ControlOutput(roll=0.0, pitch=theta_trim + math.radians(2.0),
                                     throttle=delta_trim, send=True, source="lost_leader")
            return ControlOutput(send=False, source="lost_leader", request_mode=cfg.failsafe_mode)
        if self.state == self.LOST_LEADER:
            LOG.warning("Leader telemetry back")
            self.state = self.NOMINAL

        s_now = ellipsoid_metric(rel, f.breakaway_axes)
        # closest approach within the look-ahead (a single end-point test can
        # step straight over the ellipsoid during a fast crossing)
        s_ahead, tau_cpa = ellipsoid_metric_cpa(rel, rel_vel, f.breakaway_axes, f.breakaway_lookahead_s)
        if self.state == self.BREAKAWAY:
            clear = ellipsoid_metric(rel, np.asarray(f.separation_axes) * f.rejoin_scale) > 1.0
            if f.auto_rejoin and clear and now - self._t_enter > f.breakaway_hold_s:
                LOG.warning("Breakaway complete, rejoining formation")
                self.state = self.NOMINAL
                return None
        elif min(s_now, s_ahead) < 1.0:
            # latch escape directions at entry so the manoeuvre does not flip
            self._signs = (1.0 if rel[1] >= 0.0 else -1.0,      # roll away laterally
                           1.0 if rel[2] > 0.0 else -1.0,        # below -> push down
                           1.0 if rel[0] > 0.0 else -1.0)        # ahead -> speed up
            self.state, self._t_enter = self.BREAKAWAY, now
            LOG.error("BREAKAWAY: rel=(%.1f, %.1f, %.1f) m, s_now=%.2f, s_cpa=%.2f in %.2f s",
                      rel[0], rel[1], rel[2], s_now, s_ahead, tau_cpa)

        if self.state == self.BREAKAWAY:
            sy, sz, sx = self._signs
            lim = cfg.limits
            return ControlOutput(
                roll=sy * min(math.radians(f.breakaway_roll_deg), lim.roll_max),
                pitch=float(np.clip(theta_trim - sz * math.radians(f.breakaway_pitch_deg),
                                    lim.pitch_min, lim.pitch_max)),
                throttle=float(np.clip(delta_trim + (0.2 if sx > 0 else -0.15),
                                       lim.throttle_min, lim.throttle_max)),
                send=True, source="breakaway")
        return None


# =============================================================================
# Controller core (I/O free: used by the MAVLink runner and the self-test)
# =============================================================================

class NMPCFormationController:
    """Hierarchical formation controller.

    Call ``step(now, leader_snapshot, follower_snapshot, active)`` at the control
    rate. Snapshots must already be in the follower's local NED frame. The
    returned ``ControlOutput`` holds roll/pitch/yaw [rad] and throttle [0..1]
    for SET_ATTITUDE_TARGET, plus diagnostics.
    """

    def __init__(self, cfg: ControllerConfig):
        cfg.validate()
        self.cfg = cfg
        self.model = FixedWingPointMassModel(cfg.aircraft)
        self.refgen = ReferenceGenerator(self.model, cfg.limits, cfg.formation)
        self.nmpc = FormationNMPC(self.model, cfg.limits, cfg.weights, cfg.mpc,
                                  cfg.formation, cfg.control_period)
        self.observer = DisturbanceObserver(cfg.observer, self.model)
        self.leader_tracker = LeaderTracker()
        self.wind_est = WindEstimator()
        self.safety = SafetySupervisor(cfg)
        self.history = CommandHistory()
        self.u_last: Optional[np.ndarray] = None
        self.t_last_cmd: Optional[float] = None
        self.delta_est: Optional[float] = None
        self._t_delta: Optional[float] = None
        self.solve_time_ema = 0.03
        self._prev_meas: Optional[Tuple[float, np.ndarray]] = None
        self._was_active = False

    @property
    def total_delay(self) -> float:
        """Send -> plant dead time: radio/link latency plus autopilot inner-loop delay."""
        return self.cfg.link.link_latency_s + self.cfg.aircraft.input_delay

    # -------------------------------------------------------------- helpers
    def _measured_state(self, snap: VehicleSnapshot, wind: np.ndarray, delta: float) -> np.ndarray:
        v_air = snap.vel - wind
        V = float(np.linalg.norm(v_air))
        Vh = math.hypot(v_air[0], v_air[1])
        chi = math.atan2(v_air[1], v_air[0]) if Vh > 3.0 else snap.yaw
        return np.array([snap.pos[0], snap.pos[1], snap.pos[2], V, chi,
                         snap.roll, snap.pitch, delta])

    def _throttle_state(self, now: float, snap: VehicleSnapshot) -> float:
        """Throttle state: VFR_HUD throttle if fresh, else the internal lag model."""
        if self.delta_est is not None and self.u_last is not None and self._t_delta is not None:
            h = max(now - self._t_delta, 0.0)
            self.delta_est = self.u_last[2] + (self.delta_est - self.u_last[2]) * \
                math.exp(-h / self.cfg.aircraft.tau_throttle)
        self._t_delta = now
        if snap.throttle is not None and snap.age(now, snap.t_vfr) < 0.5:
            self.delta_est = float(np.clip(snap.throttle, 0.0, 1.0))
        if self.delta_est is None:
            self.delta_est = 0.5
        return self.delta_est

    def _commit(self, t_sent: float, u: np.ndarray):
        self.u_last = np.array(u, dtype=float)
        self.t_last_cmd = t_sent
        self.history.push(t_sent, self.u_last)

    @staticmethod
    def _to_leader_frame(vec: np.ndarray, psi: float) -> np.ndarray:
        c, s = math.cos(psi), math.sin(psi)
        return np.array([vec[0] * c + vec[1] * s, -vec[0] * s + vec[1] * c, vec[2]])

    # ----------------------------------------------------------------- step
    def step(self, now: float, leader: Optional[VehicleSnapshot],
             follower: VehicleSnapshot, active: bool = True) -> ControlOutput:
        cfg, lim = self.cfg, self.cfg.limits
        tmo = cfg.link.state_timeout_s

        # --- 1. follower must be fresh, otherwise we cannot control at all
        if follower.pos is None or follower.age(now, follower.t_pos) > tmo \
                or follower.age(now, follower.t_att) > tmo:
            self.nmpc.reset()
            self._prev_meas = None
            return ControlOutput(send=False, source="no_follower_data",
                                 request_mode=cfg.failsafe_mode if active else None)

        wind = self.wind_est.update(now, leader, follower)
        delta_meas = self._throttle_state(now, follower)
        x_meas = self._measured_state(follower, wind, delta_meas)
        if not np.all(np.isfinite(x_meas)) or not np.all(np.isfinite(wind)):
            LOG.error("non-finite follower state or wind estimate; not commanding")
            self.wind_est = WindEstimator()
            self.nmpc.reset()
            return ControlOutput(send=False, source="bad_follower_data")

        D = self.total_delay
        if not active or self.u_last is None or not self._was_active:
            # ArduPlane (or the pilot) is flying: treat its actual attitude as the
            # input in flight, so the first NMPC command after engagement is
            # rate-limited from reality and the delay prediction is sensible.
            u_now = x_meas[5:8] - self.model.lag_offset(self.observer.bias)
            if not self._was_active or self.u_last is None:
                self.history.clear()
            self._commit(now - D, u_now)
            self.t_last_cmd = now
            self.nmpc.reset()
        self._was_active = active

        # --- 2. disturbance observer on measured data (not extrapolated)
        self.observer.update(follower, x_meas, wind, self._prev_meas, self.history, D)
        self._prev_meas = (follower.t_pos, x_meas.copy())
        bias = self.observer.bias.copy()

        # --- 3. delay + latency compensation: predict the state to the time the
        #        new command reaches the plant, through the commands in flight
        t_send = now + self.solve_time_ema
        t0 = t_send + D
        x0 = predict_through_history(self.model, x_meas, follower.t_pos, t0,
                                     self.history, D, wind, bias)
        lat = t0 - now

        theta_trim, delta_trim = self.model.trim(max(x_meas[3], lim.airspeed_min), 0.0, 0.0, bias)
        theta_trim, delta_trim = float(theta_trim), float(delta_trim)
        trim = (float(np.clip(theta_trim, lim.pitch_min, lim.pitch_max)),
                float(np.clip(delta_trim, lim.throttle_min, lim.throttle_max)))

        # --- 4. leader estimate and current relative geometry
        leader_age = math.inf if leader is None or leader.pos is None else leader.age(now, leader.t_pos)
        rel = rel_vel = None
        L = None
        if leader_age <= tmo + cfg.link.leader_coast_s:
            L = self.leader_tracker.update(leader, wind)
            lpos_now, lvel_now, psi_now = self.refgen.propagate_leader(L, np.array([now - L.t]), wind)
            fpos_now = follower.pos + follower.vel * (now - follower.t_pos)
            rel = self._to_leader_frame(fpos_now - lpos_now[0], psi_now[0])
            rel_vel = self._to_leader_frame(follower.vel - lvel_now[0], psi_now[0])
        diag = self._base_diag(now, follower, x_meas, bias, rel, wind)
        diag["leader_age"] = float(leader_age)
        if L is not None:
            diag.update({"leader_heading": float(wrap_pi(L.heading)), "leader_turn_rate": float(L.turn_rate),
                         "leader_V": float(L.airspeed_h)})

        # --- 5. safety supervisor (runs even when passive, to log hazards)
        override = self.safety.evaluate(now, rel, rel_vel, leader_age, trim)
        if override is not None:
            override.yaw = float(wrap_pi(x0[4]))
            override.diag = diag
            if override.send and active:
                self._commit(now, [override.roll, override.pitch, override.throttle])
                self.nmpc.reset()
            if not active:
                override.send, override.request_mode = False, None
            return override
        if not active:
            return ControlOutput(send=False, source="passive", diag=diag)

        # --- 6. reference over the horizon
        ref = self.refgen.generate(L, t0, cfg.mpc.horizon, cfg.mpc.dt, wind, bias)

        # --- 7. NMPC
        deadline = time.monotonic() + cfg.mpc.time_budget_frac * cfg.control_period
        sol = self.nmpc.solve(t0, x0, ref, self.u_last, wind, bias, deadline)
        self.solve_time_ema += 0.2 * (sol.solve_time - self.solve_time_ema)
        if sol.success:
            u0, source = sol.U[0], "nmpc"
        else:
            u0, source = ref.u_ff[0], "fallback"
            LOG.warning("NMPC failed (%s); using feedforward", sol.message)

        # --- 8. defence in depth: box + rate limit against the last command
        h = cfg.control_period if self.t_last_cmd is None else max(now - self.t_last_cmd, cfg.control_period)
        u0 = np.clip(u0, self.u_last - lim.rate_max * h, self.u_last + lim.rate_max * h)
        u0 = np.clip(u0, self.nmpc.u_lb, self.nmpc.u_ub)
        self._commit(now + sol.solve_time, u0)

        tr = sol.traj
        diag.update({
            "roll_cmd": float(u0[0]), "pitch_cmd": float(u0[1]), "throttle_cmd": float(u0[2]),
            "V_plan": float(tr["V"][0]), "V_ref": float(ref.V_ref[0]),
            "climb_plan": float(-tr["vd"][0]), "climb_ref": float(-ref.slot_vel[0, 2]),
            "chi": float(wrap_pi(x_meas[4])), "chi_ref": float(wrap_pi(ref.chi_ref[0])),
            "chi0": float(wrap_pi(x0[4])),
            "cost": sol.cost, "nit": float(sol.nit), "nfev": float(sol.nfev),
            "solve_ms": 1e3 * sol.solve_time, "solve_frac": sol.solve_time / cfg.control_period,
            "deadline_hit": float(sol.deadline_hit), "status": float(sol.status),
            "latency_ms": 1e3 * lat,
        })
        if L is not None:
            _, psi_s, slot_now, _ = self.refgen.slot(L, np.array([now - L.t]), wind)
            fpos_now = follower.pos + follower.vel * (now - follower.t_pos)
            err = self._to_leader_frame(fpos_now - slot_now[0], psi_s[0])
            diag.update({"err_x": float(err[0]), "err_y": float(err[1]), "err_z": float(err[2])})
        return ControlOutput(roll=float(u0[0]), pitch=float(u0[1]), yaw=float(wrap_pi(x0[4])),
                             throttle=float(u0[2]), send=True, source=source, diag=diag)

    def _base_diag(self, now, follower, x_meas, bias, rel, wind) -> Dict[str, float]:
        d = {
            "t": now, "V": float(x_meas[3]), "roll": float(follower.roll),
            "pitch": float(follower.pitch), "climb": float(-follower.vel[2]),
            "throttle": float(x_meas[7]),
            "b_vdot": float(bias[0]), "b_gamma": float(bias[1]), "b_chidot": float(bias[2]),
            "k_turn": float(bias[3]), "b_theta": float(bias[4]),
            "wind_n": float(wind[0]), "wind_e": float(wind[1]),
        }
        if follower.airspeed is not None:
            d["V_sensor"] = float(follower.airspeed)
        if rel is not None:
            d.update({"rel_x": float(rel[0]), "rel_y": float(rel[1]), "rel_z": float(rel[2]),
                      "sep_metric": ellipsoid_metric(rel, self.cfg.formation.separation_axes)})
        return d


# =============================================================================
# MAVLink layer
# =============================================================================

PLANE_GUIDED = 15
# ArduPlane partial-attitude convention (see module docstring): use roll and pitch
# from q plus thrust, never force yaw (it would be applied as a raw rudder command).
ATTITUDE_TYPE_MASK = 0b10000100


def _plane_mode_numbers() -> Dict[str, int]:
    if mavutil is None:
        return {"GUIDED": PLANE_GUIDED, "LOITER": 12, "RTL": 11, "CIRCLE": 1}
    mapping = mavutil.mode_mapping_bynumber(mavutil.mavlink.MAV_TYPE_FIXED_WING)
    return {name: num for num, name in mapping.items()}


class MavlinkLink:
    """One pymavlink connection with a reader thread that dispatches messages to
    subscribed vehicles. Several vehicles may share one link (e.g. a multipoint
    radio on one serial port) if their sysids are given explicitly. Note: with a
    plain ``udpin`` endpoint pymavlink replies to the last sender, so for two
    SITL instances use one UDP port per vehicle."""

    def __init__(self, url: str, source_system: int, baud: int = 57600):
        if mavutil is None:
            raise RuntimeError("pymavlink is required for live operation: pip install pymavlink")
        self.url = url
        self.conn = mavutil.mavlink_connection(
            url, baud=baud, source_system=source_system,
            source_component=mavutil.mavlink.MAV_COMP_ID_ONBOARD_COMPUTER, autoreconnect=True)
        self._send_lock = threading.Lock()
        self._subs: List["MavlinkVehicle"] = []
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, name=f"mavlink[{url}]", daemon=True)

    def subscribe(self, vehicle: "MavlinkVehicle"):
        self._subs.append(vehicle)

    def start(self):
        if not self._thread.is_alive():
            self._thread.start()

    def _run(self):
        while not self._stop.is_set():
            try:
                msg = self.conn.recv_match(blocking=True, timeout=0.2)
            except Exception as ex:  # noqa: BLE001  (link errors must not kill the thread)
                LOG.warning("%s: receive error: %s", self.url, ex)
                time.sleep(0.2)
                continue
            if msg is None:
                continue
            mtype = msg.get_type()
            if mtype == "BAD_DATA":
                continue
            t = time.monotonic()
            for v in self._subs:
                try:
                    v.handle(msg, mtype, t)
                except Exception:  # noqa: BLE001
                    LOG.exception("%s: error handling %s", v.name, mtype)

    def send(self, fn: str, *args, **kwargs):
        with self._send_lock:
            try:
                getattr(self.conn.mav, fn)(*args, **kwargs)
            except Exception as ex:  # noqa: BLE001
                LOG.warning("%s: send %s failed: %s", self.url, fn, ex)

    def close(self):
        self._stop.set()
        self._thread.join(timeout=1.0)
        try:
            self.conn.close()
        except Exception:  # noqa: BLE001
            pass


class MavlinkVehicle:
    """Telemetry state and command interface for one ArduPlane."""

    def __init__(self, name: str, link: MavlinkLink, sysid: Optional[int], compid: int = 1):
        self.name = name
        self.link = link
        self.sysid = sysid
        self.compid = compid
        self._lock = threading.Lock()
        self._snap = VehicleSnapshot()
        self._origin: Optional[Tuple[float, float, float]] = None
        self._counts: Counter = Counter()
        self._acks: Dict[int, Tuple[int, float]] = {}
        self._boot_t0 = time.monotonic()
        link.subscribe(self)

    # ------------------------------------------------------------- receive
    def handle(self, msg, mtype: str, t: float):
        src, comp = msg.get_srcSystem(), msg.get_srcComponent()
        ml = mavutil.mavlink
        if mtype == "HEARTBEAT":
            if msg.type in (ml.MAV_TYPE_GCS, ml.MAV_TYPE_ONBOARD_CONTROLLER) \
                    or msg.autopilot == ml.MAV_AUTOPILOT_INVALID:
                return
            if self.sysid is None:
                self.sysid = src
                LOG.info("%s: locked onto sysid %d", self.name, src)
            if src != self.sysid or comp != self.compid:
                return
            with self._lock:
                self._snap.t_hb = t
                self._snap.mode = int(msg.custom_mode)
                self._snap.armed = bool(msg.base_mode & ml.MAV_MODE_FLAG_SAFETY_ARMED)
            return
        if self.sysid is None or src != self.sysid:
            return
        if mtype == "STATUSTEXT":
            LOG.info("[%s] %s", self.name, msg.text)
            return
        if mtype == "COMMAND_ACK":
            self._acks[int(msg.command)] = (int(msg.result), t)
            return
        if comp != self.compid:
            return
        with self._lock:
            s = self._snap
            if mtype == "LOCAL_POSITION_NED":
                s.t_pos = t
                s.pos = np.array([msg.x, msg.y, msg.z], dtype=float)
                s.vel = np.array([msg.vx, msg.vy, msg.vz], dtype=float)
            elif mtype == "ATTITUDE":
                s.t_att = t
                s.roll, s.pitch, s.yaw = float(msg.roll), float(msg.pitch), float(msg.yaw)
                s.p, s.q, s.r = float(msg.rollspeed), float(msg.pitchspeed), float(msg.yawspeed)
            elif mtype == "VFR_HUD":
                s.t_vfr = t
                s.airspeed = float(msg.airspeed)
                s.throttle = float(msg.throttle) / 100.0
            elif mtype == "WIND":
                # ArduPilot: direction the wind comes FROM [deg], horizontal speed, vertical
                ang = math.radians(msg.direction)
                s.t_wind = t
                s.wind = np.array([-msg.speed * math.cos(ang), -msg.speed * math.sin(ang),
                                   float(msg.speed_z)])
            elif mtype == "GPS_GLOBAL_ORIGIN":
                self._origin = (msg.latitude * 1e-7, msg.longitude * 1e-7, msg.altitude * 1e-3)
            self._counts[mtype] += 1

    # ------------------------------------------------------------- queries
    def snapshot(self) -> VehicleSnapshot:
        with self._lock:
            s = self._snap
            return VehicleSnapshot(
                t_pos=s.t_pos, pos=None if s.pos is None else s.pos.copy(),
                vel=None if s.vel is None else s.vel.copy(), t_att=s.t_att,
                roll=s.roll, pitch=s.pitch, yaw=s.yaw, p=s.p, q=s.q, r=s.r,
                t_vfr=s.t_vfr, airspeed=s.airspeed, throttle=s.throttle,
                t_wind=s.t_wind, wind=None if s.wind is None else s.wind.copy(),
                t_hb=s.t_hb, mode=s.mode, armed=s.armed)

    @property
    def origin(self) -> Optional[Tuple[float, float, float]]:
        return self._origin

    def message_counts(self) -> Dict[str, int]:
        with self._lock:
            return dict(self._counts)

    # ----------------------------------------------------------------- send
    def _boot_ms(self) -> int:
        return int((time.monotonic() - self._boot_t0) * 1000) & 0xFFFFFFFF

    def _command_long(self, command: int, *params: float):
        p = list(params) + [0.0] * (7 - len(params))
        self.link.send("command_long_send", self.sysid, self.compid, command, 0, *p)

    def request_streams(self, hz: float):
        ml = mavutil.mavlink
        for msg_id, rate in ((ml.MAVLINK_MSG_ID_LOCAL_POSITION_NED, hz),
                             (ml.MAVLINK_MSG_ID_ATTITUDE, hz),
                             (ml.MAVLINK_MSG_ID_VFR_HUD, min(hz, 10.0)),
                             (ml.MAVLINK_MSG_ID_WIND, 4.0)):
            self._command_long(ml.MAV_CMD_SET_MESSAGE_INTERVAL, msg_id, 1e6 / rate)

    def request_origin(self):
        ml = mavutil.mavlink
        self._command_long(ml.MAV_CMD_REQUEST_MESSAGE, ml.MAVLINK_MSG_ID_GPS_GLOBAL_ORIGIN)

    def set_mode(self, mode_name: str) -> bool:
        num = _plane_mode_numbers().get(mode_name.upper())
        if num is None:
            LOG.error("%s: unknown ArduPlane mode %s", self.name, mode_name)
            return False
        ml = mavutil.mavlink
        LOG.warning("%s: requesting mode %s", self.name, mode_name)
        self._command_long(ml.MAV_CMD_DO_SET_MODE, ml.MAV_MODE_FLAG_CUSTOM_MODE_ENABLED, num)
        return True

    def send_attitude_target(self, roll: float, pitch: float, yaw: float, thrust: float):
        q = euler_to_quaternion(roll, pitch, yaw)
        self.link.send("set_attitude_target_send", self._boot_ms(), self.sysid, self.compid,
                       ATTITUDE_TYPE_MASK, q, 0.0, 0.0, 0.0, float(np.clip(thrust, 0.0, 1.0)))

    def send_heartbeat(self):
        ml = mavutil.mavlink
        self.link.send("heartbeat_send", ml.MAV_TYPE_ONBOARD_CONTROLLER, ml.MAV_AUTOPILOT_INVALID,
                       0, 0, ml.MAV_STATE_ACTIVE)


# =============================================================================
# Optional real-time visualiser (separate process) and CSV logger
# =============================================================================

_VIZ_PANELS = (
    ("Airspeed [m/s]", (("V", "actual"), ("V_plan", "NMPC plan k=1"), ("V_ref", "slot reference")), 1.0),
    ("Roll [deg]", (("roll", "actual"), ("roll_cmd", "commanded")), 180.0 / math.pi),
    ("Pitch [deg]", (("pitch", "actual"), ("pitch_cmd", "commanded")), 180.0 / math.pi),
    ("Climb rate [m/s]", (("climb", "actual"), ("climb_plan", "NMPC plan k=1"), ("climb_ref", "slot reference")), 1.0),
    ("Slot error [m]", (("err_x", "along"), ("err_y", "cross"), ("err_z", "vertical (down +)")), 1.0),
    ("Throttle / solve load", (("throttle_cmd", "throttle cmd"), ("solve_frac", "solve time / period")), 1.0),
)


def _visualizer_main(q, window_s: float):
    """Runs in its own process: plots commanded vs actual signals."""
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(3, 2, figsize=(13, 8.5), sharex=True)
    fig.suptitle("Formation NMPC: commanded vs actual")
    buf: Dict[str, deque] = {}
    lines = []
    for ax, (title, series, scale) in zip(axes.ravel(), _VIZ_PANELS):
        ax.set_title(title, fontsize=10)
        ax.grid(alpha=0.3)
        for key, label in series:
            (ln,) = ax.plot([], [], label=label, lw=1.4)
            lines.append((ax, ln, key, scale))
            buf[key] = deque(maxlen=4000)
        ax.legend(fontsize=8, loc="upper left")
    for ax in axes[-1]:
        ax.set_xlabel("time [s]")
    tbuf: deque = deque(maxlen=4000)
    plt.ion()
    plt.show(block=False)
    t_start = None
    running = True
    while running:
        try:
            while True:
                item = q.get_nowait()
                if item is None:
                    running = False
                    break
                if t_start is None:
                    t_start = item["t"]
                tbuf.append(item["t"] - t_start)
                for key in buf:
                    buf[key].append(item.get(key, math.nan))
        except _queue.Empty:
            pass
        if not plt.fignum_exists(fig.number):
            break
        if tbuf:
            t = np.asarray(tbuf)
            for ax, ln, key, scale in lines:
                ln.set_data(t, np.asarray(buf[key]) * scale)
            for ax in axes.ravel():
                ax.relim()
                ax.autoscale_view()
                ax.set_xlim(max(0.0, t[-1] - window_s), max(window_s, t[-1]))
        plt.pause(0.05)
    plt.close("all")


class Visualizer:
    def __init__(self, window_s: float = 40.0):
        import multiprocessing as mp
        ctx = mp.get_context("spawn")
        self._q = ctx.Queue(maxsize=5000)
        self._p = ctx.Process(target=_visualizer_main, args=(self._q, window_s), daemon=True)
        self._p.start()

    def push(self, diag: Dict[str, float]):
        try:
            self._q.put_nowait(dict(diag))
        except _queue.Full:
            pass

    def close(self):
        try:
            self._q.put_nowait(None)
        except _queue.Full:
            pass
        self._p.join(timeout=2.0)


DIAG_FIELDS = (
    "t", "active", "source", "V", "V_sensor", "V_ref", "V_plan", "roll", "roll_cmd",
    "pitch", "pitch_cmd", "throttle", "throttle_cmd", "climb", "climb_plan", "climb_ref",
    "err_x", "err_y", "err_z", "rel_x", "rel_y", "rel_z", "sep_metric",
    "b_vdot", "b_gamma", "b_chidot", "k_turn", "b_theta", "wind_n", "wind_e",
    "cost", "nit", "nfev", "solve_ms", "solve_frac", "deadline_hit", "status", "latency_ms",
    "chi", "chi_ref", "chi0", "leader_heading", "leader_turn_rate", "leader_V", "leader_age",
)


class CsvLogger:
    """CSV log with a fixed column set (rows may omit fields, e.g. while passive)."""

    def __init__(self, path: str):
        self._f = open(path, "w", newline="")
        self._w = csv.DictWriter(self._f, fieldnames=DIAG_FIELDS, extrasaction="ignore", restval="")
        self._w.writeheader()
        self._t_flush = time.monotonic()

    def write(self, row: Dict[str, Any]):
        self._w.writerow(row)
        if time.monotonic() - self._t_flush > 1.0:
            self._f.flush()
            self._t_flush = time.monotonic()

    def close(self):
        self._f.close()


# =============================================================================
# Live runner: two ArduPlanes over MAVLink
# =============================================================================

class FormationRunner:
    """Connects to leader and follower, aligns frames, engages GUIDED, and runs
    the controller loop at a fixed rate with absolute-time scheduling."""

    def __init__(self, cfg: ControllerConfig):
        self.cfg = cfg
        lk = cfg.link
        if lk.leader_url == lk.follower_url and (lk.leader_sysid is None or lk.follower_sysid is None
                                                 or lk.leader_sysid == lk.follower_sysid):
            raise ValueError("leader and follower share one link: give distinct --leader-sysid/--follower-sysid")
        links: Dict[str, MavlinkLink] = {}
        for url in {lk.leader_url, lk.follower_url}:
            links[url] = MavlinkLink(url, lk.source_system, lk.baud)
        self.links = links
        self.leader = MavlinkVehicle("leader", links[lk.leader_url], lk.leader_sysid, lk.compid)
        self.follower = MavlinkVehicle("follower", links[lk.follower_url], lk.follower_sysid, lk.compid)
        self.ctrl = NMPCFormationController(cfg)
        self.viz = Visualizer() if cfg.visualize else None
        self.csv = CsvLogger(cfg.log_csv) if cfg.log_csv else None
        self._origin_offset: Optional[np.ndarray] = None
        self._engaged_once = False
        self._was_guided = False
        self._last_mode_request: Dict[str, float] = {}
        self._stop = threading.Event()

    # ------------------------------------------------------------ startup
    def _wait_for(self, what: str, cond: Callable[[], bool], timeout: float,
                  tick: Optional[Callable[[], None]] = None) -> bool:
        t_end = time.monotonic() + timeout
        last_tick = 0.0
        while time.monotonic() < t_end and not self._stop.is_set():
            if cond():
                return True
            if tick is not None and time.monotonic() - last_tick > 2.0:
                tick()
                last_tick = time.monotonic()
            time.sleep(0.05)
        LOG.error("Timed out waiting for %s", what)
        return False

    def _request_all(self):
        for v in (self.leader, self.follower):
            if v.sysid is not None:
                v.send_heartbeat()
                v.request_streams(self.cfg.link.stream_hz)
                if v.origin is None:
                    v.request_origin()

    def startup(self) -> bool:
        for link in self.links.values():
            link.start()
        LOG.info("Waiting for heartbeats ...")
        ok = self._wait_for("heartbeats", lambda: self.leader.snapshot().t_hb is not None
                            and self.follower.snapshot().t_hb is not None, 30.0)
        if not ok:
            return False
        LOG.info("leader sysid %s, follower sysid %s", self.leader.sysid, self.follower.sysid)
        self._request_all()
        ok = self._wait_for("ATTITUDE streams", lambda: all(
            s.t_att is not None for s in (self.leader.snapshot(), self.follower.snapshot())),
            15.0, self._request_all)
        # LOCAL_POSITION_NED only exists once the EKF has an origin (GPS lock)
        ok = ok and self._wait_for("LOCAL_POSITION_NED (EKF origin set)", lambda: all(
            s.t_pos is not None for s in (self.leader.snapshot(), self.follower.snapshot())),
            self.cfg.link.origin_timeout_s, self._request_all)
        if not ok:
            return False
        if self.cfg.link.assume_common_origin:
            self._origin_offset = np.zeros(3)
            LOG.warning("Assuming both vehicles share an EKF origin")
        else:
            ok = self._wait_for("GPS_GLOBAL_ORIGIN from both vehicles",
                                lambda: self.leader.origin is not None and self.follower.origin is not None,
                                self.cfg.link.origin_timeout_s, self._request_all)
            if not ok:
                LOG.error("Cannot align local frames (use --assume-common-origin only if you are sure)")
                return False
            self._origin_offset = origin_offset_ned(self.leader.origin, self.follower.origin)
            LOG.info("Leader origin offset in follower frame: %s m", np.round(self._origin_offset, 2))
        return True

    # --------------------------------------------------------------- loop
    def _leader_snapshot(self) -> VehicleSnapshot:
        s = self.leader.snapshot()
        if s.pos is not None and self._origin_offset is not None:
            s.pos = s.pos + self._origin_offset
        return s

    def _request_mode(self, mode: Optional[str], now: float):
        if not mode:
            return
        if now - self._last_mode_request.get(mode, -1e9) > 2.0:
            self.follower.set_mode(mode)
            self._last_mode_request[mode] = now

    def _engagement(self, now: float, f: VehicleSnapshot, l: VehicleSnapshot) -> bool:
        cfg = self.cfg
        hb_ok = f.age(now, f.t_hb) < cfg.link.heartbeat_timeout_s
        guided = hb_ok and f.mode == PLANE_GUIDED and f.armed
        if self._was_guided and not guided:
            LOG.warning("Follower left GUIDED (pilot or failsafe): controller passive, will not re-engage")
        self._was_guided = guided
        if guided:
            self._engaged_once = True
            return True
        flying = f.vel is not None and float(np.linalg.norm(f.vel)) > 0.8 * cfg.limits.airspeed_min
        leader_ok = l.pos is not None and l.age(now, l.t_pos) < cfg.link.state_timeout_s
        if cfg.auto_guided and not self._engaged_once and f.armed and flying and leader_ok:
            self._request_mode("GUIDED", now)
        return False

    def run(self):
        if not self.startup():
            self.shutdown()
            return
        cfg = self.cfg
        period = cfg.control_period
        next_t = time.monotonic()
        t_hb = t_streams = t_stat = 0.0
        n_cycles = n_overruns = n_errors = 0
        LOG.info("Control loop running at %.1f Hz", cfg.control_rate_hz)
        try:
            while not self._stop.is_set():
                now = time.monotonic()
                if now - t_hb > 1.0:
                    self.leader.send_heartbeat()
                    self.follower.send_heartbeat()
                    t_hb = now
                if now - t_streams > 10.0:   # re-assert rates (a GCS may change SRx)
                    self.leader.request_streams(cfg.link.stream_hz)
                    self.follower.request_streams(cfg.link.stream_hz)
                    t_streams = now

                l_snap, f_snap = self._leader_snapshot(), self.follower.snapshot()
                active = self._engagement(now, f_snap, l_snap)
                try:
                    out = self.ctrl.step(now, l_snap, f_snap, active=active)
                    n_errors = 0
                except Exception:  # noqa: BLE001  never let a bug leave a stale command latched
                    LOG.exception("controller step failed; not commanding this cycle")
                    n_errors += 1
                    out = ControlOutput(send=False, source="exception",
                                        request_mode=cfg.failsafe_mode if n_errors >= 5 else None)
                if active and out.send:
                    self.follower.send_attitude_target(out.roll, out.pitch, out.yaw, out.throttle)
                if active and out.request_mode:
                    self._request_mode(out.request_mode, now)

                if out.diag:
                    out.diag["source"] = out.source
                    out.diag["active"] = float(active)
                    if self.viz:
                        self.viz.push(out.diag)
                    if self.csv:
                        self.csv.write(out.diag)
                n_cycles += 1
                if now - t_stat > 5.0:
                    d = out.diag
                    LOG.info("[%s|%s] err=(%+.1f,%+.1f,%+.1f) m  V=%.1f  solve=%.0f ms  wind=%s",
                             "ACTIVE" if active else "passive", out.source, d.get("err_x", math.nan),
                             d.get("err_y", math.nan), d.get("err_z", math.nan), d.get("V", math.nan),
                             d.get("solve_ms", math.nan), self.ctrl.wind_est.source)
                    t_stat = now

                next_t += period
                slack = next_t - time.monotonic()
                if slack > 0:
                    time.sleep(slack)
                else:
                    n_overruns += 1
                    if slack < -period:
                        next_t = time.monotonic()   # resync after a long stall
        except KeyboardInterrupt:
            LOG.info("Interrupted")
        finally:
            LOG.info("cycles=%d overruns=%d", n_cycles, n_overruns)
            self.shutdown()

    def stop(self):
        self._stop.set()

    def shutdown(self):
        if self._was_guided and self.cfg.exit_mode:
            self.follower.set_mode(self.cfg.exit_mode)
            time.sleep(0.2)
        if self.viz:
            self.viz.close()
        if self.csv:
            self.csv.close()
        for link in self.links.values():
            link.close()


# =============================================================================
# Offline closed-loop self-test (no autopilot): truth model != NMPC model
# =============================================================================

class _TruthPlane:
    """Higher-fidelity truth model used only by the self-test, shaped after
    ArduPlane SITL step responses: input dead time + fast attitude lags with
    rate limits, true flight-path dynamics (not quasi-steady), lift-based turn
    rate with imperfect coordination (turn efficiency), stall-clipped CL,
    throttle slew + lag and a steady pitch-tracking offset."""

    def __init__(self, p: AircraftParams, pos, V, chi, *, delay=0.40, tau_roll=0.25, tau_pitch=0.20,
                 roll_rate=math.radians(90), pitch_rate=math.radians(45), thr_slew=1.0,
                 tau_thr=0.15, CL_max=1.4, pitch_bias=math.radians(1.0), turn_eff=0.72):
        self.p = p
        self.delay, self.turn_eff, self.t = delay, turn_eff, 0.0
        self._queue: deque = deque()
        self.pos = np.array(pos, dtype=float)
        self.V, self.chi = float(V), float(chi)
        self.tau_roll, self.tau_pitch = tau_roll, tau_pitch
        self.roll_rate, self.pitch_rate = roll_rate, pitch_rate
        self.thr_slew, self.tau_thr = thr_slew, tau_thr
        self.CL_max, self.pitch_bias = CL_max, pitch_bias
        qS = 0.5 * p.rho * V * V * p.wing_area
        alpha0 = (p.mass * p.g / qS - p.CL0) / p.CL_alpha
        self.gamma, self.phi, self.theta = 0.0, 0.0, alpha0
        self.delta = self.delta_slew = 0.4
        self.phi_dot = self.theta_dot = self.chi_dot = 0.0
        self.cmd = (0.0, alpha0 - pitch_bias, 0.4)

    def command(self, t_sent: float, cmd):
        """Queue a command; it reaches the inner loops ``delay`` seconds later."""
        self._queue.append((t_sent + self.delay, cmd))

    def step(self, dt: float, wind: np.ndarray):
        p = self.p
        self.t += dt
        while self._queue and self._queue[0][0] <= self.t:
            self.cmd = self._queue.popleft()[1]
        phi_c, th_c, dl_c = self.cmd
        self.phi_dot = float(np.clip((phi_c - self.phi) / self.tau_roll, -self.roll_rate, self.roll_rate))
        self.theta_dot = float(np.clip((th_c + self.pitch_bias - self.theta) / self.tau_pitch,
                                       -self.pitch_rate, self.pitch_rate))
        self.phi += self.phi_dot * dt
        self.theta += self.theta_dot * dt
        self.delta_slew += float(np.clip(dl_c - self.delta_slew, -self.thr_slew * dt, self.thr_slew * dt))
        self.delta += (self.delta_slew - self.delta) * dt / self.tau_thr

        V = max(self.V, 3.0)
        cphi = math.cos(self.phi)
        alpha = (self.theta - self.gamma) / cphi
        CL_lin = p.CL0 + p.CL_alpha * alpha
        CL = CL_lin - math.log1p(math.exp(20.0 * (CL_lin - self.CL_max))) / 20.0   # smooth stall clip
        qS = 0.5 * p.rho * V * V * p.wing_area
        L, D, T = qS * CL, qS * (p.CD0 + p.k_induced * CL * CL), self.delta * p.thrust_max
        F_perp = L + T * math.sin(alpha)
        V_dot = (T * math.cos(alpha) - D) / p.mass - p.g * math.sin(self.gamma)
        gamma_dot = (F_perp * cphi - p.mass * p.g * math.cos(self.gamma)) / (p.mass * V)
        self.chi_dot = self.turn_eff * F_perp * math.sin(self.phi) / (p.mass * V * math.cos(self.gamma))
        cg = math.cos(self.gamma)
        self.pos += dt * np.array([V * cg * math.cos(self.chi) + wind[0],
                                   V * cg * math.sin(self.chi) + wind[1],
                                   -V * math.sin(self.gamma) + wind[2]])
        self.V += V_dot * dt
        self.gamma += gamma_dot * dt
        self.chi += self.chi_dot * dt

    def ground_vel(self, wind: np.ndarray) -> np.ndarray:
        cg = math.cos(self.gamma)
        return np.array([self.V * cg * math.cos(self.chi), self.V * cg * math.sin(self.chi),
                         -self.V * math.sin(self.gamma)]) + wind

    def snapshot(self, t: float, wind: np.ndarray, rng: np.random.Generator) -> VehicleSnapshot:
        # body rates from Euler rates (coordinated, zero sideslip, yaw = air heading)
        ph, th = self.phi, self.theta
        psi_dot = self.chi_dot
        p_ = self.phi_dot - psi_dot * math.sin(th)
        q_ = self.theta_dot * math.cos(ph) + psi_dot * math.sin(ph) * math.cos(th)
        r_ = -self.theta_dot * math.sin(ph) + psi_dot * math.cos(ph) * math.cos(th)
        n = rng.normal
        return VehicleSnapshot(
            t_pos=t, pos=self.pos + n(0, 0.05, 3), vel=self.ground_vel(wind) + n(0, 0.05, 3),
            t_att=t, roll=ph + n(0, 0.003), pitch=th + n(0, 0.003),
            yaw=wrap_pi(self.chi), p=p_, q=q_ + n(0, 0.005), r=r_ + n(0, 0.005),
            t_vfr=t, airspeed=self.V + n(0, 0.2), throttle=self.delta,
            t_wind=t, wind=wind + n(0, 0.1, 3) * np.array([1, 1, 0]),
            t_hb=t, mode=PLANE_GUIDED, armed=True)


def _leader_autopilot(lp: _TruthPlane, t: float, model: FixedWingPointMassModel,
                      state: Dict[str, float]) -> Tuple[float, float, float]:
    """Scripted leader: turns, a climb and a speed change, flown by a simple
    altitude/airspeed autopilot (stands in for the leader's own AUTO mode)."""
    def ramp(t0, t1, a, b):
        return a + (b - a) * min(max((t - t0) / max(t1 - t0, 1e-6), 0.0), 1.0)
    bank = 0.0
    for (t0, t1, deg) in ((12, 30, 25.0), (42, 58, -20.0)):
        if t0 <= t < t1:
            bank = math.radians(deg) * min(1.0, (t - t0) / 2.0, (t1 - t) / 2.0)
    alt_ref = ramp(65, 75, 100.0, 115.0)
    V_ref = ramp(78, 82, 20.0, 22.0)
    alt = -lp.pos[2]
    gamma_c = float(np.clip(0.06 * (alt_ref - alt) - 0.0, -0.15, 0.15))
    th_ff, dl_ff = model.trim(V_ref, gamma_c, bank)
    state["i_v"] = float(np.clip(state.get("i_v", 0.0) + 0.0005 * (V_ref - lp.V), -0.3, 0.3))
    throttle = float(np.clip(dl_ff + 0.08 * (V_ref - lp.V) + state["i_v"], 0.0, 1.0))
    return bank, float(th_ff), throttle


def run_selftest(cfg: ControllerConfig, duration: float = 100.0, plot: Optional[str] = None,
                 seed: int = 1) -> Dict[str, float]:
    """Closed-loop simulation with model mismatch, wind, noise and latency."""
    rng = np.random.default_rng(seed)
    nominal = cfg.aircraft
    truth_p = AircraftParams(**{**nominal.__dict__})
    truth_p.mass *= 1.08
    truth_p.CD0 *= 1.25
    truth_p.k_induced *= 1.15
    truth_p.CL0 -= 0.04
    truth_p.CL_alpha *= 0.9
    truth_p.thrust_max *= 0.92
    wind_mean = np.array([-3.0, 4.0, 0.0])          # 5 m/s

    leader = _TruthPlane(truth_p, [0.0, 0.0, -100.0], 20.0, 0.0, delay=0.0, tau_roll=0.5, pitch_bias=0.0)
    off = np.array(cfg.formation.offset)
    follower = _TruthPlane(truth_p, [off[0] - 30.0, off[1] - 15.0, -100.0 + off[2] + 8.0],
                           18.0, math.radians(20.0))
    ctrl = NMPCFormationController(cfg)
    ap_state: Dict[str, float] = {}
    nominal_model = FixedWingPointMassModel(truth_p)

    dt_sim, t, period = 0.005, 0.0, cfg.control_period
    lag_meas = 0.04
    hist: deque = deque(maxlen=200)
    pending: deque = deque()
    next_ctrl = 0.0
    log: List[Dict[str, float]] = []
    min_sep = math.inf
    while t < duration:
        wind = wind_mean + np.array([0.8 * math.sin(0.31 * t), 0.6 * math.sin(0.23 * t + 1.0), 0.0])
        leader.cmd = _leader_autopilot(leader, t, nominal_model, ap_state)
        leader.step(dt_sim, wind)
        while pending and pending[0][0] <= t:
            follower.command(t, pending.popleft()[1])     # leaves the companion now
        follower.step(dt_sim, wind)
        t += dt_sim
        hist.append((t, leader.snapshot(t, wind, rng), follower.snapshot(t, wind, rng)))

        # true geometry (leader heading frame) for metrics
        psi_l = leader.chi
        c, s = math.cos(psi_l), math.sin(psi_l)
        rel = follower.pos - leader.pos
        rel_b = np.array([rel[0] * c + rel[1] * s, -rel[0] * s + rel[1] * c, rel[2]])
        if t > min(25.0, 0.4 * duration):
            min_sep = min(min_sep, ellipsoid_metric(rel_b, cfg.formation.separation_axes))

        if t >= next_ctrl:
            next_ctrl += period
            # measurements arrive lag_meas late
            t_meas = t - lag_meas
            l_snap = f_snap = None
            for th_, ls, fs in reversed(hist):
                if th_ <= t_meas:
                    l_snap, f_snap = ls, fs
                    break
            if l_snap is None:
                continue
            out = ctrl.step(t, l_snap, f_snap, active=True)
            if out.send:   # sent after the solve; link latency is part of the plane's delay
                delay = min(out.diag.get("solve_ms", 30.0) * 1e-3, 0.9 * period)
                pending.append((t + delay, (out.roll, out.pitch, out.throttle)))
            e = rel_b - off
            row = dict(out.diag)
            row.update({"t": t, "true_ex": e[0], "true_ey": e[1], "true_ez": e[2],
                        "n_f": follower.pos[0], "e_f": follower.pos[1],
                        "n_l": leader.pos[0], "e_l": leader.pos[1], "src": out.source})
            log.append(row)

    tt = np.array([r["t"] for r in log])
    E = np.array([[r["true_ex"], r["true_ey"], r["true_ez"]] for r in log])
    solve = np.array([r.get("solve_ms", np.nan) for r in log])
    nit = np.array([r.get("nit", np.nan) for r in log])
    settle = min(25.0, 0.4 * duration)
    mask = tt > settle
    turns = mask & (((tt > 14) & (tt < 32)) | ((tt > 44) & (tt < 60)))
    res = {
        "rms_along": float(np.sqrt(np.mean(E[mask, 0] ** 2))),
        "rms_cross": float(np.sqrt(np.mean(E[mask, 1] ** 2))),
        "rms_vert": float(np.sqrt(np.mean(E[mask, 2] ** 2))),
        "max_err_norm": float(np.max(np.linalg.norm(E[mask], axis=1))),
        "rms_norm_in_turns": float(np.sqrt(np.mean(np.sum(E[turns] ** 2, axis=1)))) if turns.any() else math.nan,
        "min_sep_metric": float(min_sep),
        "solve_ms_mean": float(np.nanmean(solve)),
        "solve_ms_p95": float(np.nanpercentile(solve, 95)),
        "solve_ms_max": float(np.nanmax(solve)),
        "nit_mean": float(np.nanmean(nit)),
        "roll_cmd_rate_rms_dps": float(np.degrees(np.sqrt(np.nanmean(
            (np.diff([r.get("roll_cmd", np.nan) for r in log])[mask[1:]] / cfg.control_period) ** 2)))),
        "pitch_cmd_rate_rms_dps": float(np.degrees(np.sqrt(np.nanmean(
            (np.diff([r.get("pitch_cmd", np.nan) for r in log])[mask[1:]] / cfg.control_period) ** 2)))),
        "deadline_hits": int(np.nansum([r.get("deadline_hit", 0.0) for r in log])),
        "fallbacks": int(sum(r["src"] == "fallback" for r in log)),
        "breakaways": int(sum(r["src"] == "breakaway" for r in log)),
        "bias_vdot": float(ctrl.observer.bias[0]),
        "bias_gamma_deg": float(math.degrees(ctrl.observer.bias[1])),
        "bias_chidot_dps": float(math.degrees(ctrl.observer.bias[2])),
        "turn_efficiency": float(ctrl.observer.bias[3]),
        "pitch_offset_deg": float(math.degrees(ctrl.observer.bias[4])),
    }
    if plot:
        _plot_selftest(log, plot)
    return res


def _plot_selftest(log: List[Dict[str, float]], path: str):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    t = np.array([r["t"] for r in log])
    get = lambda k: np.array([r.get(k, np.nan) for r in log])  # noqa: E731
    fig, axes = plt.subplots(4, 2, figsize=(14, 13))
    ax = axes[0, 0]
    ax.plot(get("e_l"), get("n_l"), label="leader")
    ax.plot(get("e_f"), get("n_f"), label="follower")
    ax.set_title("Ground track [m]"); ax.set_xlabel("east"); ax.set_ylabel("north")
    ax.axis("equal"); ax.legend()
    ax = axes[0, 1]
    for k, lab in (("true_ex", "along"), ("true_ey", "cross"), ("true_ez", "vertical")):
        ax.plot(t, get(k), label=lab)
    ax.set_title("True slot error [m]"); ax.set_ylim(-6, 6); ax.legend()
    for ax, (title, series, scale) in zip(axes.ravel()[2:], _VIZ_PANELS):
        for key, lab in series:
            ax.plot(t, get(key) * scale, label=lab, lw=1.2)
        ax.set_title(title); ax.legend(fontsize=8)
    for ax in axes.ravel():
        ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


# =============================================================================
# Entry point
# =============================================================================

def build_config(args: argparse.Namespace) -> ControllerConfig:
    cfg = ControllerConfig()
    cfg.link.leader_url = args.leader
    cfg.link.follower_url = args.follower
    cfg.link.leader_sysid = args.leader_sysid
    cfg.link.follower_sysid = args.follower_sysid
    cfg.link.stream_hz = args.stream_hz
    cfg.link.assume_common_origin = args.assume_common_origin
    cfg.formation.offset = tuple(args.offset)
    cfg.control_rate_hz = args.rate
    cfg.mpc.horizon = args.horizon
    cfg.mpc.dt = args.dt
    cfg.mpc.integrator = args.integrator
    cfg.mpc.solver = args.solver
    cfg.auto_guided = not args.no_auto_guided
    cfg.visualize = args.viz
    cfg.log_csv = args.log
    return cfg


def main():
    ap = argparse.ArgumentParser(description="Fixed-wing formation NMPC for ArduPlane (pymavlink)")
    ap.add_argument("--leader", default="udpin:0.0.0.0:14550", help="leader MAVLink URL")
    ap.add_argument("--follower", default="udpin:0.0.0.0:14560", help="follower MAVLink URL")
    ap.add_argument("--leader-sysid", type=int, default=None)
    ap.add_argument("--follower-sysid", type=int, default=None)
    ap.add_argument("--offset", type=float, nargs=3, default=(-6.0, 4.0, 0.0),
                    metavar=("FWD", "RIGHT", "DOWN"), help="slot offset in leader heading frame [m]")
    ap.add_argument("--rate", type=float, default=10.0, help="control rate [Hz] (10-20)")
    ap.add_argument("--horizon", type=int, default=20)
    ap.add_argument("--dt", type=float, default=0.2)
    ap.add_argument("--integrator", choices=("rk2", "rk4"), default="rk2")
    ap.add_argument("--solver", choices=("gn", "slsqp"), default="gn",
                    help="gn: bounded Gauss-Newton (real-time default); slsqp: scipy SLSQP")
    ap.add_argument("--stream-hz", type=float, default=25.0)
    ap.add_argument("--assume-common-origin", action="store_true")
    ap.add_argument("--no-auto-guided", action="store_true", help="do not switch follower to GUIDED")
    ap.add_argument("--viz", action="store_true", help="real-time matplotlib plots")
    ap.add_argument("--log", default=None, help="CSV log file")
    ap.add_argument("--selftest", action="store_true", help="offline closed-loop simulation")
    ap.add_argument("--duration", type=float, default=100.0, help="self-test duration [s]")
    ap.add_argument("--plot", default=None, help="self-test plot output (png)")
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args()
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    cfg = build_config(args)
    if args.selftest:
        res = run_selftest(cfg, duration=args.duration, plot=args.plot)
        width = max(len(k) for k in res)
        for k, v in res.items():
            print(f"{k:<{width}} : {v:.3f}" if isinstance(v, float) else f"{k:<{width}} : {v}")
        return

    # Example: leader on UDP 14550, follower on UDP 14560, echelon right 6 m
    # behind and 4 m to the right at the same altitude (see --offset).
    runner = FormationRunner(cfg)
    runner.run()


if __name__ == "__main__":
    main()
