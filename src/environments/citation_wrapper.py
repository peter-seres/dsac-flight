import numpy as np
import environments.models.h2000_v90_windows.citation as citation
from typing import Optional


class CitationDynamics:
    """
    Wrapper class around the citation model to provide actuator controls and states.

    - initalize(): initializes the internal state
    - step(u: np.array): apply control input u (elevator, aileron, rudder) deflections in radians.
                         steps the dynamics forward.
    - state getters: [p, q, r, V, alpha, beta, phi, theta, psi, h, x_e, y_e]
    """

    # Sampling time of the compiled model
    dt: float = 0.01

    # Control deflection saturation [rad]
    defl_low: np.ndarray = np.deg2rad(np.array([-17.0, -19.0, -22.0]))
    defl_high: np.ndarray = np.deg2rad(np.array([15.0, 15.0, 22.0]))

    # Enable slots
    __slots__ = "t", "x", "u"

    def __init__(self):
        # Time tracking
        self.t: float = 0.0

        # Current dynamic state [12]
        self.x: Optional[np.ndarray] = None

        # Latest control inputs [3]
        self.u: Optional[np.ndarray] = None

    def initialize(self) -> None:
        # Initialize the compiled model
        citation.initialize()

        # Initialize zero deflections:
        self.u = np.zeros(3)

        # Make a zero input step to retreive the state (citation python model has no internal state)
        self.x = citation.step(cmd=self.pad_control_inputs(deflections=self.u))

        # Reset the time:
        self.t = 0.0

    def step(self, u: np.ndarray) -> (np.ndarray, np.ndarray):
        """Applies control input u [size 3] to the dynamic model, with deflection saturation.
        Steps the internal clock.
        Sets the next state.
        """

        # Clip the control surfaces to saturation limits:
        self.u = u.clip(min=self.defl_low, max=self.defl_high)

        # Pad the vector 3 deflections to control inputs of size 10
        control_inputs = self.pad_control_inputs(deflections=u)

        # Apply to the citation model
        self.x = citation.step(cmd=control_inputs)

        # Step the time
        self.t += self.dt

    @staticmethod
    def pad_control_inputs(deflections: np.ndarray) -> np.ndarray:
        """Pad the latter part of the control input vector with zeros from size 3 to size 10."""
        return np.pad(array=deflections, pad_width=(0, 7))

    @property
    def p(self) -> float:
        return self.x[0]

    @property
    def q(self) -> float:
        return self.x[1]

    @property
    def r(self) -> float:
        return self.x[2]

    @property
    def V(self) -> float:
        return self.x[3]

    @property
    def alpha(self) -> float:
        return self.x[4]

    @property
    def beta(self) -> float:
        return self.x[5]

    @property
    def phi(self) -> float:
        return self.x[6]

    @property
    def theta(self) -> float:
        return self.x[7]

    @property
    def psi(self) -> float:
        return self.x[8]

    @property
    def h(self) -> float:
        return self.x[9]

    @property
    def x_e(self) -> float:
        return self.x[10]

    @property
    def y_e(self) -> float:
        return self.x[11]
