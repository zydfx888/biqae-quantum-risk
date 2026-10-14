"""Validated protocol and project-local output boundary."""
from dataclasses import dataclass, field, asdict
from pathlib import Path
import hashlib
import json

ROOT = Path(__file__).resolve().parents[2]
METHODS = {"mc", "sobol", "is", "mlae", "iqae", "grid_biqae", "normal_biqae", "beta_biqae"}

def guarded(path, root=ROOT):
    root = Path(root).resolve()
    candidate = (root / path).resolve()
    if not candidate.is_relative_to(root):
        raise ValueError(f"Output path escapes project root: {candidate}")
    return candidate

def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def write_json(path, value):
    path = guarded(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")

@dataclass
class Config:
    dataset: str = "synthetic_student_t"
    assets: int = 4
    states: int = 16
    scenarios: int = 10000
    alpha: float = .975
    epsilon: float = .01
    failure_probability: float = .05
    methods: list[str] = field(default_factory=lambda: sorted(METHODS))
    shots_per_round: int = 100
    max_grover_power: int = 15
    max_rounds: int = 12
    budget: int = 4096
    seeds: list[int] = field(default_factory=lambda: [13, 29, 43, 71, 101])
    distribution_seed: int = 20260908
    backend: str = "analytic_shots"
    count_state_preparation: bool = True
    rho: float = .5
    df: float = 5
    grid_size: int = 8193
    prior: list[float] = field(default_factory=lambda: [1., 1.])
    var_budget_fraction: float = .4
    w_max: float = .6
    optimization_steps: int = 25
    learning_rate: float = .5
    discretization: str = "tail"
    noise_probability: float = .001
    record_posterior: bool = True
    stopping_rule: str = "interval"
    input_file: str | None = None
    train_end: str | None = None
    test_start: str | None = None
    output: str = "artifacts/runs"

    def __post_init__(self):
        if self.stopping_rule not in {"interval", "budget"}:
            raise ValueError("stopping_rule must be interval or budget")
        if not 0 < self.alpha < 1 or not 0 < self.failure_probability < 1:
            raise ValueError("alpha and failure_probability must be in (0, 1)")
        if not 0 < self.epsilon <= .5 or not 0 < self.var_budget_fraction < 1:
            raise ValueError("Invalid epsilon or budget split")
        if self.assets < 2 or self.states < 2 or self.scenarios < self.states:
            raise ValueError("Need >=2 assets/states and scenarios >= states")
        if not 1 / self.assets <= self.w_max <= 1:
            raise ValueError("Infeasible capped simplex")
        if self.budget < 16 or self.shots_per_round < 1 or self.max_rounds < 1 or self.max_grover_power < 0:
            raise ValueError("Invalid measurement budget")
        if self.grid_size < 1025 or self.grid_size < 16 * (2*self.max_grover_power+1):
            raise ValueError("Posterior grid too coarse for configured Grover powers")
        if len(self.prior) != 2 or min(self.prior) <= 0:
            raise ValueError("Prior beta shapes must be positive")
        if not self.methods or not set(self.methods) <= METHODS or not self.seeds:
            raise ValueError("Unknown method or empty seeds")
        if self.backend not in {"analytic_shots", "qiskit_statevector", "aer_shots", "aer_noise"}:
            raise ValueError("Unsupported backend; hardware requires a separate adapter")
        if not self.count_state_preparation:
            raise ValueError("This protocol always counts state preparation")
        if self.optimization_steps < 1 or not self.learning_rate > 0:
            raise ValueError("Optimization requires positive steps and learning rate")
        if not -1/(self.assets-1) < self.rho < 1 or self.df <= 2:
            raise ValueError("Invalid correlation or Student-t degrees of freedom")
        if self.discretization not in {"tail", "uniform"} or not 0 <= self.noise_probability < 1:
            raise ValueError("Invalid discretization/noise")
        guarded(self.output)
        if self.input_file:
            guarded(self.input_file)

    @classmethod
    def load(cls, path):
        return cls(**json.loads(Path(path).read_text(encoding="utf-8-sig")))

    def to_dict(self):
        return asdict(self)
