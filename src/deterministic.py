import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plot


def sir_deterministic(
        _t: float,
        y: list[float],
        beta: float,
        gamma: float,
    ) -> list[float]:
    """
    _t is unused since SIR doesn't factor time into the calculation
    y is the state of the SIR values
    beta is the infection rate constant in 
        ppl infected per day per infected person
    gamma is the recovery rate constant in
        ppl recovered per day per infected person
    """
    susceptible_pop, infectious_pop, recovered_pop = y
    total_pop = susceptible_pop + infectious_pop + recovered_pop

    new_susceptible_pop = -beta * infectious_pop * susceptible_pop / total_pop
    new_recovered_pop = gamma * infectious_pop
    new_infectious_pop = -new_susceptible_pop - new_recovered_pop

    return [new_susceptible_pop, new_infectious_pop, new_recovered_pop]

y0 = [4980, 20, 0]
beta = 0.4
gamma = 0.2
t_span = (0, 200)
t_eval = np.linspace(*t_span, 600)

solution = solve_ivp(
    sir_deterministic,
    t_span,
    y0,
    args=(beta, gamma),
    t_eval=t_eval,
)

plot.plot(solution.t, solution.y[0], label="Susceptible")
plot.plot(solution.t, solution.y[1], label="Infectious")
plot.plot(solution.t, solution.y[2], label="Recovered")
plot.xlabel("Time (days)")
plot.ylabel("People")
plot.title(f"SIR Deterministic model beta = {beta:.1f}, gamma = {gamma:.1f}")
plot.legend()
plot.show()
