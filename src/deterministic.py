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

    susceptible_rate = -beta * infectious_pop * susceptible_pop / total_pop
    recovered_rate = gamma * infectious_pop
    infectious_rate = -susceptible_rate - recovered_rate

    return [susceptible_rate, infectious_rate, recovered_rate]


def main():
    # variables to mess around with
    total_population = 5000
    starting_infectious_population = 50
    beta = 0.5
    gamma = 0.1
    total_days_to_simulate = 70

    y0 = [
        total_population - starting_infectious_population,
        starting_infectious_population,
        0,
    ]
    t_span = (0, total_days_to_simulate)
    t_eval = np.linspace(*t_span, 600)

    solution = solve_ivp(
        sir_deterministic,
        t_span,
        y0,
        args=(beta, gamma),
        t_eval=t_eval,
    )

    assert solution.success, solution.message

    plot.plot(solution.t, solution.y[0], label="Susceptible", color="#0077BB")
    plot.plot(solution.t, solution.y[1], label="Infectious", color="#CC3311")
    plot.plot(solution.t, solution.y[2], label="Recovered", color="#117733")
    plot.xlabel("Time (days)")
    plot.ylabel("People")
    plot.title(
        f"SIR Deterministic model beta = {beta:.1f}, gamma = {gamma:.1f}"
    )
    plot.legend()
    plot.show()


if __name__ == "__main__":
    main()