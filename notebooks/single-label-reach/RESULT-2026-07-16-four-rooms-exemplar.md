# Four-rooms reach-exemplar audit

The legacy plot chose the first label with maximum largest-basin coverage.  In
the fixed random twist (seed 11), labels N, S, and W all have coverage 0.175.
N was therefore plotted despite having maximum reach 5; W has maximum reach 7
from the right-hand side.  The migrated notebook breaks coverage ties by
maximum reach and plots W.

The evolved sigma used by the legacy figure remains the correct exemplar when
“best” is defined for this reach figure.  Among the 24 full-goal multi-goal
run-bests in the pooled decision-information/free-energy cohort it has the
largest `fp_mean_rho` (5.7625) and diameter (19), with coverage 0.85.  It is the
legacy row-shuffle decision-information run
`prod-baseline-g200-b1-four-rooms-sp3011-09-20260309`.

It is not the lowest-free-energy run and not the largest-coverage free-energy
run.  The current free-energy coverage maximum is 0.875, with `fp_mean_rho`
4.475 and diameter 14.  The paper should therefore call the plotted sigma the
*highest-reach run-best*, not simply “GA-best.”

