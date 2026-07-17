# Uniform null versus initialiser footprints

## Question

Is the twists-home-vectors paper's fresh-uniform null a useful third twist
distribution, or is it already represented by one of the paper's two
generation-zero initialisation strategies?

## Method

`00-compare-initialiser-footprints.py` generated 2,016 twists from each of
three generators on `wrap_grid`, `helical`, and `pillar_3` at 7×7:

- fresh uniform, with every non-wall state-row drawn independently from the
  4! valid permutations;
- row-shuffle (IP-00), using the current gridTwist operator;
- permutation-balanced (IP-05), using 21 complete 96-member populations so
  the production epsilon schedule is preserved.

Every twist was placed on the mean-basins-per-label × mean-cycle/basin-ratio
fingerprint plane.  A bivariate Gaussian KDE was fitted separately to each
cloud.  Its 95% and 99% footprint thresholds were set at the 5th and 1st
percentiles of the density evaluated at the sampled points.

## Result

Fresh uniform and row-shuffle are the same probability law over the rows that
enter the fingerprint.  Their independently sampled footprints coincide to
Monte Carlo/KDE accuracy:

| environment | row-shuffle inside uniform 95% | row-shuffle inside uniform 99% |
|---|---:|---:|
| `wrap_grid` | 0.936 | 0.983 |
| `helical` | 0.946 | 0.984 |
| `pillar_3` | 0.927 | 0.986 |

Permutation-balanced is not drawn from that law.  Its epsilon-stratified
population has a broad structured tail, including near-aligned individuals:

| environment | permutation-balanced inside uniform 95% | permutation-balanced inside uniform 99% |
|---|---:|---:|
| `wrap_grid` | 0.577 | 0.676 |
| `helical` | 0.580 | 0.665 |
| `pillar_3` | 0.590 | 0.692 |

On the open environments its KDE footprint also reaches the pure-cycle
identity region; on `helical` this appears as a separate high-ratio component.

## Interpretation

The fresh-uniform null need not be introduced as a third initialisation
strategy.  It is a large, independently generated sample from the same prior
as a generation-zero row-shuffle individual.  It can therefore be named the
**row-shuffle/no-selection reference** while retaining 2,000 fresh draws for a
stable contour.

Permutation-balanced answers a different question: it shows the structured
prior from which production evolution starts.  Its footprint should not be
described as equivalent to the row-shuffle null.  If both are included in a
paper figure, the row-shuffle contour tests rarity under fully shuffled local
relabelling, whereas the permutation-balanced contour tests displacement from
the production initial population.

The contour percentages describe sample mass, not confidence.  A contour is
an iso-density boundary chosen so that approximately 95% or 99% of generated
points lie on its high-density side.  Because permutation-balanced is
multimodal, one density level can produce more than one closed curve.

The diagnostic scatter and overlay figures use common numerical limits on
both fingerprint axes across all three environments.  The notebook also
renders `F-figure7-two-initialiser-footprints.png`, a paper-layout preview in
which row-shuffle and permutation-balanced contours are both placed over the
full evaluated populations.

The matching coverage-plane preview,
`F-figure8-two-initialiser-footprints.png`, carries the same priors onto
`open_grid`, `pinwheel`, and `four_rooms`.  Neither prior reaches the
`open_grid` run-best coverage median.  At or above the more modest run-best
medians there are 5 row-shuffle versus 9 permutation-balanced draws on
`pinwheel`, and 4 versus 1 on `four_rooms`, out of 2,016 per strategy.
Every run-best lies outside the outer 99% row-shuffle contour.  The broader
permutation-balanced contour contains 9 of 33 `open_grid` and 3 of 24
`four_rooms` run-bests at the 99% level, all from the modest portion of those
modes; it contains no `pinwheel` run-best.

## Open-grid run-best stratification

The notebook also audits the 33 stars in the `open_grid` panel rather than
treating their visible bands as an evolutionary time series.  Coverage is an
integer basin size divided by 49, so its rows are separated by exactly 1/49;
mean basins per label averages four integer basin counts and therefore lies on
quarter-steps.  Each star is the final best from an independent run.

The archived run metadata give:

| factor | cohort | n | median coverage | range |
|---|---|---:|---:|---:|
| objective | decision information | 22 | 0.827 | 0.429--0.939 |
| objective | free energy | 11 | 0.673 | 0.388--0.939 |
| initialiser | permutation-balanced | 31 | 0.714 | 0.388--0.939 |
| initialiser | row-shuffle | 2 | 0.878 | 0.857--0.898 |
| budget | 200 generations | 27 | 0.796 | 0.388--0.939 |
| budget | 500 generations | 3 | 0.898 | 0.878--0.939 |

The two row-shuffle results are high-coverage legacy decision-information
runs, so this cohort cannot identify a separate row-shuffle effect.  The 27
200-generation runs span almost the full coverage range; generation budget
therefore does not explain the bands, although coverage has a moderate
Spearman association with budget (0.448).  Its association with the first
generation containing the final winner is weaker (0.324).

There are 26 distinct random seeds.  Seven exact seeds are reused in matched
decision-information/free-energy pairs, but none of the 33 stars has its
same-seed partner as its nearest plotted neighbour.  Numerical proximity
between seed integers is not meaningful.  Finally, no run records a parent or
continuation lineage: the three genuine 500-generation runs log generations
0--500 independently.  One run named `prod-baseline-g500...` is actually a
0--200 run in both its configuration and log and is classified as 200.

These checks are rendered in
`figures/F-open-grid-run-best-stratification-diagnostic.png`; the exact
run-level audit is in `artifacts/open-grid-run-best-metadata.csv`.
