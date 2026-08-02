# Environment topology

Paper figure generators that show the geometry itself rather than an evolved
twist or policy.

| notebook | paper output |
|---|---|
| `00-environment-palette.py` | *Twists, home vectors*, Figure 1 |

The environment palette draws walls as vector rectangles, using the shared
`gridbench.papers.home_vectors.WALL_COLOUR`, so both PDF and PNG exports have
hard edges.  It writes the two paper assets directly when run without
arguments.

Run with the `py-3.12-grid` environment:

```bash
MPLBACKEND=Agg python notebooks/environment-topology/00-environment-palette.py
```
