# Development-gallery scalability

Recorded real-image run on 25 October 2026. Reproduce from the project root with
`python -m tools.benchmark_scalability` after supplying the original development
images and trusted stage-5/6 model files. The main notebook section 7.2.1 presents
the method and results. No models were fitted and no test images were accessed.

Eight fixed queries per split, nested seeded galleries, five repeats, single-threaded
OpenCV/BLAS. Each model/size has 40 query observations. Brackets below are the
25th–75th percentiles across queries and repeats, not confidence intervals.
Cache build has five observations per size and includes candidate ORB and HSV/D.

| Split | Candidates | Linear online seconds, median [IQR] | SVR online seconds, median [IQR] | Cache build median seconds |
| --- | ---: | ---: | ---: | ---: |
| Train | 16 | 0.0438 [0.0414, 0.0463] | 0.0463 [0.0446, 0.0490] | 0.1922 |
| Train | 32 | 0.0707 [0.0690, 0.0740] | 0.0753 [0.0738, 0.0791] | 0.3940 |
| Train | 64 | 0.1283 [0.1232, 0.1349] | 0.1357 [0.1310, 0.1434] | 0.7613 |
| Train | 128 | 0.2393 [0.2306, 0.2452] | 0.2551 [0.2467, 0.2622] | 1.5501 |
| Train | 256 | 0.4648 [0.4481, 0.4868] | 0.4954 [0.4805, 0.5178] | 3.1126 |
| Train | 305 | 0.5407 [0.5315, 0.5547] | 0.5808 [0.5701, 0.5959] | 3.6881 |
| Validation | 16 | 0.0438 [0.0418, 0.0453] | 0.0465 [0.0436, 0.0477] | 0.1923 |
| Validation | 32 | 0.0731 [0.0698, 0.0795] | 0.0769 [0.0747, 0.0834] | 0.3989 |
| Validation | 64 | 0.1278 [0.1251, 0.1319] | 0.1355 [0.1333, 0.1401] | 0.7703 |
| Validation | 112 | 0.2214 [0.2113, 0.2647] | 0.2372 [0.2247, 0.2824] | 1.4017 |

At 112 validation candidates, median SVR stages are query-image extraction 0.01347 s,
pair features 0.20558 s, prediction 0.01514 s and ranking 0.00263 s. At 305 train
candidates, pair features take 0.52226 s, SVR prediction 0.03898 s and ranking
0.00482 s. Stage medians need not sum to the total median. Linear prediction at
305 candidates is 0.00102 s, while its online total is 0.54066 s: feature matching
dominates both models, so prediction-only ratios exaggerate the end-to-end gap.

The measured curves are approximately linear with gallery size over this range.
Reusing gallery summaries avoids repeating construction for each query, but the
first query still pays it: median SVR first-query totals are 1.6338 s at 112
validation candidates and 4.2638 s at 305 train candidates. These are computed
from each observation's build-plus-query cost, not sums of separate medians.
Caching does not remove the per-candidate matching/RANSAC cost; reducing that cost
is a better-supported next optimization than optimizing the small ranking stage.
No additional optimization was selected or tested here.

Files:

- `timings.csv`: all 800 per-model query observations, including all four stages
  and online/first-query sums.
- `cache_timings.csv`: 50 gallery builds; array storage excludes Python overhead.
- `per_query_summary.csv`: five-repeat dispersion for each fixed query/size/model.
- `summary.csv`: median, quartiles, standard deviation and range over 40 observations.
- `cache_summary.csv`: five-repeat construction cost and array-storage summaries.
- `protocol.json`: workload IDs, configuration, model/source/image hashes,
  environment/thread library details, equivalence checks and timing boundaries.
- `scaling.png`: median curves with IQR shading.

An untimed pass validates all accessed image summaries against the production
extractors; all repeated model scores and tie-aware ranks agree. Query and gallery
images are decoded from original files; persistent feature-cache loads are not
timed. Hashing, persistent cache writes, model loading, assertions and output I/O
are excluded. Image decoding is shared by ORB and HSV, and image/pair stages are
shared across model measurements; each reported latency represents either model
used alone. OS cache state is uncontrolled after warmup, so these are not cold-disk
timings. All measurements are conditional on this machine, these fixed queries
and models. They do not establish training scalability, matching accuracy, peak
RAM requirements or performance beyond 305 candidates.
