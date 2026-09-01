# Reacher cross-task replication and calibrated-margin extension

Status: final results. The extension is post-confirmatory and does not replace the inherited-margin failure.

## Results

| Analysis | Target-OOD paired reward difference (95% bootstrap CI) | Maximum trip rate | Decision |
|---|---:|---:|---|
| Inherited physics margin `z=1.5` vs `z=0` | +0.721 [+0.474, +0.978] | 2.3% | Original primary gate failed |
| Reacher-calibrated margin `z=2.0` vs `z=0` | +0.692 [+0.434, +0.962] | 1.6% | Post-confirmatory extension passed |
| Inherited `z=1.5` on the same extension seeds | -- | 1.9% | Also passed the point gate |
| Calibrated `z=2.0` vs inherited `z=1.5` | -0.008 [-0.121, +0.108] | -- | No mean reward improvement established |
| Hybrid `z=1.5` vs `z=0` | +0.797 [+0.546, +1.059] | 1.4% | Secondary result |
| Monolithic RecurrentPPO vs `z=0` | -9.068 [-9.507, -8.630] | -- | Strong OOD failure in this tested family |

The inherited margin improved expected target-OOD utility but missed its original frozen point gate (2.3% versus 2.0%). A cutoff/margin pair selected only on development seeds chose cutoff 0.06 and `z=2.0`. On 100 new lifetimes, the selected policy retained a positive mean effect and reached a 1.6% maximum trip rate. The inherited setting also passed on those same fresh seeds at 1.9%, and their mean reward difference was inconclusive (-0.008 [-0.121, +0.108]). The extension therefore demonstrates a feasible development-only selection procedure, not calibration necessity, superiority, or unique gate recovery.

Only 52 of 100 paired lifetime effects were positive (exact sign test p=0.764). The supported claim concerns mean expected utility, not majority-lifetime improvement. Both tasks remain in one simulator with the same phenomenological thermal law; no real-hardware or formal-safety claim is supported.

Protocol hashes: inherited `eeea9e261d36b198897346ac5b9b3d3eda90b71936b2e9ba11ea12795f9ae925`; extension `bed29c0121ddf4b632173a360cd842faefde3e6ce1ac4a345bfb8af273aa049c`.
