# Experimental Results & Evaluation Summary

## 1. Dataset & Execution Overview
- **Total clips evaluated**: 8
- **Clips where `phase_aware` ran without fallback**: 7 of 8 (87.5%)
- **Clips with fallback to hybrid**: 1 of 8 (12.5%)
  - Fallback clip: `Straight Drive/vid100_45.avi` (wrist-speed inflection boundaries ambiguous)
- **Execution crashes**: 0

## 2. Reconstruction Error (MPJPE) and Joint-Angle Error
*Values reported as mean ± std across all 8 clips. MPJPE in normalised coordinates [0, 1]; Angle error in degrees (°). `greedy_oracle` is a theoretical upper bound that inspects ground-truth frames.*

| Method | K=4 MPJPE | K=4 Angle | K=6 MPJPE | K=6 Angle | K=8 MPJPE | K=8 Angle | K=12 MPJPE | K=12 Angle |
|---|---|---|---|---|---|---|---|---|
| **uniform** | 0.0616 ± 0.0244 | 18.2° | 0.0367 ± 0.0148 | 12.6° | 0.0270 ± 0.0121 | 9.5° | 0.0146 ± 0.0068 | 5.8° | 
| **motion_aware** | 0.0575 ± 0.0236 | 15.9° | 0.0404 ± 0.0169 | 13.2° | 0.0299 ± 0.0155 | 10.7° | 0.0160 ± 0.0084 | 5.4° | 
| **hybrid** | 0.0901 ± 0.0422 | 19.3° | 0.0653 ± 0.0314 | 15.3° | 0.0468 ± 0.0221 | 13.1° | 0.0222 ± 0.0098 | 8.9° | 
| **phase_aware** | 0.0856 ± 0.0244 | 16.5° | 0.0578 ± 0.0309 | 14.9° | 0.0426 ± 0.0277 | 12.6° | 0.0190 ± 0.0086 | 8.5° | 
| *greedy_oracle (ref)* | *0.0478 ± 0.0195* | *16.9°* | *0.0304 ± 0.0140* | *12.8°* | *0.0193 ± 0.0091* | *8.4°* | *0.0069 ± 0.0045* | *4.0°* | 

## 3. Key-Phase Recall
*Fraction of detected phases containing at least one selected frame (mean ± std across all 8 clips).*

| Method | K=4 Recall | K=6 Recall | K=8 Recall | K=12 Recall |
|---|---|---|---|---|
| **uniform** | 0.686 ± 0.181 | 0.800 ± 0.151 | 0.800 ± 0.151 | 0.886 ± 0.146 | 
| **motion_aware** | 0.657 ± 0.176 | 0.657 ± 0.176 | 0.743 ± 0.140 | 0.800 ± 0.151 | 
| **hybrid** | 0.629 ± 0.198 | 0.686 ± 0.181 | 0.800 ± 0.185 | 0.943 ± 0.090 | 
| **phase_aware** | 0.771 ± 0.167 | 0.886 ± 0.099 | 1.000 ± 0.000 | 1.000 ± 0.000 | 
| *greedy_oracle (ref)* | *0.657 ± 0.206* | *0.771 ± 0.167* | *0.857 ± 0.176* | *0.886 ± 0.181* | 

## 4. Impact-Frame Coverage (Distance in Frames to Nearest Selected Frame)
*Distance in frames from the nearest selected frame to the impact frame (mean ± std across all 8 clips). Lower is better.*

| Method | K=4 Distance | K=6 Distance | K=8 Distance | K=12 Distance |
|---|---|---|---|---|
| **uniform** | 3.12 ± 0.60 | 0.88 ± 0.78 | 1.25 ± 0.66 | 0.62 ± 0.48 | 
| **motion_aware** | 1.62 ± 1.32 | 1.25 ± 0.83 | 0.75 ± 0.83 | 0.25 ± 0.43 | 
| **hybrid** | 2.62 ± 1.32 | 2.00 ± 1.50 | 1.62 ± 1.32 | 0.62 ± 0.86 | 
| **phase_aware** | 2.50 ± 1.41 | 1.38 ± 1.41 | 1.25 ± 1.30 | 0.50 ± 0.71 | 
| *greedy_oracle (ref)* | *2.75 ± 1.92* | *1.12 ± 0.93* | *0.75 ± 0.66* | *0.38 ± 0.48* | 

## 5. Information Per Frame (IPF)
*Efficiency score: IPF = (1 - MPJPE / MPJPE_baseline) / K, where baseline uses only first+last frames (mean ± std across all 8 clips). Higher is better.*

| Method | K=4 IPF | K=6 IPF | K=8 IPF | K=12 IPF |
|---|---|---|---|---|
| **uniform** | 0.119 ± 0.050 | 0.115 ± 0.018 | 0.096 ± 0.011 | 0.073 ± 0.004 | 
| **motion_aware** | 0.128 ± 0.042 | 0.110 ± 0.019 | 0.093 ± 0.014 | 0.072 ± 0.005 | 
| **hybrid** | 0.058 ± 0.075 | 0.075 ± 0.036 | 0.075 ± 0.022 | 0.067 ± 0.007 | 
| **phase_aware** | 0.075 ± 0.032 | 0.085 ± 0.039 | 0.079 ± 0.027 | 0.069 ± 0.007 | 
| *greedy_oracle (ref)* | *0.146 ± 0.038* | *0.123 ± 0.018* | *0.104 ± 0.009* | *0.078 ± 0.003* | 

## 6. Paired Statistical Comparisons (Clips where `phase_aware` Ran Without Fallback, N=7)
*Bonferroni-corrected significance threshold: α = 0.05 / 4 = 0.0125. Only comparisons with p < 0.0125 are designated as statistically significant; otherwise win/loss/tie counts are reported.*

### 6.1 `phase_aware` vs `uniform`

| Metric | Budget K | Mean Diff (PA − U) | PA Wins | PA Losses | Ties | Wilcoxon *p*-value | Significant (α = 0.0125) |
|---|---|---|---|---|---|---|---|
| **mpjpe** | K=4 | +0.0256 | 2 | 5 | 0 | 0.0781 | No |
| **mpjpe** | K=6 | +0.0212 | 1 | 6 | 0 | 0.0312 | No |
| **mpjpe** | K=8 | +0.0153 | 1 | 6 | 0 | 0.1562 | No |
| **mpjpe** | K=12 | +0.0035 | 1 | 6 | 0 | 0.2188 | No |
| **angle_error** | K=4 | -0.2250 | 2 | 5 | 0 | 0.9375 | No |
| **angle_error** | K=6 | +1.6501 | 3 | 4 | 0 | 0.5781 | No |
| **angle_error** | K=8 | +1.7975 | 3 | 4 | 0 | 0.3750 | No |
| **angle_error** | K=12 | +1.1607 | 2 | 5 | 0 | 0.2969 | No |
| **phase_recall** | K=4 | +0.0857 | 2 | 1 | 4 | 0.7500 | No |
| **phase_recall** | K=6 | +0.0857 | 2 | 1 | 4 | 0.5000 | No |
| **phase_recall** | K=8 | +0.2000 | 5 | 0 | 2 | 0.0625 | No |
| **phase_recall** | K=12 | +0.1143 | 3 | 0 | 4 | 0.2500 | No |
| **impact_dist** | K=4 | -0.8571 | 4 | 2 | 1 | 0.3750 | No |
| **impact_dist** | K=6 | +0.4286 | 2 | 2 | 3 | 0.7500 | No |
| **impact_dist** | K=8 | -0.1429 | 3 | 2 | 2 | 1.0000 | No |
| **impact_dist** | K=12 | +0.0000 | 2 | 2 | 3 | 1.0000 | No |
| **info_per_frame** | K=4 | -0.0460 | 2 | 5 | 0 | 0.0781 | No |
| **info_per_frame** | K=6 | -0.0289 | 1 | 6 | 0 | 0.0312 | No |
| **info_per_frame** | K=8 | -0.0164 | 1 | 6 | 0 | 0.1094 | No |
| **info_per_frame** | K=12 | -0.0034 | 1 | 6 | 0 | 0.1094 | No |

### 6.2 `phase_aware` vs `hybrid`

| Metric | Budget K | Mean Diff (PA − H) | PA Wins | PA Losses | Ties | Wilcoxon *p*-value | Significant (α = 0.0125) |
|---|---|---|---|---|---|---|---|
| **mpjpe** | K=4 | -0.0051 | 5 | 1 | 1 | 0.4375 | No |
| **mpjpe** | K=6 | -0.0085 | 4 | 3 | 0 | 0.3750 | No |
| **mpjpe** | K=8 | -0.0048 | 4 | 3 | 0 | 0.5781 | No |
| **mpjpe** | K=12 | -0.0036 | 2 | 4 | 1 | 0.8438 | No |
| **angle_error** | K=4 | -3.2400 | 5 | 1 | 1 | 0.2188 | No |
| **angle_error** | K=6 | -0.5407 | 4 | 3 | 0 | 0.9375 | No |
| **angle_error** | K=8 | -0.5704 | 4 | 3 | 0 | 0.5781 | No |
| **angle_error** | K=12 | -0.4826 | 3 | 3 | 1 | 0.8438 | No |
| **phase_recall** | K=4 | +0.1429 | 3 | 0 | 4 | 0.2500 | No |
| **phase_recall** | K=6 | +0.2000 | 6 | 0 | 1 | 0.0312 | No |
| **phase_recall** | K=8 | +0.2000 | 5 | 0 | 2 | 0.0625 | No |
| **phase_recall** | K=12 | +0.0571 | 2 | 0 | 5 | 0.5000 | No |
| **impact_dist** | K=4 | -0.1429 | 2 | 3 | 2 | 1.0000 | No |
| **impact_dist** | K=6 | -0.7143 | 4 | 2 | 1 | 0.5312 | No |
| **impact_dist** | K=8 | -0.4286 | 3 | 2 | 2 | 0.7500 | No |
| **impact_dist** | K=12 | -0.1429 | 2 | 1 | 4 | 1.0000 | No |
| **info_per_frame** | K=4 | +0.0185 | 5 | 1 | 1 | 0.4375 | No |
| **info_per_frame** | K=6 | +0.0113 | 4 | 3 | 0 | 0.3750 | No |
| **info_per_frame** | K=8 | +0.0053 | 4 | 3 | 0 | 0.5781 | No |
| **info_per_frame** | K=12 | +0.0021 | 2 | 4 | 1 | 0.8438 | No |

## 7. Impact Moment Estimation vs Trivial Baseline
- **Labelled ground-truth clips**: 2 of 8
- **Body-only impact estimator**:
  - Mean timing error: 10.50 frames
  - Median timing error: 10.50 frames
  - Maximum timing error: 13 frames
- **Trivial baseline (predict middle frame = frame 12)**:
  - Mean timing error: 3.00 frames
  - Median timing error: 3.00 frames
  - Maximum timing error: 6 frames
- **Per-clip breakdown on labelled clips**:
  - `Cover Drive/vid100_41.avi`: Ground Truth = 18, Estimator = 5 (error = 13f), Middle Frame 12 (error = 6f)
  - `Straight Drive/vid100_45.avi`: Ground Truth = 12, Estimator = 4 (error = 8f), Middle Frame 12 (error = 0f)

## 8. Findings
1. **Reconstruction error (MPJPE)**:
   - At K=4: lowest mean MPJPE is **motion_aware** (0.0575) [Full ranking: motion_aware (0.0575), uniform (0.0616), phase_aware (0.0856), hybrid (0.0901)].
   - At K=6: lowest mean MPJPE is **uniform** (0.0367) [Full ranking: uniform (0.0367), motion_aware (0.0404), phase_aware (0.0578), hybrid (0.0653)].
   - At K=8: lowest mean MPJPE is **uniform** (0.0270) [Full ranking: uniform (0.0270), motion_aware (0.0299), phase_aware (0.0426), hybrid (0.0468)].
   - At K=12: lowest mean MPJPE is **uniform** (0.0146) [Full ranking: uniform (0.0146), motion_aware (0.0160), phase_aware (0.0190), hybrid (0.0222)].
2. **Joint-angle error**:
   - At K=4: lowest mean angle error is **motion_aware** (15.9°) [Full ranking: motion_aware (15.9°), phase_aware (16.5°), uniform (18.2°), hybrid (19.3°)].
   - At K=6: lowest mean angle error is **uniform** (12.6°) [Full ranking: uniform (12.6°), motion_aware (13.2°), phase_aware (14.9°), hybrid (15.3°)].
   - At K=8: lowest mean angle error is **uniform** (9.5°) [Full ranking: uniform (9.5°), motion_aware (10.7°), phase_aware (12.6°), hybrid (13.1°)].
   - At K=12: lowest mean angle error is **motion_aware** (5.4°) [Full ranking: motion_aware (5.4°), uniform (5.8°), phase_aware (8.5°), hybrid (8.9°)].
3. **Paired statistical comparisons (Bonferroni-corrected threshold: α = 0.05 / 4 = 0.0125)**:
   - **`phase_aware` vs `uniform` (MPJPE)**:
     * At K=4: 2 PA wins, 5 PA losses, 0 ties (mean diff = +0.0256, p = 0.0781).
     * At K=6: 1 PA wins, 6 PA losses, 0 ties (mean diff = +0.0212, p = 0.0312).
     * At K=8: 1 PA wins, 6 PA losses, 0 ties (mean diff = +0.0153, p = 0.1562).
     * At K=12: 1 PA wins, 6 PA losses, 0 ties (mean diff = +0.0035, p = 0.2188).
   - **`phase_aware` vs `hybrid` (Phase Recall)**:
     * At K=4: 3 PA wins, 0 PA losses, 4 ties (mean diff = +0.1429, p = 0.2500).
     * At K=6: 6 PA wins, 0 PA losses, 1 ties (mean diff = +0.2000, p = 0.0312).
     * At K=8: 5 PA wins, 0 PA losses, 2 ties (mean diff = +0.2000, p = 0.0625).
     * At K=12: 2 PA wins, 0 PA losses, 5 ties (mean diff = +0.0571, p = 0.5000).
4. **Working hypothesis for error distribution**:
   - *Untested hypothesis*: Phase-aware selection allocates frames preferentially to the downswing and impact region. While this improves key-phase recall, it leaves wider intervals in the stance and follow-through phases, which may result in higher linear interpolation error across the full clip compared to uniform spacing.
5. **Impact estimation vs trivial baseline**:
   - On the 2 manually labelled clips, the trivial baseline (predict middle frame = 12) achieved lower mean error (3.00 frames) than the body-only estimator (10.50 frames).

## 9. Limitations
1. **Sample size**: 8 clips of 24 frames each (approx. 0.8–1.0 seconds at standard frame rates). Results have high variance and limited statistical power.
2. **Ground truth proxy**: Ground truth is defined by smoothed MediaPipe Pose output with linear gap-filling, not multi-camera optical motion capture.
3. **Coordinate space**: MPJPE is evaluated in normalised image coordinates [0, 1], representing fractions of frame width and height, not metric units (mm or metres).
4. **Circularity in phase recall**: Phase recall measures coverage of rule-detected phases. Because the phase-aware method explicitly selects detected phase boundaries, higher phase recall is partly circular.
5. **Impact annotations**: Only 2 clips are manually labelled. Results cannot be extrapolated beyond these labelled instances.
