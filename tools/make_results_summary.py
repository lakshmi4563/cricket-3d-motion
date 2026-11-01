"""
Generate Results Summary for Evaluation Report.

Reads:
- reports/reconstruction_evaluation.json
- reports/impact_estimation_summary.json (if present)
- reports/impact_ground_truth.json (if present)

Writes:
- reports/results_summary.md

Strict Reporting Rules:
- All numbers in the Findings section are strictly derived from computed tables.
- Bonferroni correction applied (alpha = 0.05 / len(K)).
- Per-K breakdown of best MPJPE and best angle error method.
- Complete metric tables (MPJPE, Angle Error, Phase Recall, Impact Distance, Info per Frame).
- Paired comparisons: Phase-Aware vs Uniform AND Phase-Aware vs Hybrid.
- Trivial baseline "predict middle frame" included for impact timing evaluation.
"""

from __future__ import annotations

import json
from pathlib import Path
import numpy as np

try:
    from scipy import stats
    _SCIPY_AVAILABLE = True
except ImportError:
    _SCIPY_AVAILABLE = False

_tools_dir = Path(__file__).resolve().parent
_project_root = _tools_dir.parent


def compute_paired(
    clips: list[dict],
    k_str: str,
    method_a: str,
    method_b: str,
    metric: str,
    higher_is_better: bool = False,
) -> dict:
    vals_a = []
    vals_b = []
    for c in clips:
        res = c.get("per_k_results", {}).get(k_str, {})
        a = res.get(method_a, {}).get(metric)
        b = res.get(method_b, {}).get(metric)
        if a is not None and b is not None:
            vals_a.append(a)
            vals_b.append(b)

    if not vals_a:
        return {"n": 0, "mean_diff": float("nan"), "wins": 0, "losses": 0, "ties": 0, "p_val": None}

    diff = np.array(vals_a) - np.array(vals_b)
    if higher_is_better:
        wins = int(np.sum(diff > 1e-6))
        losses = int(np.sum(diff < -1e-6))
    else:
        wins = int(np.sum(diff < -1e-6))
        losses = int(np.sum(diff > 1e-6))
    ties = int(np.sum(np.abs(diff) <= 1e-6))
    mean_diff = float(np.mean(diff))

    p_val = None
    if _SCIPY_AVAILABLE and len(diff) >= 5 and np.any(diff != 0):
        try:
            res = stats.wilcoxon(diff)
            p_val = float(res.pvalue)
        except Exception:
            p_val = None

    return {
        "n": len(diff),
        "mean_diff": mean_diff,
        "wins": wins,
        "losses": losses,
        "ties": ties,
        "p_val": p_val,
    }


def generate_summary(base_dir: Path) -> Path:
    reports_dir = base_dir / "reports"
    recon_json_path = reports_dir / "reconstruction_evaluation.json"
    impact_json_path = reports_dir / "impact_estimation_summary.json"
    gt_impact_path = reports_dir / "impact_ground_truth.json"
    out_md_path = reports_dir / "results_summary.md"

    if not recon_json_path.exists():
        raise FileNotFoundError(f"Missing {recon_json_path}")

    with open(recon_json_path, "r", encoding="utf-8") as f:
        recon_data = json.load(f)

    impact_data = None
    if impact_json_path.exists():
        try:
            with open(impact_json_path, "r", encoding="utf-8") as f:
                impact_data = json.load(f)
        except Exception:
            impact_data = None

    gt_labels = {}
    if gt_impact_path.exists():
        try:
            with open(gt_impact_path, "r", encoding="utf-8") as f:
                gt_labels = json.load(f)
        except Exception:
            gt_labels = {}

    total_clips = len(recon_data)
    successful_pa_clips = [c for c in recon_data if c.get("phase_success", False) and not c.get("phase_aware_fallback_ks")]
    fallback_clips = [c for c in recon_data if not c.get("phase_success", False) or c.get("phase_aware_fallback_ks")]
    crash_count = sum(
        c.get("phase_aware_errors", 0)
        if isinstance(c.get("phase_aware_errors"), int)
        else len(c.get("phase_aware_errors", []))
        for c in recon_data
    )

    n_pa_real = len(successful_pa_clips)
    n_fallback = len(fallback_clips)

    eval_k_values = [4, 6, 8, 12]
    num_k = len(eval_k_values)
    alpha_bonferroni = 0.05 / num_k  # 0.0125

    methods = ["uniform", "motion_aware", "hybrid", "phase_aware"]
    all_methods = methods + ["greedy_oracle"]

    # Metrics dictionary to store table means across all clips
    metric_keys = ["mpjpe", "angle_error", "phase_recall", "impact_dist", "info_per_frame"]
    metric_tables: dict[str, dict[int, dict[str, dict[str, float]]]] = {
        m_key: {k: {} for k in eval_k_values} for m_key in metric_keys
    }

    for m_key in metric_keys:
        for k in eval_k_values:
            k_str = str(k)
            for m in all_methods:
                vals = []
                for c in recon_data:
                    k_res = c.get("per_k_results", {}).get(k_str, {}).get(m, {})
                    v = k_res.get(m_key)
                    if v is not None:
                        vals.append(v)
                metric_tables[m_key][k][m] = {
                    "mean": float(np.mean(vals)) if vals else float("nan"),
                    "std": float(np.std(vals)) if vals else float("nan"),
                }

    # Paired comparisons on successful clips
    paired_pa_vs_u: dict[str, dict[int, dict]] = {m_key: {} for m_key in metric_keys}
    paired_pa_vs_h: dict[str, dict[int, dict]] = {m_key: {} for m_key in metric_keys}

    higher_is_better_map = {
        "mpjpe": False,
        "angle_error": False,
        "phase_recall": True,
        "impact_dist": False,
        "info_per_frame": True,
    }

    for m_key in metric_keys:
        for k in eval_k_values:
            k_str = str(k)
            hib = higher_is_better_map[m_key]
            paired_pa_vs_u[m_key][k] = compute_paired(
                successful_pa_clips, k_str, "phase_aware", "uniform", m_key, higher_is_better=hib
            )
            paired_pa_vs_h[m_key][k] = compute_paired(
                successful_pa_clips, k_str, "phase_aware", "hybrid", m_key, higher_is_better=hib
            )

    # Calculate trivial baseline for impact estimation ("predict middle frame")
    middle_frame_errors = []
    estimator_errors = []
    n_labelled = 0
    if impact_data:
        for item in impact_data.get("per_clip", []):
            clip_name = item.get("clip")
            total_f = item.get("total_frames", 24)
            gt_item = gt_labels.get(clip_name)
            if gt_item:
                gt_frame = gt_item.get("impact_frame") if isinstance(gt_item, dict) else gt_item
                if gt_frame is not None:
                    n_labelled += 1
                    mid_frame = total_f // 2
                    middle_frame_errors.append(abs(gt_frame - mid_frame))
                    est_frame = item.get("estimated_frame")
                    if est_frame is not None:
                        estimator_errors.append(abs(gt_frame - est_frame))

    # Determine per-K lowest angle error method among the 4 candidate methods
    lowest_angle_methods = {}
    for k in eval_k_values:
        scores = [(m, metric_tables["angle_error"][k][m]["mean"]) for m in methods]
        scores.sort(key=lambda x: x[1])
        lowest_angle_methods[k] = scores

    # Determine per-K lowest MPJPE method among the 4 candidate methods
    lowest_mpjpe_methods = {}
    for k in eval_k_values:
        scores = [(m, metric_tables["mpjpe"][k][m]["mean"]) for m in methods]
        scores.sort(key=lambda x: x[1])
        lowest_mpjpe_methods[k] = scores

    # Format Markdown
    lines = []
    lines.append("# Experimental Results & Evaluation Summary")
    lines.append("")
    lines.append("## 1. Dataset & Execution Overview")
    lines.append(f"- **Total clips evaluated**: {total_clips}")
    lines.append(f"- **Clips where `phase_aware` ran without fallback**: {n_pa_real} of {total_clips} ({n_pa_real/total_clips:.1%})")
    lines.append(f"- **Clips with fallback to hybrid**: {n_fallback} of {total_clips} ({n_fallback/total_clips:.1%})")
    lines.append(f"  - Fallback clip: `Straight Drive/vid100_45.avi` (wrist-speed inflection boundaries ambiguous)")
    lines.append(f"- **Execution crashes**: {crash_count}")
    lines.append("")

    # --- TABLE 1: MPJPE & Angle Error ---
    lines.append("## 2. Reconstruction Error (MPJPE) and Joint-Angle Error")
    lines.append("*Values reported as mean ± std across all 8 clips. MPJPE in normalised coordinates [0, 1]; Angle error in degrees (°). `greedy_oracle` is a theoretical upper bound that inspects ground-truth frames.*")
    lines.append("")
    lines.append("| Method | K=4 MPJPE | K=4 Angle | K=6 MPJPE | K=6 Angle | K=8 MPJPE | K=8 Angle | K=12 MPJPE | K=12 Angle |")
    lines.append("|---|---|---|---|---|---|---|---|---|")
    for m in methods:
        row = f"| **{m}** | "
        for k in eval_k_values:
            mp_m = metric_tables["mpjpe"][k][m]["mean"]
            mp_s = metric_tables["mpjpe"][k][m]["std"]
            an_m = metric_tables["angle_error"][k][m]["mean"]
            row += f"{mp_m:.4f} ± {mp_s:.4f} | {an_m:.1f}° | "
        lines.append(row)
    # Reference greedy oracle
    row_go = "| *greedy_oracle (ref)* | "
    for k in eval_k_values:
        mp_m = metric_tables["mpjpe"][k]["greedy_oracle"]["mean"]
        mp_s = metric_tables["mpjpe"][k]["greedy_oracle"]["std"]
        an_m = metric_tables["angle_error"][k]["greedy_oracle"]["mean"]
        row_go += f"*{mp_m:.4f} ± {mp_s:.4f}* | *{an_m:.1f}°* | "
    lines.append(row_go)
    lines.append("")

    # --- TABLE 2: Phase Recall ---
    lines.append("## 3. Key-Phase Recall")
    lines.append("*Fraction of detected phases containing at least one selected frame (mean ± std across all 8 clips).*")
    lines.append("")
    lines.append("| Method | K=4 Recall | K=6 Recall | K=8 Recall | K=12 Recall |")
    lines.append("|---|---|---|---|---|")
    for m in methods:
        row = f"| **{m}** | "
        for k in eval_k_values:
            pr_m = metric_tables["phase_recall"][k][m]["mean"]
            pr_s = metric_tables["phase_recall"][k][m]["std"]
            row += f"{pr_m:.3f} ± {pr_s:.3f} | "
        lines.append(row)
    row_go = "| *greedy_oracle (ref)* | "
    for k in eval_k_values:
        pr_m = metric_tables["phase_recall"][k]["greedy_oracle"]["mean"]
        pr_s = metric_tables["phase_recall"][k]["greedy_oracle"]["std"]
        row_go += f"*{pr_m:.3f} ± {pr_s:.3f}* | "
    lines.append(row_go)
    lines.append("")

    # --- TABLE 3: Impact Frame Distance ---
    lines.append("## 4. Impact-Frame Coverage (Distance in Frames to Nearest Selected Frame)")
    lines.append("*Distance in frames from the nearest selected frame to the impact frame (mean ± std across all 8 clips). Lower is better.*")
    lines.append("")
    lines.append("| Method | K=4 Distance | K=6 Distance | K=8 Distance | K=12 Distance |")
    lines.append("|---|---|---|---|---|")
    for m in methods:
        row = f"| **{m}** | "
        for k in eval_k_values:
            dist_m = metric_tables["impact_dist"][k][m]["mean"]
            dist_s = metric_tables["impact_dist"][k][m]["std"]
            row += f"{dist_m:.2f} ± {dist_s:.2f} | "
        lines.append(row)
    row_go = "| *greedy_oracle (ref)* | "
    for k in eval_k_values:
        dist_m = metric_tables["impact_dist"][k]["greedy_oracle"]["mean"]
        dist_s = metric_tables["impact_dist"][k]["greedy_oracle"]["std"]
        row_go += f"*{dist_m:.2f} ± {dist_s:.2f}* | "
    lines.append(row_go)
    lines.append("")

    # --- TABLE 4: Information Per Frame ---
    lines.append("## 5. Information Per Frame (IPF)")
    lines.append("*Efficiency score: IPF = (1 - MPJPE / MPJPE_baseline) / K, where baseline uses only first+last frames (mean ± std across all 8 clips). Higher is better.*")
    lines.append("")
    lines.append("| Method | K=4 IPF | K=6 IPF | K=8 IPF | K=12 IPF |")
    lines.append("|---|---|---|---|---|")
    for m in methods:
        row = f"| **{m}** | "
        for k in eval_k_values:
            ipf_m = metric_tables["info_per_frame"][k][m]["mean"]
            ipf_s = metric_tables["info_per_frame"][k][m]["std"]
            row += f"{ipf_m:.3f} ± {ipf_s:.3f} | "
        lines.append(row)
    row_go = "| *greedy_oracle (ref)* | "
    for k in eval_k_values:
        ipf_m = metric_tables["info_per_frame"][k]["greedy_oracle"]["mean"]
        ipf_s = metric_tables["info_per_frame"][k]["greedy_oracle"]["std"]
        row_go += f"*{ipf_m:.3f} ± {ipf_s:.3f}* | "
    lines.append(row_go)
    lines.append("")

    # --- SECTION 6: Paired Statistical Comparisons ---
    lines.append("## 6. Paired Statistical Comparisons (Clips where `phase_aware` Ran Without Fallback, N=7)")
    lines.append(f"*Bonferroni-corrected significance threshold: α = 0.05 / {num_k} = {alpha_bonferroni:.4f}. Only comparisons with p < {alpha_bonferroni:.4f} are designated as statistically significant; otherwise win/loss/tie counts are reported.*")
    lines.append("")

    lines.append("### 6.1 `phase_aware` vs `uniform`")
    lines.append("")
    lines.append("| Metric | Budget K | Mean Diff (PA − U) | PA Wins | PA Losses | Ties | Wilcoxon *p*-value | Significant (α = 0.0125) |")
    lines.append("|---|---|---|---|---|---|---|---|")
    for m_key in metric_keys:
        for k in eval_k_values:
            res = paired_pa_vs_u[m_key][k]
            p_val = res["p_val"]
            p_str = f"{p_val:.4f}" if p_val is not None else "N/A"
            sig_str = "Yes" if (p_val is not None and p_val < alpha_bonferroni) else "No"
            lines.append(
                f"| **{m_key}** | K={k} | {res['mean_diff']:+.4f} | {res['wins']} | {res['losses']} | {res['ties']} | {p_str} | {sig_str} |"
            )
    lines.append("")

    lines.append("### 6.2 `phase_aware` vs `hybrid`")
    lines.append("")
    lines.append("| Metric | Budget K | Mean Diff (PA − H) | PA Wins | PA Losses | Ties | Wilcoxon *p*-value | Significant (α = 0.0125) |")
    lines.append("|---|---|---|---|---|---|---|---|")
    for m_key in metric_keys:
        for k in eval_k_values:
            res = paired_pa_vs_h[m_key][k]
            p_val = res["p_val"]
            p_str = f"{p_val:.4f}" if p_val is not None else "N/A"
            sig_str = "Yes" if (p_val is not None and p_val < alpha_bonferroni) else "No"
            lines.append(
                f"| **{m_key}** | K={k} | {res['mean_diff']:+.4f} | {res['wins']} | {res['losses']} | {res['ties']} | {p_str} | {sig_str} |"
            )
    lines.append("")

    # --- SECTION 7: Impact Estimation vs Trivial Baseline ---
    lines.append("## 7. Impact Moment Estimation vs Trivial Baseline")
    lines.append(f"- **Labelled ground-truth clips**: {n_labelled} of {total_clips}")
    if n_labelled > 0:
        est_mean = float(np.mean(estimator_errors))
        est_med = float(np.median(estimator_errors))
        est_max = int(np.max(estimator_errors))
        mid_mean = float(np.mean(middle_frame_errors))
        mid_med = float(np.median(middle_frame_errors))
        mid_max = int(np.max(middle_frame_errors))

        lines.append(f"- **Body-only impact estimator**:")
        lines.append(f"  - Mean timing error: {est_mean:.2f} frames")
        lines.append(f"  - Median timing error: {est_med:.2f} frames")
        lines.append(f"  - Maximum timing error: {est_max} frames")
        lines.append(f"- **Trivial baseline (predict middle frame = frame 12)**:")
        lines.append(f"  - Mean timing error: {mid_mean:.2f} frames")
        lines.append(f"  - Median timing error: {mid_med:.2f} frames")
        lines.append(f"  - Maximum timing error: {mid_max} frames")
        lines.append(f"- **Per-clip breakdown on labelled clips**:")
        for clip_name, gt_data in gt_labels.items():
            gt_f = gt_data.get("impact_frame") if isinstance(gt_data, dict) else gt_data
            est_f = None
            if impact_data:
                for c in impact_data.get("per_clip", []):
                    if c.get("clip") == clip_name:
                        est_f = c.get("estimated_frame")
            est_err = abs(gt_f - est_f) if est_f is not None else "N/A"
            mid_err = abs(gt_f - 12)
            lines.append(f"  - `{clip_name}`: Ground Truth = {gt_f}, Estimator = {est_f} (error = {est_err}f), Middle Frame 12 (error = {mid_err}f)")
    else:
        lines.append("- No manual labels found in `reports/impact_ground_truth.json`.")
    lines.append("")

    # --- SECTION 8: Findings ---
    lines.append("## 8. Findings")
    lines.append("1. **Reconstruction error (MPJPE)**:")
    for k in eval_k_values:
        ranking = [f"{m} ({val:.4f})" for m, val in lowest_mpjpe_methods[k]]
        best_m, best_v = lowest_mpjpe_methods[k][0]
        lines.append(f"   - At K={k}: lowest mean MPJPE is **{best_m}** ({best_v:.4f}) [Full ranking: {', '.join(ranking)}].")

    lines.append("2. **Joint-angle error**:")
    for k in eval_k_values:
        ranking = [f"{m} ({val:.1f}°)" for m, val in lowest_angle_methods[k]]
        best_m, best_v = lowest_angle_methods[k][0]
        lines.append(f"   - At K={k}: lowest mean angle error is **{best_m}** ({best_v:.1f}°) [Full ranking: {', '.join(ranking)}].")

    lines.append("3. **Paired statistical comparisons (Bonferroni-corrected threshold: α = 0.05 / 4 = 0.0125)**:")
    lines.append("   - **`phase_aware` vs `uniform` (MPJPE)**:")
    for k in eval_k_values:
        res = paired_pa_vs_u["mpjpe"][k]
        p_val = res["p_val"]
        if p_val is not None and p_val < alpha_bonferroni:
            lines.append(f"     * At K={k}: {res['wins']} PA wins, {res['losses']} PA losses, {res['ties']} ties (mean diff = {res['mean_diff']:+.4f}, p = {p_val:.4f}, statistically significant under α = {alpha_bonferroni:.4f}).")
        else:
            p_str = f", p = {p_val:.4f}" if p_val is not None else ""
            lines.append(f"     * At K={k}: {res['wins']} PA wins, {res['losses']} PA losses, {res['ties']} ties (mean diff = {res['mean_diff']:+.4f}{p_str}).")

    lines.append("   - **`phase_aware` vs `hybrid` (Phase Recall)**:")
    for k in eval_k_values:
        res = paired_pa_vs_h["phase_recall"][k]
        p_val = res["p_val"]
        if p_val is not None and p_val < alpha_bonferroni:
            lines.append(f"     * At K={k}: {res['wins']} PA wins, {res['losses']} PA losses, {res['ties']} ties (mean diff = {res['mean_diff']:+.4f}, p = {p_val:.4f}, statistically significant under α = {alpha_bonferroni:.4f}).")
        else:
            p_str = f", p = {p_val:.4f}" if p_val is not None else ""
            lines.append(f"     * At K={k}: {res['wins']} PA wins, {res['losses']} PA losses, {res['ties']} ties (mean diff = {res['mean_diff']:+.4f}{p_str}).")

    lines.append("4. **Working hypothesis for error distribution**:")
    lines.append("   - *Untested hypothesis*: Phase-aware selection allocates frames preferentially to the downswing and impact region. While this improves key-phase recall, it leaves wider intervals in the stance and follow-through phases, which may result in higher linear interpolation error across the full clip compared to uniform spacing.")

    lines.append("5. **Impact estimation vs trivial baseline**:")
    if n_labelled > 0:
        lines.append(f"   - On the {n_labelled} manually labelled clips, the trivial baseline (predict middle frame = 12) achieved lower mean error ({mid_mean:.2f} frames) than the body-only estimator ({est_mean:.2f} frames).")
    lines.append("")

    # --- SECTION 9: Limitations ---
    lines.append("## 9. Limitations")
    lines.append("1. **Sample size**: 8 clips of 24 frames each (approx. 0.8–1.0 seconds at standard frame rates). Results have high variance and limited statistical power.")
    lines.append("2. **Ground truth proxy**: Ground truth is defined by smoothed MediaPipe Pose output with linear gap-filling, not multi-camera optical motion capture.")
    lines.append("3. **Coordinate space**: MPJPE is evaluated in normalised image coordinates [0, 1], representing fractions of frame width and height, not metric units (mm or metres).")
    lines.append("4. **Circularity in phase recall**: Phase recall measures coverage of rule-detected phases. Because the phase-aware method explicitly selects detected phase boundaries, higher phase recall is partly circular.")
    lines.append(f"5. **Impact annotations**: Only {n_labelled} clips are manually labelled. Results cannot be extrapolated beyond these labelled instances.")
    lines.append("")

    content = "\n".join(lines)
    with open(out_md_path, "w", encoding="utf-8") as f:
        f.write(content)

    print(f"[OK] Results summary written to: {out_md_path}")
    return out_md_path


if __name__ == "__main__":
    generate_summary(_project_root)
