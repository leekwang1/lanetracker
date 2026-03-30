from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from ..core.types import SeedProfile, StripeCandidate
from .cross_section_profile import CrossSectionProfile
from .lane_state import LaneState


@dataclass
class CrossSectionAnalyzerV2Config:
    along_half_m: float = 0.15
    lateral_half_m: float = 0.60
    bin_size_m: float = 0.01
    min_points_total: int = 10
    min_points_section: int = 6
    max_z_step_m: float = 0.12
    smooth_kernel_size: int = 5
    peak_min_abs: float = 0.05
    peak_min_prominence: float = 0.02
    stripe_threshold_ratio: float = 0.45
    lane_width_min_m: float = 0.05
    lane_width_max_m: float = 0.28
    width_sigma_m: float = 0.04
    center_sigma_m: float = 0.06
    edge_sigma_m: float = 0.05
    score_peak_weight: float = 1.2
    score_prominence_weight: float = 0.8
    score_support_weight: float = 0.8
    score_width_consistency_weight: float = 1.2
    score_center_consistency_weight: float = 1.4
    score_edge_consistency_weight: float = 1.1
    score_identity_weight: float = 1.4
    score_signal_consistency_weight: float = 1.35
    score_switch_penalty_weight: float = 1.3
    score_abnormal_width_penalty_weight: float = 0.8
    signal_target_sigma_floor: float = 2500.0
    signal_contrast_sigma_floor: float = 1800.0
    signal_overbright_sigma_scale: float = 0.35
    signal_overcontrast_sigma_scale: float = 0.30
    gap_peak_relax_factor: float = 0.75
    gap_stripe_threshold_factor: float = 0.75
    gap_center_prior_boost: float = 1.25
    gap_width_prior_boost: float = 1.20
    crosswalk_multi_stripe_count: int = 3


class CrossSectionAnalyzerV2:
    def __init__(self, cfg: CrossSectionAnalyzerV2Config):
        self.cfg = cfg

    def analyze(
        self,
        xyz: np.ndarray,
        intensity: np.ndarray,
        indices: np.ndarray,
        center_xyz: np.ndarray,
        tangent_xy: np.ndarray,
        prev_state: LaneState | None,
        seed_profile: SeedProfile | None,
        is_gap_mode: bool = False,
    ) -> CrossSectionProfile:
        if indices.size < self.cfg.min_points_total:
            return self._empty_profile("too_few_local_points")

        local_xyz = xyz[indices]
        local_i = intensity[indices].astype(np.float64)
        z_ref = float(seed_profile.z_ref) if seed_profile is not None else float(center_xyz[2])
        z_mask = np.abs(local_xyz[:, 2] - z_ref) <= self.cfg.max_z_step_m
        if np.count_nonzero(z_mask) < self.cfg.min_points_section:
            return self._empty_profile("too_few_z_gated_points")

        pts = local_xyz[z_mask]
        vals = local_i[z_mask]
        dir_xy = self._unit2(tangent_xy)
        normal = np.array([-dir_xy[1], dir_xy[0]], dtype=np.float64)
        deltas = pts[:, :2] - center_xyz[:2]
        along = deltas @ dir_xy
        lateral = deltas @ normal

        mask = (np.abs(along) <= self.cfg.along_half_m) & (np.abs(lateral) <= self.cfg.lateral_half_m)
        if np.count_nonzero(mask) < self.cfg.min_points_section:
            return self._empty_profile("too_few_cross_section_points")

        along = along[mask]
        lateral = lateral[mask]
        vals = vals[mask]
        pts = pts[mask]

        weights = self._compute_point_weights(vals, pts, along, lateral, prev_state, seed_profile, is_gap_mode)
        if np.sum(weights) <= 1e-12:
            return self._empty_profile("zero_weights")

        bins = np.arange(-self.cfg.lateral_half_m, self.cfg.lateral_half_m + self.cfg.bin_size_m, self.cfg.bin_size_m, dtype=np.float64)
        if bins.size < 4:
            return self._empty_profile("too_few_bins")

        hist_intensity_sum, edges = np.histogram(lateral, bins=bins, weights=weights * vals)
        hist_support, _ = np.histogram(lateral, bins=bins, weights=weights)
        hist_intensity = hist_intensity_sum / np.maximum(hist_support, 1e-12)
        hist_support_norm = hist_support / np.maximum(np.max(hist_support), 1e-12)
        hist_intensity_norm = hist_intensity / np.maximum(np.max(hist_intensity), 1e-12)
        hist_combined = 0.65 * hist_intensity_norm + 0.35 * hist_support_norm
        smooth_hist = self._smooth(hist_combined)

        peak_min_abs = self.cfg.peak_min_abs * (self.cfg.gap_peak_relax_factor if is_gap_mode else 1.0)
        peak_indices = self._find_peaks(smooth_hist, peak_min_abs, self.cfg.peak_min_prominence)
        stripe_candidates = self._build_stripe_candidates(
            edges,
            smooth_hist,
            hist_support,
            peak_indices,
            prev_state,
            seed_profile,
            is_gap_mode,
            lateral,
            vals,
        )
        quality = self._estimate_profile_quality(smooth_hist, stripe_candidates)
        neighbor_count = max(0, len(stripe_candidates) - 1)
        crosswalk_like_score = self._estimate_crosswalk_like_score(stripe_candidates)
        selected_idx = self._select_best_candidate(stripe_candidates, prev_state, is_gap_mode)
        switch_risk = self._estimate_switch_risk(stripe_candidates, selected_idx)

        return CrossSectionProfile(
            bins_center=0.5 * (edges[:-1] + edges[1:]),
            hist_intensity=hist_intensity,
            hist_support=hist_support_norm,
            hist_combined=hist_combined,
            smooth_hist=smooth_hist,
            peak_indices=peak_indices,
            stripe_candidates=stripe_candidates,
            selected_idx=selected_idx,
            quality=quality,
            neighbor_count=neighbor_count,
            switch_risk=switch_risk,
            crosswalk_like_score=crosswalk_like_score,
            debug={},
        )

    def _compute_point_weights(self, vals, pts, along, lateral, prev_state, seed_profile, is_gap_mode):
        if seed_profile is not None:
            bg = float(seed_profile.background_intensity)
            tgt = float(seed_profile.target_intensity)
            denom = max(tgt - bg, 1e-6)
            w_int = np.clip((vals - bg) / denom, 0.0, 1.5)
        else:
            v35 = float(np.quantile(vals, 0.35))
            v90 = float(np.quantile(vals, 0.90))
            denom = max(v90 - v35, 1e-6)
            w_int = np.clip((vals - v35) / denom, 0.0, 1.5)
        z_ref = float(seed_profile.z_ref) if seed_profile is not None else float(np.median(pts[:, 2]))
        w_z = np.exp(-np.abs(pts[:, 2] - z_ref) / 0.05)
        along_sigma = max(self.cfg.along_half_m * 0.65, 1e-6)
        w_along = np.exp(-(along / along_sigma) ** 2)
        if prev_state is not None:
            center_sigma = self.cfg.center_sigma_m / (self.cfg.gap_center_prior_boost if is_gap_mode else 1.0)
            w_lat = np.exp(-np.abs(lateral - prev_state.stripe_center_m) / max(center_sigma, 1e-6))
        else:
            w_lat = np.ones_like(lateral, dtype=np.float64)
        return np.clip(w_int * w_z * w_along * w_lat, 0.0, None)

    def _find_peaks(self, smooth_hist: np.ndarray, min_abs: float, min_prominence: float) -> list[int]:
        peaks: list[int] = []
        if smooth_hist.size < 3:
            return peaks
        for i in range(1, smooth_hist.size - 1):
            c = float(smooth_hist[i])
            if c < min_abs or c < smooth_hist[i - 1] or c < smooth_hist[i + 1]:
                continue
            left_base = float(np.min(smooth_hist[max(0, i - 3):i + 1]))
            right_base = float(np.min(smooth_hist[i:min(smooth_hist.size, i + 4)]))
            prominence = c - max(left_base, right_base)
            if prominence >= min_prominence:
                peaks.append(i)
        return peaks

    def _build_stripe_candidates(self, edges, smooth_hist, hist_support, peak_indices, prev_state, seed_profile, is_gap_mode, lateral_samples, value_samples):
        out: list[StripeCandidate] = []
        for p in peak_indices:
            peak_val = float(smooth_hist[p])
            thr_ratio = self.cfg.stripe_threshold_ratio * (
                self.cfg.gap_stripe_threshold_factor if is_gap_mode else 1.0
            )
            active_thr = peak_val * thr_ratio
            left_idx = p
            right_idx = p
            while left_idx > 0 and smooth_hist[left_idx - 1] >= active_thr:
                left_idx -= 1
            while right_idx < smooth_hist.size - 1 and smooth_hist[right_idx + 1] >= active_thr:
                right_idx += 1
            left_idx = self._refine_left_edge(smooth_hist, left_idx)
            right_idx = self._refine_right_edge(smooth_hist, right_idx)
            left_m = float(edges[left_idx])
            right_m = float(edges[right_idx + 1])
            width_m = right_m - left_m
            center_m = 0.5 * (left_m + right_m)
            if is_gap_mode and prev_state is not None:
                target_min_width = max(
                    float(self.cfg.lane_width_min_m),
                    min(float(prev_state.lane_width_m) * 0.70, float(self.cfg.lane_width_max_m)),
                )
                if width_m < target_min_width:
                    half = 0.5 * target_min_width
                    left_m = center_m - half
                    right_m = center_m + half
                    width_m = target_min_width
            abnormal_width = width_m < self.cfg.lane_width_min_m or width_m > self.cfg.lane_width_max_m
            support_count = int(np.sum(hist_support[left_idx:right_idx + 1]))
            integrated_energy = float(np.sum(smooth_hist[left_idx:right_idx + 1]))
            left_base = float(np.min(smooth_hist[max(0, p - 3):p + 1]))
            right_base = float(np.min(smooth_hist[p:min(smooth_hist.size, p + 4)]))
            prominence = peak_val - max(left_base, right_base)
            dl = p - left_idx
            dr = right_idx - p
            symmetry_score = float(np.clip(1.0 - (abs(dl - dr) / max(dl + dr, 1)), 0.0, 1.0))
            cand = StripeCandidate(
                left_m=left_m, right_m=right_m, center_m=center_m, width_m=width_m,
                peak_value=peak_val, prominence=prominence, support_count=support_count,
                integrated_energy=integrated_energy, symmetry_score=symmetry_score,
                debug={"abnormal_width": abnormal_width},
            )
            self._score_candidate_signal(cand, seed_profile, lateral_samples, value_samples)
            self._score_candidate_identity(cand, prev_state, seed_profile, is_gap_mode)
            out.append(cand)
        return out

    def _score_candidate_signal(self, cand, seed_profile, lateral_samples, value_samples):
        if seed_profile is None or lateral_samples.size == 0 or value_samples.size == 0:
            cand.signal_consistency = 0.5
            cand.debug["target_intensity_est"] = 0.0
            cand.debug["background_intensity_est"] = 0.0
            cand.debug["contrast_est"] = 0.0
            return
        stripe_mask = (lateral_samples >= cand.left_m) & (lateral_samples <= cand.right_m)
        stripe_vals = value_samples[stripe_mask]
        if stripe_vals.size == 0:
            cand.signal_consistency = 0.0
            cand.debug["target_intensity_est"] = 0.0
            cand.debug["background_intensity_est"] = 0.0
            cand.debug["contrast_est"] = 0.0
            return
        target_est = float(np.quantile(stripe_vals, 0.85))
        bg_margin = max(0.06, 0.6 * float(cand.width_m))
        bg_mask = (
            ((lateral_samples >= cand.left_m - bg_margin) & (lateral_samples < cand.left_m))
            | ((lateral_samples > cand.right_m) & (lateral_samples <= cand.right_m + bg_margin))
        )
        bg_vals = value_samples[bg_mask]
        if bg_vals.size < 4:
            outside_mask = ~stripe_mask
            bg_vals = value_samples[outside_mask]
        background_est = float(np.quantile(bg_vals, 0.35)) if bg_vals.size else float(seed_profile.background_intensity)
        contrast_est = max(target_est - background_est, 0.0)
        seed_target = float(seed_profile.target_intensity)
        seed_contrast = max(seed_target - float(seed_profile.background_intensity), 1.0)
        target_sigma = max(float(self.cfg.signal_target_sigma_floor), seed_contrast)
        contrast_sigma = max(float(self.cfg.signal_contrast_sigma_floor), seed_contrast * 0.75)
        target_term = float(np.exp(-abs(target_est - seed_target) / target_sigma))
        contrast_term = float(np.exp(-abs(contrast_est - seed_contrast) / contrast_sigma))
        overbright_sigma = max(target_sigma * float(self.cfg.signal_overbright_sigma_scale), 1.0)
        overcontrast_sigma = max(contrast_sigma * float(self.cfg.signal_overcontrast_sigma_scale), 1.0)
        overbright_penalty = float(np.exp(-max(target_est - seed_target, 0.0) / overbright_sigma))
        overcontrast_penalty = float(np.exp(-max(contrast_est - seed_contrast, 0.0) / overcontrast_sigma))
        base_consistency = 0.45 * target_term + 0.55 * contrast_term
        penalty_term = min(overbright_penalty, overcontrast_penalty)
        cand.signal_consistency = float(np.clip(base_consistency * penalty_term, 0.0, 1.0))
        cand.debug["target_intensity_est"] = target_est
        cand.debug["background_intensity_est"] = background_est
        cand.debug["contrast_est"] = contrast_est
        cand.debug["overbright_penalty"] = overbright_penalty
        cand.debug["overcontrast_penalty"] = overcontrast_penalty

    def _score_candidate_identity(self, cand, prev_state, seed_profile, is_gap_mode):
        cand.strength_score = 1.2 * cand.peak_value + 0.8 * cand.prominence + 0.5 * cand.symmetry_score
        if prev_state is None:
            cand.width_consistency = cand.center_consistency = cand.edge_consistency = cand.identity_score = 0.5
            cand.switch_penalty = 0.0
            cand.final_score = cand.strength_score + self.cfg.score_signal_consistency_weight * cand.signal_consistency
            return
        width_sigma = self.cfg.width_sigma_m / (self.cfg.gap_width_prior_boost if is_gap_mode else 1.0)
        center_sigma = self.cfg.center_sigma_m / (self.cfg.gap_center_prior_boost if is_gap_mode else 1.0)
        edge_sigma = self.cfg.edge_sigma_m
        cand.width_consistency = float(np.exp(-abs(cand.width_m - prev_state.lane_width_m) / max(width_sigma, 1e-6)))
        cand.center_consistency = float(np.exp(-abs(cand.center_m - prev_state.stripe_center_m) / max(center_sigma, 1e-6)))
        left_cons = np.exp(-abs(cand.left_m - prev_state.left_edge_m) / max(edge_sigma, 1e-6))
        right_cons = np.exp(-abs(cand.right_m - prev_state.right_edge_m) / max(edge_sigma, 1e-6))
        cand.edge_consistency = float(0.5 * (left_cons + right_cons))
        cand.identity_score = float(
            0.34 * cand.center_consistency
            + 0.28 * cand.width_consistency
            + 0.18 * cand.edge_consistency
            + 0.20 * cand.signal_consistency
        )
        lateral_jump = abs(cand.center_m - prev_state.stripe_center_m)
        width_jump = abs(cand.width_m - prev_state.lane_width_m)
        cand.switch_penalty = float(0.7 * np.clip(lateral_jump / 0.10, 0.0, 2.0) + 0.3 * np.clip(width_jump / 0.08, 0.0, 2.0))
        abnormal_width_penalty = 1.0 if cand.debug.get("abnormal_width", False) else 0.0
        cand.final_score = float(
            self.cfg.score_peak_weight * cand.peak_value +
            self.cfg.score_prominence_weight * cand.prominence +
            self.cfg.score_support_weight * min(cand.support_count / 8.0, 1.0) +
            self.cfg.score_width_consistency_weight * cand.width_consistency +
            self.cfg.score_center_consistency_weight * cand.center_consistency +
            self.cfg.score_edge_consistency_weight * cand.edge_consistency +
            self.cfg.score_identity_weight * cand.identity_score -
            self.cfg.score_switch_penalty_weight * cand.switch_penalty -
            self.cfg.score_abnormal_width_penalty_weight * abnormal_width_penalty
        )
        cand.final_score += float(self.cfg.score_signal_consistency_weight) * cand.signal_consistency

    def _select_best_candidate(self, stripe_candidates, prev_state, is_gap_mode):
        if not stripe_candidates:
            return None
        best_idx = None
        best_score = -1e18
        for i, cand in enumerate(stripe_candidates):
            score = cand.final_score
            if is_gap_mode and prev_state is not None:
                score += 0.6 * cand.center_consistency + 0.5 * cand.width_consistency
            if score > best_score:
                best_score = score
                best_idx = i
        return best_idx

    def _estimate_switch_risk(self, stripe_candidates, selected_idx):
        if selected_idx is None or len(stripe_candidates) < 2:
            return 0.0
        scores = [c.final_score for c in stripe_candidates]
        order = np.argsort(scores)[::-1]
        best = stripe_candidates[int(order[0])]
        second = stripe_candidates[int(order[1])]
        score_gap = best.final_score - second.final_score
        center_gap = abs(best.center_m - second.center_m)
        risk = 0.0
        if score_gap < 0.35:
            risk += 0.5
        if center_gap < 0.20:
            risk += 0.3
        if len(stripe_candidates) >= 3:
            risk += 0.2
        return float(np.clip(risk, 0.0, 1.0))

    def _estimate_crosswalk_like_score(self, stripe_candidates):
        if len(stripe_candidates) < self.cfg.crosswalk_multi_stripe_count:
            return 0.0
        centers = np.array([c.center_m for c in stripe_candidates], dtype=np.float64)
        centers.sort()
        diffs = np.diff(centers)
        if diffs.size == 0:
            return 0.0
        regularity = 1.0 - float(np.std(diffs) / max(np.mean(np.abs(diffs)), 1e-6))
        regularity = float(np.clip(regularity, 0.0, 1.0))
        count_score = min(len(stripe_candidates) / 5.0, 1.0)
        return float(np.clip(0.5 * count_score + 0.5 * regularity, 0.0, 1.0))

    def _estimate_profile_quality(self, smooth_hist, stripe_candidates):
        if smooth_hist.size == 0:
            return 0.0
        peak = float(np.max(smooth_hist))
        base = float(np.mean(smooth_hist))
        sep = max(0.0, peak - base)
        if not stripe_candidates:
            return float(np.clip(0.55 * sep, 0.0, 1.5))
        best = max(stripe_candidates, key=lambda c: float(c.final_score))
        peak_term = float(np.clip(best.peak_value, 0.0, 1.0))
        identity_term = float(np.clip(best.identity_score, 0.0, 1.0))
        symmetry_term = float(np.clip(best.symmetry_score, 0.0, 1.0))
        support_term = float(np.clip(best.support_count / 8.0, 0.0, 1.0))
        cand_bonus = min(len(stripe_candidates) * 0.05, 0.15)
        quality = (
            0.30 * sep
            + 0.30 * peak_term
            + 0.20 * identity_term
            + 0.10 * symmetry_term
            + 0.10 * support_term
            + cand_bonus
        )
        return float(np.clip(quality, 0.0, 1.5))

    def _unit2(self, v):
        n = float(np.linalg.norm(v[:2]))
        if n <= 1e-12:
            return np.array([1.0, 0.0], dtype=np.float64)
        return np.array([v[0] / n, v[1] / n], dtype=np.float64)

    def _smooth(self, hist):
        k = max(1, int(self.cfg.smooth_kernel_size))
        if k <= 1 or hist.size == 0:
            return hist.copy()
        kernel = np.ones((k,), dtype=np.float64) / float(k)
        return np.convolve(hist, kernel, mode="same")

    def _refine_left_edge(self, y, left_idx):
        i = left_idx
        while i > 1:
            if y[i - 1] < y[i] and y[i - 2] >= y[i - 1]:
                break
            i -= 1
        return i

    def _refine_right_edge(self, y, right_idx):
        i = right_idx
        while i < y.size - 2:
            if y[i + 1] < y[i] and y[i + 2] >= y[i + 1]:
                break
            i += 1
        return i

    def _empty_profile(self, reason):
        return CrossSectionProfile(
            bins_center=np.empty((0,), dtype=np.float64),
            hist_intensity=np.empty((0,), dtype=np.float64),
            hist_support=np.empty((0,), dtype=np.float64),
            hist_combined=np.empty((0,), dtype=np.float64),
            smooth_hist=np.empty((0,), dtype=np.float64),
            selected_idx=None,
            quality=0.0,
            neighbor_count=0,
            switch_risk=0.0,
            crosswalk_like_score=0.0,
            debug={"reason": reason},
        )
