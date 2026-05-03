#!/usr/bin/env python3
"""
Occupancy map and supporting classes for SDSM consistency defense.

OccupancyMap      — 2-D cell grid that records which vehicles are reported at
                    each cell and by which CAV.
TrustScoreManager — per-CAV trust scores with penalty + recovery.
OccupancyMapViz   — encodes occupancy-map snapshots directly to MP4 via
                    cv2.VideoWriter (no per-frame PNG files are created).
"""

import os
import time
import numpy as np
from collections import defaultdict


# ---------------------------------------------------------------------------
# OccupancyMap
# ---------------------------------------------------------------------------

class OccupancyMap:
    """
    2-D cell grid centred on a junction.

    Grid covers map_size × map_size metres, divided into cell_size × cell_size
    metre cells.  For the default 400 m / 2 m settings this yields 200 × 200
    = 40 000 cells.

    The grid is reset at the start of every defense step via reset().
    """

    def __init__(self, cx: float, cy: float,
                 map_size: float = 400.0, cell_size: float = 2.0):
        """
        cx, cy      : world (x, y) coordinates of the intersection centre
        map_size    : side length of the square coverage area in metres
        cell_size   : side length of one cell in metres
        """
        self.cx = cx
        self.cy = cy
        self.half = map_size / 2.0
        self.cell_size = cell_size
        self.n = int(map_size / cell_size)

        # cell (row, col) → set of vehicle IDs reported at that cell
        self.occupied_by: dict = defaultdict(set)
        # cell (row, col) → set of CAV IDs that claimed a vehicle at that cell
        self.detected_by: dict = defaultdict(set)

    def reset(self):
        self.occupied_by.clear()
        self.detected_by.clear()

    def _cell(self, x: float, y: float):
        """Convert world (x, y) to grid (row, col).  Returns None if out of range."""
        col = int((x - self.cx + self.half) / self.cell_size)
        row = int((y - self.cy + self.half) / self.cell_size)
        if 0 <= row < self.n and 0 <= col < self.n:
            return (row, col)
        return None

    def cell_center(self, row: int, col: int):
        """Return world (x, y) of the centre of cell (row, col)."""
        x = self.cx - self.half + (col + 0.5) * self.cell_size
        y = self.cy - self.half + (row + 0.5) * self.cell_size
        return x, y

    def mark(self, veh_id: str, x: float, y: float, cav_id: str):
        """Record that cav_id claims to have detected veh_id at world position (x, y)."""
        cell = self._cell(x, y)
        if cell is not None:
            self.occupied_by[cell].add(veh_id)
            self.detected_by[cell].add(cav_id)

    def expected_detectors(self, cell: tuple,
                            cav_positions: dict,
                            detect_range: float) -> set:
        """
        Return the set of CAV IDs whose position is within detect_range metres of
        the centre of cell.  These are the CAVs that *should* be able to detect any
        vehicle at that cell.

        A half-cell buffer (cell_size / 2) is added to detect_range to prevent
        spurious Type 1 penalties caused by the mismatch between the actual vehicle
        position (used in build_real_sdsm) and the cell centre (used here): a vehicle
        at the far edge of a cell can be up to cell_size*sqrt(2)/2 ≈ 1.4 m away from
        the centre, which at the boundary of detect_range would otherwise flag a
        legitimately reporting CAV as out-of-range.
        """
        cx, cy = self.cell_center(*cell)
        effective_range = detect_range + self.cell_size
        return {
            cav_id
            for cav_id, (px, py) in cav_positions.items()
            if np.hypot(px - cx, py - cy) <= effective_range
        }


# ---------------------------------------------------------------------------
# TrustScoreManager
# ---------------------------------------------------------------------------

class TrustScoreManager:
    """
    Maintains a trust score in [0, 1] for every known CAV.

    Each defense step:
      - suspicious reporters are penalized (score decreases)
      - all CAVs recover slightly toward 1.0
    """

    def __init__(self, penalty: float = 0.3, recovery: float = 0.05):
        """
        penalty  : score reduction per conflict association (full penalty).
                   Reduced from 0.5 → 0.3 so a single ambiguous cell does not
                   immediately halve a real CAV's trust.
        recovery : score increase per step toward 1.0 (only for non-penalized CAVs).
                   Reduced from 0.1 → 0.05 to keep fake-CAV scores suppressed longer
                   after they are correctly identified.
        """
        self.scores: dict = {}
        self.penalty = penalty
        self.recovery = recovery

    def ensure(self, cav_id: str):
        if cav_id not in self.scores:
            self.scores[cav_id] = 1.0

    def penalize(self, cav_id: str, factor: float = 1.0):
        self.ensure(cav_id)
        self.scores[cav_id] = max(0.0, self.scores[cav_id] - self.penalty * factor)

    def recover_all(self, exclude: set = None):
        """Recover all CAVs not in `exclude` (those not penalized this step)."""
        for cid in self.scores:
            if exclude is None or cid not in exclude:
                self.scores[cid] = min(1.0, self.scores[cid] + self.recovery)

    def get(self, cav_id: str, default: float = 1.0) -> float:
        return self.scores.get(cav_id, default)


# ---------------------------------------------------------------------------
# OccupancyMapViz
# ---------------------------------------------------------------------------

class OccupancyMapViz:
    """
    Encodes occupancy-map snapshots directly into an MP4 video via
    cv2.VideoWriter.  No per-frame PNG files are written to disk.

    Usage:
        viz = OccupancyMapViz(tsc_id='62477148')
        # ... inside simulation loop ...
        viz.render(omap, cav_positions, trust_scores, fake_cav_id,
                   fake_veh_weight, timestep=t)
        # ... at simulation end ...
        viz.finalize()
    """

    def __init__(self, tsc_id: str, out_dir: str = 'exp_log',
                 fps: int = 5, figsize=(8, 8)):
        """
        tsc_id  : used in the output filename
        out_dir : directory for the MP4 file
        fps     : playback frame rate of the output video
        figsize : matplotlib figure size in inches
        """
        os.makedirs(out_dir, exist_ok=True)
        ts = time.strftime('%Y%m%d_%H%M%S')
        self.out_path = os.path.join(out_dir, f'omap_{tsc_id}_{ts}.mp4')
        self.fps = fps
        self.figsize = figsize
        self.writer = None  # lazily initialized on first render() call

    def render(self, omap: OccupancyMap,
               cav_positions: dict,
               trust_scores: dict,
               fake_cav_id,
               fake_veh_weight: float,
               timestep: int):
        """
        Draw one occupancy-map frame and write it to the video.

        Colour coding:
          black  — empty cells (background)
          green  — vehicle reported only by high-trust CAVs (score >= 0.5)
          red    — vehicle reported only by low-trust / fake CAV (score < 0.5)
          yellow — vehicle reported by both trusted and untrusted CAVs
          blue ● — real CAV positions
          orange ● — fake CAV position (if known)
        """
        # Import here to avoid pulling matplotlib into modules that don't need it.
        import matplotlib
        matplotlib.use('Agg')  # non-interactive; must be set before pyplot import
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches
        import cv2

        fig, ax = plt.subplots(figsize=self.figsize)

        # Build RGB grid (default black background)
        grid = np.zeros((omap.n, omap.n, 3), dtype=np.float32)

        for cell, reporters in omap.detected_by.items():
            r, c = cell
            has_trusted   = any(trust_scores.get(rid, 1.0) >= 0.5 for rid in reporters)
            has_untrusted = any(trust_scores.get(rid, 1.0) <  0.5 for rid in reporters)
            if has_trusted and has_untrusted:
                grid[r, c] = [1.0, 1.0, 0.0]   # yellow
            elif has_trusted:
                grid[r, c] = [0.0, 0.8, 0.0]   # green
            else:
                grid[r, c] = [0.9, 0.1, 0.1]   # red

        ax.imshow(grid, origin='lower', vmin=0, vmax=1)

        # Overlay CAV positions with trust score labels
        for cav_id, pos in cav_positions.items():
            px, py = pos
            col = (px - omap.cx + omap.half) / omap.cell_size
            row = (py - omap.cy + omap.half) / omap.cell_size
            score = trust_scores.get(cav_id, 1.0)
            color = 'orange' if cav_id == fake_cav_id else 'deepskyblue'
            ax.plot(col, row, 'o', color=color, markersize=8,
                    markeredgecolor='black', markeredgewidth=0.5)
            ax.text(col + 1.5, row + 1.5, f'{score:.2f}',
                    fontsize=6, color=color,
                    bbox=dict(boxstyle='round,pad=0.1', fc='black', alpha=0.5))

        # Pin axes limits BEFORE axis('off') and BEFORE tight_layout so nothing
        # can override them.  Fixed margins (subplots_adjust) replace tight_layout
        # to prevent per-frame margin shifts caused by varying text bbox sizes.
        ax.set_xlim(-0.5, omap.n - 0.5)
        ax.set_ylim(-0.5, omap.n - 0.5)

        fake_score = trust_scores.get(fake_cav_id, 1.0) if fake_cav_id else 1.0
        ax.set_title(
            f't={timestep}   fake_weight={fake_veh_weight:.2f}'
            f'   fake_trust={fake_score:.2f}',
            fontsize=10)
        ax.axis('off')

        patches = [
            mpatches.Patch(color='green',      label='real (trusted)'),
            mpatches.Patch(color='red',        label='fake (low trust)'),
            mpatches.Patch(color='yellow',     label='mixed'),
            mpatches.Patch(color='deepskyblue', label='real CAV'),
            mpatches.Patch(color='orange',     label='fake CAV'),
        ]
        ax.legend(handles=patches, loc='lower right', fontsize=7)

        # Fixed margins — never recalculated, so axes area is identical every frame.
        fig.subplots_adjust(left=0.02, right=0.98, top=0.94, bottom=0.02)
        fig.canvas.draw()
        w, h = fig.canvas.get_width_height()
        buf = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        frame_rgb = buf.reshape(h, w, 3)
        frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
        plt.close(fig)

        # Lazy-init VideoWriter (frame size known only after first render)
        if self.writer is None:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            self.writer = cv2.VideoWriter(self.out_path, fourcc, self.fps, (w, h))

        self.writer.write(frame_bgr)

    def finalize(self):
        """Flush and close the video file.  Safe to call multiple times."""
        if self.writer is not None:
            self.writer.release()
            self.writer = None
            print(f'[OccupancyMapViz] Video saved → {self.out_path}')
        else:
            print('[OccupancyMapViz] No frames were rendered — video not created.'
                  ' (Check: cv2 installed? sdsm_defense active? fake vehicles generated?)')
