"""
Lunar Illumination Simulator
==============================
"The Far Side Is Not The Dark Side"

Demonstrates that every square inch of the Moon's surface receives direct
sunlight roughly 50% of each lunar orbit — because the Moon rotates once
per orbit (tidal locking), its far side faces the Sun for half of every month.

Requirements:  Python 3.8+,  matplotlib,  numpy
Install:       pip install matplotlib numpy
Run:           python lunar_sim.py

Layout
------
  Top-left    : Orbital overview  (Sun / Earth / Moon)
  Top-center  : Moon surface disc — live terminator, near/far side labels
  Top-right   : Cumulative sunlight tracker (bar chart per latitude band)
  Bottom-left : Moon phase diagram
  Bottom-center: Latitude sunlight table  (instantaneous + cumulative %)
  Bottom-right : Science callout panel

Controls
--------
  Moon Axial Tilt slider  0–30°  (real value: ~1.54°; exaggerate to see polar effects)
  Speed slider            0.1–8×
  Pause / Resume button
  Reset Cumulative button
"""

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
from matplotlib.widgets import Slider, Button
from matplotlib.patches import FancyArrowPatch
from matplotlib.colors import LinearSegmentedColormap

# ── Orbital / display constants ───────────────────────────────────────────────
EARTH_ORBIT_R = 4.0
MOON_ORBIT_R  = 1.05
EARTH_RADIUS  = 0.22
MOON_RADIUS   = 0.10
SUN_RADIUS    = 0.50

MOON_TILT_REAL = 1.54   # degrees — real axial tilt of the Moon

# Latitude bands tracked (Moon latitudes)
LATITUDES = [-75, -60, -45, -30, -15, 0, 15, 30, 45, 60, 75]
N_LAT = len(LATITUDES)

# Colour maps
_DAY_NIGHT = LinearSegmentedColormap.from_list(
    "daynight", ["#0d0d1a", "#1c2a4a", "#c8a84b", "#fff2aa"], N=256
)
_MOON_SURF = LinearSegmentedColormap.from_list(
    "moonsurf", ["#111118", "#2a2a38", "#b0b0c0", "#e8e8f0"], N=256
)

# ── Colour palette ────────────────────────────────────────────────────────────
BG        = "#07071a"
PANEL_BG  = "#0d0d26"
ACCENT    = "#4a90d9"
SUN_COL   = "#FDB813"
EARTH_COL = "#1a7fd4"
MOON_COL  = "#c8c8d8"
TEXT_COL  = "#dde0f0"
DIM_TEXT  = "#6a7090"
GOLD      = "#f0c040"
RED_DARK  = "#cc3333"
GREEN     = "#44bb66"


# =============================================================================
class LunarSim:

    def __init__(self):
        self.angle_earth  = 0.0    # Earth's angle around Sun (deg)
        self.angle_moon   = 0.0    # Moon's angle around Earth (deg)
        self.moon_tilt    = MOON_TILT_REAL
        self.speed        = 1.0
        self.running      = True
        self.frame        = 0

        # Cumulative sunlight accumulators: shape (N_LAT,)
        # Each frame we add 1 if that latitude is currently lit, else 0
        self.cum_lit   = np.zeros(N_LAT, dtype=float)
        self.cum_total = np.zeros(N_LAT, dtype=float)

        self._build_figure()
        self._init_orbit_artists()
        self._connect()

    # =========================================================================
    # Figure layout
    # =========================================================================
    def _build_figure(self):
        self.fig = plt.figure(figsize=(17, 10), facecolor=BG)
        try:
            self.fig.canvas.manager.set_window_title(
                "Lunar Illumination Simulator  —  The Far Side Is Not The Dark Side"
            )
        except Exception:
            pass

        gs = gridspec.GridSpec(
            2, 3,
            figure=self.fig,
            left=0.04, right=0.98,
            top=0.92, bottom=0.13,
            wspace=0.38, hspace=0.45,
        )

        def make_ax(row, col, aspect=None):
            ax = self.fig.add_subplot(gs[row, col], facecolor=PANEL_BG)
            ax.tick_params(colors=TEXT_COL, labelsize=7)
            for spine in ax.spines.values():
                spine.set_edgecolor("#2a2a4a")
            if aspect:
                ax.set_aspect(aspect)
            return ax

        self.ax_orbit  = make_ax(0, 0, "equal")
        self.ax_moon   = make_ax(0, 1, "equal")
        self.ax_cumbar = make_ax(0, 2)
        self.ax_phase  = make_ax(1, 0, "equal")
        self.ax_table  = make_ax(1, 1)
        self.ax_info   = make_ax(1, 2)

        for ax, title in [
            (self.ax_orbit,  "Orbital Overview"),
            (self.ax_moon,   "Moon Surface Illumination"),
            (self.ax_cumbar, "Cumulative Sunlight by Latitude"),
            (self.ax_phase,  "Moon Phase  (as seen from Earth)"),
            (self.ax_table,  "Instantaneous & Cumulative Sunlight"),
            (self.ax_info,   "The Science"),
        ]:
            ax.set_title(title, color=TEXT_COL, fontsize=9, pad=6,
                         fontweight="bold")

        # Main figure title
        self.fig.text(0.5, 0.965,
                      "Lunar Illumination Simulator  ·  The Far Side Is Not The Dark Side",
                      color=GOLD, fontsize=12, ha="center", va="top", fontweight="bold")

        # ── Widgets ──────────────────────────────────────────────────────────
        def make_slider_ax(left, bottom):
            return self.fig.add_axes([left, bottom, 0.26, 0.022],
                                     facecolor="#12123a")

        def make_btn_ax(left, bottom, w=0.10):
            return self.fig.add_axes([left, bottom, w, 0.035],
                                     facecolor="#12123a")

        ax_tilt  = make_slider_ax(0.04,  0.065)
        ax_speed = make_slider_ax(0.04,  0.033)
        ax_pause = make_btn_ax(0.33,  0.043)
        ax_reset = make_btn_ax(0.45,  0.043)

        self.sl_tilt  = Slider(ax_tilt,  "Moon Axial Tilt (°)",
                               0, 30, valinit=self.moon_tilt, color=ACCENT)
        self.sl_speed = Slider(ax_speed, "Speed",
                               0.1, 8.0, valinit=self.speed, color=ACCENT)
        self.btn_pause = Button(ax_pause, "Pause",
                                color="#1e1e4a", hovercolor="#3a3a6a")
        self.btn_reset = Button(ax_reset, "Reset Cumulative",
                                color="#1e1e4a", hovercolor="#3a3a6a")

        for sl in [self.sl_tilt, self.sl_speed]:
            sl.label.set_color(TEXT_COL)
            sl.valtext.set_color(ACCENT)

        for btn in [self.btn_pause, self.btn_reset]:
            btn.label.set_color(TEXT_COL)

        # Tilt reference note
        self.fig.text(0.04, 0.018,
                      f"Real lunar axial tilt: {MOON_TILT_REAL}°  "
                      "— drag slider right to exaggerate and see polar shadow effects",
                      color=DIM_TEXT, fontsize=7, style="italic")

    # =========================================================================
    # Orbital panel static artists
    # =========================================================================
    def _init_orbit_artists(self):
        ax = self.ax_orbit
        ax.axis("off")
        lim = EARTH_ORBIT_R + MOON_ORBIT_R + 0.6
        ax.set_xlim(-lim, lim)
        ax.set_ylim(-lim, lim)

        th = np.linspace(0, 2*np.pi, 360)
        ax.plot(EARTH_ORBIT_R*np.cos(th), EARTH_ORBIT_R*np.sin(th),
                color="#ffffff18", lw=0.7, zorder=1)

        # Sun + glow
        for r, a in [(1.0, 0.04), (0.75, 0.07), (0.5, 0.12)]:
            ax.add_patch(plt.Circle((0, 0), SUN_RADIUS+r,
                                    color=SUN_COL, alpha=a, zorder=2))
        ax.add_patch(plt.Circle((0, 0), SUN_RADIUS, color=SUN_COL, zorder=3))
        ax.text(0, -SUN_RADIUS-0.35, "Sun", color=SUN_COL,
                fontsize=7, ha="center", zorder=3)

        # Earth (static circle; position updated per frame)
        self.earth_patch = plt.Circle((EARTH_ORBIT_R, 0), EARTH_RADIUS,
                                      color=EARTH_COL, zorder=5)
        ax.add_patch(self.earth_patch)

        # Moon orbit ring (ellipse, updated per frame)
        self.moon_orbit_line, = ax.plot([], [], color="#ffffff28", lw=0.6, zorder=4)

        # Moon
        self.moon_patch = plt.Circle((0, 0), MOON_RADIUS,
                                     color=MOON_COL, zorder=6)
        ax.add_patch(self.moon_patch)

        # Near-side indicator line (Moon→Earth)
        self.nearside_line, = ax.plot([], [], color="#ffff0066",
                                      lw=1.2, zorder=7)

        # Sun ray arrow toward Earth
        self.sun_ray = FancyArrowPatch((0, 0), (1, 0),
                                       arrowstyle="-|>", color=SUN_COL+"88",
                                       mutation_scale=10, lw=1, zorder=7)
        ax.add_patch(self.sun_ray)

        # Orbit info text
        self.orbit_info = ax.text(
            -lim+0.15, lim-0.2, "", color=TEXT_COL, fontsize=7, va="top",
            bbox=dict(facecolor="#00000077", pad=4, edgecolor="none"))

        ax.legend(handles=[
            mpatches.Patch(color=SUN_COL,   label="Sun"),
            mpatches.Patch(color=EARTH_COL, label="Earth"),
            mpatches.Patch(color=MOON_COL,  label="Moon"),
            mpatches.Patch(color="#ffff00", label="Near side →Earth"),
        ], loc="lower right", facecolor="#10103a", edgecolor="#3a3a6a",
           labelcolor=TEXT_COL, fontsize=6.5)

    # =========================================================================
    # Widget callbacks
    # =========================================================================
    def _connect(self):
        self.sl_tilt.on_changed(lambda v: setattr(self, "moon_tilt", v))
        self.sl_speed.on_changed(lambda v: setattr(self, "speed", v))
        self.btn_pause.on_clicked(self._toggle_pause)
        self.btn_reset.on_clicked(self._reset_cumulative)
        self._timer = self.fig.canvas.new_timer(interval=40)
        self._timer.add_callback(self._step)
        self._timer.start()

    def _toggle_pause(self, _):
        self.running = not self.running
        self.btn_pause.label.set_text("Resume" if not self.running else "Pause")
        self.fig.canvas.draw_idle()

    def _reset_cumulative(self, _):
        self.cum_lit[:]   = 0.0
        self.cum_total[:] = 0.0
        self.frame = 0

    # =========================================================================
    # Geometry
    # =========================================================================
    def _positions(self):
        """Return Earth world-pos and Moon offset from Earth (3-D)."""
        ea = np.radians(self.angle_earth)
        ex = EARTH_ORBIT_R * np.cos(ea)
        ey = EARTH_ORBIT_R * np.sin(ea)

        ma  = np.radians(self.angle_moon)
        inc = np.radians(self.moon_tilt)
        mx = MOON_ORBIT_R * np.cos(ma)
        my = MOON_ORBIT_R * np.sin(ma) * np.cos(inc)
        mz = MOON_ORBIT_R * np.sin(ma) * np.sin(inc)
        return ex, ey, mx, my, mz

    def _moon_sun_angle(self, ex, ey, mx, my):
        """
        Phase angle of the Moon:
        angle between the Sun-Moon vector and the Earth-Moon vector.
        0° = new moon (Sun behind Moon as seen from Earth)
        180° = full moon
        """
        # Vector from Moon to Sun
        moon_wx = ex + mx
        moon_wy = ey + my
        v_ms = np.array([-moon_wx, -moon_wy])   # Moon→Sun
        v_me = np.array([-mx, -my])              # Moon→Earth
        cos_a = np.dot(v_ms, v_me) / (
            np.linalg.norm(v_ms) * np.linalg.norm(v_me) + 1e-12)
        return np.degrees(np.arccos(np.clip(cos_a, -1, 1)))

    def _sun_direction_on_moon(self, ex, ey, mx, my):
        """Unit vector pointing FROM Moon TOWARD Sun (in 2-D display plane)."""
        moon_wx = ex + mx
        moon_wy = ey + my
        v = np.array([-moon_wx, -moon_wy])
        return v / (np.linalg.norm(v) + 1e-12)

    def _lit_fraction_moon_lat(self, lat_deg, sun_dir_2d):
        """
        Fraction of the current moment that this Moon latitude is in sunlight.
        We treat the instantaneous terminator:
          a latitude band at lat_deg is on the lit hemisphere if the
          component of the sun direction along the equatorial plane
          (dot with the local surface normal projected) > 0, weighted
          by the tilt.

        For the table we compute INSTANTANEOUS fraction:
          = fraction of the latitude circle currently in sunlight.
        """
        lat   = np.radians(lat_deg)
        tilt  = np.radians(self.moon_tilt)

        # Sun elevation angle at this latitude given tilt and current geometry
        # Sun direction angle in the Moon's frame
        sun_ang = np.arctan2(sun_dir_2d[1], sun_dir_2d[0])

        # Effective solar declination on the Moon (proxy via tilt × sin(season))
        decl = tilt * np.sin(sun_ang)

        cos_ha = -np.tan(lat) * np.tan(decl)
        if   cos_ha <= -1.0: return 1.0
        elif cos_ha >=  1.0: return 0.0
        else:
            return np.arccos(cos_ha) / np.pi

    def _is_lat_lit_now(self, lat_deg, sun_dir_2d):
        """
        Boolean: is the sub-solar point on this latitude band right now?
        We check the instantaneous illumination of the band's central point
        facing the Sun.  Used for cumulative accumulation.
        """
        lat  = np.radians(lat_deg)
        tilt = np.radians(self.moon_tilt)
        sun_ang = np.arctan2(sun_dir_2d[1], sun_dir_2d[0])
        decl = tilt * np.sin(sun_ang)
        # Sub-solar latitude
        sub_solar_lat = decl
        # Is sun above horizon at this lat for ANY longitude? (i.e. band has ANY daylight)
        return abs(lat - sub_solar_lat) < (np.pi/2)

    # =========================================================================
    # Main animation step
    # =========================================================================
    def _step(self):
        if not self.running:
            return

        dt = self.speed
        self.angle_earth = (self.angle_earth + 0.25 * dt) % 360
        # Moon orbits ~13× faster than Earth around Sun in reality;
        # for illustration we make it visually faster than Earth's orbit
        self.angle_moon  = (self.angle_moon  + 2.5  * dt) % 360
        self.frame += 1

        ex, ey, mx, my, mz = self._positions()
        sun_dir = self._sun_direction_on_moon(ex, ey, mx, my)

        # Accumulate cumulative sunlight
        for i, lat in enumerate(LATITUDES):
            self.cum_total[i] += 1
            frac = self._lit_fraction_moon_lat(lat, sun_dir)
            self.cum_lit[i]   += frac

        # ── Draw all panels ───────────────────────────────────────────────────
        self._draw_orbit(ex, ey, mx, my)
        self._draw_moon_surface(ex, ey, mx, my, sun_dir)
        self._draw_cumulative_bar(sun_dir)
        self._draw_phase(ex, ey, mx, my)
        self._draw_table(sun_dir)
        self._draw_info()

        self.fig.canvas.draw_idle()

    # =========================================================================
    # Panel: Orbital overview
    # =========================================================================
    def _draw_orbit(self, ex, ey, mx, my):
        self.earth_patch.center = (ex, ey)
        self.moon_patch.center  = (ex + mx, ey + my)

        # Moon orbit ellipse
        th  = np.linspace(0, 2*np.pi, 120)
        inc = np.radians(self.moon_tilt)
        self.moon_orbit_line.set_data(
            ex + MOON_ORBIT_R * np.cos(th),
            ey + MOON_ORBIT_R * np.sin(th) * np.cos(inc),
        )

        # Near-side indicator: line from Moon toward Earth
        me_unit = np.array([-mx, -my])
        me_unit /= np.linalg.norm(me_unit) + 1e-12
        tip = np.array([ex + mx, ey + my]) + me_unit * (MOON_RADIUS + 0.18)
        self.nearside_line.set_data(
            [ex + mx, tip[0]], [ey + my, tip[1]])

        # Sun ray arrow
        sun_unit = np.array([-ex, -ey])
        sun_unit /= np.linalg.norm(sun_unit) + 1e-12
        ray_start = sun_unit * (SUN_RADIUS + 0.3)
        ray_end   = np.array([ex, ey]) - sun_unit * (EARTH_RADIUS + 0.25)
        self.sun_ray.set_positions(tuple(ray_start), tuple(ray_end))

        self.orbit_info.set_text(
            f"Earth: {self.angle_earth:.0f}°\n"
            f"Moon:  {self.angle_moon:.0f}°\n"
            f"Tilt:  {self.moon_tilt:.1f}°\n"
            f"Frame: {self.frame}"
        )

    # =========================================================================
    # Panel: Moon surface disc
    # =========================================================================
    def _draw_moon_surface(self, ex, ey, mx, my, sun_dir):
        ax = self.ax_moon
        ax.cla()
        ax.set_aspect("equal")
        ax.axis("off")
        ax.set_facecolor(PANEL_BG)
        ax.set_title("Moon Surface Illumination", color=TEXT_COL,
                     fontsize=9, pad=6, fontweight="bold")

        R   = 1.0
        theta = np.linspace(0, 2*np.pi, 720, endpoint=False)
        sun_ang = np.arctan2(sun_dir[1], sun_dir[0])

        # Fill disc with day/night gradient
        dangle = 2*np.pi / len(theta)
        for ang in theta:
            diff = ang - sun_ang
            frac = (np.cos(diff) + 1) / 2
            color = _MOON_SURF(frac)
            ax.add_patch(mpatches.Wedge(
                (0, 0), R, np.degrees(ang),
                np.degrees(ang + dangle + 0.01), color=color))

        # Rim
        ax.add_patch(plt.Circle((0, 0), R, fill=False,
                                 edgecolor="#ffffff55", lw=1.5))

        # Latitude rings
        for lat in [-60, -30, 0, 30, 60]:
            r_lat = R * np.cos(np.radians(lat))
            if r_lat < 0.05:
                continue
            ax.add_patch(plt.Circle((0, 0), r_lat, fill=False,
                                     edgecolor="#ffffff30", lw=0.5, ls="--"))
            ax.text(r_lat + 0.04, 0.02, f"{lat}°",
                    color="#ffffff55", fontsize=5.5, va="center")

        # Terminator line (great circle perpendicular to sun direction)
        t_theta = np.linspace(-np.pi/2, np.pi/2, 200)
        tx = R * np.cos(t_theta + sun_ang + np.pi/2)
        ty = R * np.sin(t_theta + sun_ang + np.pi/2)
        ax.plot(tx, ty, color="#ffffffbb", lw=2, zorder=10)

        # Near side / far side labels
        # Near side always faces Earth; in our display the Moon is shown
        # with the Earth-facing hemisphere on the left (toward Earth from Moon)
        me_ang = np.arctan2(-my, -mx)   # direction from Moon toward Earth
        near_x = 0.62 * np.cos(me_ang)
        near_y = 0.62 * np.sin(me_ang)
        far_x  = -0.62 * np.cos(me_ang)
        far_y  = -0.62 * np.sin(me_ang)

        ax.text(near_x, near_y, "NEAR\nSIDE", color="#88ffaa",
                fontsize=6.5, ha="center", va="center", fontweight="bold",
                bbox=dict(facecolor="#00000088", pad=2, edgecolor="none"))
        ax.text(far_x, far_y, "FAR\nSIDE", color="#ff9966",
                fontsize=6.5, ha="center", va="center", fontweight="bold",
                bbox=dict(facecolor="#00000088", pad=2, edgecolor="none"))

        # Sun direction arrow outside disc
        ax.annotate("",
                    xy=(sun_dir[0]*1.55, sun_dir[1]*1.55),
                    xytext=(sun_dir[0]*1.22, sun_dir[1]*1.22),
                    arrowprops=dict(arrowstyle="-|>", color=SUN_COL, lw=2))
        ax.text(sun_dir[0]*1.72, sun_dir[1]*1.72, "☀",
                color=SUN_COL, fontsize=14, ha="center", va="center")

        # Earth direction arrow
        me_unit = np.array([np.cos(me_ang), np.sin(me_ang)])
        ax.annotate("",
                    xy=(me_unit[0]*1.55, me_unit[1]*1.55),
                    xytext=(me_unit[0]*1.22, me_unit[1]*1.22),
                    arrowprops=dict(arrowstyle="-|>", color=EARTH_COL, lw=1.5))
        ax.text(me_unit[0]*1.72, me_unit[1]*1.72, "🌍",
                fontsize=10, ha="center", va="center")

        # "Currently lit" fraction badge
        lit_frac = 0.5   # always ~50% of Moon is lit at any time
        ax.text(0, -1.48,
                f"~{lit_frac*100:.0f}% of Moon surface currently lit by Sun",
                color=GOLD, fontsize=7.5, ha="center", va="center",
                bbox=dict(facecolor="#1a1a3a", pad=3, edgecolor=GOLD+"88"))

        ax.set_xlim(-2.0, 2.0)
        ax.set_ylim(-1.75, 1.75)

    # =========================================================================
    # Panel: Cumulative sunlight bar chart
    # =========================================================================
    def _draw_cumulative_bar(self, sun_dir):
        ax = self.ax_cumbar
        ax.cla()
        ax.set_facecolor(PANEL_BG)
        ax.set_title("Cumulative Sunlight by Latitude", color=TEXT_COL,
                     fontsize=9, pad=6, fontweight="bold")

        if self.cum_total[0] == 0:
            ax.text(0.5, 0.5, "Waiting for data…", color=DIM_TEXT,
                    ha="center", va="center", transform=ax.transAxes)
            return

        fracs = self.cum_lit / (self.cum_total + 1e-12)
        y_pos = np.arange(N_LAT)

        colors = []
        for f in fracs:
            if f > 0.48:
                colors.append(GOLD)
            elif f > 0.35:
                colors.append("#88aaff")
            else:
                colors.append(RED_DARK)

        bars = ax.barh(y_pos, fracs * 100, color=colors, height=0.65,
                       edgecolor="#2a2a4a", linewidth=0.5)

        # 50% reference line
        ax.axvline(50, color="#ffffff44", lw=1, ls="--")
        ax.text(50.5, N_LAT - 0.3, "50%", color="#ffffff66", fontsize=6.5)

        ax.set_yticks(y_pos)
        ax.set_yticklabels([f"{lat:+d}°" for lat in LATITUDES],
                           color=TEXT_COL, fontsize=7.5)
        ax.set_xlabel("Cumulative sunlight (%)", color=DIM_TEXT, fontsize=7.5)
        ax.set_xlim(0, 100)
        ax.tick_params(axis="x", colors=DIM_TEXT, labelsize=7)
        ax.set_facecolor(PANEL_BG)
        for spine in ax.spines.values():
            spine.set_edgecolor("#2a2a4a")

        # Value labels on bars
        for bar, f in zip(bars, fracs):
            w = bar.get_width()
            ax.text(min(w + 1.5, 96), bar.get_y() + bar.get_height()/2,
                    f"{f*100:.1f}%", color=TEXT_COL, fontsize=6.5,
                    va="center")

        frames_elapsed = int(self.cum_total[0])
        ax.set_title(
            f"Cumulative Sunlight by Latitude  (n={frames_elapsed} frames)",
            color=TEXT_COL, fontsize=9, pad=6, fontweight="bold")

    # =========================================================================
    # Panel: Moon phase
    # =========================================================================
    def _draw_phase(self, ex, ey, mx, my):
        ax = self.ax_phase
        ax.cla()
        ax.set_aspect("equal")
        ax.axis("off")
        ax.set_facecolor(PANEL_BG)
        ax.set_title("Moon Phase  (as seen from Earth)", color=TEXT_COL,
                     fontsize=9, pad=6, fontweight="bold")

        R = 1.0
        phase_angle = self._moon_sun_angle(ex, ey, mx, my)  # 0=new, 180=full
        pa_rad = np.radians(phase_angle)

        theta = np.linspace(-np.pi/2, np.pi/2, 180)

        # Determine if waxing or waning via cross-product sign
        moon_wx  = ex + mx
        moon_wy  = ey + my
        cross    = ex * moon_wy - ey * moon_wx   # z-component
        waxing   = cross > 0

        # Dark base
        ax.add_patch(plt.Circle((0, 0), R, color="#2a2a2a"))

        # Build illuminated crescent/gibbous polygon
        right_edge_x = R * np.cos(theta)
        left_edge_x  = -R * np.cos(theta)
        ys           = R * np.sin(theta)
        terminator_x = R * np.cos(pa_rad) * np.cos(theta)   # ellipse terminator

        if waxing:
            # lit on right
            poly_x = np.concatenate([right_edge_x, terminator_x[::-1]])
            poly_y = np.concatenate([ys, ys[::-1]])
        else:
            # lit on left
            poly_x = np.concatenate([left_edge_x, (-terminator_x)[::-1]])
            poly_y = np.concatenate([ys, ys[::-1]])

        ax.fill(poly_x, poly_y, color="#d8d8c8", zorder=2)
        ax.add_patch(plt.Circle((0, 0), R, fill=False,
                                 edgecolor="#ffffff55", lw=1.5, zorder=3))

        # Phase name
        phase_names = [
            (0,   22,  "New Moon"),
            (22,  67,  "Waxing Crescent"),
            (67,  112, "First Quarter"),
            (112, 157, "Waxing Gibbous"),
            (157, 180, "Full Moon"),
        ]
        # mirror for waning
        if not waxing:
            pname = {
                (0,   22):  "Full Moon",
                (22,  67):  "Waning Gibbous",
                (67,  112): "Last Quarter",
                (112, 157): "Waning Crescent",
                (157, 180): "New Moon",
            }.get(next((
                (lo, hi) for lo, hi, _ in phase_names if lo <= phase_angle < hi
            ), (157, 180)), "New Moon")
        else:
            pname = next((n for lo, hi, n in phase_names
                          if lo <= phase_angle < hi), "New Moon")

        ax.text(0, -1.38, pname, color=TEXT_COL, fontsize=8.5,
                ha="center", va="center", fontweight="bold")
        ax.text(0, -1.62, f"Phase angle: {phase_angle:.0f}°",
                color=DIM_TEXT, fontsize=7, ha="center")

        # Earthshine note on dark limb
        if phase_angle < 45 or phase_angle > 135:
            ax.text(0, 0.15, "← Earthshine\n   visible here",
                    color="#6688bb", fontsize=6, ha="center", style="italic")

        ax.set_xlim(-2, 2)
        ax.set_ylim(-2, 2)

    # =========================================================================
    # Panel: Latitude table
    # =========================================================================
    def _draw_table(self, sun_dir):
        ax = self.ax_table
        ax.cla()
        ax.axis("off")
        ax.set_facecolor(PANEL_BG)
        ax.set_title("Instantaneous & Cumulative Sunlight", color=TEXT_COL,
                     fontsize=9, pad=6, fontweight="bold")

        headers = ["Lat", "Now %", "Cumul %", "Bar (cumul)"]
        col_x   = [0.01, 0.18, 0.36, 0.56]
        row_h   = 0.074
        y0      = 0.93

        for h, x in zip(headers, col_x):
            ax.text(x, y0, h, color=ACCENT, fontsize=7.5,
                    fontweight="bold", transform=ax.transAxes, va="top")
        ax.plot([0.01, 0.99], [y0-0.032, y0-0.032],
                color="#3a3a6a", lw=0.8, transform=ax.transAxes)

        for i, lat in enumerate(LATITUDES):
            now_frac  = self._lit_fraction_moon_lat(lat, sun_dir)
            cum_frac  = (self.cum_lit[i] / (self.cum_total[i] + 1e-12))
            y         = y0 - row_h * (i + 1) - 0.01

            bg = "#14143a" if i % 2 == 0 else "#0e0e2a"
            ax.add_patch(mpatches.FancyBboxPatch(
                (0.005, y - 0.004), 0.988, row_h - 0.006,
                boxstyle="round,pad=0.003", facecolor=bg, edgecolor="none",
                transform=ax.transAxes))

            now_col = GOLD if now_frac > 0.5 else ("#8899cc" if now_frac > 0.1 else RED_DARK)
            cum_col = GREEN if cum_frac > 0.45 else ("#8899cc" if cum_frac > 0.3 else RED_DARK)

            yc = y + row_h * 0.38
            ax.text(col_x[0], yc, f"{lat:+d}°",
                    color=TEXT_COL, fontsize=7.5, transform=ax.transAxes, va="center")
            ax.text(col_x[1], yc, f"{now_frac*100:4.1f}%",
                    color=now_col, fontsize=7.5, transform=ax.transAxes, va="center")
            ax.text(col_x[2], yc, f"{cum_frac*100:4.1f}%",
                    color=cum_col, fontsize=7.5, transform=ax.transAxes, va="center")

            # Bar
            bx, bw, bh = col_x[3], 0.42, row_h * 0.52
            ax.add_patch(mpatches.FancyBboxPatch(
                (bx, y + 0.005), bw, bh,
                boxstyle="round,pad=0.002", facecolor="#1a1a3a", edgecolor="none",
                transform=ax.transAxes))
            if cum_frac > 0:
                ax.add_patch(mpatches.FancyBboxPatch(
                    (bx, y + 0.005), bw * cum_frac, bh,
                    boxstyle="round,pad=0.002", facecolor=cum_col, edgecolor="none",
                    transform=ax.transAxes))
            # 50% tick on bar
            ax.plot([bx + bw*0.5]*2, [y+0.005, y+0.005+bh],
                    color="#ffffff44", lw=0.8, transform=ax.transAxes)

        ax.text(0.5, 0.01,
                "Cumulative % converges to ~50% for all non-polar latitudes over time",
                color=DIM_TEXT, fontsize=6.2, ha="center", va="bottom",
                transform=ax.transAxes, style="italic")

    # =========================================================================
    # Panel: Science callout
    # =========================================================================
    def _draw_info(self):
        ax = self.ax_info
        ax.cla()
        ax.axis("off")
        ax.set_facecolor(PANEL_BG)
        ax.set_title("The Science", color=TEXT_COL,
                     fontsize=9, pad=6, fontweight="bold")

        lines = [
            (GOLD,     "FAR SIDE ≠ DARK SIDE"),
            (TEXT_COL, ""),
            (TEXT_COL, "The Moon is tidally locked:"),
            (TEXT_COL, "  it rotates once per orbit,"),
            (TEXT_COL, "  so the same face always"),
            (TEXT_COL, "  points toward Earth."),
            (TEXT_COL, ""),
            (GOLD,     "But the Sun illuminates ALL"),
            (GOLD,     "of the Moon over one month."),
            (TEXT_COL, ""),
            (TEXT_COL, "The far side faces the Sun"),
            (TEXT_COL, "for roughly half of every"),
            (TEXT_COL, "lunar orbit (~14.75 days)."),
            (TEXT_COL, ""),
            (ACCENT,   "Exception: Polar regions"),
            (TEXT_COL, f"Moon's axial tilt is only"),
            (TEXT_COL, f"~{MOON_TILT_REAL}°.  Near-polar craters"),
            (TEXT_COL, "  sit in permanent shadow —"),
            (TEXT_COL, "  that's where NASA found"),
            (TEXT_COL, "  water ice (LCROSS, 2009)."),
            (TEXT_COL, ""),
            (ACCENT,   "Earthshine"),
            (TEXT_COL, "The Moon's night side is"),
            (TEXT_COL, "faintly lit by sunlight"),
            (TEXT_COL, "reflected off Earth."),
            (TEXT_COL, ""),
            (DIM_TEXT, "Drag the tilt slider →"),
            (DIM_TEXT, "watch polar bands darken"),
            (DIM_TEXT, "as polar night grows."),
        ]

        y = 0.97
        dy = 0.033
        for color, txt in lines:
            ax.text(0.04, y, txt, color=color, fontsize=7.2,
                    transform=ax.transAxes, va="top")
            y -= dy
            if y < 0.02:
                break

        # Decorative separator
        ax.plot([0.04, 0.96], [0.895, 0.895], color="#3a3a6a", lw=0.7,
                transform=ax.transAxes)


# =============================================================================
def main():
    matplotlib.rcParams["toolbar"] = "None"
    sim = LunarSim()
    plt.show()


if __name__ == "__main__":
    main()
