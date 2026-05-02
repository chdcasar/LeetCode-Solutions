"""
Double Pendulum Simulator
=========================
Interactive simulation using matplotlib with sliders to adjust:
  - Rod lengths (L1, L2)
  - Bob masses  (m1, m2)
  - Initial angles (θ1, θ2)

Controls
--------
  Sliders  : adjust parameters (restart animation automatically)
  Restart  : reset simulation with current slider values
  Pause    : pause / resume the animation
  Trail    : toggle the chaos trail on/off
"""

import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.widgets import Slider, Button

# ── Physical constants ─────────────────────────────────────────────────
DT = 0.02         # simulation timestep (s)
T_MAX = 30.0      # total simulation window (s)─────
G = 9.81          # m s⁻²
TRAIL_LEN = 300   # max number of trail points to draw

# ── Default parameter values ────────────────────────────────────────────────
DEFAULTS = dict(L1=1.0, L2=1.0, m1=1.0, m2=1.0, a1=120.0, a2=-20.0)

# ── Equations of motion ─────────────────────────────────────────────────────
def derivs(t, state, L1, L2, m1, m2):
    """Return [dθ1, dω1, dθ2, dω2] for the double pendulum."""
    th1, w1, th2, w2 = state
    d = th1 - th2
    sin_d, cos_d = np.sin(d), np.cos(d)
    denom = 2*m1 + m2 - m2*np.cos(2*d)

    dth1 = w1
    dw1 = (
        -G*(2*m1 + m2)*np.sin(th1)
        - m2*G*np.sin(th1 - 2*th2)
        - 2*sin_d*m2*(w2**2*L2 + w1**2*L1*cos_d)
    ) / (L1 * denom)

    dth2 = w2
    dw2 = (
        2*sin_d*(
            w1**2*L1*(m1 + m2)
            + G*(m1 + m2)*np.cos(th1)
            + w2**2*L2*m2*cos_d
        )
    ) / (L2 * denom)

    return [dth1, dw1, dth2, dw2]


def solve(L1, L2, m1, m2, a1_deg, a2_deg):
    """Integrate the ODE and return (t, x1, y1, x2, y2) arrays."""
    th1 = np.radians(a1_deg)
    th2 = np.radians(a2_deg)
    state0 = [th1, 0.0, th2, 0.0]
    t_span = (0, T_MAX)
    t_eval = np.arange(0, T_MAX, DT)

    sol = solve_ivp(
        derivs, t_span, state0,
        args=(L1, L2, m1, m2),
        t_eval=t_eval,
        method="RK45",
        rtol=1e-8, atol=1e-9,
        dense_output=False
    )
    th1s, th2s = sol.y[0], sol.y[2]

    x1 =  L1 * np.sin(th1s)
    y1 = -L1 * np.cos(th1s)
    x2 = x1 + L2 * np.sin(th2s)
    y2 = y1 - L2 * np.cos(th2s)
    return sol.t, x1, y1, x2, y2


# ── Build figure ─────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(12, 7), facecolor="#0d0d0d")
fig.canvas.manager.set_window_title("Double Pendulum Simulator")

# Main animation axes
ax = fig.add_axes([0.02, 0.15, 0.54, 0.82], facecolor="#0d0d0d")
ax.set_aspect("equal")
ax.set_xlim(-2.6, 2.6)
ax.set_ylim(-2.6, 2.6)
ax.tick_params(colors="#555555")
for spine in ax.spines.values():
    spine.set_edgecolor("#333333")
ax.set_facecolor("#0d0d0d")
ax.grid(color="#1a1a1a", linewidth=0.5)
ax.set_title("Double Pendulum", color="#cccccc", fontsize=13, pad=8)

# Energy axes (top-right)
ax_e = fig.add_axes([0.60, 0.72, 0.37, 0.20], facecolor="#111111")
ax_e.set_title("Total Energy  (J)", color="#888888", fontsize=9)
ax_e.tick_params(colors="#555555", labelsize=7)
for sp in ax_e.spines.values():
    sp.set_edgecolor("#333333")

# Phase-space axes (bottom-right)
ax_p = fig.add_axes([0.60, 0.45, 0.37, 0.20], facecolor="#111111")
ax_p.set_title("Phase Space  θ₁ vs ω₁", color="#888888", fontsize=9)
ax_p.set_xlabel("θ₁ (rad)", color="#555555", fontsize=8)
ax_p.set_ylabel("ω₁ (rad/s)", color="#555555", fontsize=8)
ax_p.tick_params(colors="#555555", labelsize=7)
for sp in ax_p.spines.values():
    sp.set_edgecolor("#333333")

# ── Slider layout ─────────────────────────────────────────────────────────────
slider_color  = "#1e1e2e"
handle_color  = "#7c6af5"
label_color   = "#aaaaaa"

slider_specs = [
    # (label,  left,   bottom, valinit, valmin, valmax, valstep)
    ("L₁ (m)",  0.60, 0.355,  DEFAULTS["L1"], 0.2, 2.0, 0.05),
    ("L₂ (m)",  0.60, 0.305,  DEFAULTS["L2"], 0.2, 2.0, 0.05),
    ("m₁ (kg)", 0.60, 0.255,  DEFAULTS["m1"], 0.1, 5.0, 0.1),
    ("m₂ (kg)", 0.60, 0.205,  DEFAULTS["m2"], 0.1, 5.0, 0.1),
    ("θ₁ (°)",  0.60, 0.155,  DEFAULTS["a1"], -180, 180, 1),
    ("θ₂ (°)",  0.60, 0.105,  DEFAULTS["a2"], -180, 180, 1),
]

sliders = {}
for label, left, bottom, valinit, vmin, vmax, vstep in slider_specs:
    sax = fig.add_axes([left, bottom, 0.37, 0.025], facecolor=slider_color)
    sl = Slider(sax, label, vmin, vmax, valinit=valinit,
                valstep=vstep, color=handle_color)
    sl.label.set_color(label_color)
    sl.label.set_fontsize(8)
    sl.valtext.set_color(label_color)
    sl.valtext.set_fontsize(8)
    sliders[label] = sl

# ── Buttons ───────────────────────────────────────────────────────────────────
btn_restart_ax = fig.add_axes([0.60, 0.040, 0.11, 0.040])
btn_pause_ax   = fig.add_axes([0.73, 0.040, 0.11, 0.040])
btn_trail_ax   = fig.add_axes([0.86, 0.040, 0.11, 0.040])

btn_restart = Button(btn_restart_ax, "↺ Restart",  color="#1a1a2e", hovercolor="#2a2a4e")
btn_pause   = Button(btn_pause_ax,   "⏸ Pause",    color="#1a1a2e", hovercolor="#2a2a4e")
btn_trail   = Button(btn_trail_ax,   "✦ Trail",    color="#1a2a1a", hovercolor="#2a4a2a")
for btn in (btn_restart, btn_pause, btn_trail):
    btn.label.set_color("#cccccc")
    btn.label.set_fontsize(9)

# ── Graphical objects ─────────────────────────────────────────────────────────
pivot_dot,  = ax.plot(0, 0, "o", color="#ffffff", ms=5, zorder=5)
rod1_line,  = ax.plot([], [], "-", color="#7c6af5", lw=2.5, zorder=3)
rod2_line,  = ax.plot([], [], "-", color="#f57c6a", lw=2.5, zorder=3)
bob1_dot,   = ax.plot([], [], "o", color="#7c6af5", ms=12, zorder=6)
bob2_dot,   = ax.plot([], [], "o", color="#f57c6a", ms=12, zorder=6)
trail_line, = ax.plot([], [], "-", color="#f57c6a", lw=0.8, alpha=0.55, zorder=2)

energy_line, = ax_e.plot([], [], color="#5cc8ff", lw=1.0)
phase_line,  = ax_p.plot([], [], color="#b8ff5c", lw=0.6, alpha=0.7)

info_text = ax.text(
    0.02, 0.98, "", transform=ax.transAxes,
    color="#888888", fontsize=8, va="top", family="monospace"
)

# ── Simulation state ──────────────────────────────────────────────────────────
state = dict(
    t=None, x1=None, y1=None, x2=None, y2=None,
    frame=0, paused=False, trail_on=True
)


def get_params():
    return (
        sliders["L₁ (m)"].val,
        sliders["L₂ (m)"].val,
        sliders["m₁ (kg)"].val,
        sliders["m₂ (kg)"].val,
        sliders["θ₁ (°)"].val,
        sliders["θ₂ (°)"].val,
    )


def compute_energy(L1, L2, m1, m2, x1, y1, x2, y2, dt):
    """Approximate total mechanical energy at each frame."""
    # Potential energy  (pivot at y=0)
    PE = -G * (m1 * y1 + m2 * y2)
    # Kinetic energy
    vx1 = np.gradient(x1, dt)
    vy1 = np.gradient(y1, dt)
    vx2 = np.gradient(x2, dt)
    vy2 = np.gradient(y2, dt)
    KE = 0.5 * (m1*(vx1**2 + vy1**2) + m2*(vx2**2 + vy2**2))
    return KE + PE


def rebuild(*_):
    """Re-solve ODE with current slider values and reset animation."""
    L1, L2, m1, m2, a1, a2 = get_params()
    t, x1, y1, x2, y2 = solve(L1, L2, m1, m2, a1, a2)

    E = compute_energy(L1, L2, m1, m2, x1, y1, x2, y2, DT)

    # Update energy plot
    energy_line.set_data(t, E)
    ax_e.set_xlim(t[0], t[-1])
    margin = max(0.1 * (E.max() - E.min()), 0.1)
    ax_e.set_ylim(E.min() - margin, E.max() + margin)

    # Update phase plot (pre-fill all data dimly)
    th1_arr = np.arctan2(x1, -y1)  # recover θ1
    w1_arr  = np.gradient(th1_arr, DT)
    phase_line.set_data(th1_arr, w1_arr)
    ax_p.relim(); ax_p.autoscale_view()

    # Store for animation
    state.update(t=t, x1=x1, y1=y1, x2=x2, y2=y2, frame=0)
    trail_line.set_data([], [])
    energy_line.set_data([], [])
    phase_line.set_data([], [])

    # Rescale main axes to fit new lengths
    lim = (L1 + L2) * 1.15
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)

    fig.canvas.draw_idle()


def animate(frame_unused):
    if state["paused"] or state["t"] is None:
        return rod1_line, rod2_line, bob1_dot, bob2_dot, trail_line, info_text, energy_line, phase_line

    i = state["frame"]
    t  = state["t"]
    x1, y1, x2, y2 = state["x1"], state["y1"], state["x2"], state["y2"]
    n = len(t)

    if i >= n:
        state["frame"] = 0
        trail_line.set_data([], [])
        energy_line.set_data([], [])
        phase_line.set_data([], [])
        return rod1_line, rod2_line, bob1_dot, bob2_dot, trail_line, info_text, energy_line, phase_line

    # Rods
    rod1_line.set_data([0, x1[i]],  [0, y1[i]])
    rod2_line.set_data([x1[i], x2[i]], [y1[i], y2[i]])
    bob1_dot.set_data([x1[i]], [y1[i]])
    bob2_dot.set_data([x2[i]], [y2[i]])

    # Trail
    if state["trail_on"]:
        lo = max(0, i - TRAIL_LEN)
        trail_line.set_data(x2[lo:i+1], y2[lo:i+1])
    else:
        trail_line.set_data([], [])

    # Energy strip
    energy_line.set_data(t[:i+1], state.get("E", np.zeros(n))[:i+1])

    # Phase strip
    phase_line.set_data(state.get("th1", np.zeros(n))[:i+1],
                        state.get("w1",  np.zeros(n))[:i+1])

    # Info
    info_text.set_text(f"t = {t[i]:.2f} s   frame {i}/{n}")

    state["frame"] = i + 1
    return rod1_line, rod2_line, bob1_dot, bob2_dot, trail_line, info_text, energy_line, phase_line


def on_restart(_):
    rebuild()
    # Also pre-compute derived arrays for live energy/phase plots
    L1, L2, m1, m2 = (sliders[k].val for k in ("L₁ (m)", "L₂ (m)", "m₁ (kg)", "m₂ (kg)"))
    x1, y1, x2, y2 = state["x1"], state["y1"], state["x2"], state["y2"]
    t = state["t"]
    E = compute_energy(L1, L2, m1, m2, x1, y1, x2, y2, DT)
    th1 = np.arctan2(x1, -y1)
    w1  = np.gradient(th1, DT)
    state["E"]   = E
    state["th1"] = th1
    state["w1"]  = w1
    # Set axis limits
    ax_e.set_xlim(t[0], t[-1])
    m = max(0.1*(E.max()-E.min()), 0.1)
    ax_e.set_ylim(E.min()-m, E.max()+m)
    ax_p.set_xlim(th1.min()-0.1, th1.max()+0.1)
    ax_p.set_ylim(w1.min()-0.5, w1.max()+0.5)


def on_pause(_):
    state["paused"] = not state["paused"]
    btn_pause.label.set_text("▶ Resume" if state["paused"] else "⏸ Pause")
    fig.canvas.draw_idle()


def on_trail(_):
    state["trail_on"] = not state["trail_on"]
    btn_trail.label.set_text("✦ Trail ✓" if state["trail_on"] else "✦ Trail ✗")
    if not state["trail_on"]:
        trail_line.set_data([], [])
    fig.canvas.draw_idle()


# Wire up callbacks
btn_restart.on_clicked(on_restart)
btn_pause.on_clicked(on_pause)
btn_trail.on_clicked(on_trail)

for sl in sliders.values():
    sl.on_changed(lambda _: on_restart(None))

# First build
on_restart(None)

# ── Start animation ───────────────────────────────────────────────────────────
ani = animation.FuncAnimation(
    fig, animate,
    interval=int(DT * 1000),   # ms per frame → real-time playback
    blit=False,
    cache_frame_data=False,
)

plt.show()