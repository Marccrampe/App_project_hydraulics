import numpy as np
import matplotlib.pyplot as plt
import streamlit as st

from matplotlib.patches import Ellipse, Rectangle
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
from mpl_toolkits.mplot3d.art3d import Poly3DCollection


# ============================================================
# 1. Core hydraulics: 3-zone bend model
# ============================================================

def compute_conveyance(A, Rh, n):
    """Conveyance K = (1/n) * A * Rh^(2/3)."""
    return (1.0 / n) * A * (Rh ** (2.0 / 3.0))


def compute_outer_velocity(
    B=20.0,
    h=3.0,
    S=0.001,
    n_i=0.035,
    n_m=0.035,
    n_o=0.035,
    Q=100.0,
    alpha=1.3,
    gamma=0.0,
):
    """
    Simple 3-zone bend model for a rectangular cross-section.
    """

    # Split width into 3 zones: inner, mid, outer
    Bi = 0.3 * B
    Bm = 0.4 * B
    Bo = 0.3 * B

    # Areas
    A_i = Bi * h
    A_m = Bm * h
    A_o = Bo * h

    # Approx hydraulic radius ~ depth
    Rh_i = Rh_m = Rh_o = h

    # Conveyances
    Ki = compute_conveyance(A_i, Rh_i, n_i)
    Km = compute_conveyance(A_m, Rh_m, n_m)
    Ko = compute_conveyance(A_o, Rh_o, n_o)
    Ktot = Ki + Km + Ko

    # Discharge distribution before vanes
    Qi = Q * (Ki / Ktot)
    Qm = Q * (Km / Ktot)
    Qo = Q * (Ko / Ktot)

    # Effect of vanes: divert gamma * Qo from outer → mid
    Qo_after = (1.0 - gamma) * Qo
    Qm_after = Qm + gamma * Qo
    Qi_after = Qi

    # Velocities after vane deflection
    Vi = Qi_after / A_i
    Vm = Qm_after / A_m
    Vo = Qo_after / A_o

    # Outer-bank velocity amplified by bend effect
    Vo_bend = alpha * Vo

    return {
        "zones_width": (Bi, Bm, Bo),
        "areas": (A_i, A_m, A_o),
        "Q_before": (Qi, Qm, Qo),
        "Q_after": (Qi_after, Qm_after, Qo_after),
        "V_after": (Vi, Vm, Vo),
        "Vo_bend": Vo_bend,
    }


# ============================================================
# 2. Material & geometry → gamma & local scour risk
# ============================================================

def material_properties(material):
    """
    Returns (n_multiplier, scour_multiplier_base) for a given vane material.
    """
    table = {
        "steel":    (0.95, 0.9),   # smooth
        "concrete": (1.00, 1.0),   # baseline
        "wood":     (1.05, 1.1),   # slightly rougher
        "rock":     (1.15, 1.4),   # very rough → more turbulence/local scour
    }
    if material not in table:
        material = "concrete"
    return table[material]


def gamma_from_geometry(
    L_rel,
    alpha_attack_deg,
    n_vanes,
    is_array,
    bevel_angle_deg,
    material,
    gamma_base=0.25,
):
    """
    Compute gamma (fraction of outer-zone discharge diverted to mid-channel)
    from vane geometry and material.
    """

    # Length factor
    L_ref = 0.5
    f_L = 1.0 + 0.6 * (L_rel - L_ref)
    f_L = np.clip(f_L, 0.3, 1.5)

    # Attack angle factor
    alpha_ref = 20.0
    f_alpha = 1.0 + 0.5 * ((alpha_attack_deg - alpha_ref) / alpha_ref)
    f_alpha = np.clip(f_alpha, 0.5, 1.5)

    # Array effect
    if is_array:
        f_array = 1.0 + 0.1 * (n_vanes - 1)
    else:
        f_array = 1.0
    f_array = np.clip(f_array, 1.0, 1.5)

    # Material effect (rougher → slightly less efficient deflection)
    n_mult_mat, _ = material_properties(material)
    f_mat_gamma = 1.0 - 0.2 * (n_mult_mat - 1.0)
    f_mat_gamma = np.clip(f_mat_gamma, 0.8, 1.1)

    # Bevel effect (mild, best around 30–45°)
    angle_grid = np.array([0.0, 20.0, 30.0, 45.0, 60.0, 70.0])
    eff_grid = np.array([1.00, 1.02, 1.05, 1.08, 1.04, 0.98])
    theta_clamped = np.clip(
        bevel_angle_deg if bevel_angle_deg is not None else 0.0, 0.0, 70.0
    )
    f_bevel_geom = np.interp(theta_clamped, angle_grid, eff_grid)

    gamma = gamma_base * f_L * f_alpha * f_array * f_mat_gamma * f_bevel_geom
    gamma = np.clip(gamma, 0.0, 0.7)
    return gamma


def local_scour_risk_from_geometry(
    L_rel,
    alpha_attack_deg,
    n_vanes,
    bevel_angle_deg,
    material,
):
    """
    Relative local scour risk indicator around the vanes.
    """

    base_risk = 1.0

    # Length effect (longer vane → larger scour footprint)
    L_ref = 0.5
    f_L = 1.0 + 0.8 * (L_rel - L_ref)
    f_L = np.clip(f_L, 0.4, 1.8)

    # Attack angle effect
    alpha_ref = 20.0
    f_alpha = 1.0 + 0.7 * ((alpha_attack_deg - alpha_ref) / alpha_ref)
    f_alpha = np.clip(f_alpha, 0.5, 1.8)

    # Number of vanes
    f_n = 1.0 + 0.1 * (n_vanes - 1)
    f_n = np.clip(f_n, 1.0, 1.6)

    # Bevel effect via scour-volume ratios (just used qualitatively)
    angle_grid = np.array([0.0, 30.0, 45.0, 60.0, 70.0])
    vol_ratio_grid = np.array([1.0, 0.555, 0.4335, 0.0646, 0.0])
    theta_clamped = np.clip(
        bevel_angle_deg if bevel_angle_deg is not None else 0.0,
        0.0,
        70.0,
    )
    scour_vol_ratio = np.interp(theta_clamped, angle_grid, vol_ratio_grid)

    # Material effect
    _, scour_mult_mat = material_properties(material)

    risk = base_risk * f_L * f_alpha * f_n * scour_vol_ratio * scour_mult_mat
    risk = np.clip(risk, 0.0, 1.5)
    return risk


# === spacing → interaction factor ============================================

def spacing_factor(spacing_rel):
    """
    Dimensionless interaction factor for vane spacing (in H units).

    - spacing_rel ≈ 1.3 H → optimum → f_s ≈ 1
    - très petit ou très grand spacing → f_s ↓
    """
    s_opt = 1.3
    f = np.exp(-((spacing_rel - s_opt) ** 2) / 0.5)
    return float(np.clip(f, 0.0, 1.0))


# ============================================================
# 3. Plot helpers
# ============================================================

def plot_scour_plan_view(
    B,
    h,
    n_vanes,
    bevel_angle,
    scour_risk,
    use_vane,
    alpha_attack_deg,
    spacing,
):
    """
    Plan-view schematic of vanes and local scour zones.

    spacing contrôle l'écartement des vanes en y (vers la berge).
    """

    Lx = 14.0
    Ly = 16.0
    fig, ax = plt.subplots(figsize=(6, 5))

    # Channel rectangle
    ax.add_patch(Rectangle((0, 0), Lx, Ly, fill=False))
    ax.text(Lx * 0.02, Ly * 0.05, "Inner side", fontsize=8, ha="left", va="bottom")
    ax.text(Lx * 0.02, Ly * 0.95, "Outer bank", fontsize=8, ha="left", va="top")

    # Main flow arrow
    ax.annotate(
        "",
        xy=(Lx * 0.8, Ly * 0.25),
        xytext=(Lx * 0.2, Ly * 0.25),
        arrowprops=dict(arrowstyle="->", linewidth=1.3),
    )
    ax.text(
        Lx * 0.5,
        Ly * 0.30,
        "Main flow direction",
        ha="center",
        va="bottom",
        fontsize=8,
    )

    if use_vane:
        x_vane = 0.5 * Lx

        # hauteur caractéristique H projetée en y (juste pour donner une échelle)
        H = 0.3 * h
        H_y = (H / B) * Ly
        dy = spacing * H_y  # spacing * H

        # centre des vanes vers la berge externe
        y_center = 0.75 * Ly
        idx = np.arange(n_vanes) - (n_vanes - 1) / 2.0
        y_positions = y_center + idx * dy
        y_positions = np.clip(y_positions, 0.35 * Ly, 0.95 * Ly)

        vane_len = 2.0
        phi = np.deg2rad(alpha_attack_deg)
        dy_vane = vane_len * np.cos(phi)
        dx_vane = vane_len * np.sin(phi)

        base_len = 1.4
        base_width = 0.6
        size_factor = min(1.0, scour_risk)
        scour_len = base_len * (0.4 + 0.6 * size_factor)
        scour_width = base_width * (0.4 + 0.6 * size_factor)

        for y0 in y_positions:
            x1 = x_vane - 0.5 * dx_vane
            y1 = y0 - 0.5 * dy_vane
            x2 = x_vane + 0.5 * dx_vane
            y2 = y0 + 0.5 * dy_vane

            ax.plot([x1, x2], [y1, y2], color="black", lw=2)

            scour_center_x = x2 + 0.6 * scour_len
            scour_center_y = y2

            ell = Ellipse(
                (scour_center_x, scour_center_y),
                width=scour_len,
                height=scour_width,
                angle=np.rad2deg(phi),
                color="red",
                alpha=0.3,
            )
            ax.add_patch(ell)

        label = "Local scour patches\n(one per vane, top view)"
        ax.text(
            Lx * 0.72,
            Ly * 0.18,
            label,
            ha="center",
            va="bottom",
            fontsize=8,
            color="red",
        )

        ax.text(
            x_vane + 0.5,
            y_positions[-1] + 0.5,
            f"{n_vanes} submerged vane(s)\n"
            f"Bevel θ = {bevel_angle:.1f}°\n"
            f"Spacing = {spacing:.2f} H",
            ha="left",
            va="bottom",
            fontsize=8,
        )

    ax.set_xlim(0, Lx)
    ax.set_ylim(0, Ly)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("Streamwise x [m] (schematic)")
    ax.set_ylabel("Cross-stream y [m] (schematic)")
    ax.set_title("Plan-view schematic of local scour around submerged vanes")
    ax.grid(False)

    return fig


def plot_velocity_field_fig(
    B,
    h,
    L_rel,
    alpha_attack,
    gamma,
    material,
    n_vanes,
    time_phase,
    red_V,
    red_tau,
    scour_risk,
    spacing,
):
    """
    Plan view 'pseudo-CFD' – spacing agit sur la position des vanes en y.
    """

    Lx = 20.0
    Ly = 4.0
    nx, ny = 250, 80
    x = np.linspace(0, Lx, nx)
    y = np.linspace(0, Ly, ny)
    X, Y = np.meshgrid(x, y)

    U0 = 1.0
    beta = 0.3
    U_base = U0 * (1 - beta * (Y - Ly / 2.0) ** 2)
    V_base = np.zeros_like(U_base)

    x0 = 0.5 * Lx

    # spacing en y basé sur H projeté
    H = 0.3 * h
    H_y = (H / B) * Ly
    dy = spacing * H_y
    y_center = 0.7 * Ly
    idx = np.arange(n_vanes) - (n_vanes - 1) / 2.0
    y_positions = y_center + idx * dy
    y_positions = np.clip(y_positions, 0.55 * Ly, 0.95 * Ly)

    Bo_phys = 0.3 * B
    scale = Ly / B
    L_geom = L_rel * Bo_phys * scale

    theta = np.deg2rad(alpha_attack)

    n_mult, _ = material_properties(material)

    A_base = 0.25
    A = A_base + 0.7 * gamma + 0.05 * (n_vanes - 1)
    A *= 1.0 + 0.15 * (n_mult - 1.0)
    A = np.clip(A, 0.1, 0.8)

    B_base = 0.35
    B_lat = B_base + 0.4 * gamma
    B_lat *= 1.0 + 0.2 * (n_mult - 1.0)
    B_lat = np.clip(B_lat, 0.1, 1.0)

    G_total = np.zeros_like(X)
    V = np.zeros_like(X)

    sigma_s = L_geom
    sigma_n = 0.3

    vane_segments = []
    halfL_geom = L_geom / 2.0

    for yv in y_positions:
        dx = X - x0
        dy_grid = Y - yv

        s = dx * np.cos(theta) + dy_grid * np.sin(theta)
        n = -dx * np.sin(theta) + dy_grid * np.cos(theta)

        mask = (s > -0.5 * L_geom) & (s < 1.5 * L_geom)
        Gk = np.exp(-((s**2) / (2 * sigma_s**2) + (n**2) / (2 * sigma_n**2))) * mask

        G_total += Gk
        V += B_lat * np.tanh(n / sigma_n) * Gk

        x1 = x0 - halfL_geom * np.cos(theta)
        y1 = yv - halfL_geom * np.sin(theta)
        x2 = x0 + halfL_geom * np.cos(theta)
        y2 = yv + halfL_geom * np.sin(theta)
        vane_segments.append(((x1, y1), (x2, y2)))

    G_total = np.clip(G_total, 0.0, 2.0)
    V /= max(1, n_vanes)

    U = U_base * (1 - A * G_total)

    U = U * (1.0 + 0.15 * np.sin(2 * np.pi * time_phase))
    V = V * (1.0 + 0.15 * np.cos(2 * np.pi * time_phase))

    speed_vane = np.sqrt(U**2 + V**2)

    fig, ax = plt.subplots(figsize=(6, 5))

    cf = ax.contourf(X, Y, speed_vane, levels=40, cmap="viridis")
    plt.colorbar(cf, ax=ax, label="|U| (m/s)")
    ax.streamplot(X, Y, U, V, density=2, color="k", linewidth=0.6)

    for (p1, p2) in vane_segments:
        ax.plot(
            [p1[0], p2[0]],
            [p1[1], p2[1]],
            color="red",
            linewidth=4,
            solid_capstyle="round",
        )

    ax.set_title("Flow WITH submerged vane array\n(reduced velocity + lateral deviation)")
    ax.set_xlabel("x (longitudinal)")
    ax.set_ylabel("y (transversal)")
    ax.set_xlim(0, Lx)
    ax.set_ylim(0, Ly)

    caption = (
        f"Outer-bank velocity reduction ≈ {red_V * 100:.1f}%   |   "
        f"Shear proxy reduction ≈ {red_tau * 100:.1f}%   |   "
        f"Local scour risk (relative) ≈ {scour_risk:.2f}"
    )
    fig.text(
        0.5,
        0.02,
        caption,
        ha="center",
        va="bottom",
        fontsize=9,
    )

    plt.tight_layout(rect=[0.0, 0.05, 1.0, 1.0])
    return fig


def plot_vane_3d(
    B,
    h,
    L_rel,
    alpha_attack,
    n_vanes,
    bevel_angle,
    material,
    spacing,
    use_vane=True,
):
    """
    3D visualisation of the channel and submerged vanes.

    spacing contrôle la distance entre les vanes en x (le long de l'écoulement).
    """

    L_reach = 40.0
    fig = plt.figure(figsize=(6, 5))
    ax = fig.add_subplot(111, projection="3d")

    color_map = {
        "steel": "silver",
        "concrete": "lightgrey",
        "wood": "saddlebrown",
        "rock": "dimgray",
    }
    vane_color = color_map.get(material, "lightgrey")

    # Channel floor
    floor_verts = [[
        (0, 0, 0),
        (L_reach, 0, 0),
        (L_reach, B, 0),
        (0, B, 0),
    ]]
    floor = Poly3DCollection(floor_verts, alpha=0.15, facecolor="lightblue")
    ax.add_collection3d(floor)

    # Water surface
    surface_verts = [[
        (0, 0, h),
        (L_reach, 0, h),
        (L_reach, B, h),
        (0, B, h),
    ]]
    surface = Poly3DCollection(surface_verts, alpha=0.05, facecolor="blue")
    ax.add_collection3d(surface)

    # Outer bank line
    ax.plot([0, L_reach], [B, B], [0, 0], "k-", lw=2)
    ax.text(L_reach * 0.05, B + 0.2, 0.0, "Outer bank", fontsize=9)

    if use_vane:
        H = 0.3 * h
        vane_height = H
        vane_length = 2.0 * h
        y_center = B * 0.5

        phi = np.deg2rad(alpha_attack)

        # spacing en x en fonction de H
        dx_spacing = spacing * H
        total_length = dx_spacing * (n_vanes - 1)
        x_center = 0.5 * L_reach
        xs = x_center + (np.arange(n_vanes) - (n_vanes - 1) / 2.0) * dx_spacing

        xs = np.clip(xs, 0.2 * L_reach, 0.8 * L_reach)

        vane_areas = []

        bevel_norm = np.clip(bevel_angle / 70.0, 0.0, 1.0)
        drop_ratio = 0.6 * bevel_norm
        z_high = H
        z_low = H * (1.0 - drop_ratio)
        z_low = max(z_low, 0.1 * H)

        first_base_for_alpha = None

        for x0 in xs:
            z0 = 0.0

            dx = vane_length * np.cos(phi)
            dy = vane_length * np.sin(phi)

            p1 = (x0,          y_center,          z0)       # upstream bottom
            p2 = (x0 + dx,     y_center + dy,     z0)       # downstream bottom
            p4 = (x0,          y_center,          z_high)   # upstream top
            p3 = (x0 + dx,     y_center + dy,     z_low)    # downstream top (bevel)

            poly = Poly3DCollection([[p1, p2, p3, p4]],
                                    facecolor=vane_color,
                                    edgecolor="k",
                                    alpha=0.9)
            ax.add_collection3d(poly)

            area = vane_length * vane_height
            vane_areas.append(area)

            if first_base_for_alpha is None:
                first_base_for_alpha = (x0, y_center)

        total_area = sum(vane_areas)

        if first_base_for_alpha is not None:
            bx, by = first_base_for_alpha
            z_alpha = 0.05 * h
            r = 0.3 * vane_length

            ax.plot([bx, bx + r], [by, by], [z_alpha, z_alpha],
                    color="k", lw=2)
            ax.plot(
                [bx, bx + r * np.cos(phi)],
                [by, by + r * np.sin(phi)],
                [z_alpha, z_alpha],
                color=vane_color,
                lw=2,
            )
            ax.text(
                bx + 0.6 * r,
                by + 0.15 * r,
                z_alpha + 0.02 * h,
                f"α = {alpha_attack:.0f}°",
                fontsize=9,
                ha="left",
                va="bottom",
            )

        txt = (
            f"{n_vanes} vane(s)\n"
            f"H ≈ {H:.2f} m\n"
            f"α = {alpha_attack:.1f}°\n"
            f"Bevel θ = {bevel_angle:.1f}°\n"
            f"Spacing = {spacing:.2f} H\n"
            f"Material = {material}\n"
            f"Base area ≈ {total_area:.1f} m²"
        )
        ax.text(
            L_reach * 0.02,
            0.1 * B,
            1.05 * h,
            txt,
            fontsize=9,
            ha="left",
            va="bottom",
        )

    ax.set_xlim(0, L_reach)
    ax.set_ylim(0, B)
    ax.set_zlim(0, 1.2 * h)

    ax.set_xlabel("Streamwise x [m]")
    ax.set_ylabel("Cross-stream y [m]")
    ax.set_zlabel("Depth z [m]")

    ax.set_title("3D view of submerged vane configuration\n(constant height H, beveled top)")
    ax.view_init(elev=25, azim=-60)
    ax.grid(True, alpha=0.2)

    plt.tight_layout()
    return fig


# ============================================================
# 4. Streamlit App
# ============================================================

def main():
    st.set_page_config(page_title="Submerged Vane Bend Lab", layout="wide")

    st.title("🌊 Submerged Vane Bend Lab")
    st.markdown(
        "Interactive toy model for **river bend hydraulics** with submerged vanes,\n"
        "materials, bevels, arrays, spacing, and local scour risk."
    )

    # ---------- Sidebar: bend config ----------
    st.sidebar.header("1️⃣ Bend configuration")

    bend_choice = st.sidebar.selectbox(
        "Select bend type",
        ["Urban channel", "Medium gravel-bed river", "Large lowland river"],
    )

    if bend_choice == "Urban channel":
        B = 15.0
        h = 2.0
        Q = 60.0
    elif bend_choice == "Medium gravel-bed river":
        B = 25.0
        h = 3.5
        Q = 150.0
    else:
        B = 40.0
        h = 5.0
        Q = 400.0

    S = 0.001
    n0 = 0.035
    alpha_bend = st.sidebar.slider(
        "Bend amplification factor α (outer-bank)",
        1.0,
        1.6,
        1.3,
        0.05,
    )

    # ---------- Sidebar: vanes ----------
    st.sidebar.markdown("---")
    st.sidebar.header("2️⃣ Vane configuration")

    use_vane = st.sidebar.checkbox("Use submerged vanes in the bend", value=True)

    material = st.sidebar.selectbox("Vane material", ["steel", "concrete", "wood", "rock"])

    L_rel = st.sidebar.slider(
        "Relative vane length L_rel (vs outer-zone width)",
        0.1,
        0.9,
        0.6,
        0.05,
    )

    alpha_attack = st.sidebar.slider(
        "Attack angle (for hydraulics model) [°]",
        5,
        30,
        20,
        1,
    )

    n_vanes = st.sidebar.slider("Number of vanes", 1, 5, 3, 1)

    use_bevel = st.sidebar.checkbox("Bevelled leading edge", value=True)
    bevel_angle = st.sidebar.slider(
        "Bevel angle θ [°]",
        0,
        70,
        30,
        10,
        disabled=not use_bevel,
    )
    if not use_bevel:
        bevel_angle = 0.0

    spacing = st.sidebar.slider(
        "Vane spacing (in vane heights H)",
        0.5,
        3.0,
        1.3,
        0.1,
        help="1.2–1.5 H ≈ optimal interaction from flume studies.",
    )

    time_phase = st.sidebar.slider(
        "Time / animation phase",
        0.0,
        1.0,
        0.0,
        0.05,
        help="Just for a small pseudo-animation of the velocity field.",
    )

    # ---------- Hydraulics ----------
    if not use_vane:
        gamma = 0.0
        gamma_raw = 0.0
        scour_risk = 0.0
        f_spacing = 0.0
        n_mult, _ = material_properties("concrete")
    else:
        f_spacing = spacing_factor(spacing)

        gamma_raw = gamma_from_geometry(
            L_rel=L_rel,
            alpha_attack_deg=alpha_attack,
            n_vanes=n_vanes,
            is_array=(n_vanes > 1),
            bevel_angle_deg=bevel_angle,
            material=material,
            gamma_base=0.25,
        )

        gamma = gamma_raw * (0.5 + 0.5 * f_spacing)
        gamma = float(np.clip(gamma, 0.0, 0.7))

        scour_base = local_scour_risk_from_geometry(
            L_rel=L_rel,
            alpha_attack_deg=alpha_attack,
            n_vanes=n_vanes,
            bevel_angle_deg=bevel_angle,
            material=material,
        )
        scour_risk = scour_base * (1.0 - 0.3 * f_spacing)
        scour_risk = float(np.clip(scour_risk, 0.0, 1.5))

        n_mult, _ = material_properties(material)

    n_o_eff = n0 * n_mult

    # Base case (no vane)
    base_res = compute_outer_velocity(
        B=B,
        h=h,
        S=S,
        n_i=n0,
        n_m=n0,
        n_o=n0,
        Q=Q,
        alpha=alpha_bend,
        gamma=0.0,
    )
    Vo_base = base_res["Vo_bend"]

    # With vanes
    res = compute_outer_velocity(
        B=B,
        h=h,
        S=S,
        n_i=n0,
        n_m=n0,
        n_o=n_o_eff,
        Q=Q,
        alpha=alpha_bend,
        gamma=gamma,
    )
    Vo = res["Vo_bend"]

    red_V = 1.0 - Vo / Vo_base
    red_tau = 1.0 - (Vo / Vo_base) ** 2

    # ---------- Quick config header ----------
    st.markdown(
        f"**Current vane setup:** "
        f"Bevel angle = `{bevel_angle:.1f}°`, "
        f"Spacing = `{spacing:.2f} H`, "
        f"n_vanes = `{n_vanes}`"
    )

    # ---------- Metrics ----------
    col1, col2, col3 = st.columns(3)
    col1.metric("Outer-bank velocity", f"{Vo:.2f} m/s")
    col2.metric("Velocity reduction vs no-vanes", f"{red_V * 100:.1f} %")
    col3.metric("Local scour risk (relative)", f"{scour_risk:.2f}")

    st.markdown("---")

    # ---------- TOP ROW: 3D vanes + scour plan view ----------
    top_left, top_right = st.columns(2)

    with top_left:
        fig_3d = plot_vane_3d(
            B=B,
            h=h,
            L_rel=L_rel,
            alpha_attack=alpha_attack,
            n_vanes=n_vanes,
            bevel_angle=bevel_angle,
            material=material,
            spacing=spacing,
            use_vane=use_vane,
        )
        st.pyplot(fig_3d)

    with top_right:
        fig_scour = plot_scour_plan_view(
            B=B,
            h=h,
            n_vanes=n_vanes,
            bevel_angle=bevel_angle,
            scour_risk=scour_risk,
            use_vane=use_vane,
            alpha_attack_deg=alpha_attack,
            spacing=spacing,
        )
        st.pyplot(fig_scour)

    # ---------- BOTTOM: velocity field ----------
    fig_vf = plot_velocity_field_fig(
        B=B,
        h=h,
        L_rel=L_rel,
        alpha_attack=alpha_attack,
        gamma=gamma,
        material=material,
        n_vanes=n_vanes,
        time_phase=time_phase,
        red_V=red_V,
        red_tau=red_tau,
        scour_risk=scour_risk,
        spacing=spacing,
    )
    st.pyplot(fig_vf)

    # ---------- Summary ----------
    st.markdown("---")
    st.subheader("Configuration summary")
    st.write(
        f"- **Bend type**: {bend_choice}  \n"
        f"- **B = {B:.1f} m**, **h = {h:.1f} m**, **Q = {Q:.1f} m³/s**, **α = {alpha_bend:.2f}**  \n"
        f"- **Vanes active**: {'Yes' if use_vane else 'No'}  \n"
        f"- **Material**: {material} (nᵒ effective ≈ {n_o_eff:.3f})  \n"
        f"- **L_rel** = {L_rel:.2f}, **attack angle** = {alpha_attack:.1f}°, "
        f"**n_vanes** = {n_vanes}, **bevel angle θ** = {bevel_angle:.1f}°  \n"
        f"- **Spacing** = {spacing:.2f} H, **spacing factor** fₛ ≈ {f_spacing:.2f}  \n"
        f"- **γ_raw (geometry)** ≈ {gamma_raw:.3f}  \n"
        f"- **γ (effective deflected Qo)** ≈ {gamma:.3f}  \n"
        f"- **Outer-bank velocity reduction** ≈ {red_V * 100:.1f}%  \n"
        f"- **Shear proxy reduction** ≈ {red_tau * 100:.1f}%  \n"
        f"- **Local scour risk (relative)** ≈ {scour_risk:.2f}"
    )


if __name__ == "__main__":
    main()
