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
    """
    Conveyance K = (1/n) * A * Rh^(2/3)
    """
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
# 2. Material and geometry → gamma & local scour risk
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

    # Length factor: more length → more interaction with outer flow
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

    # Bevel effect on guidance (small effect)
    if bevel_angle_deg is None or bevel_angle_deg <= 0:
        f_bevel_geom = 1.0
    else:
        bevel_ref = 30.0
        f_bevel_geom = 1.0 + 0.2 * ((bevel_angle_deg - bevel_ref) / bevel_ref)
        f_bevel_geom = np.clip(f_bevel_geom, 0.8, 1.2)

    # Material effect
    n_mult_mat, _ = material_properties(material)
    f_mat_gamma = 1.0 - 0.2 * (n_mult_mat - 1.0)
    f_mat_gamma = np.clip(f_mat_gamma, 0.8, 1.1)

    gamma = gamma_base * f_L * f_alpha * f_array * f_bevel_geom * f_mat_gamma
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

    # Base risk for a "classical" single non-bevelled concrete vane
    base_risk = 0.8

    # Length effect
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

    # Bevel effect
    if bevel_angle_deg is None or bevel_angle_deg <= 0:
        f_bevel = 1.0
    else:
        bevel_norm = np.clip(bevel_angle_deg / 70.0, 0.0, 1.0)
        f_bevel = 1.0 - 0.5 * bevel_norm  # jusqu'à -50% du scour à 70°
        f_bevel = np.clip(f_bevel, 0.5, 1.0)

    # Material effect
    _, scour_mult_mat = material_properties(material)

    risk = base_risk * f_L * f_alpha * f_n * f_bevel * scour_mult_mat
    risk = np.clip(risk, 0.0, 1.5)
    return risk


# ============================================================
# 3. Pseudo-CFD field builder
# ============================================================

def build_velocity_field(B, h, Q, n0, alpha_bend, gamma, n_o_eff, nx=50, ny=25):
    """
    Build a 2D velocity field in a bend reach using 3-zone velocities.
    """
    L_reach = 100.0  # [m] schematic length
    x = np.linspace(0, L_reach, nx)
    y = np.linspace(0, B, ny)
    X, Y = np.meshgrid(x, y)

    hydro = compute_outer_velocity(
        B=B,
        h=h,
        S=0.001,
        n_i=n0,
        n_m=n0,
        n_o=n_o_eff,
        Q=Q,
        alpha=alpha_bend,
        gamma=gamma,
    )
    (Bi, Bm, Bo) = hydro["zones_width"]
    (Vi, Vm, Vo) = hydro["V_after"]
    Vo_bend = hydro["Vo_bend"]

    U = np.zeros_like(X)
    Vlat = np.zeros_like(X)

    for j in range(ny):
        yj = y[j]
        if yj < Bi:
            U[j, :] = Vi
        elif yj < Bi + Bm:
            U[j, :] = Vm
        else:
            U[j, :] = Vo_bend

    vane_x_start = 0.4 * L_reach
    vane_x_end = 0.7 * L_reach

    for i in range(nx):
        for j in range(ny):
            if vane_x_start <= X[j, i] <= vane_x_end and y[j] > Bi + Bm:
                strength = gamma * 0.5
                Vlat[j, i] = -strength  # towards mid-channel

    return X, Y, U, Vlat, (Bi, Bm, Bo)


# ============================================================
# 4. Plot helpers (cross-section, velocity field, vane 3D)
# ============================================================

def plot_cross_section(B, h, Bi, Bm, Bo, scour_risk, use_vane, use_bevel):
    fig, ax = plt.subplots(figsize=(5, 3))

    # Draw zones
    ax.add_patch(Rectangle((0, 0), Bi, h, fill=False))
    ax.add_patch(Rectangle((Bi, 0), Bm, h, fill=False))
    ax.add_patch(Rectangle((Bi + Bm, 0), Bo, h, fill=False))

    ax.text(Bi / 2, h * 0.5, "Inner", ha="center", va="center", fontsize=8)
    ax.text(Bi + Bm / 2, h * 0.5, "Mid", ha="center", va="center", fontsize=8)
    ax.text(Bi + Bm + Bo / 2, h * 0.5, "Outer bank", ha="center", va="center", fontsize=8)

    # Flow direction
    ax.annotate(
        "",
        xy=(B * 0.9, h + 0.2),
        xytext=(B * 0.1, h + 0.2),
        arrowprops=dict(arrowstyle="->", linewidth=1.3),
    )
    ax.text(B * 0.5, h + 0.25, "Main flow", ha="center", va="bottom", fontsize=8)

    # Vane position (schematic)
    vane_x = Bi + Bm + Bo * 0.3
    ax.plot([vane_x, vane_x], [0, h * 0.4], lw=3)
    vane_label = "Vanes"
    if use_bevel:
        vane_label += "\n(bevelled)"
    ax.text(vane_x, h * 0.45, vane_label, ha="center", va="bottom", fontsize=8)

    # Elliptic red scour zone near vane, size ∝ scour_risk
    if use_vane and scour_risk > 0:
        max_width = 0.25 * Bo
        max_height = 0.6 * h
        w = max_width * min(1.0, scour_risk)
        h_e = max_height * min(1.0, scour_risk)
        ell = Ellipse(
            (Bi + Bm + 0.75 * Bo, h * 0.15),
            width=w,
            height=h_e,
            angle=0,
            color="red",
            alpha=0.35,
        )
        ax.add_patch(ell)
        ax.text(
            Bi + Bm + 0.75 * Bo,
            h * 0.15 + h_e * 0.6,
            "Local scour zone",
            ha="center",
            va="bottom",
            fontsize=8,
            color="red",
        )

    ax.set_xlim(-1, B + 1)
    ax.set_ylim(-0.2, h + 0.8)
    ax.set_xlabel("Cross-section width [schematic]")
    ax.set_ylabel("Depth [schematic]")
    ax.set_title("Cross-section & local scour footprint")
    ax.set_aspect("equal", adjustable="box")
    ax.grid(False)
    return fig


def plot_velocity_field_fig(
    X_base, Y_base, U_base, V_base,
    X_vane, Y_vane, U_vane, V_vane,
    Bi, Bm, Bo
):
    """
    Viz type "avant / après" à la façon de ton script :
      - gauche : écoulement sans vanes
      - droite : écoulement avec vanes
    On utilise le champ pseudo-CFD 3-zones pour les deux cas.
    """
    # Domaine (mêmes grilles)
    Lx = X_base.max()
    Ly = Y_base.max()

    speed_base = np.sqrt(U_base**2 + V_base**2)
    speed_vane = np.sqrt(U_vane**2 + V_vane**2)

    fig, axes = plt.subplots(1, 2, figsize=(9, 3), sharey=True)

    # ---------- 1) SANS vanes ----------
    ax = axes[0]
    cf1 = ax.contourf(X_base, Y_base, speed_base, levels=40, cmap="viridis")
    plt.colorbar(cf1, ax=ax, label="|U| (m/s)")
    ax.streamplot(X_base, Y_base, U_base, V_base, density=2, color="k", linewidth=0.6)
    ax.set_title("Flow WITHOUT vanes")
    ax.set_xlabel("x (streamwise)")
    ax.set_ylabel("y (cross-stream)")
    ax.set_xlim(0, Lx)
    ax.set_ylim(0, Ly)

    # ---------- 2) AVEC vanes ----------
    ax = axes[1]
    cf2 = ax.contourf(X_vane, Y_vane, speed_vane, levels=40, cmap="viridis")
    plt.colorbar(cf2, ax=ax, label="|U| (m/s)")
    ax.streamplot(X_vane, Y_vane, U_vane, V_vane, density=2, color="k", linewidth=0.6)

    # Position schématique des vanes dans la zone externe
    vane_x_start = 0.4 * Lx
    vane_x_end = 0.7 * Lx
    xv = 0.5 * (vane_x_start + vane_x_end)
    yv = Bi + Bm + 0.5 * Bo  # milieu de la zone outer

    ax.plot([vane_x_start, vane_x_end], [yv, yv],
            color="red", linewidth=4, solid_capstyle="round",
            label="Submerged vane array")
    ax.legend(loc="upper right")

    ax.set_title("Flow WITH submerged vanes")
    ax.set_xlabel("x (streamwise)")
    ax.set_xlim(0, Lx)
    ax.set_ylim(0, Ly)

    plt.tight_layout()
    return fig


def plot_vane_3d(
    B,
    h,
    L_rel,
    alpha_attack,
    n_vanes,
    bevel_angle,
    material,
    use_vane=True,
):
    """
    3D visualisation of the channel and submerged vanes.

    - vanes au milieu du canal (y ~ B/2)
    - flow vers +x
    - angle d'attaque alpha_attack = angle entre la vane et la direction du flow (plan vue)
    - bevel_angle = différence de hauteur entre amont et aval de la vane
    """

    L_reach = 40.0  # length of the plotted reach in x
    fig = plt.figure(figsize=(6, 5))
    ax = fig.add_subplot(111, projection="3d")

    # --- Colors per material ---
    color_map = {
        "steel": "silver",
        "concrete": "lightgrey",
        "wood": "saddlebrown",
        "rock": "dimgray",
    }
    vane_color = color_map.get(material, "lightgrey")

    # --- Channel floor ---
    floor_verts = [[
        (0,   0, 0),
        (L_reach, 0, 0),
        (L_reach, B, 0),
        (0,   B, 0),
    ]]
    floor = Poly3DCollection(floor_verts, alpha=0.15, facecolor="lightblue")
    ax.add_collection3d(floor)

    # --- Water surface at z = h ---
    surface_verts = [[
        (0,   0, h),
        (L_reach, 0, h),
        (L_reach, B, h),
        (0,   B, h),
    ]]
    surface = Poly3DCollection(surface_verts, alpha=0.05, facecolor="blue")
    ax.add_collection3d(surface)

    # --- Outer bank line (y = B) pour le contexte ---
    ax.plot([0, L_reach], [B, B], [0, 0], "k-", lw=2)
    ax.text(L_reach * 0.05, B + 0.2, 0.0, "Outer bank", fontsize=9)

    # --- Flow direction arrow (along +x) ---
    ax.quiver(
        0.0, B * 0.15, 0.6 * h,
        10.0, 0.0, 0.0,
        length=1.0,
        normalize=False,
        arrow_length_ratio=0.2,
    )
    ax.text(
        10.0, B * 0.15, 0.6 * h,
        "Flow direction",
        fontsize=9,
        ha="left",
        va="center",
    )

    if use_vane:
        # Outer zone width ~ 0.3 * B, on s'en sert pour calibrer la longueur
        Bo = 0.3 * B

        # vanes plus LONGUES que hautes
        vane_height = 0.3 * h           # ~30% de la profondeur
        vane_length = 2.0 * h           # longueur ≈ 2× profondeur

        y_center = B * 0.5            # vanes au milieu du canal

        # angle d’attaque en rad
        phi = np.deg2rad(alpha_attack)

        # positions en x centrées dans le reach
        xs = np.linspace(L_reach * 0.3, L_reach * 0.7, n_vanes)
        vane_areas = []

        # bevel – différence de hauteur entre les deux extrémités
        theta_b = np.deg2rad(bevel_angle)
        dz_bevel = vane_height * 0.6 * np.sin(theta_b)

        first_base_for_alpha = None

        for x0 in xs:
            z0 = 0.0

            # direction de la vane dans le plan (angle par rapport au flow x)
            dx = vane_length * np.cos(phi)
            dy = vane_length * np.sin(phi)

            # bas du panneau
            p1 = (x0,          y_center,          z0)  # amont bas
            p2 = (x0 + dx,     y_center + dy,     z0)  # aval bas

            # haut du panneau (amont plus haut si bevel>0)
            z1_top = vane_height + dz_bevel  # amont haut
            z2_top = vane_height             # aval haut

            p4 = (x0,          y_center,          z1_top)  # amont haut
            p3 = (x0 + dx,     y_center + dy,     z2_top)  # aval haut

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

        # --- petit dessin de l’angle d’attaque α dans le plan (au sol) ---
        if first_base_for_alpha is not None:
            bx, by = first_base_for_alpha
            z_alpha = 0.05 * h
            r = 0.3 * vane_length  # longueur des segments de visualisation

            # segment dans la direction du flow (x)
            ax.plot(
                [bx, bx + r],
                [by, by],
                [z_alpha, z_alpha],
                color="k",
                lw=2,
            )

            # segment dans la direction de la vane
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
            f"L_rel = {L_rel:.2f}\n"
            f"α = {alpha_attack:.1f}°\n"
            f"Bevel θ = {bevel_angle:.1f}°\n"
            f"Material = {material}\n"
            f"Total area ≈ {total_area:.1f} m²"
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

    ax.set_title("3D view of submerged vane configuration (α & bevel visible)")

    ax.view_init(elev=25, azim=-60)
    ax.grid(True, alpha=0.2)

    plt.tight_layout()
    return fig


# ============================================================
# 5. Streamlit App
# ============================================================

def main():
    st.set_page_config(page_title="Submerged Vane Bend Lab", layout="wide")

    st.title("🌊 Submerged Vane Bend Lab")
    st.markdown(
        "Interactive toy model for **river bend hydraulics** with submerged vanes,\n"
        "materials, bevels, arrays, and local scour risk."
    )

    # -------- Sidebar controls --------
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

    time_phase = st.sidebar.slider(
        "Time / animation phase",
        0.0,
        1.0,
        0.0,
        0.05,
        help="Just for a small pseudo-animation of the velocity field.",
    )

    # -------- Compute hydraulics --------
    if not use_vane:
        gamma = 0.0
        scour_risk = 0.0
        n_mult, _ = material_properties("concrete")
    else:
        gamma = gamma_from_geometry(
            L_rel=L_rel,
            alpha_attack_deg=alpha_attack,
            n_vanes=n_vanes,
            is_array=(n_vanes > 1),
            bevel_angle_deg=bevel_angle,
            material=material,
            gamma_base=0.25,
        )
        scour_risk = local_scour_risk_from_geometry(
            L_rel=L_rel,
            alpha_attack_deg=alpha_attack,
            n_vanes=n_vanes,
            bevel_angle_deg=bevel_angle,
            material=material,
        )
        n_mult, _ = material_properties(material)

    n_o_eff = n0 * n_mult

    # Base case (no vane) for comparison
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

    # With current config
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
    (Bi, Bm, Bo) = res["zones_width"]

    red_V = 1.0 - Vo / Vo_base
    red_tau = 1.0 - (Vo / Vo_base) ** 2

    # -------- Metrics --------
    col1, col2, col3 = st.columns(3)
    col1.metric("Outer-bank velocity", f"{Vo:.2f} m/s")
    col2.metric("Velocity reduction vs no-vanes", f"{red_V * 100:.1f} %")
    col3.metric("Local scour risk (relative)", f"{scour_risk:.2f}")

    st.markdown("---")

    # -------- Build velocity fields: WITHOUT and WITH vanes --------
    X0, Y0, U0, V0, (Bi2, Bm2, Bo2) = build_velocity_field(
        B=B,
        h=h,
        Q=Q,
        n0=n0,
        alpha_bend=alpha_bend,
        gamma=0.0,
        n_o_eff=n0,
        nx=80,
        ny=40,
    )

    X1, Y1, U1, V1, _ = build_velocity_field(
        B=B,
        h=h,
        Q=Q,
        n0=n0,
        alpha_bend=alpha_bend,
        gamma=gamma,
        n_o_eff=n_o_eff,
        nx=80,
        ny=40,
    )

    # petite "animation" en phase
    U1_mod = U1 * (1.0 + 0.15 * np.sin(2 * np.pi * time_phase))
    V1_mod = V1 * (1.0 + 0.15 * np.cos(2 * np.pi * time_phase))

    # -------- Plots: 3D vane, cross-section, velocity field --------
    c1, c2, c3 = st.columns(3)

    with c1:
        fig_3d = plot_vane_3d(
            B=B,
            h=h,
            L_rel=L_rel,
            alpha_attack=alpha_attack,
            n_vanes=n_vanes,
            bevel_angle=bevel_angle,
            material=material,
            use_vane=use_vane,
        )
        st.pyplot(fig_3d)

    with c2:
        fig_cs = plot_cross_section(B, h, Bi, Bm, Bo, scour_risk, use_vane, use_bevel)
        st.pyplot(fig_cs)

    with c3:
        fig_vf = plot_velocity_field_fig(
            X0, Y0, U0, V0,
            X1, Y1, U1_mod, V1_mod,
            Bi2, Bm2, Bo2,
        )
        st.pyplot(fig_vf)

    # -------- Text summary --------
    st.markdown("---")
    st.subheader("Configuration summary")
    st.write(
        f"- **Bend type**: {bend_choice}  \n"
        f"- **B = {B:.1f} m**, **h = {h:.1f} m**, **Q = {Q:.1f} m³/s**, **α = {alpha_bend:.2f}**  \n"
        f"- **Vanes active**: {'Yes' if use_vane else 'No'}  \n"
        f"- **Material**: {material} (nᵒ effective ≈ {n_o_eff:.3f})  \n"
        f"- **L_rel** = {L_rel:.2f}, **attack angle** = {alpha_attack:.1f}°, "
        f"**n_vanes** = {n_vanes}, **bevel angle θ** = {bevel_angle:.1f}°  \n"
        f"- **γ (deflected Qo)** ≈ {gamma:.3f}  \n"
        f"- **Outer-bank velocity reduction** ≈ {red_V * 100:.1f}%  \n"
        f"- **Shear proxy reduction** ≈ {red_tau * 100:.1f}%  \n"
        f"- **Local scour risk (relative)** ≈ {scour_risk:.2f}"
    )


if __name__ == "__main__":
    main()

