import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse, Rectangle
import streamlit as st

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

    B  : total width [m]
    h  : depth [m]
    S  : slope [-] (not directly used here)
    n_*: Manning roughness in inner/mid/outer zones
    Q  : total discharge [m^3/s]
    alpha : amplification factor for outer-bank velocity (bend effect)
    gamma : fraction of outer-zone discharge diverted to mid-channel by vanes
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

    material: "steel", "concrete", "wood", "rock"
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
    as a function of:
    - L_rel: relative vane length (0–1) vs outer-zone width
    - alpha_attack_deg: attack angle (plan view) [deg]
    - n_vanes: nb of vanes
    - is_array: bool
    - bevel_angle_deg: bevel angle on leading edge [deg]
    - material: string ("steel", "concrete", "wood", "rock")

    gamma_base: base deflection efficiency for a reference vane.
    """

    # Length factor: more length → more interaction with outer flow
    L_ref = 0.5
    f_L = 1.0 + 0.6 * (L_rel - L_ref)
    f_L = np.clip(f_L, 0.3, 1.5)

    # Attack angle factor: up to ~25° increases lateral deflection
    alpha_ref = 20.0
    f_alpha = 1.0 + 0.5 * ((alpha_attack_deg - alpha_ref) / alpha_ref)
    f_alpha = np.clip(f_alpha, 0.5, 1.5)

    # Array effect: more vanes = more global deflection (up to a limit)
    if is_array:
        f_array = 1.0 + 0.1 * (n_vanes - 1)  # +10% per vane beyond the first
    else:
        f_array = 1.0
    f_array = np.clip(f_array, 1.0, 1.5)

    # Bevel angle: bevelled edge can slightly improve guidance of flow
    if bevel_angle_deg is None or bevel_angle_deg <= 0:
        f_bevel_geom = 1.0
    else:
        bevel_ref = 30.0
        f_bevel_geom = 1.0 + 0.2 * ((bevel_angle_deg - bevel_ref) / bevel_ref)
        f_bevel_geom = np.clip(f_bevel_geom, 0.8, 1.2)

    # Material may also influence effective gamma via friction (small effect)
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

    - Longer vane + higher attack angle → more vortices → more scour
    - More vanes → more local scour spots
    - Bevel reduces scour
    - Rough materials (rock) → more turbulent structures → more local scour
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
        bevel_norm = np.clip(bevel_angle_deg / 45.0, 0.0, 1.0)
        f_bevel = 1.0 - 0.5 * bevel_norm  # jusqu'à -50% du scour
        f_bevel = np.clip(f_bevel, 0.5, 1.0)

    # Material effect
    _, scour_mult_mat = material_properties(material)

    risk = base_risk * f_L * f_alpha * f_n * f_bevel * scour_mult_mat
    risk = np.clip(risk, 0.0, 1.5)
    return risk


# ============================================================
# 3. Pseudo-CFD field builder
# ============================================================

def build_velocity_field(B, h, Q, n0, alpha_bend, gamma, n_o_eff, nx=40, ny=15):
    """
    Build a simple 2D velocity field in a bend reach using 3-zone velocities.
    """
    L_reach = 100.0  # [m] schematic length
    x = np.linspace(0, L_reach, nx)
    y = np.linspace(0, B, ny)
    X, Y = np.meshgrid(x, y)

    # Recompute hydraulics for given gamma + n_o_eff
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

    U = np.zeros_like(X)    # streamwise
    Vlat = np.zeros_like(X) # lateral

    for j in range(ny):
        yj = y[j]
        if yj < Bi:
            U[j, :] = Vi
        elif yj < Bi + Bm:
            U[j, :] = Vm
        else:
            U[j, :] = Vo_bend

    # lateral component in vane zone
    vane_x_start = 0.4 * L_reach
    vane_x_end = 0.7 * L_reach

    for i in range(nx):
        for j in range(ny):
            if vane_x_start <= X[j, i] <= vane_x_end and y[j] > Bi + Bm:
                strength = gamma * 0.5
                Vlat[j, i] = -strength  # towards mid-channel

    return X, Y, U, Vlat, (Bi, Bm, Bo)


# ============================================================
# 4. Streamlit App
# ============================================================

def main():
    st.set_page_config(page_title="Submerged Vane Bend Lab", layout="wide")

    st.title("🌊 Submerged Vane Bend Lab")
    st.markdown(
        "Interactive toy model for **river bend hydraulics** with "
        "submerged vanes, bevels, materials and local scour risk."
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
        "Attack angle (plan view) [°]",
        5,
        30,
        20,
        1,
    )

    n_vanes = st.sidebar.slider("Number of vanes", 1, 5, 3, 1)

    use_bevel = st.sidebar.checkbox("Bevelled leading edge", value=True)
    bevel_angle = st.sidebar.slider(
        "Bevel angle [°]",
        0,
        45,
        30,
        5,
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
    (Vi, Vm, Vo_inner) = res["V_after"]

    red_V = 1.0 - Vo / Vo_base
    red_tau = 1.0 - (Vo / Vo_base) ** 2

    # -------- Layout: metrics + plots --------
    col1, col2, col3 = st.columns(3)

    col1.metric("Outer-bank velocity (with vanes)", f"{Vo:.2f} m/s")
    col2.metric("Velocity reduction vs no-vanes", f"{red_V * 100:.1f} %")
    col3.metric("Local scour risk (relative)", f"{scour_risk:.2f}")

    st.markdown("---")

    col_left, col_right = st.columns(2)

    # ===== Left: cross-section schematic with red ellipse for scour =====
    with col_left:
        fig, ax = plt.subplots(figsize=(6, 3))

        # Draw zones
        ax.add_patch(Rectangle((0, 0), Bi, h, fill=False))
        ax.add_patch(Rectangle((Bi, 0), Bm, h, fill=False))
        ax.add_patch(Rectangle((Bi + Bm, 0), Bo, h, fill=False))

        ax.text(Bi / 2, h * 0.5, "Inner", ha="center", va="center", fontsize=9)
        ax.text(Bi + Bm / 2, h * 0.5, "Mid", ha="center", va="center", fontsize=9)
        ax.text(Bi + Bm + Bo / 2, h * 0.5, "Outer bank", ha="center", va="center", fontsize=9)

        # Flow direction
        ax.annotate(
            "",
            xy=(B * 0.9, h + 0.2),
            xytext=(B * 0.1, h + 0.2),
            arrowprops=dict(arrowstyle="->", linewidth=1.3),
        )
        ax.text(B * 0.5, h + 0.25, "Main flow", ha="center", va="bottom", fontsize=9)

        # Vane position
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

        st.pyplot(fig)

    # ===== Right: pseudo-CFD velocity field =====
    with col_right:
        X, Y, U, Vlat, (Bi2, Bm2, Bo2) = build_velocity_field(
            B=B,
            h=h,
            Q=Q,
            n0=n0,
            alpha_bend=alpha_bend,
            gamma=gamma,
            n_o_eff=n_o_eff,
            nx=40,
            ny=20,
        )

        # add a small "time" modulation
        U_mod = U * (1.0 + 0.15 * np.sin(2 * np.pi * time_phase))
        V_mod = Vlat * (1.0 + 0.15 * np.cos(2 * np.pi * time_phase))
        speed = np.sqrt(U_mod ** 2 + V_mod ** 2)

        fig2, ax2 = plt.subplots(figsize=(7, 3))
        skip = (slice(None, None, 2), slice(None, None, 2))

        q = ax2.quiver(
            X[skip],
            Y[skip],
            U_mod[skip],
            V_mod[skip],
            speed[skip],
            scale=40,
        )

        ax2.set_title("Pseudo-velocity field in the bend (plan view)")
        ax2.set_xlabel("Streamwise distance [schematic]")
        ax2.set_ylabel("Channel width [schematic]")

        # Outer bank division line
        ax2.axhline(Bi2 + Bm2, color="k", linestyle="--", alpha=0.4)
        ax2.text(5, Bi2 + Bm2 + 0.5, "Outer bank zone", va="bottom", fontsize=8)

        # Vane zone
        L_reach = 100.0
        vane_x_start = 0.4 * L_reach
        vane_x_end = 0.7 * L_reach
        ax2.axvspan(vane_x_start, vane_x_end, color="grey", alpha=0.1)
        ax2.text(
            (vane_x_start + vane_x_end) / 2,
            B - 0.5,
            "Vane reach",
            ha="center",
            fontsize=8,
        )

        fig2.colorbar(q, ax=ax2, label="Speed [relative]")
        ax2.grid(True, linestyle="--", alpha=0.3)
        st.pyplot(fig2)

    # --------- Text summary ---------
    st.markdown("---")
    st.subheader("Configuration summary")

    st.write(
        f"- **Bend type**: {bend_choice}  \n"
        f"- **B = {B:.1f} m**, **h = {h:.1f} m**, **Q = {Q:.1f} m³/s**, **α = {alpha_bend:.2f}**  \n"
        f"- **Vanes active**: {'Yes' if use_vane else 'No'}  \n"
        f"- **Material**: {material} (nᵒ effective ≈ {n_o_eff:.3f})  \n"
        f"- **L_rel** = {L_rel:.2f}, **attack angle** = {alpha_attack:.1f}°, "
        f"**n_vanes** = {n_vanes}, **bevel angle** = {bevel_angle:.1f}°  \n"
        f"- **γ (deflected Qo)** ≈ {gamma:.3f}  \n"
        f"- **Outer-bank velocity reduction** ≈ {red_V * 100:.1f}%  \n"
        f"- **Shear proxy reduction** ≈ {red_tau * 100:.1f}%  \n"
        f"- **Local scour risk (relative)** ≈ {scour_risk:.2f}"
    )


if __name__ == "__main__":
    main()
