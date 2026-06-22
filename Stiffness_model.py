# Imports
import math
from Gripper_Analysis import *

def predict_stiffness(
    # Geometry parameters
    depth: float = 12.0,
    width: float = 15.6,
    length: float = 42.0,
    # Infill / notch settings, angle in degrees
    infill_density: float = 0.1,
    infill_angle: float = 00.0,   
    notch_angle: float = 10.0,
    # Beam dimensions taken from the slicer settings
    beam_thickness_center: float = 0.45,
    beam_thickness_side: float = 0.42,
    # Because the gripper is mostly stressed within the layer plane, for young modulus (E), XY-direction is used (slightly higher)
    young_modulus: float = 2308.0,
    visualize: bool = True,
):
    """Run one stiffness analysis for a given parameter set and return K_eff, k_eff_y, and k_eff_z. Angles in degrees"""

    EA_side, EI_side = compute_section_stiffness(young_modulus, depth, beam_thickness_side)
    EA_center, EI_center = compute_section_stiffness(young_modulus, depth, beam_thickness_center)

    infill_angle = math.radians(infill_angle)
    notch_angle = math.radians(notch_angle)

    crossbeam_count = infill_density_to_count(infill_density, beam_thickness_center, length)

    width = width - beam_thickness_side # The input is the outer width, but in the code width is the distance between the center of both side beams

    beam_positions_xy, corner_positions_xy, notch_positions_xy = calculate_node_positions(
        infill_density,
        crossbeam_count,
        infill_angle,
        length,
        width,
        notch_angle=notch_angle,
    )

    K_xy_global, all_nodes_xy, long_plane_nodes_xy, short_plane_nodes_xy, node_index_xy = assemble_global_matrix(
        beam_positions_xy,
        corner_positions_xy,
        notch_positions_xy,
        EA_center,
        EI_center,
        EA_side,
        EI_side,
        width,
    )

    applied_loads_yz_notch = build_applied_loads(
        long_plane_nodes_xy,
        short_plane_nodes_xy,
        node_index_xy,
        f_y_notch=10.0,
        f_z_notch=0.0
    )
    T = create_coordinate_transformation_matrix(n_dof=len(K_xy_global), rotation_angle=notch_angle)
    K_yz_notch = T @ K_xy_global @ T.T

    U_yz_notch, F_reactions, K_reduced_yz_notch, free_dofs, fixed_dofs = solve_direct_stiffness(
        K_yz_notch, applied_loads_yz_notch, node_index_xy
    )

    if visualize:
        visualize_assembled_structure(all_nodes_xy, beam_positions_xy, notch_positions_xy, length, width)
        visualize_deformed_structure(
            all_nodes_xy,
            beam_positions_xy,
            notch_positions_xy,
            U_yz_notch,
            node_index_xy,
            length,
            width,
            notch_angle,
            scale_factor=1.0,
        )

    K_eff_yztheta, K_eff_yz, k_eff_y, k_eff_z = calculate_effective_stiffness(
        K_reduced_yz_notch,
        long_plane_nodes_xy,
        short_plane_nodes_xy,
        node_index_xy,
        free_dofs,
        visualize=visualize,
    )

    # Assumptions for remaining dofs K_xx and K_ry and coupling K_xx_ry:
    # K_xx and K_ry are assumed to be dominated by the side beams, the infill is therfore neglected
    # K_xx and K_ry are calculated based on the stiffness of two parallel beams, so all that has to be done is swap b and h in the I calculation
    _, EI_side_x = compute_section_stiffness(young_modulus, depth = beam_thickness_side ,thickness = depth)
    EI_L3_x = EI_side_x / (length**3)
    EI_L2_x = EI_side_x / (length**2)
    EI_L_x = EI_side_x / length
    k_xx = 2 * 12* EI_L3_x  # Two parallel beams, so the stiffness is doubled
    k_ry = 2 * 4 * EI_L_x   # Two parallel beams, so the stiffness is doubled
    k_xx_ry = -2 * 6 * EI_L2_x  # Two parallel beams, so the stiffness is doubled

    #K_6x6 with kx, ky, kz, krx, kry, krz. 
    K_6x6 = np.array(
		[
        [k_xx,    0, 0, 0, k_xx_ry,0],
        [0,       0, 0, 0, 0,      0],
        [0,       0, 0, 0, 0,      0],
        [0,       0, 0, 0, 0,      0],
        [k_xx_ry, 0, 0, 0, k_ry,   0],
        [0,       0, 0, 0, 0,      np.nan]
		]
	)
    # Insert the k_eff_yztheta into rows 1-4 and columns 1-4
    K_6x6[1:4, 1:4] = K_eff_yztheta

    return K_6x6, K_eff_yztheta, K_eff_yz, float(k_eff_y), float(k_eff_z)

def compute_section_stiffness(E: float, depth: float, thickness: float) -> tuple[float, float]:
    """Return (EA, EI) for a rectangular beam section."""
    EA = E * thickness * depth
    EI = E * (depth * thickness**3) / 12.0
    return EA, EI

def build_applied_loads(
    long_plane_nodes,
    short_plane_nodes,
    node_index,
    f_y_notch: float,
    f_z_notch: float,
) -> dict[int, float]:
    """Build load vector entries in notch coordinates as {dof_index: force}."""
    applied_loads: dict[int, float] = {}

    if long_plane_nodes:
        fy_per_node = f_y_notch / len(long_plane_nodes)
        for node in long_plane_nodes:
            dof_y = 3 * node_index[node]
            applied_loads[dof_y] = fy_per_node

    # Keep existing modeling choice: only first short-notch node is loaded.
    if short_plane_nodes:
        dof_z_tip = 3 * node_index[short_plane_nodes[0]] + 1
        applied_loads[dof_z_tip] = -f_z_notch

    return applied_loads

# ========== Calculate Stiffnesses ==========

def calculate_effective_stiffness(K_reduced_yz_notch, long_plane_nodes_xy, short_plane_nodes_xy, node_index_xy, free_dofs, visualize=False) -> tuple[float, float]:
    """Calculate effective stiffnesses K_eff_y and K_eff_z from the reduced stiffness matrix."""
    # Identify DOFs corresponding to long and short plane nodes
    # The effective y dofs are the y dofs of the long plane (because the y force is applied on the long plane)

    reduced_eff_y_dofs = []
    reduced_eff_z_dofs = []
    reduced_eff_theta_dofs = []

    # Sorting of the nodes is only necessary for visualization, but it doesn't change the effective stiffness values (because they are sums of the matrix entries).
    long_plane_nodes_xy = sorted(long_plane_nodes_xy, key=lambda n: n[0], reverse=True)  # sort by x

    for node in long_plane_nodes_xy:
        idx = node_index_xy[node]
        eff_y_dof = 3 * idx         # dof in the total stiffness matrix
        eff_theta_dof = 3 * idx + 2 # dof for rotation around z-axis (theta_z) in the total stiffness matrix
        if eff_y_dof in free_dofs:
            reduced_eff_y_dofs.append(free_dofs.index(eff_y_dof))   # dof in the reduced stiffness matrix (after applying boundary conditions)
        if eff_theta_dof in free_dofs:
            reduced_eff_theta_dofs.append(free_dofs.index(eff_theta_dof))   # dof in the reduced stiffness matrix (after applying boundary conditions)

    # The effective z dofs are the z dofs of the short plane (because the z force is applied on the short plane)
    short_plane_nodes_xy = sorted(short_plane_nodes_xy, key=lambda n: n[0], reverse=True)  # sort by x

    for node in short_plane_nodes_xy:
        idx = node_index_xy[node]
        eff_z_dof = 3 * idx + 1
        eff_theta_dof = 3 * idx + 2 # dof for rotation around z-axis (theta_z) in the total stiffness matrix
        if eff_z_dof in free_dofs:
            reduced_eff_z_dofs.append(free_dofs.index(eff_z_dof))   # dof in the reduced stiffness matrix (after applying boundary conditions)
        # Make sure the theta dof in the notch corner is not counted twice
        if eff_theta_dof in free_dofs and free_dofs.index(eff_theta_dof) not in reduced_eff_theta_dofs:
            reduced_eff_theta_dofs.append(free_dofs.index(eff_theta_dof))   # dof in the reduced stiffness matrix (after applying boundary conditions)

    # Condense the stiffness matrix for the notch
    # yz only fixes z-displacement for the short plane and y-displacement for the long plane, but it allows all rotations. This resembles the boundary conditions of the experimental setup.
    K_notch_yz = static_condensation(K_reduced_yz_notch, retained_dofs=reduced_eff_y_dofs + reduced_eff_z_dofs)
    # yztheta fixes the same displacements as yz, but it also fixes the rotations of the notch nodes. It essentially enforces the notch to behave like a perpendicular L-shape, that only rotates as a whole.
    K_notch_yztheta = static_condensation(K_reduced_yz_notch, retained_dofs=reduced_eff_y_dofs + reduced_eff_z_dofs + reduced_eff_theta_dofs)

    n_y = len(reduced_eff_y_dofs)
    n_z = len(reduced_eff_z_dofs)
    n_theta = len(reduced_eff_theta_dofs)

    # Effective 2x2 matrix for notch coordinates [y, z].
    K_yy = np.sum(K_notch_yz[:n_y, :n_y])
    K_zz = np.sum(K_notch_yz[n_y:, n_y:])
    K_yz = np.sum(K_notch_yz[:n_y, n_y:])

    # Effective 2x2 matrix for notch coordinates [y, z].
    K_eff_yz = np.array(
        [
            [K_yy, K_yz],
            [K_yz, K_zz]
        ],
        dtype=float,
    )

    # Effective 3x3 matrix for notch coordinates [y, z, theta].
    K_yy = np.sum(K_notch_yztheta[:n_y, :n_y])
    K_zz = np.sum(K_notch_yztheta[n_y:n_y+n_z, n_y:n_y+n_z])
    K_yz = np.sum(K_notch_yztheta[:n_y, n_y:n_y+n_z])

    K_ytheta = np.sum(K_notch_yztheta[:n_y, -n_theta:])
    K_ztheta = np.sum(K_notch_yztheta[n_y:n_y+n_z, -n_theta:])
    K_thetatheta = np.sum(K_notch_yztheta[-n_theta:,-n_theta:])

    # Effective 3x3 matrix for notch coordinates [y, z, theta].
    K_eff_yztheta = np.array(
        [
            [K_yy, K_yz, K_ytheta],
            [K_yz, K_zz, K_ztheta],
            [K_ytheta, K_ztheta, K_thetatheta]
        ],
        dtype=float,
    )

    # Condense the stiffness matrices for the notch planes
    K_eff_y = static_condensation(K_reduced_yz_notch, reduced_eff_y_dofs)
    K_eff_z = static_condensation(K_reduced_yz_notch, reduced_eff_z_dofs)
    
    # As proven in the documentation the effective stiffness value are the sums of the matrices
    k_eff_y = np.sum(K_eff_y)
    k_eff_z = np.sum(K_eff_z)

    if visualize:
        #visualize_stiffness_matrix(K_notch_yztheta, nodes_y = long_plane_nodes_xy, nodes_z = short_plane_nodes_xy, nodes_theta = long_plane_nodes_xy[:-1]+short_plane_nodes_xy,  title="Condensed Stiffness Matrix for Notch DOFs")
        visualize_stiffness_matrix(K_notch_yz,nodes_y = long_plane_nodes_xy, nodes_z = short_plane_nodes_xy,  title="Condensed Stiffness Matrix for Notch DOFs")

    return K_eff_yztheta , K_eff_yz, k_eff_y, k_eff_z

# ========== Function Application ==========

def main() -> None:
    K_6x6, K_eff_yztheta, K_eff_yz, k_eff_y, k_eff_z = predict_stiffness()

    print(f"Effective stiffness K_eff_y: {k_eff_y:.2f} N/mm")
    print(f"Effective stiffness K_eff_z: {k_eff_z:.2f} N/mm")
    print("Effective stiffness matrix K_eff_yz:")
    print(K_eff_yz)
    print("Full 6x6 stiffness matrix K_6x6 (kx, ky, kz, krx, kry, krz):")
    np.set_printoptions(precision=2, suppress=True)  # 2 decimal places, no scientific notation
    print(K_6x6)
    print("Note: Only ky,kz and krx are based on the model, the other values are based on simplified assumptions.")
    print("Torsional stiffness k_rz is undefined (NaN), as it cannot be reliably derived from the current model.")
    
    
if __name__ == "__main__":
    main()