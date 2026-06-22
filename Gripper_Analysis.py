"""
Gripper analysis helper functions:

Geometry and node generation:
- infill_density_to_count: Convert infill density to number of crossbeams to recreate slicer settings.
- calculate_node_positions: Calculate node positions for crossbeams and perimeter to create beam model of the gripper.
- create_coordinate_transformation_matrix: Create a coordinate transformation matrix. Used to convert global csys to notch csys.

Direct Stiffness method (FEA solver):
- create_2d_beam_element_matrix: Create element stiffness matrix for 2D beam element (local stiffness matrix).
- transform_element_matrix: Transform element stiffness matrix from local to global coordinates.
- assemble_global_matrix: Assemble global stiffness matrix for the structure including perimeter beams.
- apply_boundary_conditions: Apply boundary conditions by modifying stiffness matrix and force vector.
- solve_direct_stiffness: Complete direct stiffness method solver. (reduced stiffness matrix)
- static_condensation: Perform static condensation on the reduced stiffness matrix. (condensed stiffness matrix)

Visualization functions:
- visualize_assembled_structure: Visualize the assembled structure with all beam elements.
- visualize_deformed_structure: Visualize the original and deformed structure side by side.

"""
# Imports

import math
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
from scipy import linalg as sla

#--------------------------------------------------------------------------
# Geometry and node generation functions
#--------------------------------------------------------------------------

def infill_density_to_count(infill_density, beam_thickness_center, length):
	'''
	Calculate the number of crossbeams based on infill density.
	Note: number_of_beams is a theoretical number that is used to calculate the spacing between beams
		It matches the actual number of beams only for 0° infill angle
	'''
	total_beam_volume = infill_density * (length+2)								# add 2 to match 3d printer slicer settings
	number_of_beams = int(np.ceil(total_beam_volume / beam_thickness_center))+1	# add 1 because the first beam is at y=0 and not at y=spacing

	return number_of_beams


def calculate_node_positions(
	infill_density,
	crossbeam_count,
	crossbeam_angle,
	length,
	width,
	tip_length=19,
	notch_length=8.2,
	notch_depth=4,
	notch_angle=math.radians(10),
):
	'''Calculate node positions for crossbeams and perimeter from gripper geometry parameters.'''
	# Add the notch nodes.
	notch_positions = []

	# Start of the notch.
	y_notch = length - notch_length
	notch_positions.append((0, y_notch))

	# Inner corner of the notch.
	x_notch_corner = math.cos(notch_angle) * notch_depth
	y_notch_corner = y_notch - math.sin(notch_angle) * notch_depth
	notch_positions.append((x_notch_corner, y_notch_corner))

	# End of the notch.
	notch_y_altitude = length - y_notch_corner
	tip_x = x_notch_corner + math.tan(notch_angle) * notch_y_altitude
	notch_positions.append((tip_x, length))
	
	# Spacing between crossbeams (y-direction).
	# The function is partly physics imformed, partly modelled after the slicer setting results (+2)
	spacing = (length + 2) / (crossbeam_count * math.cos(crossbeam_angle))

	# adjust the crossbeam count for angles > 0°
	crossbeam_count_adjusted = int(np.floor(length / spacing))

	positions = []
	for i in range(crossbeam_count_adjusted):
		start_x = 0
		# The start y position is adjusted based on the infill density to better match the actual beam positions in the slicer settings. The factor of 0.8 was determined empirically to give a good match across densities.
		y0 = spacing * (0.8 - 1.5 * infill_density) 
		start_y = y0 + i * spacing

		# The end of the beam if the gripper was rectangular.
		end_x_hypothetical = width
		end_y_hypothetical = start_y + width * math.tan(crossbeam_angle)

		# The base of the gripper is straight.
		straight_length = length - tip_length
		if end_y_hypothetical < straight_length:
			end_x = end_x_hypothetical
			end_y = end_y_hypothetical
			positions.append(((start_x, start_y), (end_x, end_y)))
			
		else:
			# Above the base, the gripper tapers off in a circular manner towards the tip.
			m = math.tan(crossbeam_angle)
			# the radius is a function of width. That way for any tip length, the circular tip will exactly intersect with the tip
			r = (tip_length**2 + (width - tip_x)**2) / (2*(width - tip_x))
			#print("Calculated radius for circular tapering:", r)
			w = width
			s = start_y - straight_length
			end_x = (
				(
					math.sqrt(
						2 * m**2 * r * w
						+ m**2 * (-w**2)
						+ 2 * m * r * s
						- 2 * m * s * w
						+ r**2
						- s**2
					)
					- m * s
					- r
					+ w
				)
				/ (m**2 + 1)
			)
			end_y = start_y + end_x * m

			# Only add the beam if it ends within the gripper length.
			if end_y < (length -0.01): # add a small tolerance to avoid floating point issues
				positions.append(((start_x, start_y), (end_x, end_y)))

		# Add beams starting from the bottom edge (only for angles > 0°)
		if i == 0:
			# as long as there is enough space below the end of the first beam, add additional beams with the same spacing
			while end_y_hypothetical - spacing > 0 + 1: # add a small tolerance to avoid very short beams				
				end_y_hypothetical -= spacing
				# equation of a line
				start_x_extra = width - end_y_hypothetical / math.tan(crossbeam_angle)
				# for steep angles or short lengths, even the bottom beams can end within the circular tip.
				if end_y_hypothetical < straight_length:
					positions.append(((start_x_extra, 0), (width, end_y_hypothetical)))

				# This is a very similar to above, however those cases exist
				else: 
					# Calculate the end point in the circular tapering region.
					# Above the base, the gripper tapers off in a circular manner towards the tip.
					s_extra = 0 - straight_length - start_x_extra * math.tan(crossbeam_angle)
					w_extra = width
					end_x_tip = (
						(
							math.sqrt(
								2 * m**2 * r * w_extra
								+ m**2 * (-w_extra**2)
								+ 2 * m * r * s_extra
								- 2 * m * s_extra * w_extra
								+ r**2
								- s_extra**2
							)
							- m * s_extra
							- r
							+ w_extra
						)
						/ (m**2 + 1)
					)
					end_y_tip = 0 + (end_x_tip-start_x_extra) * math.tan(crossbeam_angle)
					if end_y_tip< (length - 0.01): # add a small tolerance to avoid floating point issues
						positions.append(((start_x_extra, 0), (end_x_tip, end_y_tip)))

	# Handle beams that are cut off by the notch.
	new_segments = []

	for i, ((x1, y1), (x2, y2)) in enumerate(positions):
		# Intersection of the crossbeam line and the long notch line.
		x1_original = x1
		y1_original = y1

		# First part of the beam is cut off by the long notch line.
		if (y1 + math.tan(crossbeam_angle) * x_notch_corner) > y_notch_corner:
			x1 = (
				y1
				- y_notch_corner
				+ math.tan(math.radians(90) - notch_angle) * x_notch_corner
			) / (math.tan(math.radians(90) - notch_angle) - math.tan(crossbeam_angle))
			y1 = y1 + x1 * math.tan(crossbeam_angle)
			positions[i] = ((x1, y1), (x2, y2))

		# First part of the beam is cut off by the short notch line.
		if (
			(y1_original + math.tan(crossbeam_angle) * x_notch_corner) > y_notch_corner
			and y1_original < (y_notch - 0.01) # add a small tolerance to avoid floating point issues
		):
			x2_new = (y1_original - y_notch) / (
				math.tan(-notch_angle) - math.tan(crossbeam_angle)
			)
			y2_new = y1_original + x2_new * math.tan(crossbeam_angle)
			new_segments.append(((x1_original, y1_original), (x2_new, y2_new)))

	# Add the new segments.
	positions.extend(new_segments)

	# Add missing corners.
	corner_positions_extra = [(0,0), (width, 0)]

	return positions, corner_positions_extra, notch_positions


def create_coordinate_transformation_matrix(n_dof, rotation_angle):
	'''Create a block diagonal transformation matrix for the given number of degrees of freedom and rotation angle.
	Assumes 3 DOF per node (u, v, theta) and that all nodes are rotated by the same angle.
	The matrix rotates the coordinate system in clockwise direction. This is equivalent to rotating a point in counterclockwise direction.'''
	n_nodes = n_dof // 3
	T = np.zeros((n_dof, n_dof))

	c = np.cos(rotation_angle)
	s = np.sin(rotation_angle)

	T_node = np.array(
		[
			[c, -s, 0],
			[s,  c, 0],
			[0,  0, 1]
		]
	)

	for i in range(n_nodes):
		dof_start = 3 * i
		dof_end = 3 * (i + 1)
		T[dof_start:dof_end, dof_start:dof_end] = T_node

	return T


#--------------------------------------------------------------------------
# Direct Stiffness method (FEM solver) functions
#--------------------------------------------------------------------------


def create_2d_beam_element_matrix(EA, EI, beam_length):
	"""
	Create stiffness matrix for 2D beam element (6 DOF: 3 per node).
	DOF order: [u1, v1, theta1, u2, v2, theta2]
	
	Parameters:
    - EA: axial stiffness (E * A)
    - EI: bending stiffness (E * I)
    - beam_length: element length
	"""
	EA_L = EA / beam_length
	EI_L3 = EI / (beam_length**3)
	EI_L2 = EI / (beam_length**2)
	EI_L = EI / beam_length

	k = np.array(
		[
        [EA_L,      0,         0,       -EA_L,     0,          0],
        [0,         12*EI_L3,  6*EI_L2, 0,         -12*EI_L3,  6*EI_L2],
        [0,         6*EI_L2,   4*EI_L,  0,         -6*EI_L2,   2*EI_L],
        [-EA_L,     0,         0,       EA_L,      0,          0],
        [0,         -12*EI_L3, -6*EI_L2,0,         12*EI_L3,   -6*EI_L2],
        [0,         6*EI_L2,   2*EI_L,  0,         -6*EI_L2,   4*EI_L]
		]
	)

	return k


def transform_element_matrix(k, angle):
	"""
	Transform element stiffness matrix from local to global coordinates.
	angle: angle of element from horizontal (radians)
	"""
	c = np.cos(angle)
	s = np.sin(angle)

	# Transformation matrix (2D beam with 3 DOF per node).
	T = np.zeros((6, 6))
	T[0:2, 0:2] = [[ c, s],
				   [-s, c]]
	T[2, 2] = 1
	T[3:5, 3:5] = [[ c, s],
				   [-s, c]]
	T[5, 5] = 1

	# Transform: K_global = T^T * K_local * T.
	k_global = T.T @ k @ T
	return k_global


def assemble_global_matrix(
	node_positions,
	corner_positions,
	notch_positions,
	EA_center,
	EI_center,
	EA_side,
	EI_side,
	width,
):
	"""
	Assemble global stiffness matrix for the structure including perimeter beams.
	Avoid duplicate beams between corners and crossbeam intersection points.
	"""
	TOLERANCE = 1e-3

	beam_nodes = set()
	for (x1, y1), (x2, y2) in node_positions:
		beam_nodes.add((x1, y1))
		beam_nodes.add((x2, y2))

	perimeter_nodes = set(corner_positions + notch_positions)

	all_nodes = list(beam_nodes) + list(perimeter_nodes)
	n_nodes = len(all_nodes)
	# 3 DOF per node (u, v, theta)
	n_dof = 3 * n_nodes

    # Create node index dictionary
	node_index = {node: idx for idx, node in enumerate(all_nodes)}
	# Initialize global stiffness matrix
	K_global = np.zeros((n_dof, n_dof))

    # Helper function to add element to global matrix
	def add_element_to_matrix(node1, node2, EA, EI):
		"""
        Add element stiffness to global matrix.
        The element direction (node1 -> node2) defines the local x-axis.
        
        Parameters:
        - EA: axial stiffness (E * A)
        - EI: bending stiffness (E * I)
        """
				
		idx1 = node_index[node1]
		idx2 = node_index[node2]

		dx = node2[0] - node1[0]
		dy = node2[1] - node1[1]
		beam_length = np.sqrt(dx**2 + dy**2)


        # Skip if nodes are at same position
		if beam_length < TOLERANCE:
			return

		angle = np.arctan2(dy, dx)
		k_local = create_2d_beam_element_matrix(EA, EI, beam_length)
		k_global_elem = transform_element_matrix(k_local, angle)

        # Map element DOFs to global DOFs
        # Element DOF order: [u1, v1, theta1, u2, v2, theta2]
		dof_indices = [3 * idx1, 3 * idx1 + 1, 3 * idx1 + 2, 3 * idx2, 3 * idx2 + 1, 3 * idx2 + 2]
		ix = np.ix_(dof_indices, dof_indices)
		K_global[ix] += k_global_elem

	# Assemble crossbeams.
	crossbeam_connections = set()
	for (x1, y1), (x2, y2) in node_positions:
		node1 = (x1, y1)
		node2 = (x2, y2)
		add_element_to_matrix(node1, node2, EA_center, EI_center)
		connection = tuple(sorted([node1, node2]))
		crossbeam_connections.add(connection)

    # Create perimeter beams connecting all nodes on the boundary
	# Bottom edge: y=0.
	bottom_nodes = sorted([(x, y) for (x, y) in all_nodes if abs(y) < TOLERANCE], key=lambda n: n[0])
	for i in range(len(bottom_nodes) - 1):
		connection = tuple(sorted([bottom_nodes[i], bottom_nodes[i + 1]]))
		# Only add if this connection is NOT already a crossbeam
		if connection not in crossbeam_connections:
			add_element_to_matrix(bottom_nodes[i], bottom_nodes[i + 1], EA_side, EI_side)

	# Left edge: x=0
	left_nodes = sorted([(x, y) for (x, y) in all_nodes if abs(x) < TOLERANCE], key=lambda n: n[1])
	for i in range(len(left_nodes) - 1):
		connection = tuple(sorted([left_nodes[i], left_nodes[i + 1]]))
		if connection not in crossbeam_connections:
			add_element_to_matrix(left_nodes[i], left_nodes[i + 1], EA_side, EI_side)

	# Right edge: x=width.
	right_nodes = sorted(
		[(x, y) for (x, y) in all_nodes if abs(x - width) < TOLERANCE],
		key=lambda n: n[1],
	)
	for i in range(len(right_nodes) - 1):
		connection = tuple(sorted([right_nodes[i], right_nodes[i + 1]]))
		if connection not in crossbeam_connections:
			add_element_to_matrix(right_nodes[i], right_nodes[i + 1], EA_side, EI_side)

	# Circular tip. (all nodes above the last right node  right of the tip node)
	x_tip = notch_positions[-1][0]
	y_last_right_node = right_nodes[-1][1]
	tip_nodes = sorted(
		[(x, y) for (x, y) in all_nodes if (x >= x_tip and y >= y_last_right_node)],
		key=lambda n: n[1],
	)
	for i in range(len(tip_nodes) - 1):
		connection = tuple(sorted([tip_nodes[i], tip_nodes[i + 1]]))
		if connection not in crossbeam_connections:
			add_element_to_matrix(tip_nodes[i], tip_nodes[i + 1], EA_side, EI_side)

	# Short notch plane.
	short_plane_nodes = sorted(
		[
			(x, y)
			for (x, y) in all_nodes
			if (notch_positions[0][0] < x <= notch_positions[1][0] and y >= notch_positions[1][1])
		],
		key=lambda n: n[1],
	)
	short_plane_nodes = [notch_positions[0]] + short_plane_nodes
	for i in range(len(short_plane_nodes) - 1):
		connection = tuple(sorted([short_plane_nodes[i], short_plane_nodes[i + 1]]))
		if connection not in crossbeam_connections:
			add_element_to_matrix(short_plane_nodes[i], short_plane_nodes[i + 1], EA_side, EI_side)

	# Long notch plane.
	long_plane_nodes = sorted(
		[
			(x, y)
			for (x, y) in all_nodes
			if (notch_positions[1][0] <= x <= notch_positions[2][0] and y >= notch_positions[1][1])
		],
		key=lambda n: n[1],
	)
	for i in range(len(long_plane_nodes) - 1):
		connection = tuple(sorted([long_plane_nodes[i], long_plane_nodes[i + 1]]))
		if connection not in crossbeam_connections:
			add_element_to_matrix(long_plane_nodes[i], long_plane_nodes[i + 1], EA_side, EI_side)

	return K_global, all_nodes, long_plane_nodes, short_plane_nodes, node_index


def apply_boundary_conditions(K, F, fixed_dofs):
	"""
	Apply boundary conditions by modifying stiffness matrix and force vector.
	(create reduced system by removing fixed DOFs)
	fixed_dofs: list of DOF indices that are fixed (displacement = 0)
	"""
	free_dofs = [i for i in range(len(F)) if i not in fixed_dofs]

	K_reduced = K[np.ix_(free_dofs, free_dofs)]
	F_reduced = F[free_dofs]

	return K_reduced, F_reduced, free_dofs, fixed_dofs


def solve_direct_stiffness(
	K_yz_notch,
	applied_loads_yz_notch,
	node_index_xy
	
):
	"""
	Direct stiffness method solver. Returns are in the notch coordinate system (x along long notch, y along short notch).
	Note: Boundary conditions are automatically applied to all nodes with y~=0
	(fixed in all DOFs).
	"""
	TOLERANCE = 1e-3

    # Initialize global force vector to apply loads
	n_dof = len(K_yz_notch)
	F = np.zeros(n_dof)
	for dof, load in applied_loads_yz_notch.items():
		F[dof] = load

    # Automatically identify fixed DOFs: all nodes with y=0 are fixed (all 3 DOFs)
	fixed_dofs = []
	for node, node_idx in node_index_xy.items():
		if abs(node[1]) < TOLERANCE:    # y-coordinate is approximately 0
			fixed_dofs.extend([3 * node_idx, 3 * node_idx + 1, 3 * node_idx + 2])

    # Apply boundary conditions and solve for displacements at free DOFs
	K_reduced_yz_notch, F_reduced, free_dofs, fixed_dofs = apply_boundary_conditions(K_yz_notch, F, fixed_dofs)
	# There was a numerical robustness issue for very few combinations with np.linalg.solve, which is why we switched to scipy's sparse solver with symmetric matrix assumption and check_finite=False for better performance on large systems.
	# U_free = np.linalg.solve(K_reduced_yz_notch, F_reduced)
	U_free = sla.solve(K_reduced_yz_notch, F_reduced, assume_a="sym", check_finite=False)

    # Reconstruct full displacement vector
	U_notch_cs = np.zeros(n_dof)
	for i, dof_idx in enumerate(free_dofs):
		U_notch_cs[dof_idx] = U_free[i]

    # Calculate reactions
	F_reactions_notch_cs = K_yz_notch @ U_notch_cs

	return U_notch_cs, F_reactions_notch_cs, K_reduced_yz_notch, free_dofs, fixed_dofs


def static_condensation(K_reduced, retained_dofs):
	"""
	Perform static condensation on the reduced stiffness matrix.

	Parameters:
	- K_reduced: Reduced stiffness matrix from apply_boundary_conditions
	- retained_dofs: List of indices corresponding to the DOFs to retain
	"""
	eliminated_dofs = [i for i in range(K_reduced.shape[0]) if i not in retained_dofs]

    # Partition the stiffness matrix
	K_cc = K_reduced[np.ix_(retained_dofs, retained_dofs)]
	K_ce = K_reduced[np.ix_(retained_dofs, eliminated_dofs)]
	K_ec = K_reduced[np.ix_(eliminated_dofs, retained_dofs)]
	K_ee = K_reduced[np.ix_(eliminated_dofs, eliminated_dofs)]

    # Compute the condensed stiffness matrix

	# There was a numerical robustness issue for very few combinations with np.linalg.solve
	# K_condensed = K_cc - K_ce @ np.linalg.solve(K_ee, K_ec)
	K_condensed = K_cc - K_ce @ sla.solve(K_ee, K_ec, assume_a="sym", check_finite=False)

	return K_condensed


#--------------------------------------------------------------------------
# Visualization functions
#--------------------------------------------------------------------------


def visualize_assembled_structure(all_nodes, node_positions, notch_positions, length, width):
	"""
	Visualize all beam elements in the assembled structure.
	
	Parameters:
    - all_nodes: list of all node coordinates
    - node_positions: list of beam element endpoints
    - notch_positions: list of notch node coordinates
    - length, width: structure dimensions
    """
	fig, ax = plt.subplots(figsize=(8, 12))
	ax.set_aspect("equal")

	beam_index = 0

	# Draw crossbeams (blue).
	for (x1, y1), (x2, y2) in node_positions:
		ax.plot([x1, x2], [y1, y2], "b-", linewidth=1.5, alpha=0.7)
		mid_x, mid_y = (x1 + x2) / 2, (y1 + y2) / 2
		ax.text(
			mid_x,
			mid_y,
			str(beam_index),
			fontsize=9,
			ha="center",
			bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue", alpha=0.7),
		)
		beam_index += 1

	# Draw perimeter beams (red).
	bottom_nodes = sorted([(x, y) for (x, y) in all_nodes if abs(y) < 0.01], key=lambda n: n[0])
	for i in range(len(bottom_nodes) - 1):
		ax.plot(
			[bottom_nodes[i][0], bottom_nodes[i + 1][0]],
			[bottom_nodes[i][1], bottom_nodes[i + 1][1]],
			"r-",
			linewidth=2,
			alpha=0.8,
		)
		mid_x = (bottom_nodes[i][0] + bottom_nodes[i + 1][0]) / 2
		mid_y = (bottom_nodes[i][1] + bottom_nodes[i + 1][1]) / 2
		ax.text(
			mid_x,
			mid_y - 0.5,
			str(beam_index),
			fontsize=9,
			ha="center",
			bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow", alpha=0.7),
		)
		beam_index += 1

    # Left edge: x=0
	left_nodes = sorted([(x, y) for (x, y) in all_nodes if abs(x) < 0.01], key=lambda n: n[1])
	for i in range(len(left_nodes) - 1):
		ax.plot(
			[left_nodes[i][0], left_nodes[i + 1][0]],
			[left_nodes[i][1], left_nodes[i + 1][1]],
			"r-",
			linewidth=2,
			alpha=0.8,
		)
		mid_x = (left_nodes[i][0] + left_nodes[i + 1][0]) / 2 - 0.5
		mid_y = (left_nodes[i][1] + left_nodes[i + 1][1]) / 2
		ax.text(
			mid_x,
			mid_y,
			str(beam_index),
			fontsize=9,
			ha="center",
			bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow", alpha=0.7),
		)
		beam_index += 1

    # Right edge: x=width
	right_nodes = sorted([(x, y) for (x, y) in all_nodes if abs(x - width) < 0.01], key=lambda n: n[1])
	for i in range(len(right_nodes) - 1):
		ax.plot(
			[right_nodes[i][0], right_nodes[i + 1][0]],
			[right_nodes[i][1], right_nodes[i + 1][1]],
			"r-",
			linewidth=2,
			alpha=0.8,
		)
		mid_x = (right_nodes[i][0] + right_nodes[i + 1][0]) / 2 + 0.5
		mid_y = (right_nodes[i][1] + right_nodes[i + 1][1]) / 2
		ax.text(
			mid_x,
			mid_y,
			str(beam_index),
			fontsize=9,
			ha="center",
			bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow", alpha=0.7),
		)
		beam_index += 1

    # Circular tip. (all nodes above the last right node  right of the tip node)
	x_tip = notch_positions[-1][0]
	y_last_right_node = right_nodes[-1][1]
	tip_nodes = sorted([(x, y) for (x, y) in all_nodes if (x >= x_tip and y >= y_last_right_node)], key=lambda n: n[1])
	for i in range(len(tip_nodes) - 1):
		ax.plot(
			[tip_nodes[i][0], tip_nodes[i + 1][0]],
			[tip_nodes[i][1], tip_nodes[i + 1][1]],
			"r-",
			linewidth=2,
			alpha=0.8,
		)
		mid_x = (tip_nodes[i][0] + tip_nodes[i + 1][0]) / 2
		mid_y = (tip_nodes[i][1] + tip_nodes[i + 1][1]) / 2
		ax.text(
			mid_x,
			mid_y + 0.5,
			str(beam_index),
			fontsize=9,
			ha="center",
			bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow", alpha=0.7),
		)
		beam_index += 1

    # Short notch plane.
	short_plane_nodes = sorted(
		[
			(x, y)
			for (x, y) in all_nodes
			if (notch_positions[0][0] < x <= notch_positions[1][0] and y >= notch_positions[1][1])
		],
		key=lambda n: n[1],
	)
	short_plane_nodes = [notch_positions[0]] + short_plane_nodes
	for i in range(len(short_plane_nodes) - 1):
		ax.plot(
			[short_plane_nodes[i][0], short_plane_nodes[i + 1][0]],
			[short_plane_nodes[i][1], short_plane_nodes[i + 1][1]],
			"r-",
			linewidth=2,
			alpha=0.8,
		)
		mid_x = (short_plane_nodes[i][0] + short_plane_nodes[i + 1][0]) / 2
		mid_y = (short_plane_nodes[i][1] + short_plane_nodes[i + 1][1]) / 2
		ax.text(
			mid_x,
			mid_y + 0.5,
			str(beam_index),
			fontsize=9,
			ha="center",
			bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow", alpha=0.7),
		)
		beam_index += 1

    # Long notch plane.
	long_plane_nodes = sorted(
		[
			(x, y)
			for (x, y) in all_nodes
			if (notch_positions[1][0] <= x <= notch_positions[2][0] and y >= notch_positions[1][1])
		],
		key=lambda n: n[1],
	)
	for i in range(len(long_plane_nodes) - 1):
		ax.plot(
			[long_plane_nodes[i][0], long_plane_nodes[i + 1][0]],
			[long_plane_nodes[i][1], long_plane_nodes[i + 1][1]],
			"r-",
			linewidth=2,
			alpha=0.8,
		)
		mid_x = (long_plane_nodes[i][0] + long_plane_nodes[i + 1][0]) / 2
		mid_y = (long_plane_nodes[i][1] + long_plane_nodes[i + 1][1]) / 2
		ax.text(
			mid_x,
			mid_y + 0.5,
			str(beam_index),
			fontsize=9,
			ha="center",
			bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow", alpha=0.7),
		)
		beam_index += 1

    # Plot all nodes
	x_coords = [node[0] for node in all_nodes]
	y_coords = [node[1] for node in all_nodes]
	ax.scatter(x_coords, y_coords, color="black", s=30, zorder=5)

    # Highlight fixed nodes (y=0) with red squares.
	fixed_nodes = [(x, y) for (x, y) in all_nodes if y == 0]
	if fixed_nodes:
		fixed_x = [node[0] for node in fixed_nodes]
		fixed_y = [node[1] for node in fixed_nodes]
		ax.scatter(
			fixed_x,
			fixed_y,
			marker="s",
			color="red",
			s=100,
			facecolors="none",
			edgecolors="red",
			linewidth=2,
			zorder=6,
			label="Fixed nodes",
		)

	ax.set_xlim(-5, width + 15)
	ax.set_ylim(-1, length + 1)
	ax.invert_yaxis()
	ax.set_xlabel("Width (x)")
	ax.set_ylabel("Length (y)")
	ax.set_title(f"Assembled Structure with All Beam Elements ({len(all_nodes)} nodes)")
	ax.grid(True, alpha=0.3)

	# Create custom legend
	legend_elements = [
		Line2D([0], [0], color="b", linewidth=1.5, label="Crossbeams"),
		Line2D([0], [0], color="r", linewidth=2, label="Perimeter beams"),
		Line2D([0], [0], marker="o", color="w", markerfacecolor="black", markersize=6, label="Nodes"),
		Line2D(
			[0],
			[0],
			marker="s",
			color="w",
			markerfacecolor="none",
			markeredgecolor="red",
			markersize=8,
			markeredgewidth=2,
			label="Fixed nodes",
		),
	]
	ax.legend(handles=legend_elements, loc="upper right")
	plt.show()


def visualize_deformed_structure(
	all_nodes,
	node_positions,
	notch_positions,
	U_notch_cs,
	node_index,
	length,
	width,
    notch_angle,
	scale_factor=1.0,
):
	"""
	Visualize the original and deformed structure side by side.
    
    Parameters:
    - all_nodes: list of all node coordinates
    - node_positions: list of beam element endpoints
    - notch_positions: list of notch node coordinates
    - U_notch_cs: displacement vector in notch coordinate system
    - node_index: dictionary mapping node coordinates to indices
    - length, width: structure dimensions
    - scale_factor: amplification factor for displacements (for visualization)
	"""
	fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 8))

    # Pre-compute all edge nodes for reuse
	bottom_nodes = sorted([(x, y) for (x, y) in all_nodes if abs(y) < 0.01], key=lambda n: n[0])
	left_nodes = sorted([(x, y) for (x, y) in all_nodes if abs(x) < 0.01], key=lambda n: n[1])
	right_nodes = sorted([(x, y) for (x, y) in all_nodes if abs(x - width) < 0.01], key=lambda n: n[1])
	x_tip = notch_positions[-1][0]
	y_last_right_node = right_nodes[-1][1]
	tip_nodes = sorted([(x, y) for (x, y) in all_nodes if (x >= x_tip and y >= y_last_right_node)], key=lambda n: n[1])

	# ===== ORIGINAL STRUCTURE =====
	ax1.set_aspect("equal")
	ax1.plot([0, width, width, 0, 0], [0, 0, length, length, 0], "k-", linewidth=1, alpha=0.3, label="Boundary")

    # Draw original beams
	for (x1, y1), (x2, y2) in node_positions:
		ax1.plot([x1, x2], [y1, y2], "b-", linewidth=1.5, alpha=0.7)

    # Draw original perimeter beams
	for i in range(len(bottom_nodes) - 1):
		ax1.plot(
			[bottom_nodes[i][0], bottom_nodes[i + 1][0]],
			[bottom_nodes[i][1], bottom_nodes[i + 1][1]],
			"r-",
			linewidth=2,
			alpha=0.8,
		)

    # circular tip
	for i in range(len(tip_nodes) - 1):
		ax1.plot(
			[tip_nodes[i][0], tip_nodes[i + 1][0]],
			[tip_nodes[i][1], tip_nodes[i + 1][1]],
			"r-",
			linewidth=2,
			alpha=0.8,
		)

    # left edge
	for i in range(len(left_nodes) - 1):
		ax1.plot(
			[left_nodes[i][0], left_nodes[i + 1][0]],
			[left_nodes[i][1], left_nodes[i + 1][1]],
			"r-",
			linewidth=2,
			alpha=0.8,
		)

    # right edge
	for i in range(len(right_nodes) - 1):
		ax1.plot(
			[right_nodes[i][0], right_nodes[i + 1][0]],
			[right_nodes[i][1], right_nodes[i + 1][1]],
			"r-",
			linewidth=2,
			alpha=0.8,
		)

    # short notch plane
	short_plane_nodes = sorted(
		[
			(x, y)
			for (x, y) in all_nodes
			if (notch_positions[0][0] < x <= notch_positions[1][0] and y >= notch_positions[1][1])
		],
		key=lambda n: n[1],
	)
	short_plane_nodes = [notch_positions[0]] + short_plane_nodes
	for i in range(len(short_plane_nodes) - 1):
		ax1.plot(
			[short_plane_nodes[i][0], short_plane_nodes[i + 1][0]],
			[short_plane_nodes[i][1], short_plane_nodes[i + 1][1]],
			"r-",
			linewidth=2,
			alpha=0.8,
		)

    # long notch plane
	long_plane_nodes = sorted(
		[
			(x, y)
			for (x, y) in all_nodes
			if (notch_positions[1][0] <= x <= notch_positions[2][0] and y >= notch_positions[1][1])
		],
		key=lambda n: n[1],
	)
	for i in range(len(long_plane_nodes) - 1):
		ax1.plot(
			[long_plane_nodes[i][0], long_plane_nodes[i + 1][0]],
			[long_plane_nodes[i][1], long_plane_nodes[i + 1][1]],
			"r-",
			linewidth=2,
			alpha=0.8,
		)

	# Plot all nodes
	x_coords = [node[0] for node in all_nodes]
	y_coords = [node[1] for node in all_nodes]
	ax1.scatter(x_coords, y_coords, color="black", s=50, zorder=5)

	ax1.set_xlim(-1, width + 1)
	ax1.set_ylim(-1, length + 10)
	ax1.invert_yaxis()
	ax1.set_xlabel("Width (x)")
	ax1.set_ylabel("Length (y)")
	ax1.set_title("Original Structure")
	ax1.grid(True, alpha=0.3)

	# ===== DEFORMED STRUCTURE =====
	ax2.set_aspect("equal")
	ax2.plot([0, width, width, 0, 0], [0, 0, length, length, 0], "k-", linewidth=1, alpha=0.3, label="Original boundary")

    # Calculate deformed node positions
	deformed_nodes = []
    # Transform displacements back to global coordinate system
	T = create_coordinate_transformation_matrix(n_dof = len(U_notch_cs), rotation_angle = notch_angle)
	U_xy_global = T.T @ U_notch_cs  

	for node in all_nodes:
		node_idx = node_index[node]
        # Extract displacements (u, v components only, ignore rotation)
		u_disp = U_xy_global[3 * node_idx] * scale_factor
		v_disp = U_xy_global[3 * node_idx + 1] * scale_factor
		deformed_node = (node[0] + u_disp, node[1] + v_disp)
		deformed_nodes.append(deformed_node)

    # Draw deformed beams
	for (x1, y1), (x2, y2) in node_positions:
		node1 = (x1, y1)
		node2 = (x2, y2)
		idx1 = node_index[node1]
		idx2 = node_index[node2]
		def_node1 = deformed_nodes[idx1]
		def_node2 = deformed_nodes[idx2]
		ax2.plot([def_node1[0], def_node2[0]], [def_node1[1], def_node2[1]], "b-", linewidth=1.5, alpha=0.7)

	for i in range(len(bottom_nodes) - 1):
		idx1 = node_index[bottom_nodes[i]]
		idx2 = node_index[bottom_nodes[i + 1]]
		def_node1 = deformed_nodes[idx1]
		def_node2 = deformed_nodes[idx2]
		ax2.plot([def_node1[0], def_node2[0]], [def_node1[1], def_node2[1]], "r-", linewidth=2, alpha=0.8)

	for i in range(len(tip_nodes) - 1):
		idx1 = node_index[tip_nodes[i]]
		idx2 = node_index[tip_nodes[i + 1]]
		def_node1 = deformed_nodes[idx1]
		def_node2 = deformed_nodes[idx2]
		ax2.plot([def_node1[0], def_node2[0]], [def_node1[1], def_node2[1]], "r-", linewidth=2, alpha=0.8)

	for i in range(len(left_nodes) - 1):
		idx1 = node_index[left_nodes[i]]
		idx2 = node_index[left_nodes[i + 1]]
		def_node1 = deformed_nodes[idx1]
		def_node2 = deformed_nodes[idx2]
		ax2.plot([def_node1[0], def_node2[0]], [def_node1[1], def_node2[1]], "r-", linewidth=2, alpha=0.8)

	for i in range(len(right_nodes) - 1):
		idx1 = node_index[right_nodes[i]]
		idx2 = node_index[right_nodes[i + 1]]
		def_node1 = deformed_nodes[idx1]
		def_node2 = deformed_nodes[idx2]
		ax2.plot([def_node1[0], def_node2[0]], [def_node1[1], def_node2[1]], "r-", linewidth=2, alpha=0.8)

	short_plane_nodes = sorted(
		[
			(x, y)
			for (x, y) in all_nodes
			if (notch_positions[0][0] < x <= notch_positions[1][0] and y >= notch_positions[1][1])
		],
		key=lambda n: n[1],
	)
	short_plane_nodes = [notch_positions[0]] + short_plane_nodes
	for i in range(len(short_plane_nodes) - 1):
		idx1 = node_index[short_plane_nodes[i]]
		idx2 = node_index[short_plane_nodes[i + 1]]
		def_node1 = deformed_nodes[idx1]
		def_node2 = deformed_nodes[idx2]
		ax2.plot([def_node1[0], def_node2[0]], [def_node1[1], def_node2[1]], "r-", linewidth=2, alpha=0.8)

	long_plane_nodes = sorted(
		[
			(x, y)
			for (x, y) in all_nodes
			if (notch_positions[1][0] <= x <= notch_positions[2][0] and y >= notch_positions[1][1])
		],
		key=lambda n: n[1],
	)
	for i in range(len(long_plane_nodes) - 1):
		idx1 = node_index[long_plane_nodes[i]]
		idx2 = node_index[long_plane_nodes[i + 1]]
		def_node1 = deformed_nodes[idx1]
		def_node2 = deformed_nodes[idx2]
		ax2.plot([def_node1[0], def_node2[0]], [def_node1[1], def_node2[1]], "r-", linewidth=2, alpha=0.8)

    # Plot deformed nodes
	def_x_coords = [node[0] for node in deformed_nodes]
	def_y_coords = [node[1] for node in deformed_nodes]
	ax2.scatter(def_x_coords, def_y_coords, color="green", s=50, zorder=5)

    # Find max displacement for scaling axes
	max_disp = np.max(np.abs(U_xy_global))
	ax2.set_xlim(-max_disp * scale_factor - 2, width + max_disp * scale_factor)
	ax2.set_ylim(-1, length + 10)
	ax2.invert_yaxis()
	ax2.set_xlabel("Width (x)")
	ax2.set_ylabel("Length (y)")
	ax2.set_title(f"Deformed Structure (Scale Factor: {scale_factor}x)")
	ax2.grid(True, alpha=0.3)

    # Add legend
	legend_elements = [
		Line2D([0], [0], color="black", linewidth=1, alpha=0.3, label="Original boundary"),
		Line2D([0], [0], color="b", linewidth=1.5, label="Deformed crossbeams"),
		Line2D([0], [0], color="r", linewidth=2, label="Deformed perimeter beams"),
		Line2D([0], [0], marker="o", color="w", markerfacecolor="green", markersize=8, label="Deformed nodes"),
	]
	ax2.legend(handles=legend_elements, loc="upper left")
	# Disclaimer shown inside the deformed-structure graph
	ax2.text(
		0.02,
		0.02,
		"Linear FEM: Assumption: small deflections only (linear kinematics)\n"
		"and elastic bending behavior. (vertical loads cause vertical deflections only)\n"
		"Only displacements for nodes are shown, deformation of beams is not visualized\n",
		transform=ax2.transAxes,
		fontsize=9,
		color="darkred",
		ha="left",
		va="bottom",
		bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="darkred", alpha=0.85),
	)

	plt.subplots_adjust(wspace=0.2)
	plt.tight_layout()
	plt.show()
	
    # Print displacement statistics
	# print("\n=== DISPLACEMENT STATISTICS Overall ===")
	# print(f"  Maximum displacement (overall structure): {np.max(np.abs(U_xy_global)):.6f}")
	# print(f"  Maximum x-displacement (overall): {np.max(np.abs(U_xy_global[::3])):.6f}")
	# print(f"  Maximum y-displacement (overall): {np.max(np.abs(U_xy_global[1::3])):.6f}")
	# print(f"  Maximum rotation (overall): {np.max(np.abs(U_xy_global[2::3])):.6f}")

    # # Print displacement at tip node
	# print("\n=== DISPLACEMENT AT TIP NODE ===")
	# loading_node_idx = node_index[notch_positions[-1]]
	# print(f"Loading node: {notch_positions[-1]}")
	# print(f"  x-displacement: {U_xy_global[3 * loading_node_idx]:.6f}")
	# print(f"  y-displacement: {U_xy_global[3 * loading_node_idx + 1]:.6f}")
	# print(f"  rotation: {U_xy_global[3 * loading_node_idx + 2]:.6f}")
	# print(
	# 	"  total displacement magnitude: "
	# 	f"{np.sqrt(U_xy_global[3 * loading_node_idx] ** 2 + U_xy_global[3 * loading_node_idx + 1] ** 2):.6f}"
	# )


def visualize_stiffness_matrix(K_matrix, nodes_y=None, nodes_z=None, nodes_theta=None, abs_max=None, title="Stiffness Matrix Heatmap"):
	"""
	Visualize a stiffness matrix as a 2D heatmap.
	
	Parameters:
    - K_matrix: stiffness matrix (numpy array)
    - nodes: list of node tuples (x, y) corresponding to the DOFs in K_matrix (optional)
    - abs_max: maximum absolute value for color scaling (optional)
    - title: title for the plot
    """
	fig, ax = plt.subplots(figsize=(12, 10))
	abs_max = np.max(np.abs(K_matrix)) if abs_max is None else abs_max

    # Plot heatmap
	im = ax.imshow(K_matrix, cmap="seismic", aspect="auto", vmin=-abs_max, vmax=abs_max)

    # Add colorbar
	cbar = plt.colorbar(im, ax=ax)
	cbar.set_label("Stiffness Value", rotation=270, labelpad=20)

    # Create custom tick labels if nodes are provided
	if nodes_y is not None and nodes_z is not None:
        # Create tick labels showing node coordinates and DOF type
        # Order: all horizontal DOFs first, then all vertical DOFs
		tick_labels = []
		# Horizontal (u) DOFs
		for node in nodes_y:
			x, y = node
			tick_labels.append(f"({x:.1f},{y:.1f})-y")
        # Vertical (v) DOFs
		for node in nodes_z:
			x, y = node
			tick_labels.append(f"({x:.1f},{y:.1f})-z")

		if nodes_theta is not None:
			for node in nodes_theta:
				x, y = node
				tick_labels.append(f"({x:.1f},{y:.1f})-theta")
			
        # Only show a subset of ticks if there are too many
		n_ticks = len(tick_labels)
		if n_ticks < 20:
            # Show all ticks
			ax.set_xticks(range(n_ticks))
			ax.set_yticks(range(n_ticks))
			ax.set_xticklabels(tick_labels, rotation=90, fontsize=9)
			ax.set_yticklabels(tick_labels, fontsize=9)

		ax.set_xlabel("DOF (Node Coordinates and Direction)")
		ax.set_ylabel("DOF (Node Coordinates and Direction)")
	else:
		ax.set_xlabel("DOF Index")
		ax.set_ylabel("DOF Index")

	ax.set_title(title)
	ax.grid(True, alpha=0.3)

	plt.tight_layout()
	plt.show()