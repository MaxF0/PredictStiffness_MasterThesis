# PredictStiffness_MasterThesis
Python code for the Master's Thesis "Analytical Modeling and Analysis of Structured Compliant Grippers for High-Speed Assembly Tasks"

Gripper_Analysis.py includes most of the calculations (Geometry and node generation, Direct Stiffness Method functions) as well as the visualization functions, stiffness_model.py processes the inputs and combines the functions from Gripper_Analysis.py to obtain the global stiffness matrix for the gripper. It then takes the global stiffness matrix and condenses and assembles it into the relevant outputs (k_eff,y and k_eff,z, K_eff).

The most important function is the predict_stiffness function inside stiffness_model.py that allows to predict the stiffness based on the parameter inputs. It also includes the option to visualize the deformation. The visualize_stiffness_matrix function can be commented out in the code if not needed.

An example use case of PySR is included. The script performs symbolic regression to derive interpretable approximation functions for stiffness as a function of infill angle and infill density. The resulting candidate equations are saved in PySR_hall_of_fame.csv.

The reported functions are normalized by $(E \cdot d)$. To obtain the actual predicted stiffness values, the function output must be multiplied by $(E \cdot d)$.


```mermaid
flowchart TD

    IN[Input]
    OUT[Output]
    IN
    --"depth, width, length,
    infill_density, infill_angle, notch_angle,
    beam_thickness_center, beam_thickness_side, E, visualize (T/F)"
     --> P

    %%IN --"E, depth, thickness" --> CS
    %%IN -- "density, thickness, length" -->  ID2C#

  subgraph P[predict_stiffness]
  direction TD
    CS[compute_section_stiffness]
    ID2C[infill_density_to_count]
    CNP[calculate_node_positions]
    BAL[build_applied_loads]
    CTM[create_coordinate_transformation_matrix]
    SDOF[solve_direct_stiffness]
    VISA[visualize_assembled_structure]
    VISD[visualize_deformed_structure]

    
    ID2C --"number of crossbeams"--> CNP
    CNP -."node coordinates, ..."..-> VISA
    CNP --"node coordinates, ..."--> AGM_logic
    CS --"EA,EI"---> AGM_logic


    AGM_logic --"K_global, node list, ..."--> BAL
    AGM_logic --"K_global, node list, ... "--> SDOF
    AGM_logic --"n_dof, notch_angle"--> CTM
    BAL --"load vector"--> SDOF
    CTM --"T (global csys -> notch csys)"--> SDOF
    

   
    SDOF --"K_reduced, U(notch cs), F(notch cs), ... "--> CE_logic
  
    SDOF -."K_reduced, U(notch cs), F(notch cs),...".-> VISD
    CTM -."T (global csys -> notch csys)"..-> VISD

    subgraph AGM_logic["assemble_global_matrix"]

        subgraph ADD_logic["add_element_to_matrix"]
            direction LR
            ELEM[create_2d_beam_element_matrix]
            XFORM[transform_element_matrix]
            ELEM --"local element matrix"--> XFORM
            XFORM --"global element matrix"--> ELEM
        end

    end

    subgraph CE_logic["calculate_effective_stiffness"]
        direction TD
      SC[static_condensation]
    end


  end


 P--"k_{eff,y}, k_{eff,z}, K_6x6"--> OUT

```
