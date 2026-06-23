# %%
from pysr import PySRRegressor
from Stiffness_model import predict_stiffness
from pathlib import Path
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np


# %%
density_vals = np.linspace(0.1, 0.4, 4)
angle_deg_vals = np.linspace(0.0, 60.0, 13)
angle_rad_vals = np.radians(angle_deg_vals)
young_modulus = 2308.0
depth = 12.0

# Build the training set directly from the stiffness solver.
density_grid, angle_deg_grid = np.meshgrid(density_vals, angle_deg_vals, indexing="ij")
angle_rad_grid = np.radians(angle_deg_grid)

X = np.column_stack([
    density_grid.ravel(),
    angle_rad_grid.ravel(),
])
y = np.empty(X.shape[0], dtype=float)

for index, (rho, ang_rad) in enumerate(X):
    y[index] = float(
        predict_stiffness(
            infill_density=float(rho),
            infill_angle=float(np.degrees(ang_rad)),
            beam_thickness_side=0.42,
            visualize=False,
        )[3]
    )

y_fit = y / (young_modulus * depth)

print(f"Training samples: {X.shape[0]}")
print(f"X shape: {X.shape}")
print(f"y shape: {y.shape}")
print(f"y_fit shape: {y_fit.shape}")


# %%
model = PySRRegressor(
    niterations=3000,
    populations=16,
    population_size=60,
    model_selection="best",
    early_stop_condition="stop_if(loss, complexity) = loss < 0.03 && complexity < 6",
    timeout_in_seconds=60 * 10,
    binary_operators=["+", "-", "*", "/","^"],
    unary_operators=["exp", "sin", "cos"],
    constraints={"exp": 3,
                "^": (-3, 3)},
    nested_constraints={
        "exp": {"cos": 1, "sin": 1, "exp": 1},
        "sin": {"cos": 1, "sin": 1, "exp": 1},
        "cos": {"cos": 1, "sin": 1, "exp": 1},
    },
    parsimony=1e-4,
    elementwise_loss="loss(x, y) = abs((x - y) / y)",
)


# %%
model.fit(X, y_fit, variable_names=["density", "angle_rad"])

# %%
# Save the discovered equations so they can be reused later.

output_dir = Path("outputs")
output_dir.mkdir(exist_ok=True)

equations = model.equations_.copy()
equations_path = output_dir / "pysr_equations.csv"
equations.to_csv(equations_path, index=False)
print(f"Saved all equations to {equations_path}")


# %%
selected_equation_index = equations.sort_values(["complexity"], ascending=[True]).index[6]

print("Selected equation:")
selected_equation = model.sympy(selected_equation_index)
print(selected_equation)

selected_equation_path = output_dir / "pysr_selected_equation.txt"
selected_equation_path.write_text(str(selected_equation), encoding="utf-8")
print(f"Saved selected equation to {selected_equation_path}")

y_pred_fit = model.predict(X, index=selected_equation_index)
y_pred = y_pred_fit * (young_modulus * depth)
abs_error = np.abs(y_pred - y)
rel_error = abs_error / np.maximum(np.abs(y), 1e-12)

mae = float(np.mean(abs_error))
mape_percent = float(np.mean(rel_error) * 100.0)
rmse = float(np.sqrt(np.mean((y_pred - y) ** 2)))

print(f"Training MAE [N/mm]: {mae:.6f}")
print(f"Training MAPE [%]: {mape_percent:.4f}")
print(f"Training RMSE [N/mm]: {rmse:.6f}")


# %%
angle_deg_plot = np.linspace(0.0, 60.0, 101)
angle_rad_plot = np.radians(angle_deg_plot)
density_levels_plot = [0.10, 0.20, 0.30, 0.40]

colors = plt.cm.viridis(np.linspace(0.1, 0.9, len(density_levels_plot)))
fig, ax = plt.subplots(figsize=(7.2, 4.2), constrained_layout=True)

for color, rho in zip(colors, density_levels_plot):
    X_plot = np.column_stack([
        np.full_like(angle_rad_plot, rho, dtype=float),
        angle_rad_plot,
    ])

    y_pred_plot = model.predict(X_plot, index=selected_equation_index) * (young_modulus * depth)

    y_meas_plot = []
    for ang_deg in angle_deg_plot:
        y_meas_plot.append(
            float(
                predict_stiffness(
                    infill_density=float(rho),
                    infill_angle=float(ang_deg),
                    beam_thickness_side=0.42,
                    visualize=False,
                )[3]
            )
        )
    y_meas_plot = np.asarray(y_meas_plot, dtype=float)

    ax.plot(angle_deg_plot, y_meas_plot, color=color, linewidth=2.0, label=f"Measured, ρ={rho:.2f}")
    ax.plot(angle_deg_plot, y_pred_plot, color=color, linestyle="--", linewidth=1.8, label=f"PySR fit, ρ={rho:.2f}")

ax.set_xlabel("Infill angle [deg]")
ax.set_ylabel(r"Stiffness $k_{\mathrm{eff,y}}$ [N/mm]")
ax.set_title("Symbolic regression of stiffness vs infill angle and density")
ax.set_xticks([0, 10, 20, 30, 40, 50, 60])
ax.grid(True, alpha=0.3)
ax.legend(ncol=2, fontsize=9)

print(f"Final model form: k_eff_y = {young_modulus:.1f} * {depth:.1f} * f(density, angle_rad)")

plt.show()



