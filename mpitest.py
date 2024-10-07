from pde import UnitGrid, ScalarField, DiffusionPDE

grid = UnitGrid([32, 32])  # smaller grid for quick testing
field = ScalarField.random_uniform(grid)
eq = DiffusionPDE()

result = eq.solve(field, t_range=1, dt=0.1, solver='explicit_mpi')
if result is not None:
    result.plot()
