using JuliaDG
using CairoMakie

exact(x, y) = sin(pi * x) * sin(pi * y)
f(x, y) = 2 * pi^2 * exact(x, y)
g(x, y) = 0.0

result = solve_poisson(f; nx=16, ny=16, g=g, penalty=20.0)
fig = plot_solution(result)
save("solution.png", fig)

println("DOFs: ", length(result.coeffs))
println("L2 error: ", l2_error(result, exact))
println("Saved solution.png")
