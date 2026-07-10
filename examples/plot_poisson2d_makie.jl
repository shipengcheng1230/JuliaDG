using JuliaDG
using CairoMakie

exact(x, y) = sin(pi * x) * sin(pi * y)
f(x, y) = 2 * pi^2 * exact(x, y)
g(x, y) = 0.0

result = JuliaDG.Poisson.solve(f; nx=16, ny=16, g=g, penalty=20.0)
fig = JuliaDG.Poisson.plot(result)
save("solution.png", fig)

println("DOFs: ", length(result.coeffs))
println("L2 error: ", JuliaDG.Poisson.l2_error(result, exact))
println("Saved solution.png")
