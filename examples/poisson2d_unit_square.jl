using JuliaDG

exact(x, y) = sin(pi * x) * sin(pi * y)
f(x, y) = 2 * pi^2 * exact(x, y)
g(x, y) = 0.0

result = solve_poisson(f; nx=8, ny=8, g=g, penalty=20.0)
error = l2_error(result, exact)

println("DOFs: ", length(result.coeffs))
println("L2 error: ", error)
