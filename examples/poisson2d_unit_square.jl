using JuliaDG
using Gridap

model = unit_square_model(16, 16)
exact(x) = sin(pi * x[1]) * sin(pi * x[2])
source(x) = 2 * pi^2 * exact(x)

result = Poisson.solve(model, source; dirichlet_tags = "boundary", dirichlet = x -> 0.0)

domain = Triangulation(model)
writevtk(domain, "poisson_solution", cellfields = ["u" => result.solution])
println("L2 error: ", Poisson.l2_error(result, exact))
