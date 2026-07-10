using JuliaDG
using CairoMakie

material = JuliaDG.Elastodynamics.Material(1.0, 1.0, 0.5)

function initial_pulse(x, y)
    r2 = (x - 0.5)^2 + (y - 0.5)^2
    amplitude = exp(-120 * r2)
    return (vx=0.0, vy=amplitude, sxx=0.0, syy=0.0, sxy=0.0)
end

result = JuliaDG.Elastodynamics.solve(
    initial_pulse;
    nx=8,
    ny=8,
    material=material,
    tspan=(0.0, 0.02),
    cfl=0.05,
    boundary=:reflecting,
    save_history=true,
)

JuliaDG.Elastodynamics.record(
    result,
    "elastic_velocity_magnitude.gif";
    field=:velocity_magnitude,
    framerate=20,
)

println("Elastic DOFs: ", length(result.state))
println("Final time: ", result.times[end])
println("Final energy: ", JuliaDG.Elastodynamics.energy(result))
println("Saved elastic_velocity_magnitude.gif")
