# %%
using OrdinaryDiffEq, CUDA, LinearAlgebra

# %%
u0 = cu(rand(1000))
A  = cu(randn(1000,1000))
# %%
f(du,u,p,t)  = mul!(du,A,u)
prob = ODEProblem(f,u0,(0.0f0,1.0f0)) # Float32 is better on GPUs!
# %%
sol = solve(prob,Tsit5())
# %%
s0 = sol.u[1] |> Array{Float32}
# %%
sf = sol.u[end] |> Array{Float32}

# %%
using GLMakie
# %%
plot(s0)
# %%
plot(sf)

# %%
using DiffEqGPU, OrdinaryDiffEq
function lorenz(du,u,p,t)
    du[1] = p[1]*(u[2]-u[1])
    du[2] = u[1]*(p[2]-u[3]) - u[2]
    du[3] = u[1]*u[2] - p[3]*u[3]
end
# %%

u0 = Float32[1.0;0.0;0.0]
tspan = (0.0f0,100.0f0)
p = [10.0f0,28.0f0,8/3f0]


prob = ODEProblem(lorenz,u0,tspan,p)
# %%

prob_func = (prob,i,repeat) -> remake(prob,p=rand(Float32,3).*p)
monteprob = EnsembleProblem(prob, prob_func = prob_func, safetycopy=false)
@time sol = solve(monteprob,Tsit5(),EnsembleGPUArray(),trajectories=10_000,saveat=1.0f0)
# %%
