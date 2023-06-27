using Random
using Distributions
#using Statistics
#using LinearAlgebra
using ProgressMeter

using Markdown

@doc doc"""
    simulate(h)

simulation of the stochastic Langevin equation

```
\dot{\rho}(t)=h-(\tau-\alpha+h(1+\beta))\rho(t)-b\rho^2(t)+\sigma\sqrt{\rho(t)/N}\eta(t)
```
by splitting it into two steps.

Analogue to paper first rewrite

```
\dot{\rho}(t)=h+a\rho(t)-b\rho^2(t)+\sigma\sqrt{\rho(t)/N}\eta(t)
```

Step 1: Sample from the exact solution for
```
\dot{\rho}(t)=h+a\rho(t)+\sigma\sqrt{\rho(t)/N}\eta(t)
```

Step 2: Evolve remaining deterministic part

```
\dot{\rho}(t)=-b\rho^2(t)
```

Reference:
Dornic, Chaté, and Munoz
"Integration of Langevin equations with multiplicative noise and the viability
of field theories for absorbing phase transitions."
Physical review letters, 94 (2005).
DOI: 10.1103/PhysRevLett.94.100601
"""
function simulate(
        h::Float64;
        #optional arguments
        time_sim::Float64=50.0,
        tau::Float64=10.0,
        alpha::Float64=30.0,
        beta::Float64=15.0,
        b::Float64=25.0,
        sigma::Float64=50.0,
        N::Int=512,
        seed::Int=1000,
        dt::Float64=1e-7,
        dstep_measure::Int=Int(1e4)
    )
    rng = MersenneTwister(seed)

    # convert to units of paper
    h = h
    a = -(tau-alpha+h*(1+beta))
    b = b
    sigma  = sigma/sqrt(N)
    sigma2 = sigma*sigma

    # constant coefficients
    weight = exp(a*dt)
    lambda = 2*a/sigma2/(weight-1)
    #poisson_weight = lambda*weight
    mu = -1 + 2*h/sigma2

    # initial condition
    rho0 = 0
    time = 0
    num_steps = floor(Int, time_sim/dt)
    num_measure = floor(Int, num_steps/dstep_measure)
    rhos =zeros(num_measure)
    times=zeros(num_measure)
    j=1
    @showprogress 1 for i in 1:num_steps
        #measure
        if i%dstep_measure==0
            j+=1
        end
        if j <= num_measure
            rhos[j] += rho0/dstep_measure
            times[j]+= time/dstep_measure
        end

        #step-1: sampling from exact solution of Fokker Planck
        # first need to sample from Poisson distribution, Eq.(6),
        # where beta  = a
        #       alpha = h
        rand_poisson = rand(rng,Poisson(lambda*rho0*weight))
        rho_int = rand(rng, Gamma(mu+1+rand_poisson))/lambda

        #step-2: evolve from rho_int the remaining deterministic part
        rho_new = rho_int/(1+rho_int*b*dt)

        #advance time
        time += dt
        #swap
        rho0 = rho_new
    end
    return times, rhos
end

