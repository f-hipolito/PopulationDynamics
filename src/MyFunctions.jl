module MyFunctions

export σ, γi_a, γi_l, γi_u
export FL, fL, FΓ, fΓ, IΓ

using SpecialFunctions

"""
    σ(x)

Evaluates the sigmoid function defined as
``σ(x) = 1/\\big(1+e^{-x}\\big)``.


# Examples
```

julia> σ(0)
0.5

julia> σ(1)
0.7310585786300049

julia> σ(Inf)
1.0

julia> σ([-Inf,-1,-1.,-1/2,-0.5,0,0.,sqrt(2),π,Inf])
10-element Array{Float64,1}:
 0.0
 0.2689414213699951
 0.2689414213699951
 0.3775406687981454
 0.3775406687981454
 0.5
 0.5
 0.8044296825069569
 0.9585761678336371
 1.0
```
"""
@. function σ(x)
    return 1.0/ ( 1.0 +exp( -x ) )
end

σ(1)
σ(Inf)
σ([-Inf,-1,-1.,-1/2,-0.5,0,0.,sqrt(2),π,Inf])


"""
    γi_a(a,x[;IND=0])

Compute the lower incomple normalized Gamma function.

Invokes SpecialFunctions gamma_inc(a,x,IND) and returns the array [l,u]

# Sources:
1. https://dlmf.nist.gov/8.2


# Examples
```julia-repl
julia> γi_a( 2, 3 )
2-element Array{Float64,1}:
 0.8008517265285442
 0.19914827347145578

julia> γi_a( 2, 0.301, IND=1 ) -γi_a( 2, 0.301, IND=0 )
2-element Array{Float64,1}:
 -4.2643048814294815e-11
  4.264311126433995e-11

julia> γi_a.( 2, [3,4,5,6,7] )
5-element Array{Array{Float64,1},1}:
 [0.8008517265285442, 0.19914827347145578]
 [0.9084218055563291, 0.0915781944436709]
 [0.9595723180054871, 0.040427681994512805]
 [0.9826487347633355, 0.01735126523666451]
 [0.9927049442755639, 0.00729505572443613]
```
"""
function γi_a(a,x;IND=0)
    l::Float64,u::Float64 = gamma_inc(a,x,IND)
    return [l,u]
end

γi_a( 2, 3 )
γi_a( 2, 0.301, IND=1 ) -γi_a( 2, 0.301, IND=0 )
γi_a.( 2, [3,4,5,6,7] )


"""
    γi_l(a,x[;IND=0])

Compute the lower incomple normalized Gamma function.

Invokes SpecialFunctions gamma_inc(a,x,IND) and returns only l

# Sources:
1. https://dlmf.nist.gov/8.2


# Examples
```julia-repl
julia> γi_l( 2, 3 )
0.8008517265285442

julia> γi_l( 2, 0.301, IND=1 ) -γi_l( 2, 0.301, IND=0 )
-4.2643048814294815e-11

julia> γi_l.( 2, [3,4,5,6,7] )
5-element Array{Array{Float64,1},1}:
 [0.8008517265285442]
 [0.9084218055563291]
 [0.9595723180054871]
 [0.9826487347633355]
 [0.9927049442755639]
```
"""
function γi_l(a,x;IND=0)
    l::Float64,u::Float64 = gamma_inc(a,x,IND)
    return l
end

γi_l( 2, 3 )
γi_l( 2, 0.301, IND=1 ) -γi_l( 2, 0.301, IND=0 )
γi_l.( 2, [3,4,5,6,7] )


"""
    γi_u(a,x[;IND=0])

Compute the lower incomple normalized Gamma function.

Invokes SpecialFunctions gamma_inc(a,x,IND) and returns only l

# Sources:
1. https://dlmf.nist.gov/8.2


# Examples
```julia-repl
julia> γi_u( 2, 3 )
0.19914827347145578

julia>  γi_u( 2, 0.301, IND=1 ) -γi_u( 2, 0.301, IND=0 )
4.264311126433995e-11

julia> γi_u.( 2, [3,4,5,6,7] )
5-element Array{Float64,1}:
 0.19914827347145578
 0.0915781944436709
 0.040427681994512805
 0.01735126523666451
 0.00729505572443613
```
"""
function γi_u(a,x;IND=0)
    l::Float64,u::Float64 = gamma_inc(a,x,IND)
    return l
end

γi_u( 2, 3 )
γi_u( 2, 0.301, IND=1 ) -γi_u( 2, 0.301, IND=0 )
γi_u.( 2, [3,4,5,6,7] )


# notation for distributions
# fX => probability density function (PDF) for distribution X
# FX => cummulative distribution function (CDF) for distribution X
# IX => integral of CDF for distribution X

#   -----------------------------------------------------------------------   #
#
#   Logistic distribution
#       PDF, CDF (Logistic function)
#
#   -----------------------------------------------------------------------   #

# CDF
"""
    FL(x[; p])

Compute the CDF for a variable with logistic distribution,
    ``F_L(x;x_0,k) = 1/\\big[1+e^{-k(x-x_0)}\\big]``
    parametrized via keyword argument ``p = [x_0,k]``, with default
    values ``[ x_0, k] = [0,1]``.

Alternatively, the CDF can be cast in a closed-form expression
    ``f_L(x;x_0,k) = σ( k(x-x_0) )``.

# Sources:
1. https://en.wikipedia.org/wiki/Logistic_distribution
2. https://mathworld.wolfram.com/LogisticDistribution.html


# Examples
```

julia> FL( 1 )
0.7310585786300049

julia> FL( 1.0 )
0.7310585786300049

julia> FL( [1,3/2,2.5] )
3-element Array{Float64,1}:
 0.7310585786300049
 0.8175744761936437
 0.9241418199787566

julia> FL( 1; p=[2,1] )
0.2689414213699951

julia> FL( 1.0; p=[2.0,1.0] )
0.2689414213699951

julia> FL( [1,3/2,2.5]; p=[2.0,1.0] )
3-element Array{Float64,1}:
 0.2689414213699951
 0.3775406687981454
 0.6224593312018546
```
"""
@. function FL(x; p=[ 0.0, 1.0])
    return σ( p[2]*(x-p[1]) )
end

FL( 1 )
FL( 1.0 )
FL( [1,3/2,2.5] )

FL( 1; p=[2,1] )
FL( 1.0; p=[2.0,1.0] )
FL( [1,3/2,2.5]; p=[2.0,1.0] )

# PDF: full form and condensed form using the CDF, ie σ
"""
    fL(x[; p])

Compute the PDF for a variable with logistic distribution,
    ``f_L(x;x_0,k) = k e^{-k(x-x_0)}/\\Big[ \\Big( 1 +e^{-k(x-x_0)} \\big)^2 \\Big]``
    parametrized via keyword argument ``p = [x_0,k]``, with default values
    ``[ x_0, k] = [0,1]``.
    Alternatively, the PDF can be cast in a closed-form expression
    ``f_L(x;x_0,k) = k σ( k(x-x_0) )σ( -k(x-x_0) )``.

# Sources:
1. https://en.wikipedia.org/wiki/Logistic_distribution
2. https://mathworld.wolfram.com/LogisticDistribution.html


# Examples
```julia-repl
julia> fL(0.)
0.25

julia> fL([3/2,2,2.5]; p=[2,7/2])
3-element Array{Float64,1}:
 0.4414522881532932
 0.875
 0.44145228815329324
```
"""
@. function fL(x; p=[0.0, 1.0])
    return  p[2] *σ( p[2]*(x-p[1]) ) *σ( -p[2]*(x-p[1]) )
end

fL( 1 )
fL( 1.0 )
fL( [1,3/2,2.5] )

fL( 1; p=[2,1] )
fL( 1.0; p=[2.0,1.0] )
fL( [1,3/2,2.5]; p=[2.0,1.0] )



#   -----------------------------------------------------------------------   #
#
#   Gamma distribution
#       CDF, PDF and integral of CDF
#
#   -----------------------------------------------------------------------   #

# I could not implement directly a vectorize function with multiple arguments,
# while also using an if statement, so I define a non-nonvectorized version
# which is invoked by the vectorize version
function FΓ_def(x; p=[2,1], IND=0)::Float64
    if x >= 0
        return γi_l( p[1], p[2] *x, IND=IND )
    else
        return 0.0
    end
end

"""
    FΓ(x[; p=[2,1], IND=0])

Compute the CDF for a variable with gamma distribution,
    ``F_Γ(x;α,β) = γ(α,β x) \\big/ Γ(α)``
    parametrized via keyword argument ``p = [α,β]``, with default
    values ``[α,β] = [2,1]``. Keyword IND=0 argument required for
    gamma_inc function, defaults to maximum accuracy.

``γ(α,β x)`` and ``Γ(α)`` are the lower incomplete and the complete
    Gamma functions, respectively

To facilitate implementation of code using this function we extend
    this the domain to ℝ by setting ``F_Γ(x;α,β) = 0 ∀ x < 0``.

Requires SpecialFunctions package and makes use of gamma_inc
    function.

FΓ invokes the source function FΓ_def to allow for trivial broadcast
    over the variable x. FΓ_def is not exported by default.

# Sources:
1. https://en.wikipedia.org/wiki/Gamma_distribution
2. https://mathworld.wolfram.com/GammaDistribution.html
3. https://dlmf.nist.gov/5.2
4. https://dlmf.nist.gov/8.2


# Examples
```julia-repl
julia> FΓ(0.)
0.0

julia> FΓ([-1,3/2,2.5]; p=[2,1])
3-element Array{Float64,1}:
 0.0
 0.4421745996289254
 0.7127025048163542

julia> FΓ([1,3/2,2.5]; p=[2,1], IND=1) - FΓ([1,3/2,2.5]; p=[2,1])
3-element Array{Float64,1}:
 -8.316106692163316e-10
 -7.803298285313787e-9
  0.0
```
"""
@. function FΓ(x; p=[2,1], IND=0)
    FΓ_def(x; p=p, IND=IND)
end

FΓ( 1 )
FΓ( 1.0 )
FΓ( [1,3/2,2.5] )

FΓ( 1; p=[2,1] )
FΓ( 1.0; p=[2.0,1.0] )
FΓ( [1,3/2,2.5]; p=[2.0,1.0] )

FΓ( 1; p= [2,1], IND=0 )
FΓ( 1.0; p= [2.0,1.0], IND=0 )
FΓ( [1,3/2,2.5]; p= [2.0,1.0], IND=0 )


# PDF: full form and condensed form using the CDF, ie σ
function fΓ_def(x; p=[2,1])
    if x >=0
        return p[2]^p[1] *x^(p[1]-1.0) *exp( -p[2]*x ) /gamma(p[1])
    else
        return 0.0
    end
end

"""
    fΓ(x[; p])

Compute the PDF at for a variable with gamma distribution,
    ``f_L(x;α,β) = β^α x^{α-1} e^{-β x} \\big/  Γ(α)``
    parametrized by p = [α,β], with default values
    ``[α,β] = [2,1]`` and ``Γ(α)`` is the complete Gamma
    function.


To facilitate implementation of code using this function
    we extend this the domain to ℝ by setting
    ``f_Γ(x;α,β) = 0 ∀ x < 0``.

fΓ invokes the source function fΓ_def to allow for trivial broadcast
    over the variable x. fΓ_def is not exported by default.

# Sources:
1. https://en.wikipedia.org/wiki/Gamma_distribution
2. https://mathworld.wolfram.com/GammaDistribution.html
3. https://dlmf.nist.gov/5.2


# Examples
```julia-repl
julia> fΓ(0)
0.0

julia> fΓ([-1,3/2,2.5]; p=[2,1])
3-element Array{Float64,1}:
 0.0
 0.33469524022264474
 0.205212496559747
```
"""
@. function fΓ(x; p=[2,1])
    fΓ_def(x; p=p)
end


fΓ( 1 )
fΓ( 1.0 )
fΓ([1.0, 1.3, 2.8])

fΓ( 1; p=[2,1] )
fΓ( 1.0; p=[2.0,1.0] )
fΓ([1.0, 1.3, 2.8]; p=[2.0, 1.0])


function IΓ_def(t; ρ=[2.0,1.0], IND=0)::Float64
    if t >= 0.0
        α, β = ρ
        a = ( t -α/β )*γi_l(α,β*t;IND=IND)
        b = β^(α-1.0) *t^α *exp( -β*t ) /gamma(α)
        return a+b
    else
        return 0.0
    end
end;
"""
    IΓ(x[; ρ, IND])

Compute the integral of the CDF at for a variable with gamma
    distribution
    ``I_Γ(x;α,β) = \\big[ (x-α/β)γ(α,βt) +β^{α-1} t^α e^{-βt} \\big] \\big/ Γ(α)``
    parametrized via keyword argument ρ = [α,β], with default values
    ``[α,β] = [2,1]`` and ``Γ(α)`` is the gamma function. Keyword
    IND=0 argument required for gamma_inc function, defaults to
    maximum accuracy.

``γ(α,β x)`` and ``Γ(α)`` are the lower incomplete and the complete
    Gamma functions, respectively.

To facilitate implementation of code using this function we extend
    this the domain to ℝ by setting ``I_Γ(x<0;α,β) = 0``.

Requires SpecialFunctions package and makes use of gamma_inc
    function.



IΓ invokes the source function IΓ_def to allow for trivial broadcast
    over the variable x. IΓ_def is not exported by default.

To facilitate implementation of code using this function
    we extend its the domain to ℝ by setting
    ``I_Γ(x;α,β) = 0 ∀ x < 0``.

# Sources:
1. https://en.wikipedia.org/wiki/Gamma_distribution
2. https://mathworld.wolfram.com/GammaDistribution.html
3. https://dlmf.nist.gov/5.2
4. https://dlmf.nist.gov/8.2


# Examples
```julia-repl
julia> IΓ(0)
0.0

julia> IΓ([-1,3/2,2.5]; ρ=[2,1])
3-element Array{Float64,1}:
 0.0
 0.2809555605195043
 0.8693824938075446

julia> IΓ([1,3/2,2.5]; ρ=[2,1], IND=1) -IΓ([1,3/2,2.5]; ρ=[2,1])
3-element Array{Float64,1}:
 8.316106692163316e-10
 3.901649170412469e-9
 0.0
```
"""
@. function IΓ(x;ρ=[2,1],IND=0)
    return  IΓ_def(x; ρ=ρ,IND=IND)
end

IΓ( 1 )
IΓ( 1.0 )
IΓ( [1,3/2,2.5] )

IΓ( 1; ρ=[2,1] )
IΓ( 1.0; ρ=[2.0,1.0] )
IΓ( [1,3/2,2.5]; ρ=[2.0,1.0] )

IΓ( 1; ρ=[2,1], IND=0 )
IΓ( 1.0; ρ=[2.0,1.0], IND=0 )
IΓ( [1,3/2,2.5]; ρ=[2.0,1.0], IND=0 )

IΓ([1,3/2,2.5]; ρ=[2,1], IND=1) -IΓ([1,3/2,2.5]; ρ=[2,1])


end
