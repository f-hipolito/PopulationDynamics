module Mrate

export modelm, modelm_j, modelm_h
export sir_Γ!, sird_Γ!

using SpecialFunctions, MyFunctions

function modelm_def(t;p=[0.25,0],ρ=[2,1])::Float64
    #compute m and verify that length of p is even
    m,r = fldmod( length(p), 2) .-(1,0)
    @assert r==0 "Length of p must be even"
    @assert iseven( length( ρ ) ) "Length of ρ must be even"

    a0 = p[1]*t -p[2] # evaluate common part
    if m == 0
        return σ( a0 )
    elseif m == 1   # hand typed for m=1, but the general case would works
        am = (p[3]-p[1])*IΓ(t-p[4];ρ=ρ)
        return σ( a0 +am )
    else
        am = 0.0
        for i=1:m
            j=2*i
            am += (p[j+1]-p[j-1])*IΓ(t-p[j+2];ρ=ρ)
        end
        return σ( a0 +am )
    end
end
"""
    modelm(t,p,ρ)

Compute the solution for the Losgistic differential equation with m+1
    constant rates ``r = {r_0, r_1, \\ldots, r_m }`` and off-set
    ``t_0``. The switching between the constant rates occures at
    times ``t = {r_1, \\ldots, t_m }`` and is modelled according to a
    Gamma distribution

Requires SpecialFunctions package and makes use of gamma_inc
    function.

modelm invokes the source function modelm_def to allow for trivial
    broadcast over the variable x, model_def is not exported by
    default.

Note that LsqFit.curve_fit requires a model function that supports
    two arguments. The first, an abstract array for the indepenendent
    variables. The second, another abstract array for the parameters
    that will be optimized.

# Sources:
1. private notes contact me via https://github.com/f-hipolito


# Examples
```julia-repl
julia> p0 = [0.25, 0. ];

julia> ρ0=[6.2,.97];

julia> modelm([-10, 0.0, 10.0], p0, ρ0)
3-element Array{Float64,1}:
 0.07585818002124355
 0.5
 0.9241418199787566

julia> p2 = [0.25, 18.8, 0.075, 45. ];

julia> modelm([0.0, 50.0, 200.0], p2, ρ0)
3-element Array{Float64,1}:
 6.8432709753876305e-9
 0.0017103219749579988
 0.9944784329234642
```
"""
@. function modelm(t,p,ρ)
    modelm_def(t;p=p,ρ=ρ)
end





"""
    Rt!(R,m,x,p,ρ)

Evaluates the argument of the m-rate logistic function, where R is an
array (mutable by default).

# Sources:
1. private notes contact me via https://github.com/f-hipolito


# Examples
```julia-repl
julia> Rt!(R0,0,[-10, 0.0, 10.0],[0.25, 0. ],[6.2,.97])
3-element Array{Float64,1}:
 -2.5
  0.0
  2.5
```
"""
function Rt!(R,m,x::Array,p,ρ)
    for i=0:m
        if i==0
            R[:] += p[1] .*x .-p[2]
        elseif i>0
            j=2*i
            R[:] += ( p[j+1] -p[j-1] ) .*IΓ.( x .-p[j+2]; ρ=ρ )
        end
    end
    return R
end
"""
    Rt(m,x,p,ρ)

Evaluates the argument of the m-rate logistic function, where R is a
scalar Float64.

# Sources:
1. private notes contact me via https://github.com/f-hipolito


# Examples
```julia-repl
julia> Rt(0,10,[.25, 0],[6.2,0.97])
2.5
```
"""
function Rt(m,x::Number,p,ρ)
    R::Float64 = 0
    for i=0:m
        if i==0
            R += p[1] .*x .-p[2]
        elseif i>0
            j=2*i
            R += ( p[j+1] -p[j-1] ) .*IΓ.( x .-p[j+2]; ρ=ρ )
        end
    end
    return R
end


"""
    dr( i,x,p,ρ )

Evaluates the ith partial derivative of the argument of the m-rate
logistic function, with respect to the 'rate' parameters

# Sources:
1. private notes contact me via https://github.com/f-hipolito


# Examples
```julia-repl
julia> dr( 0,0,[0, 25, 100],p0,ρ0 )
3-element Array{Float64,1}:
   0.0
  25.0
 100.0

julia> dr( 0,0,25,p0,ρ0 )
25.0
```
"""
function dr( i,m,x::Array,p,ρ )
    out = zeros(Float64, length(x) )
    if i == 0 && m == 0
        @. out = x
    elseif i == 0 && m >= 1
        out = x -IΓ.( x .-p[4]; ρ=ρ )
    elseif i >= 1 && i < m
        j=2*i
        out = IΓ.( x .-p[j+2]; ρ=ρ ) -IΓ.( x .-p[j+4]; ρ=ρ )
    elseif i != 0 && i == m
        out = IΓ.( x .-p[2*i+2]; ρ=ρ )
    end
    return out
end
function dr( i,m,x::Number,p,ρ )
    out::Float64 = 0
    if i == 0 && m == 0
        out = x
    elseif i == 0 && m >= 1
        out = x -IΓ( x -p[4]; ρ=ρ )
    elseif i >= 1 && i < m
        j=2*i
        out = IΓ( x -p[j+2]; ρ=ρ ) -IΓ( x -p[j+4]; ρ=ρ )
    elseif i != 0 && i == m
        out = IΓ( x -p[2*i+2]; ρ=ρ )
    end
    return out
end

"""
    dt( i,x,p,ρ )

Evaluates the ith partial derivative of the argument of the m-rate
logistic function, with respect to the 'time' parameters

# Sources:
1. private notes contact me via https://github.com/f-hipolito


# Examples
```julia-repl
julia> dt( 0,[0.0, 50.0, 200.0],[0.25, 0. ],[6.2,.97] )
3-element Array{Float64,1}:
 -1.0
 -1.0
 -1.0

julia> dt( 0,25,p0,ρ0 )
-1.0
```
"""
function dt( i,x::Array,p,ρ )
    out = zeros(Float64, length(x) )
    if i == 0
        @. out = -1
    else
        j=2*i
        @. out = (p[j-1]-p[j+1]) *FΓ( x-p[j+2]; p=ρ, IND=0 )
    end
    return out
end
function dt( i,x::Number,p,ρ )
    out::Float64 = 0
    if i == 0
        out = -1
    else
        j=2*i
        out = (p[j-1]-p[j+1]) *FΓ( x-p[j+2]; p=ρ, IND=0 )
    end
    return out
end

"""
    modelm_j(t,p,ρ)

Compute the Jacobian for modelm(t,p,ρ) with respect to parameters p,
    in the form:
    ``p = \\{ r_0, t_0, r_1, t_1, \\ldots , r_m, t_m \\}``, thus
    generating a  ``n_t \\, \\times \\, 2m+1`` matrix, where ``n_t``
    is the number of elements in input argument ``t``.

# Sources:
1. private notes contact me via https://github.com/f-hipolito


# Examples
```julia-repl
julia> x1 = [0.0, 50.0, 200.0]

julia> p0 = [0.25, 0. ];

julia> ρ0=[6.2,.97];

julia> modelm_j(x1, p0, ρ0)
3×2 Array{Float64,2}:
 0.0          -0.25
 0.000186331  -3.72663e-6
 3.8575e-20   -1.92875e-22

julia> modelm_j(x1, p1, ρ0)
3×4 Array{Float64,2}:
 0.0        -6.84327e-9  0.0          0.0
 0.0846931  -0.0017074   0.000676733  9.74565e-5
 0.282196   -0.00549108  0.81602      0.000960939

julia> modelm_j(x1, p2, ρ0)
3×6 Array{Float64,2}:
 0.0        -6.84327e-9   0.0          0.0         0.0        -0.0
 0.0846931  -0.0017074    0.000676733  9.74565e-5  0.0        -0.0
 0.0166569  -0.000324115  0.011344     5.67202e-5  0.0368222  -8.10288e-6
```
"""
function modelm_j(x,p,ρ)
    #compute m and verify that length of p is even
    lp::Int = length(p)
    lx::Int = length(x)
    m::Int,r::Int = fldmod( lp, 2) .- (1,0)
    # @assert r==0 "Length of p must be even"
    # @assert iseven( length( ρ ) ) "Length of ρ must be even"

    J = zeros(Float64, lx, lp )
    R  = zeros(Float64, lx )
    dn = zeros(Float64, lx )

    j::Int = 0

    Rt!(R,m,x,p,ρ)

    dn = σ.(R) .* σ.(.-R)

    for i = 0:m
        j=2*i
        J[:,j+1] = dn .* dr(i,m,x,p,ρ)
        J[:,j+2] = dn .* dt(i,x,p,ρ)
    end

    return J
end





"""
    d2rr( k,i,m,x,p,ρ )

Evaluates the k,i-th 2nd order partial derivative of the argument of
the m-rate logistic function, with respect to the 'rate' parameters.

# Sources:
1. private notes contact me via https://github.com/f-hipolito


# Examples
```julia-repl
julia> d2rr(0,0,0,-10,[0.25, 0. ],[6.2,.97])
0.0

julia> d2rr(0,0,0,[-10, 0.0, 10.0],[0.25, 0. ],[6.2,.97])
3-element Array{Float64,1}:
 0.0
 0.0
 0.0
```
"""
function d2rr( k,i,m,x::Array,p,ρ )
    out = zeros(Float64, length(x) )
    # all are identically zero in the current model
    if k == 0 && i == 0
        #@. out = 0
    elseif k != 0 && i == 0
        #@. out = 0
    elseif k == 0 && i != 0
        #@. out = 0
    elseif k != 0 && i != 0
        #@. out = 0
    end
    return out
end
function d2rr( k,i,m,x::Number,p,ρ )
    out::Float64 = 0
    # all are identically zero in the current model
    if k == 0 && i == 0
        #@. out = 0
    elseif k != 0 && i == 0
        #@. out = 0
    elseif k == 0 && i != 0
        #@. out = 0
    elseif k != 0 && i != 0
        #@. out = 0
    end
    return out
end

"""
    d2tr( k,i,m,x,p,ρ )

Evaluates the k,i-th 2nd order partial derivative of the argument of
the m-rate logistic function, with respect to 'time' and 'rate' parameters.
Note, the partial derivative is defined as follows:
``\\partial^2 R / \\partial t_k \\partial r_i \\equiv
\\partial / \\partial t_k  \\big( \\partial R / \\partial r_i \\big)``

# Sources:
1. private notes contact me via https://github.com/f-hipolito


# Examples
```julia-repl
julia> d2tr( 1,1,1,200.0,[0.25, 18.8, 0.075, 45. ],[6.2,.97] )
-1.0

julia> d2tr( 1,1,1,[0.0, 50.0, 200.0],[0.25, 18.8, 0.075, 45. ],[6.2,.97] )
3-element Array{Float64,1}:
 -0.0
 -0.3261656196046895
 -1.0
```
"""
function d2tr( k,i,m,x::Array,p,ρ )
    out = zeros(Float64, length(x) )
    if k == 0 && i == 0
        #@. out = 0     #no instruction required for the present model
    elseif k != 0 && i == 0
        #@. out = 0     #no instruction required for the present model
    elseif k == 0 && i != 0
        #@. out = 0     #no instruction required for the present model
    elseif k != 0 && i != 0  # ∂^2 R / ( ∂t_k ∂r_i )
        if k == i       # (-1) δ_{k,i}
            @. out = -FΓ( x -p[2*k+2];p=ρ, IND=0 )
        elseif k == i+1 # δ_{k,i+1}
            @. out = FΓ( x -p[2*k+2];p=ρ, IND=0 )
        end
    end
    return out
end
function d2tr( k,i,m,x::Number,p,ρ )
    out::Float64 = 0
    if k == 0 && i == 0
        #@. out = 0     #no instruction required for the present model
    elseif k != 0 && i == 0
        #@. out = 0     #no instruction required for the present model
    elseif k == 0 && i != 0
        #@. out = 0     #no instruction required for the present model
    elseif k != 0 && i != 0  # ∂^2 R / ( ∂t_k ∂r_i )
        if k == i       # (-1) δ_{k,i}
            out = -FΓ( x -p[2*k+2];p=ρ, IND=0 )
        elseif k == i+1 # δ_{k,i+1}
            out = FΓ( x -p[2*k+2];p=ρ, IND=0 )
        end
    end
    return out
end


"""
    d2rt( k,i,m,x,p,ρ )

Evaluates the k,i-th 2nd order partial derivative of the argument of
the m-rate logistic function, with respect to 'rate' and 'time' parameters.
Note, the partial derivative is defined as follows:
``\\partial^2 R / \\partial r_k \\partial t_i \\equiv
\\partial / \\partial r_k  \\big( \\partial R / \\partial t_i \\big)``

# Sources:
1. private notes contact me via https://github.com/f-hipolito


# Examples
```julia-repl
julia> d2rt( 1,1,1,200,[0.25, 18.8, 0.075, 45. ],[6.2,.97] )
-1.0

julia> d2rt( 1,1,1,[0.0, 50.0, 200.0],[0.25, 18.8, 0.075, 45. ],[6.2,.97] )
3-element Array{Float64,1}:
 -0.0
 -0.3261656196046895
 -1.0
```
"""
function d2rt( k,i,m,x::Array,p,ρ )
    out = zeros(Float64, length(x) )
    if k == 0 && i == 0
        #@. out = 0     #no instruction required for the present model
    elseif k != 0 && i == 0
        #@. out = 0     #no instruction required for the present model
    elseif k == 0 && i != 0     # ∂^2 R / ( ∂r_0 ∂t_i )
        if i==1
            @. out = FΓ( x -p[2*i+2] ; p=ρ, IND=0)
        end
    elseif k != 0 && i != 0     # ∂^2 R / ( ∂r_k ∂t_i )
        if k == i       # (-1) δ_{k,i}
            @. out = -FΓ( x -p[2*i+2] ; p=ρ, IND=0)
        elseif k+1 == i # δ_{k+1,i}
            @. out = FΓ( x -p[2*i+2] ; p=ρ, IND=0)
        end
    end
    return out
end
function d2rt( k,i,m,x::Number,p,ρ )
    out::Float64 = 0
    if k == 0 && i == 0
        #@. out = 0     #no instruction required for the present model
    elseif k != 0 && i == 0
        #@. out = 0     #no instruction required for the present model
    elseif k == 0 && i != 0     # ∂^2 R / ( ∂r_0 ∂t_i )
        if i==1
            out = FΓ( x -p[2*i+2] ; p=ρ, IND=0)
        end
    elseif k != 0 && i != 0     # ∂^2 R / ( ∂r_k ∂t_i )
        if k == i       # (-1) δ_{k,i}
            out = -FΓ( x -p[2*i+2] ; p=ρ, IND=0)
        elseif k+1 == i # δ_{k+1,i}
            out = FΓ( x -p[2*i+2] ; p=ρ, IND=0)
        end
    end
    return out
end

"""
    d2tt( k,i,m,x,p,ρ )

Evaluates the k,i-th 2nd order partial derivative of the argument of
the m-rate logistic function, with respect to the 'time' parameters.

# Sources:
1. private notes contact me via https://github.com/f-hipolito


# Examples
```julia-repl
julia> d2tt( 1,1,1,50,[0.25, 18.8, 0.075, 45. ],[6.2,.97] )
-0.028867524522792173

julia> d2tt( 1,1,1,[0.0, 50.0, 200.0],[0.25, 18.8, 0.075, 45. ],[6.2,.97] )
3-element Array{Float64,1}:
 -0.0
 -0.028867524522792173
 -1.0608237504005124e-57
```
"""
function d2tt( k,i,m,x::Number,p,ρ )
    out::Float64 = 0
    if k == 0 && i == 0
        #@. out = 0     #no instruction required for the present model
    elseif k != 0 && i == 0
        #@. out = 0     #no instruction required for the present model
    elseif k == 0 && i != 0
        #@. out =       #no instruction required for the present model
    elseif k != 0 && i == k     # ∂^2 R / ( ∂t_k ∂t_i )
        j = 2*i
        out = ( p[j+1] -p[j-1] ) * fΓ( x -p[j+2]; p=ρ )
    end
    return out
end
function d2tt( k,i,m,x::Array,p,ρ )
    out = zeros(Float64, length(x) )
    if k == 0 && i == 0
        #@. out = 0     #no instruction required for the present model
    elseif k != 0 && i == 0
        #@. out = 0     #no instruction required for the present model
    elseif k == 0 && i != 0
        #@. out =       #no instruction required for the present model
    elseif k != 0 && i == k     # ∂^2 R / ( ∂t_k ∂t_i )
        j = 2*i
        @. out = ( p[j+1] -p[j-1] ) * fΓ( x -p[j+2]; p=ρ )
    end
    return out
end

"""
    modelm_h(t,p,ρ)

Compute the Hessian for modelm(t,p,ρ).
Compute the Hessian for modelm(t,p,ρ) with respect to parameters p,
    in the form:
    ``p = \\{ r_0, t_0, r_1, t_1, \\ldots , r_m, t_m \\}``, thus
    generating a ``n_t \\, \\times \\, 2m+1 \\, \\times \\, 2m+1``
    matrix, where ``n_t`` is the number of elements in input
    argument ``t``.

# Sources:
1. private notes contact me via https://github.com/f-hipolito


# Examples
```julia-repl
julia> modelm_h( 10, [0.25, 0. ], [6.2,.97] )
2×2 Array{Float64,2}:
-5.94678    0.594678
 0.594678  -0.0594678

julia> modelm_h( [-10, 0.0, 10.0], [0.25, 0. ], [6.2,.97] )
3×2×2 Array{Float64,3}:
[:, :, 1] =
  5.94678  0.594678
  0.0      0.0
 -5.94678  0.594678

[:, :, 2] =
 0.594678   0.0594678
 0.0        0.0
 0.594678  -0.0594678
```
"""
function modelm_h(x::Array,p,ρ)
    #compute m and verify that length of p is even
    lp::Int = length(p)
    lx::Int = length(x)
    m::Int,r::Int = fldmod( lp, 2) .-(1,0)
    # @assert r==0 "Length of p must be even"
    # @assert iseven( length( ρ ) ) "Length of ρ must be even"

    H   = zeros(Float64, lx, lp, lp )
    d2R = zeros(Float64, lx, lp, lp )
    dR = zeros(Float64, lx, lp )
    R  = zeros(Float64, lx )
    dn = zeros(Float64, lx )
    δn = zeros(Float64, lx )
    j::Int=0

    # evaluate R and all terms involving the Logistic function
    Rt!(R,m,x,p,ρ)
    dn = σ.(  R) .* σ.(.-R)
    δn = σ.(.-R) .- σ.(  R)

    # evaluate the first and second order derivatives
    for i=0:m
        j=2*i
        dR[:,j+1] = dr(i,m,x,p,ρ)
        dR[:,j+2] = dt(i,x,p,ρ)
        for k=0:m
            l=2*k
            # ∂^2 R / ∂r^2   # all are identically zero in the current model
            # d2R[:,l+1,j+1] = d2rr( k,i,m,x,p,ρ )

            # ∂^2 R / ( ∂t ∂r )
            d2R[:,l+2,j+1] = d2tr( k,i,m,x,p,ρ )

            # ∂^2 R / ( ∂t ∂r )
            d2R[:,l+1,j+2] = d2rt( k,i,m,x,p,ρ )

            # ∂^2 R / ∂r^2
            d2R[:,l+2,j+2] = d2tt( k,i,m,x,p,ρ )
        end
    end

    for j=1:lp
        for l=1:lp
            @. H[:,l,j] = dn *( d2R[:,l,j] +δn*dR[:,l]*dR[:,j] )
        end
    end

    return H
end

function modelm_h(x::Number,p,ρ)
    #compute m and verify that length of p is even
    lp::Int = length(p)
    m::Int,r::Int = fldmod( lp, 2) .-(1,0)
    # @assert r==0 "Length of p must be even"
    # @assert iseven( length( ρ ) ) "Length of ρ must be even"

    H   = zeros(Float64, lp, lp )
    d2R = zeros(Float64, lp, lp )
    dR = zeros(Float64, lp )
    R::Float64  = 0
    dn::Float64 = 0
    δn::Float64 = 0
    j::Int=0

    # evaluate R and all terms involving the Logistic function
    R = Rt(m,x,p,ρ)
    dn = σ( R) * σ(-R)
    δn = σ(-R) - σ( R)

    # evaluate the first and second order derivatives
    for i=0:m
        j=2*i
        dR[j+1] = dr(i,m,x,p,ρ)
        dR[j+2] = dt(i,x,p,ρ)
        for k=0:m
            l=2*k
            # ∂^2 R / ∂r^2   # all are identically zero in the current model
            # d2R[:,l+1,j+1] = d2rr( k,i,m,x,p,ρ )

            # ∂^2 R / ( ∂t ∂r )
            d2R[l+2,j+1] = d2tr( k,i,m,x,p,ρ )

            # ∂^2 R / ( ∂t ∂r )
            d2R[l+1,j+2] = d2rt( k,i,m,x,p,ρ )

            # ∂^2 R / ∂r^2
            d2R[l+2,j+2] = d2tt( k,i,m,x,p,ρ )
        end
    end

    for j=1:lp
        for l=1:lp
            H[l,j] = dn *( d2R[l,j] +δn*dR[l]*dR[j] )
        end
    end

    return H
end




x0 = [-10, 0.0, 10.0]
x1 = [0.0, 50.0, 200.0]

p0 = [0.25, 0. ]
p1 = [0.25, 18.8, 0.075, 45. ]
p2 = [0.25, 18.8, 0.075, 45., 0.1, 80 ]

ρ0 = [6.2,.97]

R0 = zeros(Float64, length(x0))
R1 = zeros(Float64, length(x0))
R2 = zeros(Float64, length(x0))

@assert Rt!(R0,0,x0,p0,ρ0) == [ -2.5, 0.0, 2.5 ]
@assert Rt!(R1,1,x0,p1,ρ0) == [ -21.3, -18.8, -16.3 ]
@assert Rt!(R2,2,x0,p2,ρ0) == [ -21.3, -18.8, -16.3 ]

@assert dr( 1,1,x1,p1,ρ0 ) == [ 0.0, 0.3963534314129777, 148.60824742268042 ]
@assert dt( 1,x1,p2,ρ0 ) == [ 0.0, 0.05707898343082066, 0.175 ]

@assert d2rr( 0,0,0,x0,p0,ρ0 ) == zeros(Float64,length(x0))
@assert d2tr( 1,1,1,x1,p1,ρ0 ) == [ -0.0, -0.3261656196046895, -1.0 ]
@assert d2rt( 1,1,1,x1,p1,ρ0 ) == [ -0.0, -0.3261656196046895, -1.0 ]
@assert d2tt( 1,1,1,x1,p1,ρ0 ) == [ -0.0, -0.028867524522792173, -1.0608237504005124e-57 ]


modelm_j(x1, p2, ρ0);
modelm_h( x1, p2, ρ0 );



"""
    sir_Γ!(du,u,t; p, ν=0.01, ρ=[2.0,1.0])

Computes the differential equations for susceptible, infected and removed
    fractions of the population, using a Gamma distribution for the incubation
    times.
Computes the differentials ``du = [ ds, di, dr ]`` for ``u = [ 1-s, i, r ]``
    evaluated at time ``t`` and parametrized via
    ``p_0 = \\{ \\omega_0, \\omega_1, t_1, \\ldots \\omega_m, t_m \\}`` where
    ``\\omega_i`` and ``t_i`` are the ``i^\\mathrm{th}`` growth rate and
    transition times, respectively. ``\\nu`` is the combined recovery/death
    decay-like time scale.
    The distribution of incubation times is parametrized with default values
    ``\\rho_0 = [ \\alpha_0, \\beta_0 ] = [2,1]``.

Note: we normalized variables with respect to the total population ``N_t``, i.e.
    population densities, namely
1. ``s+i+r = 1``
2. ``s = S/N_t`` susceptible individuals
3. ``i = I/N_t`` infected    individuals
4. ``r = R/N_t`` removed     individuals

Furthermore, the equations are written considering for ``1-s`` rather than
    ``s``, to facilitate the optimization process, as at the initial instant
    ``s`` is, typically, ``N_t \\sim 10^{6-8}`` times larger than ``i`` and
    ``r``.

# Sources:
1. see https://github.com/f-hipolito/PopulationDynamics/blob/master/Population_dynamics.ipynb

# Examples
```julia-repl
julia> du0 = zeros(3); u0 = [ 1e-7, 1e-7, 0. ]; t0 = 1.0;

julia> p0 = [ 0.26, 0.01, 34.9 ]; νμ0 = 1/21.0; ρ0 = [ 6.25, 0.96 ];

julia> sir_Γ!(du0,u0,t0; p=p0, ν=νμ0, ρ=ρ0)

julia> @assert isapprox( du0, [2.59999974e-8,2.1238092638e-8,4.761904761e-9] )
```
"""
function sir_Γ!(du,u,t; p, ν=0.01, ρ=[2.0,1.0])

    # basic testing of parameters
    #
    len = length(p) +1

    if len < 4
        error( "insufficient parameters for sir_Γ!" )
    elseif isodd( len )
        println( len )
        error( "requires even number of parameters!" )
    end

    # loading parameters
    #
    #   the growth rates ωᵢ for the m-steady state model
    m = fld( len, 2) -1     # number m-steady states
    ω = zeros( len )
    ω[1] = p[1]
    ω[3:end] = p[2:end]

    #   the incubation time paramters
    ρ0 = ρ # ρ₀ = ρ = [ α₀, β₀ ]

    #   compute the "m-steady state" rates
    #
    Ω = ω[1]
    for i=1:m
        j=2*i
        Ω += (ω[j+1]-ω[j-1])*FΓ(t-ω[j+2];p=ρ0)
    end

    #   the differential equations
    #
    dr = ν *u[2]
    ds = Ω *u[2] *(1.0 -u[1])
    du[1] =  ds                 #   u[1] ≣ 1 -susceptible
    du[2] =  di = ds -dr        #   u[2] ≣ infected
    du[3] =  dr                 #   u[3] ≣ removed
end

du0 = zeros(3); u0 = [ 1e-7, 1e-7, 0. ]; t0 = 1.0;

p0 = [ 0.26, 0.01, 34.9 ]; νμ0 = 1/21.0; ρ0 = [ 6.25, 0.96 ];

sir_Γ!(du0,u0,t0; p=p0, ν=νμ0, ρ=ρ0)

@assert isapprox( du0, [ 2.59999974e-8, 2.1238092638e-8, 4.761904761e-9 ] )






"""
    sird_Γ!(du,u,t; p, νμ=[1e-2,1e-4], ρ=[2.0,1.0] )

Computes the differential equations for susceptible, infected, recovered and
    deceased fractions of the population, using a Gamma distribution for the
    incubation times.
Computes the differentials ``du = [ ds, di, dr, dd ]`` for
    ``u = [ 1-s, i, r, d ]`` evaluated at time ``t`` and parametrized via
    ``p = \\{ \\omega_0, \\omega_1, t_1, \\ldots \\omega_m, t_m \\}``
    where ``\\omega_i`` and ``t_i`` are the ``i^\\mathrm{th}`` growth rate and
    transition times, respectively. ``\\nu,\\, \\mu`` are the recovery and death
    decay-like time scales.
    The distribution of incubation times is parametrized with default values
    ``\\rho_0 = [ \\alpha_0, \\beta_0 ] = [2,1]``.

Note: we normalized variables with respect to the total population ``N_t``, i.e.
    population densities, namely
1. ``s+i+r = 1``
2. ``s = S/N_t`` susceptible individuals
3. ``i = I/N_t`` infected    individuals
4. ``r = R/N_t`` recovered   individuals
5. ``d = D/N_t`` deceased    individuals

Furthermore, the equations are written considering for ``1-s`` rather than
    ``s``, to facilitate the optimization process, as at the initial instant
    ``s`` is, typically, ``N_t \\sim 10^{6-8}`` times larger than ``i`` and
    ``r``.

# Sources:
1. see https://github.com/f-hipolito/PopulationDynamics/blob/master/Population_dynamics.ipynb

# Examples
```julia-repl
julia> du0 = zeros(4); u0 = [ 1e-7,1e-7,0.,0. ]; t0 = 1.0;

julia> p0 = [ 0.26, 0.01, 34.9 ]; νμ0 = [ 1/21.0, 0.5/21 ]; ρ0 = [ 6.25, 0.96 ];

julia> sird_Γ!(du0,u0,t0; p=p0, νμ=νμ0, ρ=ρ0)

julia> @assert isapprox(du0,[2.59999974e-8,1.885714025e-8,4.76190476e-9,2.38095238e-9])
```
"""
function sird_Γ!(du,u,t; p, νμ = [ 1e-2, 1e-4 ], ρ=[2.0,1.0])
        # basic testing of parameters
    #
    len = length(p) +1

    if len < 4
        error( "insufficient parameters for sird_Γ!" )
    elseif isodd( len )
        println( len )
        error( "requires even number of p parameters!" )
    end

    # loading parameters
    #
    #   the growth rates ωᵢ for the m-steady state model
    m = fld( len, 2) -1     # number m-steady states
    ω = zeros( len )
    ω[1] = p[1]
    ω[3:end] = p[2:end]

    #   the incubation time paramters
    ρ0 = ρ # ρ₀ = ρ = [ α₀, β₀ ]

    #   the infection decay time
    ν = νμ[1]
    μ = νμ[2]

    #   compute the "m-steady state" rates
    #
    Ω = ω[1]
    for i=1:m
        j=2*i
        Ω += (ω[j+1]-ω[j-1])*FΓ(t-ω[j+2];p=ρ0)
    end

    #   the differential equations
    #
    dr = ν *u[2]
    dd = μ *u[2]
    ds = Ω *u[2] *( 1.0 -u[1] )
    du[1] =  ds                 #   u[1] ≣ 1 -susceptible
    du[2] =  di = ds -dr -dd    #   u[2] ≣ infected
    du[3] =  dr                 #   u[3] ≣ recovered
    du[4] =  dd                 #   u[3] ≣ deceased
end

du0 = zeros(4); u0 = [ 1e-7, 1e-7, 0., 0. ]; t0 = 1.0;

p0 = [ 0.26, 0.01, 34.9 ]; νμ0 = [ 1/21.0, 0.5/21 ]; ρ0 = [ 6.25, 0.96 ];

sird_Γ!(du0,u0,t0; p=p0, νμ=νμ0, ρ=ρ0)

@assert isapprox(du0,[2.59999974e-8,1.885714025e-8,4.76190476e-9,2.38095238e-9])

end
