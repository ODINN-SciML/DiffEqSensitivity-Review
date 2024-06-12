@kwdef struct DualNumber{F <: AbstractFloat}
    value::F
    derivative::F = 0.0
    # Inner constructors
    # DualNumber(value, derivative) = new(value, derivative)
    # function DualNumber(value::F) where {F <: AbstractFloat}
    #     new(value, 0.0)
    # end
end

# Outer constructor
function DualNumber(value::F) where {F <: AbstractFloat}
    DualNumber(value, F(0.0))
end 

# Chain rules for binary opperators

# Binary sum
Base.:(+)(a::DualNumber, b::DualNumber) = DualNumber(value      = a.value + b.value,
                                                     derivative = a.derivative + b.derivative)

# Binary product
Base.:(*)(a::DualNumber, b::DualNumber) = DualNumber(value      = a.value * b.value,
                                                     derivative = a.value*b.derivative + a.derivative*b.value)

# Power
Base.:(^)(a::DualNumber, b::AbstractFloat) = DualNumber(value      = a.value ^ b, 
                                                        derivative = b * a.value^(b-1) * a.derivative)


# Now we define a series of variables. We are interested in computing the derivative with respect to the variable "a":

a = DualNumber(value=1.0, derivative=1.0)

b = DualNumber(value=2.0, derivative=0.0)
c = DualNumber(value=3.0, derivative=0.0)

# Now, we can evaluate a new DualNumber
result = a * b * c
# println("The derivative of a*b*c with respect to a is: ", result.derivative)