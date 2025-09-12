using Base.Broadcast, FillArrays

struct CirculantArrayStyle <: Broadcast.BroadcastStyle end

Broadcast.BroadcastStyle(::Type{<:Circulant}) = CirculantArrayStyle()

Broadcast.BroadcastStyle(::Broadcast.AbstractArrayStyle{0}, b::CirculantArrayStyle) = b
Broadcast.BroadcastStyle(a::Broadcast.AbstractArrayStyle, ::CirculantArrayStyle) = a
Broadcast.BroadcastStyle(a::CirculantArrayStyle, ::CirculantArrayStyle) = a

function Base.similar(bc::Broadcast.Broadcasted{CirculantArrayStyle}, ::Type{ElType}) where {ElType}
    Circulant(similar(find_aac(bc).v))
end

_getcol(a::Circulant, i::Int) = a.v[i]
_getcol(a::Ref, i) = a[]
_getcol(a, i) = a

function Base.copyto!(dest::Circulant, bc::Broadcast.Broadcasted{Nothing})
    _, ax = axes(bc)
    bc2 = Broadcast.flatten(bc)
    for i in axes(dest)[1]
        dest.v[i] = bc2.f([_getcol(a, i) for a in bc2.args]...)
    end
    dest
end

find_aac(bc::Broadcast.Broadcasted) = find_aac(bc.args)
find_aac(args::Tuple) = find_aac(args[1], Base.tail(args))
find_aac(::Tuple{}) = nothing
find_aac(a::Circulant, rest) = a
find_aac(::Any, rest) = find_aac(rest)

function Base.:*(A::Circulant, b::Ones)
    if size(A, 2) == size(b, 1)
        Fill(sum(A.v), (size(A, 1), Base.tail(size(b))...))
    else
        DimensionMismatch(A, b)
    end
end

function Base.:*(A::Circulant, b::Fill)
    if size(A, 2) == size(b, 1)
        Fill(b.value * sum(A.v), (size(A, 1), Base.tail(size(b))...))
    else
        DimensionMismatch(A, b)
    end
end
