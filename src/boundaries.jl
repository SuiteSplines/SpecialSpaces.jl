"""
    boundary_number(s::Symbol)

Return the boundary number corresponding to boundary `s`.

```
                                               5
                            4                 back
                           top                  . . . . . . .
                     . . . . . . . .            . .         . .
y                    .             .            .   .       .   .
.                    .             .            .     . . . . . . .
.             left   .             .  right     .     .     .     .
.               1    .             .    2       .     .     .     .
· · · ·  x           .             .            . . . . . . .     .
  ·                  . . . . . . . .              .   .       .   .
    ·                    bottom                     . .         . .
      z                     3                         . . . . . . .
                                                                 front
                                                                   6
```
"""
boundary_number(s::Symbol) = boundary_number(Val(s))
boundary_number(::Val{:left}) = 1
boundary_number(::Val{:right}) = 2
boundary_number(::Val{:bottom}) = 3
boundary_number(::Val{:top}) = 4
boundary_number(::Val{:back}) = 5
boundary_number(::Val{:front}) = 6

"""
    boundary_symbol(s::Int)

Return the boundary label (symbol) for boundary number `s`.
"""
boundary_symbol(s::Int) = boundary_symbol(Val(s))
boundary_symbol(::Val{1}) = :left
boundary_symbol(::Val{2}) = :right
boundary_symbol(::Val{3}) = :bottom
boundary_symbol(::Val{4}) = :top
boundary_symbol(::Val{5}) = :back
boundary_symbol(::Val{6}) = :front

check_boundary_label(::Val{2}, s::Symbol) = s in [:left, :right, :top, :bottom]
check_boundary_label(::Val{3}, s::Symbol) = s in [:left, :right, :top, :bottom, :back, :front]
