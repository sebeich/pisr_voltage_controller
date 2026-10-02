module IncWrap
export line_current, voltage_drop

# Custom operators used during training and inference
line_current(V_send::Complex, S::Complex) = conj(S / V_send)
voltage_drop(I::Complex, Z::Complex) = I * Z

end # module IncWrap
