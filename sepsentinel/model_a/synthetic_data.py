# Synthetic calibration data for Model A development.
#
# Generates simulated electrochemical sensor outputs paired with
# known analyte concentrations. Used for pipeline development only —
# real calibration requires experimental data.
#
# Planned generators:
#   generate_swv_data(n_samples)      → (voltammograms, il6_concentrations)
#   generate_amperometric_data(n)     → (current_traces, lactate_concentrations)
#   generate_potentiometric_data(n)   → (voltage_traces, ph_values)
