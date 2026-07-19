# Bluetooth Low Energy data reception — Module 10 implementation.
#
# Hardware data flow:
#   Microneedle electrochemical sensors measure analytes in ISF (no extraction).
#     - IL-6: E-AB three-electrode sensor, SWV protocol
#     - Lactate: likely amperometric (final hardware TBD)
#     - pH: likely potentiometric (final hardware TBD)
#   -> Potentiostat (electrochemical protocol control)
#   -> ADC / analog front end
#   -> Microcontroller (signal processing, calibration)
#   -> BLE transmission of processed biomarker values
#
#   Physiological sensors (PPG, thermistor, impedance)
#   -> ADC
#   -> Microcontroller
#   -> BLE transmission of processed values (HR, RR, Temp, SpO2)
#
# The software pipeline receives calibrated values (concentrations, pH, bpm, etc.)
# regardless of the underlying sensing modality.
#
# This module will handle:
#   - BLE connection to the wearable patch
#   - Parsing incoming data packets (7 signal values + timestamps)
#   - Buffering readings into the rolling history window
#   - Feeding the ML model for real-time inference
