# MIMIC-IV data loader — Module 9 implementation.
#
# Will load and preprocess data from:
#   - MIMIC-IV Clinical Database (chartevents, labevents)
#   - MIMIC-IV Waveform Database (continuous physiological signals)
#
# Output format will match generate_episodes() so the rest of the pipeline
# (windowing, normalization, model training) works identically.
#
# Requires PhysioNet credentialed access.
