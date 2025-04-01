# Secure safety filter for nonlinear systems
This is an accompanying code repo for our IEEE CDC 2025 submission **Secure Safety Filter Design for Sampled-data Nonlinear Systems under Sensor Spoofing Attacks** by Xiao Tan, Pio Ong, Paulo Tabuada, and Aaron D. Ames.

## To replicate the results
run `main.py` to obtain the results without a secure safety filter

run `main.py --use_filter True` to obtain the results with a secure safety filter

## Acknowledgement
Parts of the code in this repo are based on existing works. The time derivative estimation library in `\lib` is taken from [nonlinear_observers](https://github.com/rahalnanayakkara/nonlinear_observers). Part of the zero-order CBF code in `unicycle_zocbf.py` is modified based on [Zero-Order-CBFs](https://github.com/ersindas/Zero-order-CBFs). Thanks to Rahal Nanayakkara and Ersin Das.