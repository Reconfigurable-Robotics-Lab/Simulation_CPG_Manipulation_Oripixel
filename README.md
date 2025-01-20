# Simulation_CPG_Manipulation_Oripixel

This repository includes the supplementary material for paper "CPG-Based Manipulation with Multi-Module Origami Robot Surface".

## Files Description
- Oripixel.py - Code for generating simulation model
- parameter_optimization.py - For running optimizations for finding optimal CPG parameter sets
- simulation_model.xml - Example model generated using the Oripixel.py

## Optimized CPG Parameters for Manipulations

### Fast Manipulations
| Parameter | Y+ | Y- | X+ | X- | CW | CCW |
|-----------|-----|-----|-----|-----|-----|-----|
| h_amp | 0.0091 | 0.0140 | 0.0091 | 0.0157 | 0.0118 | 0.0118 |
| ψ_amp | 39.800 | 33.532 | 42.120 | 34.802 | 22.095 | 22.095 |
| f | 0.7008 | 0.74590 | 0.7312 | 0.7911 | 0.5782 | 0.5782 |
| h₀ | 0.0356 | 0.0358 | 0.0382 | 0.0326 | -0.0383 | -0.0383 |
| ψ₀ | -0.4031 | -1.0269 | -0.3702 | -1.8514 | -0.0616 | 0.0616 |
| σ | 1.6525 | 4.7527 | 1.5470 | 4.6462 | 0.7958 | 3.9374 |
| φ | π | π | π | π | π | π |
| ε | 0.2170 | 0.1563 | 0.2434 | 0.228 | 0.1 | 0.1 |

### Smooth Manipulations
| Parameter | Y+ | Y- | X+ | X- | CW | CCW |
|-----------|-----|-----|-----|-----|-----|-----|
| h_amp | 0.0124 | 0.0124 | 0.0080 | 0.0086 | - | - |
| ψ_amp | 22.807 | 22.807 | 26.734 | 23.264 | - | - |
| f | 0.6576 | 0.6576 | 0.6624 | 0.6803 | - | - |
| h₀ | 0.0323 | 0.0323 | 0.0329 | 0.0335 | - | - |
| ψ₀ | -1.4558 | -1.4558 | 1.4765 | -2.3114 | - | - |
| σ | 1.8311 | 4.9727 | 5.3801 | 2.3289 | - | - |
| φ | 2.7647 | 2.7647 | 3.0008 | 2.9393 | - | - |
| ε | 0.1827 | 0.1827 | 0.1701 | 0.1880 | - | - |
