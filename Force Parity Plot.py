#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul  8 09:48:19 2025

@author: emiliolazcano
"""

import re,os
import matplotlib.pyplot as plt
import numpy as np
from ase.io import read,write

from orb_models.forcefield import pretrained
from orb_models.forcefield.calculator import ORBCalculator

from sklearn.metrics import mean_squared_error
from pathlib import Path
import pandas as pd

from ase.io import read, write
from ase.visualize import view
from ase import units




### Written with help from chatGPT ###

def format_forces(forces):
    return [str(f.tolist()) for f in forces]

# Step 0: set model 
output_file = "energies_forces_macempa0medium.txt"

device = "cpu"  # or "cuda"

# Load a pre-trained OrbNet model
orbff = pretrained.orb_v3_conservative_inf_omat(
    device=device,
    precision="float32-high",  # can be "float32-highest" or "float64"
)

# Wrap the model as an ASE calculator
calc = ORBCalculator(orbff, device=device)
# Step 1: read in OUTCAR file
dft_file = read('/Users/emiliolazcano/Desktop/MD500/OUTCAR.5', format="vasp-out")

dft_energies = []
ml_energies = []
dft_forces = []
dft_forcesASE = []
ml_forces = []
sources = []

with open(dft_file) as f:
    lines = f.readlines()

i = 0
while i < len(lines):
    try:
        natoms = int(lines[i])  # First line of frame
    except ValueError:
        raise ValueError(f"Expected atom count at line {i}, got: {lines[i]}")
    
    comment_line = lines[i + 1]
    match = re.search(r'energy=([-+]?[0-9]*\.?[0-9]+)', comment_line)
    struct = re.search(r'source=([^\s]+)', comment_line)
    if match:
        dft_energies.append(float(match.group(1)))
    else:
        raise ValueError(f"No energy found in line: {comment_line}")
    if struct:
        sources.append(struct.group(1))
    else:
        raise ValueError(f"No source found in line: {comment_line}")
    i += 2
    forces = []
    for _ in range(natoms):
        parts = lines[i].split()
        # Assuming format: Symbol x y z fx fy fz
        fx, fy, fz = map(float, parts[-3:])
        forces.append([fx, fy, fz])
        i += 1
    dft_forces.extend(forces) 

dft_energies = np.array(dft_energies)
dft_forces = np.array(dft_forces)

print("Extracted", len(dft_energies), "DFT energies.")

# Step 2: Loop through structures and extract energies and forces
structures = read(dft_file,index=":")
for config in structures:
    try:
        atoms = config
        # save dft_forces
        dft_force = atoms.get_forces()
        dft_forcesASE.append(dft_force)
        
        # mlff calculations
        #atoms.set_calculator(grace_fm(model))
        atoms.set_calculator(device)
        ml_energy = atoms.get_potential_energy()
        ml_force = atoms.get_forces()

        # Attach results to atoms object for XYZ metadata
        atoms.info['energy'] = ml_energy
        atoms.arrays['forces'] = ml_force
        atoms.calc = None
        print(f"Processed structure {i}: Energy = {ml_energy:.6f} eV")

        # Append data
        #dft_energies.append(dft_energy)
        ml_energies.append(ml_energy)
        #dft_forces.append(dft_force)
        ml_forces.append(ml_force)

    except Exception as e:
        print(f"Error processing config: {e}")

# Step 3: Flatten force arrays for comparison

print(len(dft_energies),len(ml_energies),len(dft_forces),len(ml_forces),len(sources))

dft_forces_flat = np.concatenate([f.reshape(-1) for f in dft_forces])
ml_forces_flat = np.concatenate([f.reshape(-1) for f in ml_forces])

df = pd.DataFrame({
    'dft_energy': dft_energies,
    'ml_energy': ml_energies,
    'dft_force': format_forces(dft_forcesASE),
    'ml_force': format_forces(ml_forces),
    'source': sources
})

df.to_csv('dft_ml_comparison.csv', index=False)

# === Write to extended XYZ format ===
write(output_file, structures, format="extxyz")

# Step 4: Compute RMSE
rmse_energy = np.sqrt(mean_squared_error(dft_energies, ml_energies))
rmse_force = np.sqrt(mean_squared_error(dft_forces_flat, ml_forces_flat))

# Step 5: Parity plot - Energy
plt.figure(figsize=(6, 6))
plt.scatter(dft_energies, ml_energies, alpha=0.7)
e_min, e_max = min(dft_energies + ml_energies), max(dft_energies + ml_energies)
plt.plot([e_min, e_max], [e_min, e_max], 'k--', label='y = x')
plt.xlabel("DFT Energy (eV)")
plt.ylabel("GraceFM Energy (eV)")
plt.title("Energy Parity Plot")
plt.text(0.05, 0.95, f"RMSE = {rmse_energy:.4f} eV", transform=plt.gca().transAxes,
         fontsize=12, verticalalignment='top')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# Step 6: Parity plot - Forces
plt.figure(figsize=(6, 6))
plt.scatter(dft_forces_flat, ml_forces_flat, alpha=0.5, s=5)
f_min, f_max = min(dft_forces_flat.min(), ml_forces_flat.min()), max(dft_forces_flat.max(), ml_forces_flat.max())
plt.plot([f_min, f_max], [f_min, f_max], 'k--', label='y = x')
plt.xlabel("DFT Forces (eV/Å)")
plt.ylabel("GraceFM Forces (eV/Å)")
plt.title("Force Parity Plot")
plt.text(0.05, 0.95, f"RMSE = {rmse_force:.4f} eV/Å", transform=plt.gca().transAxes,
         fontsize=12, verticalalignment='top')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()