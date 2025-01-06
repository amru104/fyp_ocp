#!/usr/bin/env python
# coding: utf-8

# In[3]:


from fairchem.core.models.model_registry import available_pretrained_models
print(available_pretrained_models)


# In[2]:


from fairchem.core.models.model_registry import model_name_to_local_file
checkpoint_path = model_name_to_local_file('EquiformerV2-31M-S2EF-OC20-All+MD', local_cache='/tmp/fairchem_checkpoints/')
checkpoint_path


# In[4]:


pip install torch-sparse

Optimization of OH Adsorption on Pt(111) Surface
# In[3]:


from ase.build import fcc111, add_adsorbate
from ase import Atoms
from ase.optimize import BFGS
from fairchem.core.common.relaxation.ase_utils import OCPCalculator
import matplotlib.pyplot as plt
from ase.visualize.plot import plot_atoms

# Define the model atomic system for Pt(111) with OH adsorbate
slab = fcc111('Pt', size=(3, 3, 5), vacuum=10.0)  

# Create the OH molecule as a separate Atoms object
OH = Atoms(['O', 'H'], positions=[(0, 0, 0), (0, 0, 1)])  

# Add the OH molecule to the surface 
add_adsorbate(slab, OH, height=1.3, position='fcc')  

# Load the pre-trained calculator
checkpoint_path = '/tmp/fairchem_checkpoints/eq2_31M_ec4_allmd.pt'
calc = OCPCalculator(checkpoint_path=checkpoint_path, cpu=False)
slab.set_calculator(calc)

# Run the optimization
opt = BFGS(slab)
opt.run(fmax=0.05, steps=100)

# Visualize the result
fig, axs = plt.subplots(1, 2)
plot_atoms(slab, axs[0])  # Left side view
plot_atoms(slab, axs[1], rotation=('-90x'))  # Right side view, rotated
axs[0].set_axis_off()
axs[1].set_axis_off()

# Show the plot
plt.show()


# In[5]:


from ase import Atoms
from ase.build import fcc111
from ase.visualize import view

try:
    slab = fcc111('Pt', size=(2, 2, 5), vacuum=10.0)
    view(slab)
except Exception as e:
    print("An error occurred:", e)
    
from ase import Atoms
from ase.optimize import BFGS
from fairchem.core.common.relaxation.ase_utils import OCPCalculator
import matplotlib.pyplot as plt
from ase.visualize.plot import plot_atoms

# Define the model atomic system with OH adsorbate for different catalysts
def optimize_catalyst(material, checkpoint_path):
    # Set up the atomic system for the catalyst 
    slab = fcc111(material, size=(2, 2, 5), vacuum=10.0)

    # Create the OH molecule as a separate Atoms object
    OH = Atoms(['O', 'H'], positions=[(0, 0, 0), (0, 0, 1)])  # Positions of O and H atoms for OH

    # Add the OH molecule to the surface (adjust height and position as needed)
    add_adsorbate(slab, OH, height=1.2, position='fcc')

    # Load the pre-trained calculator
    calc = OCPCalculator(checkpoint_path=checkpoint_path, cpu=False)
    slab.set_calculator(calc)

    # Run the optimization
    opt = BFGS(slab)
    opt.run(fmax=0.02, steps=200)

    # Visualize the result
    fig, axs = plt.subplots(1, 2)
    plot_atoms(slab, axs[0])
    plot_atoms(slab, axs[1], rotation=('-90x'))
    axs[0].set_axis_off()
    axs[1].set_axis_off()

    # Show the plot
    plt.show()


checkpoint_path = '/tmp/fairchem_checkpoints/eq2_153M_ec4_allmd.pt'

# Call the function with different materials
optimize_catalyst('Pd', checkpoint_path)  # For Pd(111)
optimize_catalyst('Au', checkpoint_path)  # For Au(111)
optimize_catalyst('Ag', checkpoint_path)  # For Ag(111)
optimize_catalyst('Ag', checkpoint_path)  # For Ag(100)
optimize_catalyst('Pt', checkpoint_path)  # For Pt(100)




# In[4]:


import json
from ase import Atoms
from ase.optimize import BFGS
from fairchem.core.common.relaxation.ase_utils import OCPCalculator
from ase.build import fcc111, add_adsorbate
from ase.constraints import FixAtoms

# Load the reference energies from the provided JSON file 
with open('./ocp/docs/tutorials/energies.json') as f:
    edata = json.load(f)

checkpoint_path = '/tmp/fairchem_checkpoints/eq2_31M_ec4_allmd.pt'

# Step 1: Compute energy of the clean Pt(111) slab
clean_slab = fcc111('Pt', size=(2, 2, 5), vacuum=10.0)
calc = OCPCalculator(checkpoint_path=checkpoint_path, cpu=False)
clean_slab.set_calculator(calc)

# Relax the clean slab
opt = BFGS(clean_slab)
opt.run(fmax=0.05, steps=100)
E_slab = clean_slab.get_potential_energy()
print(f"E_slab: {E_slab} eV")

# Step 2: Compute energy of isolated OH molecule
OH = Atoms(['O', 'H'], positions=[(0, 0, 0), (0, 0, 1.0)])
OH.set_calculator(calc)

# Relax OH 
opt = BFGS(OH)
opt.run(fmax=0.05, steps=50)
E_OH = OH.get_potential_energy()
print(f"E_OH: {E_OH} eV")

# Step 3: Create slab with OH adsorbed
slab_OH = clean_slab.copy()  # Start from the relaxed clean slab
add_adsorbate(slab_OH, Atoms(['O', 'H'], OH.get_positions()), height=1.5, position='fcc')

# Fix the slab atoms if desired, so only OH relaxes
constraint = FixAtoms(indices=[atom.index for atom in slab_OH if atom.symbol == 'Pt'])
slab_OH.set_constraint(constraint)

slab_OH.set_calculator(calc)
opt = BFGS(slab_OH)
opt.run(fmax=0.05, steps=100)
E_slab_OH = slab_OH.get_potential_energy()
print(f"E_slab_OH: {E_slab_OH} eV")

# 4. Calculate adsorption energy
E_adsorption_OH = E_slab_OH - (E_slab + E_OH)
print(f"Calculated adsorption energy for OH on Pt(111): {E_adsorption_OH} eV")






# In[9]:


import matplotlib.pyplot as plt
from ase import Atoms
from ase.optimize import BFGS
from fairchem.core.common.relaxation.ase_utils import OCPCalculator
from ase.build import fcc111, fcc100, add_adsorbate
from ase.constraints import FixAtoms

def build_surface(metal, facet='111', size=(2, 2, 5), vacuum=10.0):
    """
    Build the specified surface. Extend this function for other metals/facets if needed.
    """
    if facet == '111':
        slab = fcc111(metal, size=size, vacuum=vacuum)
    elif facet == '100':
        slab = fcc100(metal, size=size, vacuum=vacuum)
    else:
        raise ValueError(f"Unsupported facet: {facet}")
    return slab

def compute_oh_adsorption_energy(metal='Pt', facet='111', size=(2, 2, 5), vacuum=10.0, ads_height=1.5, ads_site='fcc', checkpoint_path='/tmp/fairchem_checkpoints/eq2_31M_ec4_allmd.pt'):
    # Initialize calculator
    calc = OCPCalculator(checkpoint_path=checkpoint_path, cpu=False)
    
    # 1. Compute energy of the clean metal slab
    clean_slab = build_surface(metal, facet=facet, size=size, vacuum=vacuum)
    clean_slab.set_calculator(calc)
    
    opt = BFGS(clean_slab, logfile=None)
    opt.run(fmax=0.05, steps=100)
    E_slab = clean_slab.get_potential_energy()

    # 2. Compute energy of isolated OH molecule
    OH = Atoms(['O', 'H'], positions=[(0, 0, 0), (0, 0, 1.0)])
    OH.set_calculator(calc)
    opt = BFGS(OH, logfile=None)
    opt.run(fmax=0.05, steps=50)
    E_OH = OH.get_potential_energy()

    # 3. Compute energy of slab with OH adsorbed
    slab_OH = clean_slab.copy()  # start from the relaxed clean slab
    add_adsorbate(slab_OH, Atoms(['O','H'], OH.get_positions()), height=ads_height, position=ads_site)
    
    # Fix the slab atoms
    constraint = FixAtoms(indices=[atom.index for atom in slab_OH if atom.symbol == metal])
    slab_OH.set_constraint(constraint)

    slab_OH.set_calculator(calc)
    opt = BFGS(slab_OH, logfile=None)
    opt.run(fmax=0.05, steps=100)
    E_slab_OH = slab_OH.get_potential_energy()

    # 4. Calculate adsorption energy:
    E_ads = E_slab_OH - (E_slab + E_OH)

    return E_ads

# The requested metals and facets:
systems = [
    ('Pt', '111'),  
    ('Pd', '111'),
    ('Au', '111'),
    ('Ag', '111'),
]

ads_energies = []
labels = []

# Compute and collect adsorption energies
checkpoint_path = '/tmp/fairchem_checkpoints/eq2_31M_ec4_allmd.pt'
for metal, facet in systems:
    E_ads = compute_oh_adsorption_energy(metal=metal, facet=facet, checkpoint_path=checkpoint_path)
    ads_energies.append(E_ads)
    labels.append(f"{metal}({facet})")
    print(f"OH Adsorption on {metal}({facet}): {E_ads:.3f} eV")
    
# Plot the results as a line graph 
plt.figure(figsize=(8,4))
plt.plot(labels, ads_energies, marker='o', color='skyblue', linewidth=2)  # Line plot with markers
plt.xlabel('Metal(Facet)')
plt.ylabel('OH Adsorption Energy (eV)')
plt.title('OH Adsorption Energies on Various Metals and Facets')
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
plt.show()


# In[10]:


import json
from ase import Atoms
from ase.optimize import BFGS
from fairchem.core.common.relaxation.ase_utils import OCPCalculator
from ase.build import fcc111, add_adsorbate
from ase.constraints import FixAtoms

# Set up the surface
metal = 'Pt'
facet = '111'
size = (2, 2, 5)
vacuum = 10.0
ads_height = 1.5
ads_site = 'fcc'
checkpoint_path = '/tmp/fairchem_checkpoints/eq2_31M_ec4_allmd.pt'

# Build the slab (for example, a Pt(111) surface)
slab = fcc111(metal, size=size, vacuum=vacuum)

# Create the adsorbate (OH)
OH = Atoms(['O', 'H'], positions=[(0, 0, 0), (0, 0, 1.0)])

# Add OH to the slab at the desired site
add_adsorbate(slab, OH, height=ads_height, position=ads_site)

# Fix the slab atoms if needed
constraint = FixAtoms(indices=[atom.index for atom in slab if atom.symbol == metal])
slab.set_constraint(constraint)

# Set up the OCP calculator
calc = OCPCalculator(checkpoint_path=checkpoint_path, cpu=False)
slab.set_calculator(calc)

# Relax the structure if desired
opt = BFGS(slab, logfile=None)
opt.run(fmax=0.05, steps=100)

# Now, just get the potential energy
# According to OCP documentation, this is already the adsorption energy.
E_adsorption = slab.get_potential_energy()

print(f"OH Adsorption Energy on {metal}({facet}): {E_adsorption:.3f} eV")


# In[19]:


import json

with open('./ocp/docs/tutorials/structures.json') as f:
    sdata = json.load(f)

# Print the structure to inspect the keys and data
print(json.dumps(sdata, indent=4))  # This will print the JSON data in a human-readable format


# In[ ]:


from __future__ import annotations

import argparse
import glob
import logging
import os

import fairchem.core

"""
This script provides users with an automated way to download, preprocess (where
applicable), and organize data to readily be used by the existing config files.
"""

DOWNLOAD_LINKS_is2re: dict[str, str] = {
    "is2re": "https://dl.fbaipublicfiles.com/opencatalystproject/data/is2res_train_val_test_lmdbs.tar.gz",
}


def get_data(datadir: str, task: str, del_intmd_files: bool) -> None:
    os.makedirs(datadir, exist_ok=True)

    if task != "is2re":
        raise Exception(f"Unrecognized task {task}")
    
    download_link = DOWNLOAD_LINKS_is2re[task]

    os.system(f"wget {download_link} -P {datadir}")
    filename = os.path.join(datadir, os.path.basename(download_link))
    logging.info("Extracting contents...")
    os.system(f"tar -xvf {filename} -C {datadir}")
    
    dirname = os.path.join(datadir, os.path.basename(filename).split(".")[0])
    os.system(f"mv {dirname}/data/is2re {datadir}")

    if del_intmd_files:
        cleanup(filename, dirname)


def cleanup(filename: str, dirname: str) -> None:
    import shutil

    if os.path.exists(filename):
        os.remove(filename)
    if os.path.exists(dirname):
        shutil.rmtree(dirname)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=str, default="is2re", help="Only is2re is supported")
    parser.add_argument(
        "--keep",
        action="store_true",
        help="Keep intermediate directories and files upon data retrieval/processing",
    )
    parser.add_argument(
        "--data-path",
        type=str,
        default=os.path.join(os.path.dirname(fairchem.core.__path__[0]), "data"),
        help="Specify path to save dataset. Defaults to 'fairchem.core/data'",
    )

    args: argparse.Namespace
    args, _ = parser.parse_known_args()
    
    get_data(
        datadir=args.data_path,
        task=args.task,
        del_intmd_files=not args.keep,
    )


# In[5]:


import json

with open('./ocp/docs/tutorials/energies.json') as f:
    edata = json.load(f)
    
with open('./ocp/docs/tutorials/structures.json') as f:
    sdata = json.load(f)
    
edata['Pt']['O']['fcc']['0.25']


# In[7]:


with open('./ocp/docs/tutorials/structures.json') as f:
    s = json.load(f)
    
sfcc = s['Pt']['O']['fcc']['0.25']    




# In[ ]:




