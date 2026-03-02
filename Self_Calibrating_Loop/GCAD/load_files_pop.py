# Adopted from GCAD Repo
# Edited by Phoebe Kuang

import pickle
import numpy as np

# ==========================================
# 1. LOAD THE ESSENTIALS
# ==========================================
with open("promo.pkl", "rb") as fid:
    promo = pickle.load(fid)

with open("parts.pkl", "rb") as fid:
    parts = pickle.load(fid)

with open("Ref.pkl", "rb") as fid:
    Ref = pickle.load(fid)

# ==========================================
# 2. DYNAMIC LISTS
# ==========================================

# Restrict to synTF1/2 Case
# tf_list = ['Z1', 'Z2']
# inhibitor_list = ['I1', 'I2']

# Free-search Case
# This automatically grabs Z1-Z15 and I1-I15 from the parts dictionary
tf_list = [k for k in parts.keys() if k[0] == 'Z']
inhibitor_list = [k for k in parts.keys() if k[0] == 'I']
print(f"Loaded {len(tf_list)} Activators and {len(inhibitor_list)} Inhibitors.")

# ==========================================
# 3. FAKE THE POPULATION MATRICES
# ==========================================
# We keep these as placeholders so the rest of the code doesn't crash 
# looking for variables, since we are doing Single-Cell analysis right now.
Ref_pop20 = None
Ref_pop200 = None
Z_20 = np.zeros((20, 15))   
Z_200 = np.zeros((200, 15))