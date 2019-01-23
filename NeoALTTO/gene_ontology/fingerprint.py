import rdkit.Chem as Chem
from rdkit.Chem import AllChem
import numpy as np


def get_fp(s):
    mol = Chem.MolFromSmiles(s)
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, 3, nBits=1024)
    return np.array(fp, dtype=int)


# canonical SMILES taken from pubchem
cisplatin = "N.N.[Cl-].[Cl-].[Pt+2]"
paclitaxel = "CC1=C2C(C(=O)C3(C(CC4C(C3C(C(C2(C)C)(CC1OC(=O)C(C(C5=CC=CC=C5)NC(=O)C6=CC=CC=C6)O)O)OC(=O)C7=CC=CC=C7)(CO4)OC(=O)C)O)C)OC(=O)C"
carboplatin = "C1CC(C1)(C(=O)O)C(=O)O.N.N.[Pt]"
fluorouracil = "C1=C(C(=O)NC(=O)N1)F"
lapatinib = "CS(=O)(=O)CCNCC1=CC=C(O1)C2=CC3=C(C=C2)N=CN=C3NC4=CC(=C(C=C4)OCC5=CC(=CC=C5)F)Cl"
trast = "C1C(N(CC2=C(N1CC3=CN=CN3)C=CC(=C2)C#N)S(=O)(=O)C4=CC=CS4)CC5=CC=CC=C5" #https://pubchem.ncbi.nlm.nih.gov/compound/bms-214662#section=InChI

drugs = [cisplatin, paclitaxel, carboplatin, fluorouracil, lapatinib, trast]

fps = None
for drug in drugs:
    if fps is None:
        fps = get_fp(drug)
    else:
        fps = np.vstack((fps, get_fp(drug)))
print(fps)
np.savetxt("drug_fp.csv", fps, delimiter=",", fmt="%d")
