import autograd.numpy as np
import networkx as nx
from rdkit import Chem
from util import one_of_k_encoding, one_of_k_encoding_unk


def aa_features(aa_node, pin_graph):
    """
    Returns an array of encoded features from each amino acid node.

    Parameters:
    ===========
    - aa_node:      A NetworkX node from a Protein Interaction Graph.
    - pin_graph:    The Protein Interaction Graph's graph object, accessed
                    through the ProteinInteractionNetwork.masterG attribute.

    Returns:
    ========
    - an concatenated numpy array of encoded features, encompassing:
        - one-of-K encoding for amino acid identity at that node [23 cells]
        - the node degree, i.e. the number of other nodes it is connected to
          [1 cell]
        - the sum of all distances on each edge connecting those nodes [1 cell]
        - the molecular weight of the amino acid [1 cell]
        - the pKa of the amino acid [1 cell]

    Note to future self: Add more features as you think of them!
    """
    



def atom_features(atom):
    return np.array(one_of_k_encoding_unk(atom.GetSymbol(),
                                      ['C', 'N', 'O', 'S', 'F', 'Si', 'P', 'Cl', 'Br', 'Mg', 'Na',
                                       'Ca', 'Fe', 'As', 'Al', 'I', 'B', 'V', 'K', 'Tl', 'Yb',
                                       'Sb', 'Sn', 'Ag', 'Pd', 'Co', 'Se', 'Ti', 'Zn', 'H',    # H?
                                       'Li', 'Ge', 'Cu', 'Au', 'Ni', 'Cd', 'In', 'Mn', 'Zr',
                                       'Cr', 'Pt', 'Hg', 'Pb', 'Unknown']) +
                    one_of_k_encoding(atom.GetDegree(), [0, 1, 2, 3, 4, 5]) +
                    one_of_k_encoding_unk(atom.GetTotalNumHs(), [0, 1, 2, 3, 4]) +
                    one_of_k_encoding_unk(atom.GetImplicitValence(), [0, 1, 2, 3, 4, 5]) +
                    [atom.GetIsAromatic()])

def bond_features(bond):
    bt = bond.GetBondType()
    return np.array([bt == Chem.rdchem.BondType.SINGLE,
                     bt == Chem.rdchem.BondType.DOUBLE,
                     bt == Chem.rdchem.BondType.TRIPLE,
                     bt == Chem.rdchem.BondType.AROMATIC,
                     bond.GetIsConjugated(),
                     bond.IsInRing()])

def num_atom_features():
    # Return length of feature vector using a very simple molecule.
    m = Chem.MolFromSmiles('CC')
    alist = m.GetAtoms()
    a = alist[0]
    return len(atom_features(a))

def num_bond_features():
    # Return length of feature vector using a very simple molecule.
    simple_mol = Chem.MolFromSmiles('CC')
    Chem.SanitizeMol(simple_mol)
    return len(bond_features(simple_mol.GetBonds()[0]))

