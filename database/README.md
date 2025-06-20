# Database
1. Ionic Liquids (`IL-anions.csv`, `IL-cations.csv`, `collected-ILs.csv`, and `collected-ILs.txt`): These databases are collected from ILThermo, and other literature resources (the details can be found in the paper).
2. Ionic Liquids Melting Point (`IL-melting-point.csv`): This is from [Venkatraman et al.](https://www.sciencedirect.com/science/article/pii/S0167732218315186).
3. Exhausted Cation-Anion Combinations: This is created by combining all the IL cations and all the IL anions.
4. Expanded ILs (`expanded-ILs.csv` and `expanded-ILs_vocab.pkl`): This is from link prediction by running `python step1_link_prediction/run.py`. The vocab file corresponds to the unique characters in all the IL SMILES strings.
