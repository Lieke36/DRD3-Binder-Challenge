#%%
#importeren van de juiste libraries 
import pandas as pd

#variabelen
filename = "data_raw.csv"

#%% 
def Read_me(filename):
    """ 
    Read a data file from disk 
    and return its full contents as a string
    """
    inf = open(filename, "r")  # open het bestand
    data = inf.read()          # lees inhoud van het bestand
    inf.close()                # sluit het bestand
    return data



#%%
def num_atoms(smiles):
    """
    Calculate the number of (non-hydrogen) atoms
    in a SMILES string.
    """
    nr_atoms = {}            # dictionary om atoomfrequentie bij te houden
    i = 0                    # index

    while i<len(smiles):     # loop door elke teken in de SMILES string
        letter = smiles[i]
        
        if letter.isupper():                                           # controleer of het een hoofdletter is
            if i+1<len(smiles) and smiles[i+1].islower():              # controleer of er een kleine letter volgt, zoals bijvoorbeeld bij Cl of Na
                next_letter = smiles[i+1]
                if i+2<len(smiles) and smiles[i+2].isdigit():          # controleer of er een getal naar de kleine letter volgt. Dit is een aromatisch atoom 
                    atom = letter                                      # als dit zo is, tel je de hoofdletter onafhankelijk als atoom
                else:                                                  
                    atom = letter + next_letter                        # anders combineer je hoofdletter en kleine letter.
                    i += 1
            else:
                atom = letter                                          # als er geen kleine letter volgt, tel je ook alleen de hoofdletter.

            if atom in nr_atoms:                                       # voeg atoom toe aan de dictionary of verhoog de teller
                nr_atoms[atom] += 1
            else:
                nr_atoms[atom] = 1
        
        elif letter.islower():                                         # als de atoom een kleine letter is die niet bij een hoofdletter hoort.
            atom = letter
            if atom in nr_atoms:
                nr_atoms[atom] += 1
            else: 
                nr_atoms[atom] = 1
        
        i += 1                                                         # ga naar het volgende letter/teken
    return sum(nr_atoms.values())                                      # som van alle atomen teruggeven 

#%%
def num_aromatic(smiles):
    """
    Calculate the number of aromatic rings in a SMILES string.
    Aromatic rings are represented by lowercase followed by a number
    """

    rings = []                # lijst om open ringen bij the houden
    aromatic_rings = 0        # teller voor het aantal aromatische ringen
    i = 0                     # index

    while i<len(smiles):      # loop weer door elke teken in de SMILES string
        letter = smiles[i]

        if letter.islower() and i+1<len(smiles) and smiles[i+1].isdigit():              # controleer op een kleine letter gevolgd door een getal (kenmerk van een aromatische ring) 
            ring_number = smiles[i+1]

            if ring_number in rings:                                                    # als ringnummer al in de lijst zit, is de ring gesloten
                aromatic_rings += 1                                                     # een aromatische ring is compleet 
                rings.remove(ring_number)                                               # verwijder de gesloten ring uit de lijst     
            else: 
                rings.append(ring_number)                                               # open een nieuwe ring

        i += 1                                                                          # naar het volgende teken         
        
    return aromatic_rings       

#%% 
def feature_generation(infile, descriptor_type):                                 # infile is een parameter voor de naam van de invoer bestand. descriptor_type is de type descriptor die moet worden berekend.
    """
    Function to generate discriptors from a SMILES dataset
    and saves results in an Excel.
    """

    data = pd.read_csv(infile)                                                   # lees de data in met pandas
    if 'SMILES' not in data.columns:                                             # controleer of de smiles kolom aanwezig is
        print("Error: 'SMILES' column not found in the dataset")
        return                                                                   # stop de functie als smiles kolom ontbreekt
    
    if descriptor_type == 'num_atoms':                                           # controleert welke descriptor type berekent moet worden        
        data[descriptor_type] = data['SMILES'].apply(num_atoms)                  # voeg een nieuwe kolom toe voor het aantal atomen
    elif descriptor_type == 'num_aromatic_rings':
        data[descriptor_type] = data['SMILES'].apply(num_aromatic)               # voeg een nieuwe kolom toe voor het aantal aromatische ringen 
    else: 
        print(f"Error: Descriptor type '{descriptor_type}' not recognized.")     # geeft een foutmelding als het descriptor_type niet wordt herkend
        return
    
    outfile = f"{descriptor_type}_results.xlsx"                                  # bepaalt de naam van het output bestand aan de hand van de descriptor_type
    data.to_excel(outfile, index=False)                                          # slaat de resultaten op in een excel bestand
    print(f"Results saved to {outfile}")                                         # Een bevestigings berichtje 

#%%
# genereert de moleculaire descriptoren
feature_generation(infile='data_raw.csv', descriptor_type='num_atoms')             
feature_generation(infile='data_raw.csv', descriptor_type='num_aromatic_rings')
