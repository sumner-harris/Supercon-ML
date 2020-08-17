"""
This function is designed to take in a compound name, in the form of a string like 'Fe2O3', and generate 60 machine
learning features that can be used to predict material properties.

Currently, elements in group 2 and 12 have their valence electrons manually assigned to be 2 s shell electrons
pymatgen gives an error with these elements: "ambigious valence". There are possibly more elements with the same problem

Sumner B. Harris 10/15/19

"""
import pymatgen as mg
import numpy as np
from HW1_CompoundSplit import ChemSplit

def material_features_generator(compound):
    formula = compound
    #Split the compound into a list of elements and compositions
    ele,num = ChemSplit(compound)
    num = np.array(num)  
    comp = mg.Composition(compound)
        
    ####################################################
    # Stoichoimetric Attributes: L**p norm for p = 0,1,2
    # (3 featues)
    #
    L0 = len(mg.Composition(compound).elements)

    x = []
    for element in ele:
        x.append(comp.get_atomic_fraction(element))
    xarray=np.array(x)   
    L1 = sum(xarray)**1
    L2 = sum(xarray**2)**(1/2)
    L3 = sum(xarray**3)**(1/3)
    
    ####################################################
    #Valance Orbital Occupation Attributes: s,p,d,f orbitals
    # (4 featues)
    # The pymatgen Elements.valence object returns (l, n) where
    # is orbital quantum numner s,p,d,f = 0,1,2,3 and n is # of electrons
    valence_with_fraction=[]
    s, p, d, f = [],[],[],[]
    Fs_num, Fp_num, Fd_num, Ff_num = 0,0,0,0
    
    for j in ele:
        if mg.Element(j).group == 2 or mg.Element(j).group == 12: #group 2 elements have an error for some reason so this is a manual assignment
            valence_with_fraction.append(((0,2),x[ele.index(j)]))
            continue
        if j =='Mo':
            valence_with_fraction.append(((2,5),x[ele.index(j)]))
            continue

        if j =='Ru':
            valence_with_fraction.append(((2,7),x[ele.index(j)]))
            continue

        if j =='Nb':
            valence_with_fraction.append(((2,4),x[ele.index(j)]))
            continue

        if j =='Ce':
            valence_with_fraction.append(((3,1),x[ele.index(j)]))
            continue

        if j =='Gd':
            valence_with_fraction.append(((3,7),x[ele.index(j)]))
            continue

        if j =='Pd':
            valence_with_fraction.append(((2,10),x[ele.index(j)]))
            continue

        if j =='Rh':
            valence_with_fraction.append(((2,8),x[ele.index(j)]))
            continue

        if j =='Np':
            valence_with_fraction.append(((3,4),x[ele.index(j)]))
            continue

        if j =='Pt':
            valence_with_fraction.append(((2,9),x[ele.index(j)]))
            continue

        if j =='Pa':
            valence_with_fraction.append(((3,2),x[ele.index(j)]))
            continue

        if j =='Zn':
            valence_with_fraction.append(((2,10),x[ele.index(j)]))
            continue

        if j =='Yb':
            valence_with_fraction.append(((3,14),x[ele.index(j)]))
            continue

        if j =='Cr':
            valence_with_fraction.append(((2,5),x[ele.index(j)]))
            continue

        if j == "U":
            valence_with_fraction.append(((3,5),x[ele.index(j)]))

        else:
            valence_with_fraction.append((mg.Element(j).valence, x[ele.index(j)]))
    # final has the form contains number of valence electrons in each orbital with total at the end:
    #(s,p,d,f,total)
    final=[]
    for k in valence_with_fraction:
        if k[0][0] == 0:
            s.append(k)
            final.append([k[0][1],0,0,0,k[0][1]])
        if k[0][0] == 1:
            p.append(k)
            final.append([2,k[0][1],0,0,2+k[0][1]])           
        if k[0][0] == 2:
            d.append(k)
            final.append([2,0,k[0][1],0,2+k[0][1]])
        if k[0][0] == 3:
            f.append(k)
            final.append([2,0,0,k[0][1],2+k[0][1]])

    weighted_ave = 0
    for i in range(len(final)):
        weighted_ave = weighted_ave + (valence_with_fraction[i][1]*final[i][4])
    for i in range(len(final)):
        Fs_num = Fs_num + valence_with_fraction[i][1]*final[i][0]
        Fp_num = Fp_num + valence_with_fraction[i][1]*final[i][1]
        Fd_num = Fd_num + valence_with_fraction[i][1]*final[i][2]
        Ff_num = Ff_num + valence_with_fraction[i][1]*final[i][3]
    Fs,Fp,Fd,Ff = Fs_num/weighted_ave, Fp_num/weighted_ave, Fd_num/weighted_ave, Ff_num/weighted_ave
    
    ####################################################
    # Ionic compound attributes (3 features)
    # 
    # Elements with multiple common oxi states are: H, N, P, S, Cl, Cr, Mn, Fe, Co, Ge, As, Se, Br, Mo, Tc, Ru, Pd,
    # Sn, Sb, Te, I, Ce, Eu, W, Ir, Pt, Hg, Tl, Pb, Po 
    #
    electroneg = []
    Ilist = []
    Ibarsum=0
    for i in ele:
        electroneg.append(mg.Element(i).X)
    if len(ele) != 1:
        for i in electroneg:
            for j in electroneg:
                Ibarsum = Ibarsum + x[electroneg.index(i)]*x[electroneg.index(j)]*i*j
                Ilist.append(1-np.exp(-(i-j)**2/4))
        I = max(Ilist)
        Ibar = Ibarsum/2
    else:
        I = 0
        Ibar = 0
    #print(I, Ibar, Ilist)
    
    Ibar =comp.average_electroneg #I decided to just use this average electroneg attribute
    
    #The variable isionic provided by Dr. Chen's code
    numelement=len(comp.element_composition)
    elist=[]
    flist=[]
    clist=[]

    tempcount=1
    
    #This statement sets isionic = 0 if the compound contains a single element, addition to Dr. Chen's code
    if numelement == 1:
        isionic = 0
    else:    
        for ii in range(numelement):
            etemp=str(comp.elements[ii])
            etemp=mg.Element(etemp)    
            elist.append(etemp)

            ftemp=comp.get_atomic_fraction(etemp)
            flist.append(ftemp)

            clist.append(ftemp*np.array(etemp.common_oxidation_states))

            tempcount=tempcount*len(clist[ii])

        for ii in range(numelement):
            factor = int(tempcount/len(clist[ii]))
            clist[ii]=np.array(list(clist[ii])*factor)
            #print(clist[ii])

        for ii in range(1,numelement):
            clist[0] = np.vstack((clist[0], clist[ii]))

        isionic = 0
        if(min(abs(np.sum(clist[0],axis=0))) < 10**-8): isionic = 1    
    
    ####################################################
    # Elemental property based attributes 
    # (50 features)
    # electroneg is defined above
    atomic_number,atomic_mass,column,row,atomic_radius,s_valence,p_valence,d_valence,f_valence= [],[],[],[],[],[],[],[],[]
    #create lists of all the values we will need
    for e in ele:
        atomic_number.append(mg.Element(e).Z),atomic_mass.append(mg.Element(e).atomic_mass),column.append(mg.Element(e).group)
        row.append(mg.Element(e).row),atomic_radius.append(mg.Element(e).atomic_radius)
    for i in range(len(final)):
        s_valence.append(final[i][0]),p_valence.append(final[i][1]),d_valence.append(final[i][2]),f_valence.append(final[i][3])
        
    # Get max, min, and range for each   
    max_atomic_number,min_atomic_number,range_atomic_number = max(atomic_number),min(atomic_number),max(atomic_number)-min(atomic_number)
    max_atomic_mass, min_atomic_mass, range_atomic_mass = max(atomic_mass),min(atomic_mass),max(atomic_mass)-min(atomic_mass)
    max_column, min_column, range_column = max(column), min(column), max(column)-min(column)
    max_row, min_row, range_row = max(row), min(row), max(row)-min(row)
    max_atomic_radius, min_atomic_radius, range_atomic_radius = max(atomic_radius), min(atomic_radius), max(atomic_radius)-min(atomic_radius)
    max_electroneg, min_electroneg, range_electroneg = max(electroneg), min(electroneg), max(electroneg)-min(electroneg)
    max_s, min_s, range_s = max(s_valence), min(s_valence), max(s_valence)-min(s_valence)
    max_p, min_p, range_p = max(p_valence), min(p_valence), max(p_valence)-min(p_valence)
    max_d, min_d, range_d = max(d_valence), min(d_valence), max(d_valence)-min(d_valence)
    max_f, min_f, range_f = max(f_valence), min(f_valence), max(f_valence)-min(f_valence)
    
    # Get fraction weighted mean f and deviation dev for each
    f_atomic_number, dev_atomic_number = sum(xarray*np.array(atomic_number)), abs(sum(np.array(atomic_number)-sum(xarray*np.array(atomic_number))))
    f_atomic_mass, dev_atomic_mass = sum(xarray*np.array(atomic_mass)), abs(sum(np.array(atomic_mass)-sum(xarray*np.array(atomic_mass))))
    f_column, dev_column = sum(xarray*np.array(column)), abs(sum(np.array(column)-sum(xarray*np.array(column))))
    f_row, dev_row = sum(xarray*np.array(row)), abs(sum(np.array(row)-sum(xarray*np.array(row))))
    f_atomic_radius, dev_atomic_radius = sum(xarray*np.array(atomic_radius)), abs(sum(np.array(atomic_radius)-sum(xarray*np.array(atomic_radius))))
    f_electroneg, dev_electroneg = sum(xarray*np.array(electroneg)), abs(sum(np.array(electroneg)-sum(xarray*np.array(electroneg))))
    f_s, dev_s = sum(xarray*np.array(s_valence)), abs(sum(np.array(s_valence)-sum(xarray*np.array(s_valence))))
    f_p, dev_p = sum(xarray*np.array(p_valence)), abs(sum(np.array(p_valence)-sum(xarray*np.array(p_valence))))
    f_d, dev_d = sum(xarray*np.array(d_valence)), abs(sum(np.array(d_valence)-sum(xarray*np.array(d_valence))))
    f_f, dev_f = sum(xarray*np.array(f_valence)), abs(sum(np.array(f_valence)-sum(xarray*np.array(f_valence))))
     
    ###########################################################
    # Brute force dictionary creation of 50 features to output
    material_features ={
        'formula': compound,
        'L0':L0,
        'L1':L1,
        'L2':L2,
        'L3':L3,
        'Fs':Fs,
        'Fp':Fp,
        'Fd':Fd,
        'Ff':Ff,
        'Ionic':isionic,
        'Ionic character':I,
        'Mean ionic character': Ibar,
        'Max atomic number': max_atomic_number,
        'Min atomic number': min_atomic_number,
        'Range atomic number': range_atomic_number,
        'Weighted mean atomic number':f_atomic_number,
        'Average deviation atomic number': dev_atomic_number,
        'Max atomic mass (amu)': max_atomic_mass,
        'Min atomic mass (amu)': min_atomic_mass,
        'Range atomic mass (amu)': range_atomic_mass,
        'Weighted mean atomic mass (amu)':f_atomic_mass,
        'Average deviation atomic mass (amu)': dev_atomic_mass,
        'Max atomic radius': max_atomic_radius,
        'Min atomic radius': min_atomic_radius,
        'Range atomic radius': range_atomic_radius,
        'Weighted mean atomic radius':f_atomic_radius,
        'Average deviation atomic radius': dev_atomic_radius,
        'Max column': max_column,
        'Min column': min_column,
        'Range column': range_column,
        'Weighted column':f_column,
        'Average deviation column': dev_column,
        'Max row': max_row,
        'Min row': min_row,
        'Range row': range_row,
        'Weighted row':f_row,
        'Average deviation row': dev_row,
        'Max electronegativity': max_electroneg,
        'Min electronegativity': min_electroneg,
        'Range electronegativity': range_electroneg,
        'Weighted electronegativity':f_electroneg,
        'Average deviation electronegativity': dev_electroneg,
        'Max s valence electrons': max_s,
        'Min s valence electrons': min_s,
        'Range s valence electrons': range_s,
        'Weighted s valence electrons':f_s,
        'Average deviation s valence electrons': dev_s,
        'Max p valence electrons': max_p,
        'Min p valence electrons': min_p,
        'Range p valence electrons': range_p,
        'Weighted p valence electrons':f_p,
        'Average deviation p valence electrons': dev_p,
        'Max d valence electrons': max_d,
        'Min d valence electrons': min_d,
        'Range d valence electrons': range_d,
        'Weighted d valence electrons':f_d,
        'Average deviation d valence electrons': dev_d,
        'Max f valence electrons': max_f,
        'Min f valence electrons': min_f,
        'Range f valence electrons': range_f,
        'Weighted f valence electrons':f_f,
        'Average deviation f valence electrons': dev_f,    
    }
    return material_features