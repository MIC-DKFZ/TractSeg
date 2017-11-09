# TractSeg
 
Tool for fast and accurate white matter bundle segmentation.

More documentation will follow soon. Work in Progress.

## Install

Pytorch: Conda install uses all CPU cores. Pip uninstall not.
 
```
git clone https://phabricator.mitk.org/source/dldabg.git
cd dldabg
git checkout 0c01469
pip install .
```

```
pip install https://github.com/MIC-DKFZ/TractSeg/zipball/master
```

## Usage

```
TractSeg.py -i Diffusion.nii.gz
```
This create a 

List of extracted bundles. The number shows the index of the bundle in the output file.
```
0: Background
1: AF_left         (Arcuate fascicle)
2: AF_right
3: ATR_left        (Anterior Thalamic Radiation)
4: ATR_right
5: CA              (Commissure Anterior)
6: CC_1            (Rostrum)
7: CC_2            (Genu)
8: CC_3            (Rostral body (Premotor))
9: CC_4            (Anterior midbody (Primary Motor))
10: CC_5           (Posterior midbody (Primary Somatosensory))
11: CC_6           (Isthmus)
12: CC_7           (Splenium)
13: CG_left        (Cingulum left)
14: CG_right   
15: CST_left       (Corticospinal tract
16: CST_right 
17: EMC_left       (Extreme capsule)
18: EMC_right 
19: MLF_left       (Middle longitudinal fascicle)
20: MLF_right
21: FPT_left       (Fronto-pontine tract)
22: FPT_right 
23: FX_left        (Fornix)
24: FX_right
25: ICP_left       (Inferior cerebellar peduncle)
26: ICP_right 
27: IFO_left       (Inferior occipito-frontal fascicle) 
28: IFO_right
29: ILF_left       (Inferior longitudinal fascicle) 
30: ILF_right 
31: MCP            (Middle cerebellar peduncle)
32: OR_left        (Optic radiation) 
33: OR_right
34: POPT_left      (Parieto‐occipital pontine)
35: POPT_right 
36: SCP_left       (Superior cerebellar peduncle)
37: SCP_right 
38: SLF_I_left     (Superior longitudinal fascicle I)
39: SLF_I_right 
40: SLF_II_left    (Superior longitudinal fascicle II)
41: SLF_II_right
42: SLF_III_left   (Superior longitudinal fascicle III)
43: SLF_III_right 
44: STR_left       (Superior Thalamic Radiation)
45: STR_right 
46: UF_left        (Uncinate fascicle) 
47: UF_right 
48: CC             (Corpus Callosum - all)
49: T_PREF_left    (Thalamo-prefrontal)
50: T_PREF_right 
51: T_PREM_left    (Thalamo-premotor)
52: T_PREM_right 
53: T_PREC_left    (Thalamo-precentral)
54: T_PREC_right 
55: T_POSTC_left   (Thalamo-postcentral)
56: T_POSTC_right 
57: T_PAR_left     (Thalamo-parietal)
58: T_PAR_right 
59: T_OCC_left     (Thalamo-occipital)
60: T_OCC_right 
61: ST_FO_left     (Striato-fronto-orbital)
62: ST_FO_right 
63: ST_PREF_left   (Striato-prefrontal)
64: ST_PREF_right 
65: ST_PREM_left   (Striato-premotor)
66: ST_PREM_right 
67: ST_PREC_left   (Striato-precentral)
68: ST_PREC_right 
69: ST_POSTC_left  (Striato-postcentral)
70: ST_POSTC_right
71: ST_PAR_left    (Striato-parietal)
72: ST_PAR_right 
73: ST_OCC_left    (Striato-occipital)
74: ST_OCC_right
```

## Train your own model