# UCI Machine Learning Repository

This directory contains data from the UCI machine learning repository. The corresponding BibTeX entry is below.

```
@misc{dua2019uci,
    author = {Dua, Dheeru and Graff, Casey},
    year = {2017},
    title = {UCI Machine Learning Repository},
    url = {https://archive.ics.uci.edu/ml},
    institution = {University of California, Irvine, School of Information and Computer Sciences}
} 
```

## Datasets

### Boston Housing (`boston-housing`)

No information found on UCI. Copied from [yaringal/DropoutUncertaintyExps](https://github.com/yaringal/DropoutUncertaintyExps).

### [Concrete Compressive Strength](https://archive.ics.uci.edu/ml/datasets/Concrete+Compressive+Strength) (`concrete.csv`)

| Data Set Characteristic | Attribute Characteristic | Number of Instances | Number of Inputs | Number of Outputs |
|:-----------------------:|:------------------------:|:-------------------:|:----------------:|:-----------------:|
|       Multivariate      |           Real           |         1030        |         8        |         1         |

1. Cement
2. Blast Furnace Slag
3. Fly Ash
4. Water
5. Superplasticizer
6. Coarse Aggregate
7. Fine Aggregate
8. Age
9. **Concrete Compressive Strength**

```
@article{yeh1998modeling,
    title = {Modeling of strength of high-performance concrete using artificial neural networks},
    author = {Yeh, I-C},
    journal = {Cement and Concrete research},
    volume = {28},
    number = {12},
    pages = {1797--1808},
    year = {1998},
    publisher = {Elsevier}
}
```

### [Energy Efficiency](https://archive.ics.uci.edu/ml/datasets/Energy+efficiency) (`energy`)

| Data Set Characteristic | Attribute Characteristic | Number of Instances | Number of Inputs | Number of Outputs |
|:-----------------------:|:------------------------:|:-------------------:|:----------------:|:-----------------:|
|       Multivariate      |           Real           |         768         |         6        |         2         |

1. Relative Compactness
2. Surface Area
3. Wall Area
4. Roof Area
5. Overall Height
6. Orientation
7. Glazing Area
8. Glazing Area Distribution
9. **Heating Load**
10. **Cooling Load**

```
@article{tsanas2012accurate,
    title = {Accurate quantitative estimation of energy performance of residential buildings using statistical machine learning tools},
    author = {Tsanas, Athanasios and Xifara, Angeliki},
    journal = {Energy and Buildings},
    volume = {49},
    pages = {560--567},
    year = {2012},
    publisher = {Elsevier}
}
```

### Kin8NM (`kin8nm`)

No information found on UCI. Copied from [yaringal/DropoutUncertaintyExps](https://github.com/yaringal/DropoutUncertaintyExps).

### [Condition Based Maintenance of Naval Propulsion Plants](https://archive.ics.uci.edu/ml/datasets/Condition+Based+Maintenance+of+Naval+Propulsion+Plants) (`naval-propulsion-plant.csv`)

| Data Set Characteristic | Attribute Characteristic | Number of Instances | Number of Inputs | Number of Outputs |
|:-----------------------:|:------------------------:|:-------------------:|:----------------:|:-----------------:|
|       Multivariate      |           Real           |        11934        |        16        |         2         |

1. Lever Position (lp)
2. Ship Speed (v)
3. Gas Turbine Shaft Torque (GTT)
4. Gas Turbine Rate of Revolutions (GTn)
5. Gas Generator Rate of Revolutions (GGn)
6. Starboard Propeller Torque (Ts)
7. Port Propeller Torque (Tp)
8. HP Turbine Exit Temperature (T48)
9. GT Compressor Inlet Air Temperature (T1)
10. GT Compressor Outlet Air Temperature (T2)
11. HP Turbine Exit Pressure (P48)
12. GT Compressor Inlet Air Pressure (P1)
13. GT Compressor Outlet Air Pressure (P2)
14. Gas Turbine Exhaust Gas Pressure (Pexh)
15. Turbine Injecton Control (TIC)
16. Fuel Flow (mf)
17. **GT Compressor Decay State Coefficient**
18. **GT Turbine Decay State Coefficient**

```
@article{coraddu2016machine,
    title = {Machine learning approaches for improving condition-based maintenance of naval propulsion plants},
    author = {Coraddu, Andrea and Oneto, Luca and Ghio, Aessandro and Savio, Stefano and Anguita, Davide and Figari, Massimo},
    journal = {Proceedings of the Institution of Mechanical Engineers, Part M: Journal of Engineering for the Maritime Environment},
    volume = {230},
    number = {1},
    pages = {136--153},
    year = {2016},
    publisher = {SAGE Publications Sage UK: London, England}
}
```

### [Combined Cycle Power Plant Data Set](https://archive.ics.uci.edu/ml/datasets/Combined+Cycle+Power+Plant) (`power-plant`)

| Data Set Characteristic | Attribute Characteristic | Number of Instances | Number of Inputs | Number of Outputs |
|:-----------------------:|:------------------------:|:-------------------:|:----------------:|:-----------------:|
|       Multivariate      |           Real           |         9568        |         4        |         1         |

1. Temperature (T)
2. Exhaust Vacuum (V)
3. Ambient Pressure (AP)
4. Relative Humidity (RH)
5. **Net Hourly Electrical Energy Output (PE)**

```
@article{tufekci2014prediction,
    title = {Prediction of full load electrical power output of a base load operated combined cycle power plant using machine learning methods},
    author = {T{\"u}fekci, P{\i}nar},
    journal = {International Journal of Electrical Power \& Energy Systems},
    volume = {60},
    pages = {126--140},
    year = {2014},
    publisher = {Elsevier}
}

@inproceedings{kaya2012local,
    title = {Local and global learning methods for predicting power of a combined gas \& steam turbine},
    author = {Kaya, Heysem and T{\"u}fekci, Pmar and G{\"u}rgen, Fikret S},
    booktitle = {Proceedings of the international conference on emerging trends in computer and electronics engineering icetcee},
    pages = {13--18},
    year = {2012}
}
```

### [Physicochemical Properties of Protein Tertiary Structure](https://archive.ics.uci.edu/ml/datasets/Physicochemical+Properties+of+Protein+Tertiary+Structure) (`protein-tertiary-structure`)

| Data Set Characteristic | Attribute Characteristic | Number of Instances | Number of Inputs | Number of Outputs |
|:-----------------------:|:------------------------:|:-------------------:|:----------------:|:-----------------:|
|       Multivariate      |           Real           |        45730        |         9        |         1         |

1. Total Surface Area
2. Non-Polar Exposed Area
3. Fractional Area of Exposed Non-Polar Residue
4. Fractional Area of Exposed Non-Polar Part of residue
5. Molecular Mass Weighted Exposed Area
6. Average Deviation from Standard Exposed Area of Residue
7. Euclidian Distance
8. Secondary Structure Penalty
9. Spacial Distribution constraints
10. **Size of the Residue**

### [Wine Quality](https://archive.ics.uci.edu/ml/datasets/Wine+Quality) (`wine-quality-red`)

| Data Set Characteristic | Attribute Characteristic |    Number of Instances    | Number of Inputs | Number of Outputs |
|:-----------------------:|:------------------------:|:-------------------------:|:----------------:|:-----------------:|
|       Multivariate      |           Real           | 1599 (red) + 4898 (white) |        11        |         1         |

1. Fixed Acidity
2. Volatile Acidity
3. Citric Acid
4. Residual Sugar
5. Chlorides
6. Free Sulfur Dioxide
7. Total Sulfur Dioxide
8. Density
9. pH
10. Sulphates
11. Alcohol
12. **Quality (between 0 and 10)**

```
@article{cortez2009modeling,
    title = {Modeling wine preferences by data mining from physicochemical properties},
    author = {Cortez, Paulo and Cerdeira, Ant{\'o}nio and Almeida, Fernando and Matos, Telmo and Reis, Jos{\'e}},
    journal = {Decision support systems},
    volume = {47},
    number = {4},
    pages = {547--553},
    year = {2009},
    publisher = {Elsevier}
}
```

### [Yacht Hydrodynamics](https://archive.ics.uci.edu/ml/datasets/Yacht+Hydrodynamics) (`yacht`)

| Data Set Characteristic | Attribute Characteristic | Number of Instances | Number of Inputs | Number of Outputs |
|:-----------------------:|:------------------------:|:-------------------:|:----------------:|:-----------------:|
|       Multivariate      |           Real           |         308         |         6        |         1         |

1. Longitudinal Position of Center of Buoyancy
2. Prismatic Coefficient
3. Length-Displacement Ratio
4. Beam-Draught Ratio
5. Length-Beam Ratio
6. Froude Number
7. **Residuary Resistance per Unit Weight of Displacement**
