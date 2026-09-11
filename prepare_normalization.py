import pandas as pd
import pickle


normalization_method = {
     'MEAN_Elev': {
         'centering': None,
         'scaling': None
     },
     'MAX_Elev': {
         'centering': None,
         'scaling': None
     },
     'SD_Elev': {
         'centering': None,
         'scaling': None
     },
     'Percent_Area_Land_0': {
         'centering': None,
         'scaling': None
     },
     'Percent_Area_Land_1': {
         'centering': None,
         'scaling': None
     },
     'Percent_Area_Land_4': {
         'centering': None,
         'scaling': None
     },
     'Percent_Area_Land_5': {
         'centering': None,
         'scaling': None
     },
     'Percent_Area_Land_6': {
         'centering': None,
         'scaling': None
     },
     'Percent_Area_Land_7': {
         'centering': None,
         'scaling': None
     },
     'Percent_Area_Land_8': {
         'centering': None,
         'scaling': None
     },
     'Percent_Area_Land_9': {
         'centering': None,
         'scaling': None
     },
     'Percent_Area_Land_10': {
         'centering': None,
         'scaling': None
     },
     'Percent_Area_Land_11': {
         'centering': None,
         'scaling': None
     },
     'Percent_Area_Land_14': {
         'centering': None,
         'scaling': None
     },
     'Percent_Area_Soil_1': {
         'centering': None,
         'scaling': None
     },
     'Percent_Area_Soil_2': {
         'centering': None,
         'scaling': None
     },
     'Percent_Area_Soil_3': {
         'centering': None,
         'scaling': None
     },
     'Percent_Area_Soil_4': {
         'centering': None,
         'scaling': None
     },
     'Percent_Area_Soil_6': {
         'centering': None,
         'scaling': None
     },
     'Percent_Area_Soil_7': {
         'centering': None,
         'scaling': None
     },
     'Percent_Area_Soil_8': {
         'centering': None,
         'scaling': None
     },
     'Percent_Area_Soil_9': {
         'centering': None,
         'scaling': None
     },
     'Percent_Area_Soil_11': {
         'centering': None,
         'scaling': None
     },
     'Percent_Area_Soil_12': {
         'centering': None,
         'scaling': None
     },
     'Percent_Area_Soil_14': {
         'centering': None,
         'scaling': None
     },
     'Percent_Area_Soil_16': {
         'centering': None,
         'scaling': None
     }
}


with open('normalization.pickle', 'wb') as handle:
    pickle.dump(normalization_method, handle, protocol=pickle.HIGHEST_PROTOCOL)
