import pandas as pd
import pickle


dates = {
     '2': {
         'start_dates': [pd.to_datetime('01/10/1999')],
         'end_dates': [pd.to_datetime('30/09/2019')]
     },
     '5': {
         'start_dates': [pd.to_datetime('01/10/1987')],
         'end_dates': [pd.to_datetime('30/09/2019')]
     },
     '15': {
         'start_dates': [pd.to_datetime('01/10/1987')],
         'end_dates': [pd.to_datetime('30/09/2019')]
     },
     '11': {
         'start_dates': [pd.to_datetime('01/10/1987')],
         'end_dates': [pd.to_datetime('30/09/2019')]
     },
     '6': {
         'start_dates': [pd.to_datetime('01/10/1988')],
         'end_dates': [pd.to_datetime('30/09/2019')]
     },
     '10': {
         'start_dates': [pd.to_datetime('01/10/1988')],
         'end_dates': [pd.to_datetime('30/09/2019')]
     },
     '7': {
         'start_dates': [pd.to_datetime('01/10/1987')],
         'end_dates': [pd.to_datetime('30/09/2019')]
     },
     '8': {
         'start_dates': [pd.to_datetime('01/10/1987')],
         'end_dates': [pd.to_datetime('30/09/2019')]
     },
     '3': {
         'start_dates': [pd.to_datetime('01/10/1987')],
         'end_dates': [pd.to_datetime('30/09/2019')]
     },
     '12': {
         'start_dates': [pd.to_datetime('01/10/1987')],
         'end_dates': [pd.to_datetime('30/09/2019')]
     },
     '14': {
         'start_dates': [pd.to_datetime('01/10/1988')],
         'end_dates': [pd.to_datetime('30/09/2019')]
     },
     '1': {
         'start_dates': [pd.to_datetime('01/10/1987')],
         'end_dates': [pd.to_datetime('30/09/2019')]
     },
     '9': {
         'start_dates': [pd.to_datetime('01/10/1986')],
         'end_dates': [pd.to_datetime('30/09/2019')]
     },
     '13': {
         'start_dates': [pd.to_datetime('01/10/1987')],
         'end_dates': [pd.to_datetime('30/09/2019')]
     },
     '4': {
         'start_dates': [pd.to_datetime('01/10/1987')],
         'end_dates': [pd.to_datetime('30/09/2019')]
     }
}


with open('filename.pickle', 'wb') as handle:
    pickle.dump(a, handle, protocol=pickle.HIGHEST_PROTOCOL)
