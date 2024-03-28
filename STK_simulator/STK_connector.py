"""
Initial connection between python and STK
Only need to be executed once
"""

import comtypes
from comtypes.client import CreateObject

# STK instance creation
app = CreateObject("STK11.Application")
app.Visible = True
print('Type of this app: ', type(app))

# the root object of this app: IAgStkObjectRoot
root = app.Personality2
print('Type of this root:', type(root))

# module related to Astrogator: AgStkGatorLib
comtypes.client.GetModule((comtypes.GUID("{090D317C-31A7-4AF7-89CD-25FE18F4017C}"), 1, 0))

print('python initial connection with STK finished.')
print('The python module of STK Object Model API has been created under comtypes\gen.')