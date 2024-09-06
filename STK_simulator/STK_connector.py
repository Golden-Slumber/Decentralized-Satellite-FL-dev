import comtypes
from comtypes.client import CreateObject

# STK instance creation
app = CreateObject("STK11.Application")
app.Visible = True
app.UserControl = True
print("Launching STK Successful")

# manually load scenario or create constellation for analysis here

# # only need to be executed for the initial connection with python
# root = app.Personality2
# print('Type of this root:', type(root))

# # module related to Astrogator: AgStkGatorLib
# comtypes.client.GetModule((comtypes.GUID("{090D317C-31A7-4AF7-89CD-25FE18F4017C}"), 1, 0))
# print('python initial connection with STK finished.')
# print('The python module of STK Object Model API has been created under comtypes\gen.')


