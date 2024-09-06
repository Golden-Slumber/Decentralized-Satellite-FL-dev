import pythoncom

context = pythoncom.CreateBindCtx(0)

running_coms = pythoncom.GetRunningObjectTable()

monikers = running_coms.EnumRunning()

for moniker in monikers:
    print('-'*100)
    print(moniker.GetDisplayName(context, moniker))