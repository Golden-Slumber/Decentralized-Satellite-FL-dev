import time
from tqdm import tqdm

start_time = time.time()
from comtypes.gen import STKObjects, STKUtil, AgStkGatorLib
from comtypes.client import CreateObject, GetActiveObject, GetEvents, CoGetObject, ShowEvents

# stkApp = CreateObject("STK11.Application")
uiApp = GetActiveObject("STK11.Application")
# uiApp = CoGetObject("STK11.Application")
uiApp.Visible = True
uiApp.UserControl = True
stkRoot = uiApp.Personality2

stkRoot.UnitPreferences.SetCurrentUnit("DateFormat", "UTCG")

scenarioObj = stkRoot.CurrentScenario
scenarioIAF = scenarioObj.QueryInterface(STKObjects.IAgScenario)

print("Launching Scenario Successful")


def Add_transmitter_receiver(sat_list):
    for each in sat_list:
        Instance_name = each.InstanceName
        #  new transmitter and receiver
        each.Children.New(STKObjects.eTransmitter, "Transmitter_" + Instance_name)
        each.Children.New(STKObjects.eReceiver, "Reciver_" + Instance_name)


def Set_Transmitter_Parameter(transmitter, frequency=12, EIRP=20, DataRate=14):
    transmitter2 = transmitter.QueryInterface(STKObjects.IAgTransmitter)  # 建立发射机的映射，以便对其进行设置
    transmitter2.SetModel('Simple Transmitter Model')
    txModel = transmitter2.Model
    txModel = txModel.QueryInterface(STKObjects.IAgTransmitterModelSimple)
    txModel.Frequency = frequency  # GHz range:10.7-12.7GHz
    txModel.EIRP = EIRP  # dBW
    txModel.DataRate = DataRate  # Mb/sec


def Set_Receiver_Parameter(receiver, GT=20, frequency=12):
    receiver2 = receiver.QueryInterface(STKObjects.IAgReceiver)  # 建立发射机的映射，以便对其进行设置
    receiver2.SetModel('Simple Receiver Model')
    recModel = receiver2.Model
    recModel = recModel.QueryInterface(STKObjects.IAgReceiverModelSimple)
    recModel.AutoTrackFrequency = False
    recModel.Frequency = frequency  # GHz range:10.7-12.7GHz
    recModel.GOverT = GT  # dB/K
    return receiver2


def Get_sat_receiver(sat, GT=20, frequency=12):
    receiver = sat.Children.GetElements(STKObjects.eReceiver)[0]  # 找到该卫星的接收机
    receiver2 = Set_Receiver_Parameter(receiver=receiver, GT=GT, frequency=frequency)
    # return receiver2
    return receiver


def Compute_access(access):
    access.ComputeAccess()
    # stkRoot.UnitPreferences.SetCurrentUnit("Time", "Min")
    AccessData = access.DataProviders.Item('Access Data')
    AccessData_ProvG = AccessData.QueryInterface(STKObjects.IAgDataPrvInterval)
    # print(scenarioIAF.StartTime)
    # print(scenarioIAF.StopTime)
    # StartTime = "29 Aug 2024 06:00:00.000"
    # StopTime = "30 Aug 2024 04:00:00.000"
    AccessData_results = AccessData_ProvG.Exec(scenarioIAF.StartTime, scenarioIAF.StopTime)
    # AccessData_results = AccessData_ProvG.Exec(StartTime, StopTime)
    accessStartTime = AccessData_results.DataSets.GetDataSetByName('Start Time').GetValues()
    accessStopTime = AccessData_results.DataSets.GetDataSetByName('Stop Time').GetValues()
    duration = AccessData_results.DataSets.GetDataSetByName('Duration').GetValues()
    # print(scenarioIAF.StartTime, scenarioIAF.StopTime)
    print(accessStartTime, accessStopTime, duration)

    # accessDataProvider = access.DataProviders.GetDataPrvIntervalFromPath("Access Data")
    # elements = ["Start Time", "Stop Time", "Duration"]
    # accessResults = accessDataProvider.ExecElements(
    #     scenarioIAF.StartTime, scenarioIAF.StopTime, elements
    # )
    #
    # startTimes = accessResults.DataSets.GetDataSetByName("Start Time").GetValues()
    # stopTimes = accessResults.DataSets.GetDataSetByName("Stop Time").GetValues()
    # durations = accessResults.DataSets.GetDataSetByName("Duration").GetValues()
    #
    # print(startTimes, stopTimes, durations)

    # accessDP = access.DataProviders.Item('Link Information')
    # accessDP2 = accessDP.QueryInterface(STKObjects.IAgDataPrvTimeVar)
    # Elements = ["Time", 'Link Name', 'EIRP', 'Prop Loss', 'Rcvr Gain', "Xmtr Gain", "Eb/No", "BER"]
    # results = accessDP2.ExecElements(scenarioIAF.StartTime, scenarioIAF.StopTime, 3600, Elements)
    # Times = results.DataSets.GetDataSetByName('Time').GetValues()  # 时间
    # EbNo = results.DataSets.GetDataSetByName('Eb/No').GetValues()  # 码元能量
    # BER = results.DataSets.GetDataSetByName('BER').GetValues()  # 误码率
    # Link_Name = results.DataSets.GetDataSetByName('Link Name').GetValues()
    # Prop_Loss = results.DataSets.GetDataSetByName('Prop Loss').GetValues()
    # Xmtr_Gain = results.DataSets.GetDataSetByName('Xmtr Gain').GetValues()
    # EIRP = results.DataSets.GetDataSetByName('EIRP').GetValues()
    # # Range = results.DataSets.GetDataSetByName('Range').GetValues()
    # return Times, Link_Name, BER, EbNo, Prop_Loss, Xmtr_Gain, EIRP


def Creating_All_Access():
    # 首先清空场景中所有的链接
    print('Clearing All Access')
    stkRoot.ExecuteCommand('RemoveAllAccess /')
    # 计算某个卫星与其通信的四颗卫星的链路质量，并生成报告
    for each_sat in tqdm(sat_list):
        now_sat_name = each_sat.InstanceName
        # now_plane_num = int(now_sat_name.split('_')[0][3:])
        # if now_sat_name != 'Sat':
        if now_sat_name == 'Sat101':
            now_plane_num = int(now_sat_name[3]) - 1
            # now_sat_num = int(now_sat_name.split('_')[1])
            now_sat_transmitter = each_sat.Children.GetElements(STKObjects.eTransmitter)[0]  # 找到该卫星的发射机
            # Set_Transmitter_Parameter(now_sat_transmitter, EIRP=20)
            # 发射机与接收机相连
            for each_plane in Plane_num:
                if each_plane != now_plane_num:
                    for each_neighbor in Sat_num:
                        tail_str = str(each_neighbor + 1)
                        if int(tail_str) < 10:
                            tail_str = '0' + tail_str
                        rec_obj = \
                            sat_dic['Sat' + str(each_plane + 1) + tail_str].Children.GetElements(STKObjects.eReceiver)[
                                0]
                        # rec_obj = Get_sat_receiver(sat_dic['Sat' + str(each_plane + 1) + tail_str])
                        access = rec_obj.GetAccessToObject(now_sat_transmitter)
                        print('Sat' + str(each_plane + 1) + tail_str)
                        Compute_access(access)


sat_list = stkRoot.CurrentScenario.Children.GetElements(STKObjects.eSatellite)
# Add_transmitter_receiver(sat_list)
sat_dic = {}
print('Creating Satellite Dictionary')
for sat in tqdm(sat_list):
    # print(sat.InstanceName)
    sat_dic[sat.InstanceName] = sat
Plane_num = []
for i in range(0, 7):
    Plane_num.append(i)
Sat_num = []
for i in range(0, 20):
    Sat_num.append(i)
# Creating_All_Access()

stkRoot.ExecuteCommand('RemoveAllAccess /')

stkRoot.UnitPreferences.SetCurrentUnit("DateFormat", "EpSec")

# transmitter = sat_dic['Sat101'].Children.GetElements(STKObjects.eTransmitter)[0]
# trans_obj = transmitter.QueryInterface(STKObjects.IAgStkObject)
# print(transmitter.InstanceName)
# receiver = sat_dic['Sat201'].Children.GetElements(STKObjects.eReceiver)[0]
# recv_obj = receiver.QueryInterface(STKObjects.IAgStkObject)
# print(receiver.InstanceName)
# # access = receiver.GetAccessToObject(transmitter)
# access = recv_obj.GetAccessToObject(trans_obj)
# access.ComputeAccess()
#
# results = access.ComputedAccessIntervalTimes
# results = results.ToArray(0, results.Count)
#
# for accessTime in results:
#     print(accessTime)
#
#
# LinkInfo = access.DataProviders.Item('Link Information')
# LinkInfo_TimeVar = LinkInfo.QueryInterface(STKObjects.IAgDataPrvTimeVar)
LinkInfo_elements = ['Prop Loss', 'EIRP', 'Rcvd. Frequency', 'Freq. Doppler Shift']
StepTm = 3600

# LinkInfo_results = LinkInfo_TimeVar.ExecElements(scenarioIAF.StartTime, scenarioIAF.StopTime, StepTm, LinkInfo_elements)
# PropLoss = list(LinkInfo_results.DataSets.GetDataSetByName('Prop Loss').GetValues())
# EIRP = list(LinkInfo_results.DataSets.GetDataSetByName('EIRP').GetValues())
# RcvdFrequency = list(LinkInfo_results.DataSets.GetDataSetByName('Rcvd. Frequency').GetValues())
# FreqDopplerShift = list(LinkInfo_results.DataSets.GetDataSetByName('Freq. Doppler Shift').GetValues())
# print(PropLoss)
# print(FreqDopplerShift)
#

transmitter = sat_dic['Sat101'].Children.GetElements(STKObjects.eTransmitter)[0]
trans_obj = transmitter.QueryInterface(STKObjects.IAgStkObject)
print(transmitter.InstanceName)
receiver = sat_dic['Sat205'].Children.GetElements(STKObjects.eReceiver)[0]
recv_obj = receiver.QueryInterface(STKObjects.IAgStkObject)
print(receiver.InstanceName)
# access = receiver.GetAccessToObject(transmitter)
access = recv_obj.GetAccessToObject(trans_obj)
access.ComputeAccess()

results = access.ComputedAccessIntervalTimes
print(results.Count)
results = results.ToArray(0, results.Count)
print(results)

AccessData = access.DataProviders.Item('Access Data')
AccessData_ProvG = AccessData.QueryInterface(STKObjects.IAgDataPrvInterval)
AccessData_results = AccessData_ProvG.Exec(scenarioIAF.StartTime, scenarioIAF.StopTime)
# AccessData_results = AccessData_ProvG.Exec(StartTime, StopTime)
accessStartTime = list(AccessData_results.DataSets.GetDataSetByName('Start Time').GetValues())
accessStopTime = list(AccessData_results.DataSets.GetDataSetByName('Stop Time').GetValues())
duration = list(AccessData_results.DataSets.GetDataSetByName('Duration').GetValues())
# print(scenarioIAF.StartTime, scenarioIAF.StopTime)
# print(accessStartTime, accessStopTime, duration)
for idx in range(len(duration)):
    print(str(accessStartTime[idx])+"--"+str(accessStopTime[idx])+"--duration: "+str(duration[idx]))
    LinkInfo = access.DataProviders.Item('Link Information')
    LinkInfo_TimeVar = LinkInfo.QueryInterface(STKObjects.IAgDataPrvTimeVar)
    LinkInfo_results = LinkInfo_TimeVar.ExecElements(accessStartTime[idx], accessStopTime[idx], StepTm,
                                                     LinkInfo_elements)
    PropLoss = list(LinkInfo_results.DataSets.GetDataSetByName('Prop Loss').GetValues())
    EIRP = list(LinkInfo_results.DataSets.GetDataSetByName('EIRP').GetValues())
    RcvdFrequency = list(LinkInfo_results.DataSets.GetDataSetByName('Rcvd. Frequency').GetValues())
    FreqDopplerShift = list(LinkInfo_results.DataSets.GetDataSetByName('Freq. Doppler Shift').GetValues())
    print(PropLoss)
    print(EIRP)
    print(FreqDopplerShift)

# access_list = stkRoot.CurrentScenario.Children.GetElements(STKObjects.eAccess)
# for access in tqdm(access_list):
#     Times, Link_Name, BER, EbNo, Prop_Loss, Xmtr_Gain, EIRP = Compute_access(access=access)
#     print(Times, Link_Name, BER, EbNo, Xmtr_Gain, EIRP, Prop_Loss)
