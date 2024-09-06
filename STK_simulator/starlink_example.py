import time
from tqdm import tqdm

startTime = time.time()
from comtypes.gen import STKObjects, STKUtil, AgStkGatorLib
from comtypes.client import CreateObject, GetActiveObject, GetEvents, CoGetObject, ShowEvents

useSTKEngine = False
Read_Scenario = False

if useSTKEngine:
    print("Launching STK Engine...")
    stkApp = CreateObject("STKX11.Application")
    stkApp.NoGraphics = True
    stkRoot = CreateObject('AgStkObjects11.AgStkObjectRoot')
else:
    print("Launching STK...")
    if not Read_Scenario:
        uiApp = CreateObject("STK11.Application")
    else:
        uiApp = GetActiveObject("STK11.Application")
    uiApp.Visible = True
    uiApp.UserControl = True

    stkRoot = uiApp.Personality2

stkRoot.UnitPreferences.SetCurrentUnit("DateFormat", "UTCG")

print("Creating Scenario...")
if not Read_Scenario:
    stkRoot.NewScenario('Starlink')
scenario = stkRoot.CurrentScenario
scenario2 = scenario.QueryInterface(STKObjects.IAgScenario)

totalTime = time.time() - startTime
splitTime = time.time()
print("--- Scenario creation: {a:4.3f} sec\t\tTotal time: {b:4.3f} sec ---".format(a=totalTime, b=totalTime))


def create_satellite(numOrbitPlanes=7, numSatePerPlane=20, height=550, inclination=85, name='Sat'):
    constellation = scenario.Children.New(STKObjects.eConstellation, name)
    constellation2 = constellation.QueryInterface(STKObjects.IAgConstellation)

    for orbitPlaneNum in range(numOrbitPlanes):
        for satNum in range(numSatePerPlane):
            satellite = scenario.Children.New(STKObjects.eSatellite, f"{name}{orbitPlaneNum}_{satNum}")
            satellite2 = satellite.QueryInterface(STKObjects.IAgSatellite)

            satellite2.SetPropagatorType(STKObjects.ePropagatorTwoBody)

            twoBodyPropagator = satellite2.Propagator.QueryInterface(STKObjects.IAgVePropagatorTwoBody)
            keplarian = twoBodyPropagator.InitialState.Representation.ConvertTo(
                STKUtil.eOrbitStateClassical).QueryInterface(STKObjects.IAgOrbitStateClassical)

            keplarian.SizeShapeType = STKObjects.eSizeShapeSemimajorAxis
            keplarian.SizeShape.QueryInterface(
                STKObjects.IAgClassicalSizeShapeSemimajorAxis).SemiMajorAxis = height + 6371  # km
            keplarian.SizeShape.QueryInterface(STKObjects.IAgClassicalSizeShapeSemimajorAxis).Eccentricity = 0

            keplarian.Orientation.Inclination = int(inclination)  # degrees
            keplarian.Orientation.ArgOfPerigee = 0  # degrees
            keplarian.Orientation.AscNodeType = STKObjects.eAscNodeRAAN
            RAAN = 360 / numOrbitPlanes * orbitPlaneNum
            keplarian.Orientation.AscNode.QueryInterface(STKObjects.IAgOrientationAscNodeRAAN).Value = RAAN  # degrees

            keplarian.LocationType = STKObjects.eLocationTrueAnomaly
            trueAnomaly = 360 / numSatePerPlane * satNum
            keplarian.Location.QueryInterface(STKObjects.IAgClassicalLocationTrueAnomaly).Value = trueAnomaly

            satellite2.Propagator.QueryInterface(STKObjects.IAgVePropagatorTwoBody).InitialState.Representation.Assign(
                keplarian)
            satellite2.Propagator.QueryInterface(STKObjects.IAgVePropagatorTwoBody).Propagate()

            constellation2.Objects.AddObject(satellite)


def create_walker_star_satellites(numOrbitPlanes=7, numSatePerPlane=20, height=550, inclination=85):
    satellite = scenario.Children.New(STKObjects.eSatellite, 'Sat')
    satelliteIAF = satellite.QueryInterface(STKObjects.IAgSatellite)
    proIAF = satelliteIAF.Propagator
    proTwoBodyIAF = proIAF.QueryInterface(STKObjects.IAgVePropagatorTwoBody)
    proTwoBodyIAF.InitialState.Representation.AssignClassical(3, 7200, 0, 90, 0, 0, 0)
    stkRoot.ExecuteCommand(
        'Walker */Satellite/Sat Type Star NumPlanes 7 NumSatsPerPlane 20 InterPlanePhaseIncrement 1 ColorByPlane Yes')


def Add_transmitter_receiver(sat_list):
    for each in sat_list:
        Instance_name = each.InstanceName
        #  new transmitter and receiver
        transmitter = each.Children.New(STKObjects.eTransmitter, "Transmitter_" + Instance_name)
        reciver = each.Children.New(STKObjects.eReceiver, "Reciver_" + Instance_name)


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
    return receiver2


def Compute_access(access):
    access.ComputeAccess()
    accessDP = access.DataProviders.Item('Link Information')
    accessDP2 = accessDP.QueryInterface(STKObjects.IAgDataPrvTimeVar)
    Elements = ["Time", 'Link Name', 'EIRP', 'Prop Loss', 'Rcvr Gain', "Xmtr Gain", "Eb/No", "BER"]
    results = accessDP2.ExecElements(scenario2.StartTime, scenario2.StopTime, 3600, Elements)
    Times = results.DataSets.GetDataSetByName('Time').GetValues()  # 时间
    EbNo = results.DataSets.GetDataSetByName('Eb/No').GetValues()  # 码元能量
    BER = results.DataSets.GetDataSetByName('BER').GetValues()  # 误码率
    Link_Name = results.DataSets.GetDataSetByName('Link Name').GetValues()
    Prop_Loss = results.DataSets.GetDataSetByName('Prop Loss').GetValues()
    Xmtr_Gain = results.DataSets.GetDataSetByName('Xmtr Gain').GetValues()
    EIRP = results.DataSets.GetDataSetByName('EIRP').GetValues()
    # Range = results.DataSets.GetDataSetByName('Range').GetValues()
    return Times, Link_Name, BER, EbNo, Prop_Loss, Xmtr_Gain, EIRP


def Creating_All_Access():
    # 首先清空场景中所有的链接
    print('Clearing All Access')
    stkRoot.ExecuteCommand('RemoveAllAccess /')
    # 计算某个卫星与其通信的四颗卫星的链路质量，并生成报告
    for each_sat in tqdm(sat_list):
        now_sat_name = each_sat.InstanceName
        # now_plane_num = int(now_sat_name.split('_')[0][3:])
        now_plane_num = int(now_sat_name[3]) - 1
        # now_sat_num = int(now_sat_name.split('_')[1])
        now_sat_transmitter = each_sat.Children.GetElements(STKObjects.eTransmitter)[0]  # 找到该卫星的发射机
        Set_Transmitter_Parameter(now_sat_transmitter, EIRP=20)
        # 发射机与接收机相连
        for each_plane in Plane_num:
            if each_plane != now_plane_num:
                for each_neighbor in Sat_num:
                    access = now_sat_transmitter.GetAccessToObject(
                        Get_sat_receiver(sat_dic['Sat' + str(each_plane + 1) + str(each_neighbor + 1)]))
                    Compute_access(access)
        # # 与后面的卫星的接收机相连
        # access_backward = now_sat_transmitter.GetAccessToObject(
        #     Get_sat_receiver(sat_dic['Sat' + str(now_plane_num) + '_' + str((now_sat_num + 1) % 22)]))
        # # 与前面的卫星的接收机相连
        # access_forward = now_sat_transmitter.GetAccessToObject(
        #     Get_sat_receiver(sat_dic['Sat' + str(now_plane_num) + '_' + str((now_sat_num - 1) % 22)]))
        # # 与左面的卫星的接收机相连
        # access_left = now_sat_transmitter.GetAccessToObject(
        #     Get_sat_receiver(sat_dic['Sat' + str((now_plane_num - 1) % 72) + '_' + str(now_sat_num)]))
        # # 与右面的卫星的接收机相连
        # access_right = now_sat_transmitter.GetAccessToObject(
        #     Get_sat_receiver(sat_dic['Sat' + str((now_plane_num + 1) % 72) + '_' + str(now_sat_num)]))
        # B_Times, B_Link_Name, B_BER, B_EbNo, B_Prop_Loss, B_Xmtr_Gain, B_EIRP = Compute_access(access_backward)
        # F_Times, F_Link_Name, F_BER, F_EbNo, F_Prop_Loss, F_Xmtr_Gain, F_EIRP = Compute_access(access_forward)
        # L_Times, L_Link_Name, L_BER, L_EbNo, L_Prop_Loss, L_Xmtr_Gain, L_EIRP = Compute_access(access_left)
        # R_Times, R_Link_Name, R_BER, R_EbNo, R_Prop_Loss, R_Xmtr_Gain, R_EIRP = Compute_access(access_right)
        # print('{0}\r', R_Times, R_Link_Name, R_BER, R_EbNo, R_Prop_Loss, R_Xmtr_Gain, R_EIRP)
    # stkRoot.ExecuteCommand('RemoveAllAccess /')


def Change_Sat_color(sat_list):
    # 修改卫星及其轨道的颜色
    print('Changing Color of Satellite')
    for each_sat in tqdm(sat_list):
        now_sat_name = each_sat.InstanceName
        now_plane_num = int(now_sat_name.split('_')[0][3:])
        now_sat_num = int(now_sat_name.split('_')[1])
        satellite = each_sat.QueryInterface(STKObjects.IAgSatellite)
        graphics = satellite.Graphics
        graphics.SetAttributesType(1)  # eAttributesBasic
        attributes = graphics.Attributes
        attributes_color = attributes.QueryInterface(STKObjects.IAgVeGfxAttributesBasic)
        attributes_color.Inherit = False
        # 16436871 浅蓝色
        # 2330219 墨绿色
        # 42495 橙色
        # 9234160 米黄色
        # 65535 黄色
        # 255 红色
        # 16776960 青色
        color_sheet = [16436871, 2330219, 42495, 9234160, 65535, 255, 16776960]
        if now_sat_name[2] == 'A':
            color = 255
        elif now_sat_name[2] == 'B':
            color = 42495
        elif now_sat_name[2] == 'C':
            color = 16436871
        attributes_color.Color = color
        # 找出轨道对应的属性接口
        # orbit = attributes.QueryInterface(STKObjects.IAgVeGfxAttributesOrbit)
        # orbit.IsOrbitVisible = False  # 将轨道设置为不可见


if not Read_Scenario:
    # create_satellite(numOrbitPlanes=7, numSatePerPlane=20, height=550, inclination=85)  # Starlink
    create_walker_star_satellites(numOrbitPlanes=7, numSatePerPlane=20, height=550, inclination=85)
    # Kuiper
    # Creat_satellite(numOrbitPlanes=34, numSatsPerPlane=34, hight=630, Inclination=51.9, name='KPA')  # Phase A
    # Creat_satellite(numOrbitPlanes=32, numSatsPerPlane=32, hight=610, Inclination=42, name='KPB')  # Phase B
    # Creat_satellite(numOrbitPlanes=28, numSatsPerPlane=28, hight=590, Inclination=33, name='KPC')  # Phase C
    sat_list = stkRoot.CurrentScenario.Children.GetElements(STKObjects.eSatellite)
    Add_transmitter_receiver(sat_list)
    # Creating_All_Access()

sat_list = stkRoot.CurrentScenario.Children.GetElements(STKObjects.eSatellite)
sat_dic = {}
print('Creating Satellite Dictionary')
for sat in tqdm(sat_list):
    sat_dic[sat.InstanceName] = sat
Plane_num = []
for i in range(0, 7):
    Plane_num.append(i)
Sat_num = []
for i in range(0, 20):
    Sat_num.append(i)
Creating_All_Access()
access_list = stkRoot.CurrentScenario.Children.GetElements(STKObjects.eAccess)
for access in tqdm(access_list):
    Times, Link_Name, BER, EbNo, Prop_Loss, Xmtr_Gain, EIRP = Compute_access(access=access)
    print(Times, Link_Name, BER, EbNo, Xmtr_Gain, EIRP, Prop_Loss)
