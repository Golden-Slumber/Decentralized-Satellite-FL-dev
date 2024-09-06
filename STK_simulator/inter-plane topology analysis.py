import time
import numpy
from tqdm import tqdm
from comtypes.gen import STKObjects, STKUtil, AgStkGatorLib
from comtypes.client import CreateObject, GetActiveObject, GetEvents, ShowEvents

PLANE_NUM = 7
SAT_NUM = 20
CARRIER_FREQUENCY = 2.4  # GHz
EIRP = 10  # dbW
STEP_SEC = 600  # sec
LINK_INFO_ELEMENTS = ['Prop Loss', 'EIRP', 'Rcvd. Frequency', 'Freq. Doppler Shift', 'Bandwidth Overlap',
                      'Rcvd. Iso. Power', 'Flux Density', 'g/T', 'C/No', 'Bandwidth', 'C/N', 'Spectral Flux Density',
                      'Eb/No', 'BER']
DOPPLER_THRESHOLD = 0.05  # GHz


def set_transmitter_receiver(sat_list):
    for sat in sat_list:
        instance_name = sat.InstanceName
        transmitter = sat.Children.New(STKObjects.eTransmitter, "Transmitter_" + instance_name)
        receiver = sat.Children.New(STKObjects.eReceiver, "Reciver_" + instance_name)

        transmitter_obj = transmitter.QueryInterface(STKObjects.IAgTransmitter)
        transmitter_obj.SetModel('Simple Transmitter Model')
        transmitter_model = transmitter_obj.Model
        transmitter_model_obj = transmitter_model.QueryInterface(STKObjects.IAgTransmitterModelSimple)
        transmitter_model_obj.Frequency = CARRIER_FREQUENCY
        transmitter_model_obj.EIRP = EIRP

        receiver_obj = receiver.QueryInterface(STKObjects.IAgReceiver)
        receiver_obj.SetModel('Simple Receiver Model')
        receiver_model = receiver_obj.Model
        receiver_model_obj = receiver_model.QueryInterface(STKObjects.IAgReceiverModelSimple)
        receiver_model_obj.AutoTrackFrequency = True


def compute_access(access):
    access.ComputeAccess()
    valid_access_start_list = []
    valid_access_stop_list = []
    valid_snr_list = []
    if access.ComputedAccessIntervalTimes.Count != 0:
        access_data = access.DataProviders.Item('Access Data')
        access_data_obj = access_data.QueryInterface(STKObjects.IAgDataPrvInterval)
        access_results = access_data_obj.Exec(scenario_obj.StartTime, scenario_obj.StopTime)
        access_start_time_list = list(access_results.DataSets.GetDataSetByName('Start Time').GetValues())
        access_stop_time_list = list(access_results.DataSets.GetDataSetByName('Stop Time').GetValues())
        duration_list = list(access_results.DataSets.GetDataSetByName('Duration').GetValues())

        for idx in range(len(duration_list)):
            print(str(access_start_time_list[idx]) + "--" + str(access_stop_time_list[idx]) + "--duration: " + str(
                duration_list[idx]))
            link_info = access.DataProviders.Item('Link Information')
            link_info_obj = link_info.QueryInterface(STKObjects.IAgDataPrvTimeVar)
            link_info_results = link_info_obj.ExecElements(access_start_time_list[idx], access_stop_time_list[idx],
                                                           STEP_SEC, LINK_INFO_ELEMENTS)
            doppler_list = list(link_info_results.DataSets.GetDataSetByName('Freq. Doppler Shift').GetValues())
            snr_list = list(link_info_results.DataSets.GetDataSetByName('Eb/No').GetValues())
            if max(doppler_list) < DOPPLER_THRESHOLD:
                valid_access_start_list.append(access_start_time_list[idx])
                valid_access_stop_list.append(access_stop_time_list[idx])
                valid_snr_list.append(sum(snr_list) / len(snr_list))
    return valid_access_start_list, valid_access_stop_list, valid_snr_list


def check_access(plane_idx, neighbor_idx, sat_dic):
    stk_root.ExecuteCommand('RemoveAllAccess /')

    connection_start_time = 86400.0
    connection_stop_time = 0.0
    avg_snr = 0.0
    access_flag = False
    trans_sat = sat_dic['Sat' + str(plane_idx + 1) + '01'].Children.GetElements(STKObjects.eTransmitter)[0]
    trans_obj = trans_sat.QueryInterface(STKObjects.IAgStkObject)
    for i in range(SAT_NUM):
        tail_str = str(i + 1)
        if i + 1 < 10:
            tail_str = '0' + tail_str
        recv_sat = sat_dic['Sat' + str(neighbor_idx + 1) + tail_str].Children.GetElements(STKObjects.eReceiver)[0]
        recv_obj = recv_sat.QueryInterface(STKObjects.IAgStkObject)

        access = recv_obj.GetAccessToObject(trans_obj)
        valid_access_start_list, valid_access_stop_list, valid_snr_list = compute_access(access)
        for j in range(len(valid_access_start_list)):
            if valid_access_start_list[j] <= connection_start_time and valid_access_stop_list[j] > connection_stop_time:
                connection_start_time = valid_access_start_list[j]
                connection_stop_time = valid_access_stop_list[j]
                if avg_snr != 0.0:
                    avg_snr = (avg_snr + valid_snr_list[j]) / 2
                else:
                    avg_snr = valid_snr_list[j]
    if connection_start_time == 0 and connection_stop_time == 86400:
        access_flag = True
    return access_flag, avg_snr


if __name__ == '__main__':
    app = GetActiveObject("STK11.Application")
    app.Visible = True
    app.UserControl = True
    stk_root = app.Personality2

    # of most importance, necessary for obtaining access information
    stk_root.UnitPreferences.SetCurrentUnit("DateFormat", "EpSec")

    scenario = stk_root.CurrentScenario
    scenario_obj = scenario.QueryInterface(STKObjects.IAgScenario)

    print("Launching Scenario Successful")

    sat_list = stk_root.CurrentScenario.Children.GetElements(STKObjects.eSatellite)
    set_transmitter_receiver(sat_list)

    sat_dic = {}
    print("Creating Satellite Dictionary")
    for sat in tqdm(sat_list):
        sat_dic[sat.InstanceName] = sat

    connectivity_matrix = numpy.zeros((PLANE_NUM, PLANE_NUM))
    for i in range(PLANE_NUM):
        for j in range(PLANE_NUM):
            access_flag, avg_snr = check_access(i, j, sat_dic)
            if access_flag:
                connectivity_matrix[i, j] = avg_snr
    print(connectivity_matrix)
