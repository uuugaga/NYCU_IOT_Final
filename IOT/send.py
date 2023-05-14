import IOT.DAN as DAN

ServerURL = 'https://2.iottalk.tw'
Reg_addr = 'IOT_Fianl'  #if None, Reg_addr = MAC address

DAN.profile['dm_name']='Dummy_Device'
DAN.profile['df_list']=['Dummy_Sensor', 'Dummy_Control',]
DAN.profile['d_name']= 'IOT_Fianl' 
DAN.device_registration_with_retry(ServerURL, Reg_addr)

def send_data(data):
   DAN.push('Dummy_Sensor', data)
