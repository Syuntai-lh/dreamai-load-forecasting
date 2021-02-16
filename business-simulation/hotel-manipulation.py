import pandas as pd

hotel_name = 'City Hotel'

hotel_data_raw = pd.read_csv('data/hotel_bookings.csv')

date = pd.date_range('2015-06-01','2017-12-31',freq='D')
data = pd.DataFrame(index=date, columns=['adults','children','baby'] + list(range(1,13)))

# cancel된 데이터 제외
is_cancel = hotel_data_raw['is_canceled'].values == 0
# hotel 선택
is_resort = hotel_data_raw['hotel'].values == hotel_name

# data filtering
filter_idx = is_cancel * is_resort
hotel_data = hotel_data_raw.iloc[filter_idx,:].copy()

# 머문 날 수
stay_days = (hotel_data['stays_in_weekend_nights'] + hotel_data['stays_in_week_nights']).values

# checkout day
checkout_date = pd.to_datetime(hotel_data['reservation_status_date'])
# checkin day
import datetime
days = [datetime.timedelta(int(s)) for s in stay_days]
checkin_date = checkout_date.copy()
for i in range(checkin_date.shape[0]):
    checkin_date.iloc[i] = checkin_date.iloc[i] - days[i]

#%% data를 채워넣음
data.iloc[:,:] = 0
num_guests_tot = []
for i in checkin_date.index:
    num_guests = hotel_data.loc[i,'adults':'babies'].values
    data.loc[checkin_date.loc[i]:checkout_date.loc[i],'adults':'baby'] += num_guests
    n = int(num_guests.sum())
    if n != 0:
        data.loc[checkin_date.loc[i]:checkout_date.loc[i], n] += 1

#%% nan을 0으로 바꾸고 앞뒤로 자름
import numpy as np
is_zero = (data.sum(axis=1) == 0).values
data = data.iloc[40:823,:]
data['sum'] = data.loc[:,'adults':'baby'].sum(axis=1)
data = data.astype(int)

data.to_csv(f'data/{hotel_name}.csv')

# #%% 결과 비교
# import matplotlib.pyplot as plt
# data_1 = pd.read_csv('data/ResortHotel.csv')
# data_2 = pd.read_csv('data/CityHotel.csv')
#
# plt.plot(data_1['sum'])
# plt.plot(data_2['sum'])
# plt.show()