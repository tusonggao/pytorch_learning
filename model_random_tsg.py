import os
import sys
import numpy as np
import pandas as pd

gift_card = {0:0, 1:50, 2:50, 3:100, 4:200, 5:200, 6:300, 7:300, 8:400, 9:500, -1:500}
extra_for_each_member = {0:0, 1:0, 2:9, 3:9, 4:9, 5:18, 6:18, 7:36, 8:36, 9:36+199, -1:36+398}

choices = {}
family_people_num = {}
with open('./atad/family_data.csv') as file_r:
    line_cnt = 0
    for line in file_r:
        line_cnt += 1
        if line_cnt==1:
            continue
        family_id = int(line.strip().split(',')[0])
        n_people = int(line.strip().split(',')[-1])
        family_people_num[family_id] = n_people
        choices[family_id] = {}
        for n, day in enumerate(line.strip().split(',')[1:-1]):
            choices[family_id][int(day)] = n

def compute_score(assignment, days_people_num):
    global gift_card, extra_for_each_member
    total_cost = 0
    for family_id, day in assignment.items():
        option_num = choices[family_id].get(day, -1)
        n_people = family_people_num[family_id]
        total_cost += gift_card[option_num] + extra_for_each_member[option_num]*n_people
    for i in range(1, 101, 1):
        N_d = days_people_num[i]
        N_d_plus_1 = days_people_num[i+1] if i < 100 else days_people_num[i]
        total_cost += max(0, (N_d - 125) / 400 * (N_d ** (0.5 + abs(N_d - N_d_plus_1) / 50)))
    return total_cost

def compute_days_people_num(assignment):
    days_people_num = {i:0 for i in range(1, 101, 1)}
    for family_id, day in assignment.items():
        days_people_num[day] += family_people_num[family_id]
    check = all(125 <= days_people_num[i] <= 300 for i in range(1, 101, 1))
    return days_people_num, check

def generate_random_assignment():
    days = np.random.randint(low=1, high=101, size=5000)
    return {i:days[i] for i in range(0, 5000)}

#assign_df = pd.read_csv('./submission_672254.0276683343.csv')
#assignment = {family_id : day for family_id, day in 
#              zip(assign_df['family_id'], assign_df['assigned_day'])}

while True:
    assignment = generate_random_assignment()
    days_people_num, check = compute_days_people_num(assignment)
    print('check is ', check)
    if check is True:
        break

score = compute_score(assignment, days_people_num)
print('scroe is ', score)

outcome_df = pd.DataFrame({'family_id': list(assignment.keys()), 
                           'assigned_day': list(assignment.values())
             })
outcome_df = outcome_df[['family_id', 'assigned_day']]
outcome_df.to_csv('./submission_tsg_{}.csv'.format(score), index=False)

print('prog ends here!')


#data = pd.read_csv('./atad/family_data.csv', index_col='family_id')
#data = pd.read_csv('./atad/family_data.csv')
#data = data.sort_values(by='n_people', ascending=False)
#data.to_csv('./atad/family_data_sorted.csv', index=False)
#print('total number of people is ', data['n_people'].sum())


#family_n_people = {}
#for family_id, n_people in zip(data['family_id'], data['n_people']):
#    family_n_people[family_id] = n_people
#file_w_content = 'family_id,assigned_day\n'

#file_w = open('./atad/submission_first_options_tsg.csv', 'w')
#file_w.write(file_w_content)
#file_w.close()
