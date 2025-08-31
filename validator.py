import pandas as pd
import argparse
import os
import time
from collections import defaultdict

class AllocationValidator:
    # Initialize the validator.
    def __init__(self, dir, file):

        self.dir = dir
        self.file = file
        self.errors = []
        
        # Load all required files
        self.df_tas = pd.read_csv(os.path.join(dir, 'tas.csv'))
        self.df_sessions = pd.read_csv(os.path.join(dir, 'sessions.csv'))
        self.df_approvals = pd.read_csv(os.path.join(dir, 'ta_approve_status.csv'))
        self.df_unavailability = pd.read_csv(os.path.join(dir, 'ta_unavailability.csv'))
        self.df_allocation = pd.read_csv(file)

        # Preprocess data
        self.session_info = self.df_sessions.set_index('session_id').to_dict('index')
        self.ta_info = self.df_tas.set_index('ta_id').to_dict('index')
        
        # Only consider records with approval_level > 0 as valid approved.
        df_approved_only = self.df_approvals[self.df_approvals['approval_level'] > 0]
        self.approved_info = df_approved_only.groupby('module_id')['ta_id'].apply(set).to_dict()

        self.unavailability_info = defaultdict(set)
        for i, row in self.df_unavailability.iterrows():
            self.unavailability_info[row['ta_id']].add((row['week'], row['day'], int(row['start_time']), int(row['end_time'])))
        
        self.allocations_by_ta = self.df_allocation.groupby('ta_id')['session_id'].apply(list).to_dict()

    # check if session demand is satisfied
    def check_demand_constraint(self):
        alloc_counts = self.df_allocation.groupby('session_id').size().to_dict()
        for sid, session in self.session_info.items():
            demand = session['demand']
            count = alloc_counts.get(sid, 0)
            if demand != count:
                self.errors.append(f"Session {sid}does not meet the demand ")

    # check if TA is approved
    def check_approval_constraint(self):
        for i, row in self.df_allocation.iterrows():
            sid, tid = row['session_id'], row['ta_id']
            if sid not in self.session_info:
                continue
            module_id = self.session_info[sid]['module_id']
            if tid not in self.approved_info.get(module_id, set()):
                self.errors.append(f"TA {tid} is ineligible for module {module_id}")

    # check time conflict.
    def check_time_conflict(self):
        for tid, session_list in self.allocations_by_ta.items():
            if len(session_list) < 2: continue
            
            for i in range(len(session_list)):
                for j in range(i + 1, len(session_list)):
                    s1_id, s2_id = session_list[i], session_list[j]
                    if s1_id not in self.session_info or s2_id not in self.session_info: continue
                    s1, s2 = self.session_info[s1_id], self.session_info[s2_id]
                    
                    if not (s1['week'] == s2['week'] and s1['day'] == s2['day']):
                        continue
                    
                    start1, end1 = int(s1['timeslot']), int(s1['timeslot']) + s1['duration'] * 100
                    start2, end2 = int(s2['timeslot']), int(s2['timeslot']) + s2['duration'] * 100
                    
                    # Check if the check time periods overlap.
                    if max(start1, start2) < min(end1, end2):
                        self.errors.append(f" TA {tid} Time conflict between Session {s1_id} and {s2_id}")

    # check the travel constraint
    def check_travel_constraint(self):
        for tid, session_list in self.allocations_by_ta.items():
            if len(session_list) < 2: continue

            for i in range(len(session_list)):
                for j in range(i + 1, len(session_list)):
                    s1_id, s2_id = session_list[i], session_list[j]
                    if s1_id not in self.session_info or s2_id not in self.session_info: continue
                    s1, s2 = self.session_info[s1_id], self.session_info[s2_id]

                    if not (s1['week'] == s2['week'] and s1['day'] == s2['day']):
                        continue
                    
                    # Check only when campuses are different.
                    if s1['campus_id'] != s2['campus_id']:
                        start1, end1 = int(s1['timeslot']), int(s1['timeslot']) + s1['duration'] * 100
                        start2, end2 = int(s2['timeslot']), int(s2['timeslot']) + s2['duration'] * 100
                        
                        # Determine the sequence of sessions
                        first_end = end1 if start1 < start2 else end2
                        second_start = start2 if start1 < start2 else start1
                        
                        # Check whether the gap between the two sessions meets the requirements 
                        if second_start < first_end + 200:
                            self.errors.append(f"TA do not have enough time to travel between session {s1_id} and {s2_id}")

    # Check TA availability constraint
    def check_availability(self):
        for i, row in self.df_allocation.iterrows():
            sid, tid = row['session_id'], row['ta_id']
            if sid not in self.session_info: continue
            session = self.session_info[sid]
            # 检查该助教的每一个不可用时间段
            for week, day, start_time, end_time in self.unavailability_info.get(tid, []):
                if (session['week'] == week and session['day'] == day and
                    int(session['timeslot']) >= start_time and int(session['timeslot']) < end_time):
                    self.errors.append(f"TA {id} is no available for session {sid}")

    # check the working hour limits
    def check_work_hour_limits(self):
        weekly_hours = defaultdict(lambda: defaultdict(float))
        annual_hours = defaultdict(float)
        for i, row in self.df_allocation.iterrows():
            sid, tid = row['session_id'], row['ta_id']
            if sid not in self.session_info: continue
            if tid not in self.ta_info:
                self.errors.append(f"Data error, TA {tid}")
                continue
            
            session = self.session_info[sid]
            hours_to_add = session['duration'] * 2
            
            # Check weekly working hours
            weekly_hours[tid][session['week']] += hours_to_add
            if weekly_hours[tid][session['week']] > self.ta_info[tid]['max_hours_per_week']:
                self.errors.append(f"TA {tid} has reach their weekly working limit")

            # Check annual working hours
            annual_hours[tid] += hours_to_add
            if annual_hours[tid] > self.ta_info[tid]['max_hours_per_year']:
                self.errors.append(f"TA {tid} has reach their annual working limit")

    # check there some session in reading week.
    def check_reading_week(self):
        reading_weeks = {11, 27}
        for i, row in self.df_allocation.iterrows():
            sid = row['session_id']
            if sid not in self.session_info: continue
            if self.session_info[sid]['week'] in reading_weeks:
                self.errors.append(f"Reading week conflict")


    # Perform all hard constraint checks in order and output the final report.
    def run_all_checks(self):

        self.check_demand_constraint()
        self.check_approval_constraint()
        self.check_time_conflict()
        self.check_travel_constraint()
        self.check_availability()
        self.check_work_hour_limits()
        self.check_reading_week()
        
        if not self.errors:
            print("This allocation plan is valid。")
        else:
            print("Error:")
            for i, error in enumerate(self.errors, 1):
                print(f"  {i}. {error}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-dir", type=str, required=True)
    parser.add_argument("-file", type=str, required=True)
    
    args = parser.parse_args()

    validator = AllocationValidator(args.dir, args.file)
    validator.run_all_checks()