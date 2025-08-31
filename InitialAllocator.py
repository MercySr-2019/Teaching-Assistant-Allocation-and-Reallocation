import pandas as pd
import argparse
import os
from collections import defaultdict
from ortools.sat.python import cp_model

class InitialAllocator:
    # Initialize the allocator and configures
    def __init__(self, dir, weights):
        self.dir = dir
        self.weights = weights
        self.model = cp_model.CpModel()
        self.x = {}

        self.load_data()
        self.data_preprocess()

    # Data loading
    def load_data(self):
        self.df_tas = pd.read_csv(os.path.join(self.dir, 'tas.csv'))
        self.df_sessions = pd.read_csv(os.path.join(self.dir, 'sessions.csv'))
        self.df_approvals = pd.read_csv(os.path.join(self.dir, 'ta_approve_status.csv'))
        self.df_unavailability = pd.read_csv(os.path.join(self.dir, 'ta_unavailability.csv'))

    # data preprocessing
    def data_preprocess(self):
        self.session_info = self.df_sessions.set_index('session_id').to_dict('index')
        self.ta_info = self.df_tas.set_index('ta_id').to_dict('index')
        
        approved_only = self.df_approvals[self.df_approvals['approval_level'] > 0]
        self.approval_info = approved_only.groupby('module_id')['ta_id'].apply(set).to_dict()
        
        preferred_only = self.df_approvals[self.df_approvals['approval_level'] == 2]
        self.preference_info = preferred_only.groupby('module_id')['ta_id'].apply(set).to_dict()

        self.unavailability_info = defaultdict(set)
        for i, row in self.df_unavailability.iterrows():
            self.unavailability_info[row['ta_id']].add((row['week'], row['day'], int(row['start_time']), int(row['end_time'])))

    # Hard constraint implementation
    # demand constraint, each session must meet TA requirements.
    def demand_constraint(self):
        for session_id, session in self.session_info.items():
            self.model.Add(sum(self.x[(ta_id, session_id)] for ta_id in self.ta_info) == session['demand'])
    
    # approval constraint, only approved TAs can be allocated to sessions in this module.
    def approval_constraint(self):
        for (ta_id, session_id), var in self.x.items():
            module_id = self.session_info[session_id]['module_id']
            if ta_id not in self.approval_info.get(module_id, set()):
                self.model.Add(var == 0)

    # time conflict constrain, the same TA cannot be allocated to two sessions simultaneously if they overlap in time.
    def time_conflict_constraint(self):
        for ta_id in self.ta_info:
            sessions_by_day = defaultdict(list)
            for session_id in self.session_info:
                sessions_by_day[(self.session_info[session_id]['week'], self.session_info[session_id]['day'])].append(session_id)
            for day_sessions in sessions_by_day.values():
                for i in range(len(day_sessions)):
                    for j in range(i + 1, len(day_sessions)):
                        s1_id, s2_id = day_sessions[i], day_sessions[j]
                        s1, s2 = self.session_info[s1_id], self.session_info[s2_id]
                        start1, end1 = int(s1['timeslot']), int(s1['timeslot']) + s1['duration'] * 100
                        start2, end2 = int(s2['timeslot']), int(s2['timeslot']) + s2['duration'] * 100
                        if max(start1, start2) < min(end1, end2):
                            self.model.AddBoolOr([self.x[(ta_id, s1_id)].Not(), self.x[(ta_id, s2_id)].Not()])

    # travel constraint
    def travel_constraint(self):
        for ta_id in self.ta_info:
            sessions_by_day = defaultdict(list)
            for session_id in self.session_info:
                sessions_by_day[(self.session_info[session_id]['week'], self.session_info[session_id]['day'])].append(session_id)
            
            for day_sessions in sessions_by_day.values():
                for i in range(len(day_sessions)):
                    for j in range(i + 1, len(day_sessions)):
                        s1_id, s2_id = day_sessions[i], day_sessions[j]
                        s1, s2 = self.session_info[s1_id], self.session_info[s2_id]

                        if s1['campus_id'] != s2['campus_id']:
                            start1, end1 = int(s1['timeslot']), int(s1['timeslot']) + s1['duration'] * 100
                            start2, end2 = int(s2['timeslot']), int(s2['timeslot']) + s2['duration'] * 100
                            
                            first_end = end1 if start1 < start2 else end2
                            second_start = start2 if start1 < start2 else start1
                            
                            if second_start < first_end + 200:
                                self.model.AddBoolOr([self.x[(ta_id, s1_id)].Not(), self.x[(ta_id, s2_id)].Not()])

    # availability constraint
    def availability_constraint(self):
        for ta_id, unavailable_slots in self.unavailability_info.items():
            for week, day, start_time, end_time in unavailable_slots:
                for session_id, session in self.session_info.items():
                    if (session['week'] == week and session['day'] == day and
                        int(session['timeslot']) >= start_time and int(session['timeslot']) < end_time):
                        self.model.Add(self.x[(ta_id, session_id)] == 0)
    
    # Maximum weekly working hours and annual working hours
    def work_hour_constraint(self):
        self.weekly_hours_vars = {}
        all_weeks = self.df_sessions['week'].unique()
        for ta_id, ta in self.ta_info.items():
            annual_hours = sum(self.x[(ta_id, s_id)] * s['duration'] * 2 for s_id, s in self.session_info.items())
            self.model.Add(annual_hours <= ta['max_hours_per_year'])
            for week in all_weeks:
                sessions_in_week = [s_id for s_id, s in self.session_info.items() if s['week'] == week]
                weekly_hours = sum(self.x[(ta_id, s_id)] * self.session_info[s_id]['duration'] * 2 for s_id in sessions_in_week)
                self.weekly_hours_vars[(ta_id, week)] = weekly_hours
                self.model.Add(weekly_hours <= ta['max_hours_per_week'])
    
    # find the cost for continuity, used for maximize the TA and session continuity
    def find_continuity_cost(self):
        consecutive_pairs = []
        continuity_cost = []
        continuity_records = []
        grouped_sessions = self.df_sessions.groupby(['module_id', 'type_id'])
        for i, group in grouped_sessions:
            if len(group) < 2:
                continue
            sorted_group = group.sort_values(by='week')
            session_ids = sorted_group['session_id'].tolist()
            weeks = sorted_group['week'].tolist()
            for i in range(len(weeks) - 1):
                if weeks[i+1] - weeks[i] == 1:
                    consecutive_pairs.append((session_ids[i], session_ids[i+1]))
    
        for s1_id, s2_id in consecutive_pairs:
            for ta_id in self.ta_info:
                var = self.model.NewBoolVar(f'change_{ta_id}i{s1_id}i{s2_id}')
                self.model.Add(self.x[(ta_id, s1_id)] != self.x[(ta_id, s2_id)]).OnlyEnforceIf(var)
                self.model.Add(self.x[(ta_id, s1_id)] == self.x[(ta_id, s2_id)]).OnlyEnforceIf(var.Not())
                continuity_cost.append(var)
                continuity_records.append((ta_id, s1_id, s2_id, var))
        self.continuity_records = continuity_records
        return continuity_cost

    # find the cost for preference, used for maximize the lecturer's preference, by allocation preferred TA as many as possible
    def find_preference_cost(self):
        preference_cost = []
        preference_records = []
        all_preferred_tas = set()
        for module_id, ta_set in self.preference_info.items():
            for ta_id in ta_set:
                all_preferred_tas.add(ta_id)

        for ta_id in all_preferred_tas:
            available_session_vars = []
            for session_id, session in self.session_info.items():
                module_id = session['module_id']
                if ta_id not in self.preference_info.get(module_id, set()):
                    continue
                unavailable = False
                week, day = session['week'], session['day']
                timeslot, duration = int(session['timeslot']), session['duration']
                for slot in self.unavailability_info.get(ta_id, []):
                    slot_week, slot_day, slot_start, slot_end = slot
                    if week == slot_week and day == slot_day:
                        session_start = timeslot
                        session_end = timeslot + duration * 100
                        if max(slot_start, session_start) < min(slot_end, session_end):
                            unavailable = True
                            break
                if not unavailable and (ta_id, session_id) in self.x:
                    available_session_vars.append(self.x[(ta_id, session_id)])
            if available_session_vars:
                assigned_indicator = self.model.NewBoolVar(f"preferred_used_{ta_id}")
                self.model.AddMaxEquality(assigned_indicator, available_session_vars)
                preference_cost.append(1 - assigned_indicator)
                preference_records.append({'ta_id': ta_id, 'assigned_indicator': assigned_indicator})
        self.preference_records = preference_records
        return preference_cost


    def build_model(self):
        # Creating decision variables
        for session_id in self.session_info:
            for ta_id in self.ta_info:
                self.x[(ta_id, session_id)] = self.model.NewBoolVar(f'x_{ta_id}i{session_id}')

        # Apply all hard constraints
        self.demand_constraint()
        self.approval_constraint()
        self.time_conflict_constraint()
        self.travel_constraint()
        self.availability_constraint()
        self.work_hour_constraint()

    # Solve the model using the CP-SAT solver and process the results.
    def solve(self, time_limit):
        solver = cp_model.CpSolver()
        solver.parameters.max_time_in_seconds = time_limit 
        continuity_cost = self.find_continuity_cost()
        preference_cost = self.find_preference_cost()
        total_cost = (self.weights['continuity'] * sum(continuity_cost) + 
                      self.weights['preference'] * sum(preference_cost))
        self.model.Minimize(total_cost)
        status = solver.Solve(self.model)
        
        if status == cp_model.OPTIMAL or status == cp_model.FEASIBLE:
            print(f"Status: {solver.StatusName(status)}")
            allocations = [{'session_id': sid, 'ta_id': tid} for (tid, sid), var in self.x.items() if solver.Value(var) == 1]
            df_alloc = pd.DataFrame(allocations)
            df_alloc.sort_values(by=['session_id', 'ta_id'], inplace=True)
            output_path = os.path.join(self.dir, 'initial_allocation.csv')
            df_alloc.to_csv(output_path, index=False)
            print("Allocation solution successfully found")

        else:
            print("Error: No feasible solution found.")
        
        

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-dir", type=str, required=True)
    parser.add_argument("-w_continuity", type=int, default=10)
    parser.add_argument("-w_preference", type=int, default=5)
    args = parser.parse_args()

    weights = {
        'continuity': args.w_continuity,
        'preference': args.w_preference
    }
    allocator = InitialAllocator(args.dir, weights)
    allocator.build_model()
    allocator.solve(600)
