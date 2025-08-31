import pandas as pd
import yaml
import argparse
import os
import time
from collections import defaultdict
from ortools.sat.python import cp_model
import numpy as np

class Reallocator:
    # Initialize the allocator and configures
    def __init__(self, dir, weights):
        self.dir = dir
        self.weights = weights
        self.model = cp_model.CpModel()
        self.x = {}

        self.load_data()
        self.apply_changes()
        self.data_preprocess()

    # data loading
    def load_data(self):
        # Read data from each file
        self.df_tas = pd.read_csv(os.path.join(self.dir, 'tas.csv')) #TA info 
        self.df_sessions = pd.read_csv(os.path.join(self.dir, 'sessions.csv') ) #Session info
        self.df_approvals = pd.read_csv(os.path.join(self.dir, 'ta_approve_status.csv')) # TA approve status
        self.df_unavailability = pd.read_csv(os.path.join(self.dir, 'ta_unavailability.csv')) # TA unavailability info

        self.df_initial_allocation = pd.read_csv(os.path.join(self.dir, 'initial_allocation.csv')) # The initial allocation

        # Read change requests
        with open(os.path.join(self.dir, 'change_request.yaml'), 'r') as f:
            self.change_requests = yaml.safe_load(f)

    # Apply Change Request
    # This function is used to apply change requests to the unavailability info. Currently, it mainly handles TA_UNAVAILABLE
    # requests by updating the system state by appending new unavailable time slots to the df_unavailability.
    def apply_changes(self):
        self.affected_sessions = {}
        if not self.change_requests:
            return
        for change in self.change_requests:
            details = change.get('details', {})
            if change.get('change_type') == 'TA_UNAVAILABLE':
                ta_id = details.get('ta_id')
                sid = details.get('session_id')
                new_unavailability = pd.DataFrame([details])
                new_unavailability.rename(columns={'start': 'start_time', 'end': 'end_time'}, inplace=True)
                self.df_unavailability = pd.concat([self.df_unavailability, new_unavailability], ignore_index=True)
                if sid and ta_id:
                    self.affected_sessions[sid] = ta_id
             

    # Data preprocessing
    # This function is to preprocessing all data required for model. it converts the original DataFrame into dictionaries and collections
    # that are easy to query quickly, and organizes TA' approved status, lecturer's preferences, unavailable times, and initial allocation conflicts
    # caused by change requests.
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
			
        self.initial_allocation = set(tuple(x) for x in self.df_initial_allocation.to_numpy())

    # Hard constraint implementation
    # demand constraint, each session must meet TA requirements.
    def demand_constraint(self):
        for session_id, session in self.session_info.items():
            self.model.Add(sum(self.x[(ta_id, session_id)] for ta_id in self.ta_info) == session['demand'])

    # This function is an approval constraint, which traverses all possible TA and course pairs and prohibits any TA who 
    # does not have the corresponding module approval from being allocated to a session under that module.
    def approval_constraint(self):
        for (ta_id, session_id), var in self.x.items():
            module_id = self.session_info[session_id]['module_id']
            if ta_id not in self.approval_info.get(module_id, set()):
                self.model.Add(var == 0)

    # This function is a time conflict constraint, which ensures that no TA can be allocate to two sessions that overlap in time. This is achieved by
    # finding all time-conflicting course pairs (s1, s2) and adding the logical constraint TA cannot be allocate to s1 or TA cannot be allocate to s2 to
    # each TA.
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

    # This function is a travel constraint. If two sessions are on the same day but located on different campuses, and the gap
    # between them is less than two hours, the same TA cannot be assigned to both sessions to ensure sufficient commuting time between campuses.
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
    # This function is an availability constraint. It reads the individual unavailability times submitted by each TA and prohibits the
    # model from allocating that TA to any sessions that conflict with those times.
    def availability_constraint(self):
        for ta_id, unavailable_slots in self.unavailability_info.items():
            for week, day, start_time, end_time in unavailable_slots:
                for session_id, session in self.session_info.items():
                    if (session['week'] == week and session['day'] == day and
                        int(session['timeslot']) >= start_time and int(session['timeslot']) < end_time):
                        self.model.Add(self.x[(ta_id, session_id)] == 0)


    # This function is a work hour constraint that ensures that each TA's allocation complies with the work hour restrictions specified in their contract.
    # There are two type of limits:
    # 1。 Annual Hour Limit: Calculates the total number of hours for each TA for the entire academic year and ensures that the 
    # total does not exceed their maximum annual hours.
    # 2. Weekly Hour Limit: Each week, calculates the hours for the TA for that week and ensures 
    # that they do not exceed the maximum weekly hours.
    # It is worth noting that the hour calculation here includes a multiplier factor of * 2, because each hour of 
    # teaching corresponds to an additional hour of preparation time.
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


    # find the cost of continuity
    # This function defines the cost of teaching continuity (continuity cost).
    # Encourage the same TA to teach the same course continuously.Specifically, 
    # It identifies all session pairs where teaching occurs in continuous sessions, then introduces a Boolean variable for the case where the TA changes. 
    # If the TA remains the same, the variable is 0; if there is a change, it is 1. The function returns a list of all these change variables, which are
    # then minimized in the main optimization objective.
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
    
    # find the cost of preference
    # This function defines the preference cost, which aims to prioritize the preferred TA
    # This cost only applies to courses that need to be reallocated due to the temporary absence. For a vacant course, if a TA on the preference list is 
    # not selected to fill in, the model generates a cost.
    def find_preference_cost(self):
        preference_cost = []
        preference_records = []

        for session_id, orig_ta_id in self.affected_sessions.items():
            session = self.session_info[session_id]
            module_id = session['module_id']
            preferred_tas = set(self.preference_info.get(module_id, set()))

            eligible_tas = []
            week, day = session['week'], session['day']
            timeslot, duration = int(session['timeslot']), session['duration']

            for ta_id in self.approval_info.get(module_id, set()):
                if ta_id == orig_ta_id:
                    continue

                unavailable = False
                for slot in self.unavailability_info.get(ta_id, []):
                    slot_week, slot_day, slot_start, slot_end = slot
                    if week == slot_week and day == slot_day:
                        session_start = timeslot
                        session_end = timeslot + duration * 100
                        if max(slot_start, session_start) < min(slot_end, session_end):
                            unavailable = True
                            break
                if not unavailable and (ta_id, session_id) in self.x:
                    eligible_tas.append(ta_id)

            preferred_eligible = preferred_tas.intersection(eligible_tas)
            preferred_vars = []
            for ta_id in preferred_eligible:
                conflict = False
                for other_session_id, other_session in self.session_info.items():
                    if other_session_id == session_id:
                        continue
                    week2, day2 = other_session['week'], other_session['day']
                    timeslot2, duration2 = int(other_session['timeslot']), other_session['duration']
                    if week == week2 and day == day2:
                        session_start2 = timeslot2
                        session_end2 = timeslot2 + duration2 * 100
                        session_start = timeslot
                        session_end = timeslot + duration * 100
                        if max(session_start2, session_start) < min(session_end2, session_end):
                            if (ta_id, other_session_id) in self.x:
                                conflict = True
                                break
                if not conflict:
                    preferred_vars.append(self.x[(ta_id, session_id)])
            if preferred_vars:
                assigned_preferred = self.model.NewBoolVar(f"preferred_used_{session_id}")
                self.model.AddMaxEquality(assigned_preferred, preferred_vars)
                preference_cost.append(1 - assigned_preferred)
                preference_records.append({'session_id': session_id, 'assigned_indicator': assigned_preferred})
        self.preference_records = preference_records
        return preference_cost


    
    # find the cost of disruption
    # This function defines the disruption cost, which aims to maintain the original allocation as much as possible while satisfying all hard constraints.
    # By using a Boolean variable to identify and penalize each change to the initial allocation, whether it is the cancellation of an existing arrangement
    # or the addition of a new arrangement that did not exist before
    def find_disruption_cost(self):
        disruption_cost = []
        for (ta_id, session_id), var in self.x.items():
            is_initial = (session_id, ta_id) in self.initial_allocation
            change_var = self.model.NewBoolVar(f'change_{ta_id}i{session_id}')
            self.model.Add(var != is_initial).OnlyEnforceIf(change_var)
            self.model.Add(var == is_initial).OnlyEnforceIf(change_var.Not())
            disruption_cost.append(change_var)
        return disruption_cost

    # find the cost of balance
    # This function is the workload balancing cost, to distribute work fairly and evenly among all TAs.
    # It calculates the absolute difference in the total annual working hours between any two TAs and defines this difference as a cost to be optimized.
    def find_balance_cost(self):
        balance_cost = []
        ta_ids = list(self.ta_info.keys())
        all_weeks = self.df_sessions['week'].unique()
        ta_annual_hours = {
            ta_id: sum(self.weekly_hours_vars.get((ta_id, week), 0) for week in all_weeks)
            for ta_id in ta_ids
        }

        for i in range(len(ta_ids)):
            for j in range(i + 1, len(ta_ids)):
                ta1_id, ta2_id = ta_ids[i], ta_ids[j]
                h1 = ta_annual_hours[ta1_id]
                h2 = ta_annual_hours[ta2_id]
                max_annual_hours = self.ta_info[ta1_id]['max_hours_per_year']
                diff_var = self.model.NewIntVar(0, int(max_annual_hours), f'balance_diff_annual_{ta1_id}i{ta2_id}')
                self.model.AddAbsEquality(diff_var, h1 - h2)
                balance_cost.append(diff_var)
        return balance_cost

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

    # This solution process for this program. It uses a two-stage optimization strategy to find the solution。
    # First stage is to find the minimize disturbance. it solve the problem with the sole objective of minimizing changes to the initial schedule to 
    # find the most stable feasible solution. 
    # Second stage is to optimize secondary objectives. after first stage, it add the minimum disturbance value found in the first stage as a hard constraint into the model. Based on this, optimize a total cost function composed of preference, continuity, and workload balance weighted terms.
    def solve(self, time_limit):
        start_time = time.time()
        solver = cp_model.CpSolver()
        solver.parameters.max_time_in_seconds = float(time_limit)
        disruption_cost = self.find_disruption_cost()
        preference_cost = self.find_preference_cost()
        continuity_cost = self.find_continuity_cost()
        balance_cost = self.find_balance_cost()
        
        self.model.Minimize(sum(disruption_cost))
        status = solver.Solve(self.model)
        if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
            print("Failure: No feasible solution found in the first stage (min disruption).")
            return
        
        min_disruption_value = int(solver.ObjectiveValue())
        print(f"First Stage: Minimum Disruption Found = {min_disruption_value}")
        self.model.Add(sum(disruption_cost) == min_disruption_value)
        total_cost = (self.weights['preference'] * sum(preference_cost) +
                    self.weights['continuity'] * sum(continuity_cost) +
                    self.weights['balance'] * sum(balance_cost))
		
        self.model.Minimize(total_cost)
        status = solver.Solve(self.model)
        end_time = time.time()
        total_time = end_time - start_time

        if status == cp_model.OPTIMAL or status == cp_model.FEASIBLE:
            print(f"Second Stage: Solver Status: {solver.StatusName(status)}")
            self.solution = [{'session_id': sid, 'ta_id': tid} for (tid, sid), var in self.x.items() if solver.Value(var) == 1]
            df_alloc = pd.DataFrame(self.solution)
            df_alloc.sort_values(by=['session_id', 'ta_id'], inplace=True)
            output_path = os.path.join(self.dir, 'new_allocation.csv')
            df_alloc.to_csv(output_path, index=False)
            print("Allocation successfully saved to new_allocation.csv")

            final_continuity_cost = sum(solver.Value(var) for var in continuity_cost)
            final_preference_cost = sum(solver.Value(var) for var in preference_cost)
            final_balance_cost = sum(solver.Value(var) for var in balance_cost)
            
            # print(f"Continuity cost: {final_continuity_cost}")
            print(f"Preference cost: {final_preference_cost}")
            print(f"Balance Cost: {final_balance_cost}")
            print(f"Total solving time: {total_time:.2f} seconds")
            # print(f"Minimum number of personnel changes (disruptions): {min_disruption_value}")

            changed_sessions = []
            for (tid, sid), var in self.x.items():
                is_initial = (sid, tid) in self.initial_allocation
                if solver.Value(var) == 1 and not is_initial:
                    changed_sessions.append({'session_id': sid, 'new_ta_id': tid})
            print(f"Changed assignments (not in initial allocation): {len(changed_sessions)}")

            self.report_costs_comparison(solver)
        else:
            print("Failure: No feasible solution was found in the second stage.")


    def report_costs_comparison(self, solver):
        ta_ids = list(self.ta_info.keys())
        all_weeks = self.df_sessions['week'].unique()

        initial_weekly_hours = { (ta_id, week): 0 for ta_id in ta_ids for week in all_weeks }
        for sid, tid in self.initial_allocation:
            session = self.session_info[sid]
            week = session['week']
            initial_weekly_hours[(tid, week)] += session['duration'] * 2
        initial_annual_hours = { ta_id: sum(initial_weekly_hours[(ta_id, week)] for week in all_weeks) for ta_id in ta_ids }
        initial_balance_cost = []
        for i in range(len(ta_ids)):
            for j in range(i+1, len(ta_ids)):
                ta1_id, ta2_id = ta_ids[i], ta_ids[j]
                diff = abs(initial_annual_hours[ta1_id] - initial_annual_hours[ta2_id])
                initial_balance_cost.append(diff)

        final_weekly_hours = { (ta_id, week): 0 for ta_id in ta_ids for week in all_weeks }
        for (ta_id, session_id), var in self.x.items():
            if solver.Value(var) == 1:
                session = self.session_info[session_id]
                week = session['week']
                final_weekly_hours[(ta_id, week)] += session['duration'] * 2
        final_annual_hours = { ta_id: sum(final_weekly_hours[(ta_id, week)] for week in all_weeks) for ta_id in ta_ids }
        final_balance_cost = []
        for i in range(len(ta_ids)):
            for j in range(i+1, len(ta_ids)):
                ta1_id, ta2_id = ta_ids[i], ta_ids[j]
                diff = abs(final_annual_hours[ta1_id] - final_annual_hours[ta2_id])
                final_balance_cost.append(diff)

        max_initial_hours = max(initial_annual_hours.values())
        min_initial_hours = min(initial_annual_hours.values())
        max_final_hours = max(final_annual_hours.values())
        min_final_hours = min(final_annual_hours.values())
        initial_max_ta = max(initial_annual_hours, key=initial_annual_hours.get)
        initial_min_ta = min(initial_annual_hours, key=initial_annual_hours.get)
        final_max_ta = max(final_annual_hours, key=final_annual_hours.get)
        final_min_ta = min(final_annual_hours, key=final_annual_hours.get)

        initial_preference_cost = 0
        final_preference_cost = 0
        for session_id, orig_ta_id in self.affected_sessions.items():
            session = self.session_info[session_id]
            module_id = session['module_id']
            preferred_tas = set(self.preference_info.get(module_id, set()))

            eligible_tas = []
            week, day = session['week'], session['day']
            timeslot, duration = int(session['timeslot']), session['duration']
            for ta_id in self.approval_info.get(module_id, set()):
                if ta_id == orig_ta_id:
                    continue
                unavailable = False
                for slot in self.unavailability_info.get(ta_id, []):
                    slot_week, slot_day, slot_start, slot_end = slot
                    if week == slot_week and day == slot_day:
                        session_start = timeslot
                        session_end = timeslot + duration * 100
                        if max(slot_start, session_start) < min(slot_end, session_end):
                            unavailable = True
                            break
                if not unavailable:
                    eligible_tas.append(ta_id)
            preferred_eligible = preferred_tas.intersection(eligible_tas)

            assigned_preferred = False
            for ta_id in preferred_eligible:
                if (ta_id, session_id) in self.initial_allocation:
                    assigned_preferred = True
                    break
            if preferred_eligible and not assigned_preferred:
                initial_preference_cost += 1

            assigned_preferred_final = False
            for ta_id in preferred_eligible:
                if (ta_id, session_id) in self.x and solver.Value(self.x[(ta_id, session_id)]) == 1:
                    assigned_preferred_final = True
                    break
            if preferred_eligible and not assigned_preferred_final:
                final_preference_cost += 1

        consecutive_pairs = []
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
        initial_continuity_cost = 0
        final_continuity_cost = 0
        for s1_id, s2_id in consecutive_pairs:
            for ta_id in ta_ids:
                init1 = (ta_id, s1_id) in self.initial_allocation
                init2 = (ta_id, s2_id) in self.initial_allocation
                if init1 != init2:
                    initial_continuity_cost += 1
                final1 = (ta_id, s1_id) in self.x and solver.Value(self.x[(ta_id, s1_id)]) == 1
                final2 = (ta_id, s2_id) in self.x and solver.Value(self.x[(ta_id, s2_id)]) == 1
                if final1 != final2:
                    final_continuity_cost += 1

        print("\n=== Cost Comparison Before & After ===")
        print(f"TA annual working hours absolute maximum difference: {max_initial_hours - min_initial_hours} (TA{initial_max_ta}-{initial_min_ta})"
            f" -> {max_final_hours - min_final_hours} (TA{final_max_ta}-{final_min_ta})")
        print(f"Balance cost sum: {sum(initial_balance_cost)} -> {sum(final_balance_cost)}")
        print(f"Balance cost max: {max(initial_balance_cost) if initial_balance_cost else 0} -> {max(final_balance_cost) if final_balance_cost else 0}")
        print(f"Preference cost: {initial_preference_cost} -> {final_preference_cost}")
        # print(f"Continuity cost: {initial_continuity_cost} -> {final_continuity_cost}")

        print_workload_stats(initial_annual_hours, final_annual_hours)

def print_workload_stats(initial_annual_hours, final_annual_hours):
    init_vals = np.array(list(initial_annual_hours.values()))
    final_vals = np.array(list(final_annual_hours.values()))

    def stat(arr):
        return {
            "ave": np.mean(arr),
            "std": np.std(arr, ddof=0),
            "min": np.min(arr),
            "max": np.max(arr),
            "worst": np.max(arr) - np.min(arr)
        }

    init_stats = stat(init_vals)
    final_stats = stat(final_vals)

    print("\n=== Workload balance indicator ===")
    print("indicator|   initial  |reallocation")
    print("---------|------------|------------")
    for k in ["ave", "std", "min", "max", "worst"]:
        print(f"{k:<8} | {init_stats[k]:<10.2f} | {final_stats[k]:<10.2f}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-dir", type=str, required=True)
    parser.add_argument("-preference", type=int, default=5) # weight of preference, default is 20
    parser.add_argument("-continuity", type=int, default=1) # weight of continuity, default is 1
    parser.add_argument("-balance", type=int, default=20) # weigh of continuity, default is 10
    args = parser.parse_args()

    weights = {
        'preference': args.preference,
        'continuity': args.continuity,
        'balance': args.balance
    }
    reallocator = Reallocator(args.dir, weights)
    reallocator.build_model()
    reallocator.solve(600)
 