import pandas as pd
import random
import os
import argparse

class DataGenerator:
    # Initialize the Generator class and configure the relevant parameters. 
    def __init__(self, config):
        self.config = config # Save configuration parameters
        self.dir = config.get("dir") # Get the output folder path
        
        # Define the number of weeks in a semester and exclude the 11th and 27th weeks, which are reading weeks.
        term1_teaching_weeks = [w for w in range(6, 17) if w != 11]
        term2_teaching_weeks = [w for w in range(22, 33) if w != 27]
        self.all_terms = [term1_teaching_weeks, term2_teaching_weeks]

        # Check and create output directory
        if not os.path.exists(self.dir):
            os.makedirs(self.dir)

        self.dataframes = {}

    # Internal utility function that randomly generates a start time and end time
    def generate_time_slot(self):
        start_hour = random.randint(9, 16)
        end_hour = start_hour + random.randint(1, 4)
        return f"{start_hour:02d}00", f"{end_hour:02d}00"
    
    # Generate TA information
    def generate_tas(self):
        num_of_tas = self.config.get("num_of_tas")
        tas = [{'ta_id': i, 'max_hours_per_week': 20, 'max_hours_per_year': 312} for i in range(1, num_of_tas + 1)]
        self.dataframes['tas'] = pd.DataFrame(tas)

    # Generate session information
    def generate_sessions(self):
        num_of_modules = self.config.get("num_of_modules")
        total_sessions = self.config.get("total_sessions")
        
        # Predetermine the number of sessions that each module should generate.
        module_session_counts = {mod_id: 0 for mod_id in range(1, num_of_modules + 1)}
        remaining_sessions = total_sessions
        
        # Randomly select a module that has not yet reached the session limit of 20.(this number is set by me, not a requirement, 
        # just wanna make sure that single module will no have too much sessions)
        while remaining_sessions > 0:
            modules = [mid for mid, count in module_session_counts.items() if count < 20]
            if not modules:
                break
            
            selected_module = random.choice(modules)
            module_session_counts[selected_module] += 1
            remaining_sessions -= 1

        # Generate the number of sessions for each module according to the predetermined plan ---
        sessions = []
        session_id_counter = 1
        
        for mod_id, num_of_sessions in module_session_counts.items():
            sessions_generated_for_module = 0
            type_id_counter = 1
            used_time_slots = set()

            # Set a semester lock for the module
            module_term_weeks = random.choice(self.all_terms)

            while sessions_generated_for_module < num_of_sessions:
                remaining_sessions = num_of_sessions - sessions_generated_for_module
                if remaining_sessions <= 0:
                    break

                # Set a timeslot for a new session type
                while True:
                    day, timeslot = random.randint(1, 5), self.generate_time_slot()[0]
                    if (day, timeslot) not in used_time_slots:
                        used_time_slots.add((day, timeslot))
                        break

                template = {
                    'day': day, 'timeslot': timeslot, 'duration': random.randint(1, 2),
                    'campus_id': random.randint(1, 2), 'demand': random.randint(1, 2),
                    'room_id': f"Room_{random.randint(1,2)}0{random.randint(1,9)}"
                }

                rates = random.random()
                
                # Decide whether the session is recurrent or one-time.
                if rates < 0.7:
                    duration_weeks = random.randint(3, 8)
                else:
                    duration_weeks = 1
                
                duration_weeks  = min(duration_weeks, remaining_sessions)
                start_week = random.randint(0, len(module_term_weeks) - duration_weeks)
                run_weeks = module_term_weeks[start_week : start_week + duration_weeks]
                
                for week in run_weeks:
                    sessions.append({
                        'session_id': session_id_counter,
                        'module_id': mod_id,
                        'type_id': type_id_counter,
                        'week': week,
                        **template
                    })
                    session_id_counter += 1
                
                sessions_generated_for_module += len(run_weeks)
                type_id_counter += 1
                    
        self.dataframes['sessions'] = pd.DataFrame(sessions)

    # Generate TA approval data information
    def generate_approval_status(self):
        approvals = []
        ta_ids = self.dataframes['tas']['ta_id'].tolist()
        num_of_modules = self.config.get("num_of_modules")
        all_module_ids = list(range(1, num_of_modules + 1))
        guaranteed_allocate = set()

        # Ensure that each module has at least two TA (to better reflect real-world data).
        for module_id in all_module_ids:
            guaranteed_tas = random.sample(ta_ids, k=2)
            for ta_id in guaranteed_tas:
                approvals.append({'ta_id': ta_id, 'module_id': module_id, 'approval_level': random.choice([1, 2])})
                guaranteed_allocate.add((ta_id, module_id))
        
        # Simulate remaining applications
        approval_rate = self.config.get("approval_rate")
        for ta_id in ta_ids:
            # Each TA may apply 3 to 5 module. This number is just set up for testing
            num_applications = random.randint(3, 5)
            applied_modules = random.sample(all_module_ids, num_applications)

            # Check whether the application is already in guaranteed_allocate. If it is, skip it to avoid duplicate applications.
            for module_id in applied_modules:
                if (ta_id, module_id) in guaranteed_allocate:
                    continue
                approval_level = random.choice([1, 2]) if random.random() < approval_rate else 0
                approvals.append({'ta_id': ta_id, 'module_id': module_id, 'approval_level': approval_level})
        self.dataframes['ta_approve_status'] = pd.DataFrame(approvals)

    # Generate TA unavailability data
    def generate_unavailability(self):
        unavailability, ta_ids = [], self.dataframes['tas']['ta_id'].tolist()
        unavailability_rate = self.config.get("unavailability_rate")
        num_tas_with_unavailability = int(len(ta_ids) * unavailability_rate)

        ta_subset = random.sample(ta_ids, num_tas_with_unavailability)
        all_teaching_weeks = self.all_terms[0] + self.all_terms[1]

        for ta_id in ta_subset:
            for i in range(random.randint(1, 3)):
                start_time, end_time = self.generate_time_slot()
                unavailability.append({
                    'ta_id': ta_id,
                    'week': random.choice(all_teaching_weeks),
                    'day': random.randint(1, 5),
                    'start_time': start_time,
                    'end_time': end_time
                })
        self.dataframes['ta_unavailability'] = pd.DataFrame(unavailability)
        
    # Export all as CSV files.
    def save_to_csv(self):
        for name, df in self.dataframes.items():
            file_path = os.path.join(self.dir, f"{name}.csv")
            df.to_csv(file_path, index=False)
        print("Generated Files")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-num_of_tas", type=int, default=200)
    parser.add_argument("-num_of_modules", type=int, default=100)
    parser.add_argument("-total_sessions", type=int, default=100)
    parser.add_argument("-dir", type=str, default="synthetic_data_large")
    parser.add_argument("-approval_rate", type=float, default=0.8)
    parser.add_argument("-unavailability_rate", type=float, default=0.1)

    args = parser.parse_args()
    
    config = {
        "num_of_tas": args.num_of_tas,
        "num_of_modules": args.num_of_modules,
        "total_sessions": args.total_sessions,
        "dir": args.dir,
        "approval_rate": args.approval_rate,
        "unavailability_rate": args.unavailability_rate,
    }

    generator = DataGenerator(config)
    generator.generate_tas()
    generator.generate_sessions()
    generator.generate_approval_status()
    generator.generate_unavailability()
    generator.save_to_csv()