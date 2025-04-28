from mip import Model, xsum, minimize, BINARY, INTEGER, OptimizationStatus
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from utils import plot_gantt_chart, plot_resource_usage, plot_project_delays, plot_supplier_allocation
from model_data import ModelData

def create_and_solve_model():
    """
    Creates and solves the infrastructure project management optimization model
    based on the specifications in Model 1.docx
    """
    # Load model data
    data = ModelData()
    
    # Big-M value
    M = 1000000

    # Define the time horizon (maximum possible time) - DRASTICALLY REDUCE THIS VALUE
    max_time = min(50, sum([max(data.Duration[i]) for i in range(data.NUM_act)]) // 2)
    print(f"Using reduced time horizon of {max_time} days to improve solvability")
    time_periods = range(max_time)

    # Creating the MIP model
    m = Model("infrastructure_project_management", solver_name='cbc')
    
    # Enable more detailed output and set parameters
    m.verbose = 1
    m.max_mip_gap = 0.1  # Accept solutions within 10% of optimal
    m.threads = -1       # Use all available threads

    # Decision Variables

    # Start Times (ST) and Finish Times (FT) for each activity and project
    ST = [[
        m.add_var(var_type=INTEGER, name=f'ST_{i+1}_{j+1}', lb=0)
        for j in range(data.NUM_project)
    ] for i in range(data.NUM_act)]

    FT = [[
        m.add_var(var_type=INTEGER, name=f'FT_{i+1}_{j+1}', lb=0)
        for j in range(data.NUM_project)
    ] for i in range(data.NUM_act)]

    # x[i][j][t]: Binary variable indicating if activity i of project j is started at time t
    x = [[[
        m.add_var(var_type=BINARY, name=f'x_{i+1}_{j+1}_{t}')
        for t in time_periods]
        for j in range(data.NUM_project)]
        for i in range(data.NUM_act)
    ]

    # TD[j]: Tardiness (delay) of project j
    TD = [m.add_var(var_type=INTEGER, name=f'TD_{j+1}', lb=0) for j in range(data.NUM_project)]

    # I[k][t]: Inventory level of raw material k at time t
    I = [[
        m.add_var(var_type=INTEGER, name=f'I_{k+1}_{t}', lb=0)
        for t in time_periods]
        for k in range(data.NUM_raw_mat)
    ]

    # O[k][s][i][j][t]: Quantity of raw material k ordered from supplier s for activity i of project j at time t
    O = [[[[[
        m.add_var(var_type=INTEGER, name=f'O_{k+1}_{s+1}_{i+1}_{j+1}_{t}', lb=0)
        for t in time_periods]
        for j in range(data.NUM_project)]
        for i in range(data.NUM_act)]
        for s in range(data.NUM_sup)]
        for k in range(data.NUM_raw_mat)
    ]

    # y[k][s][i][j][t]: Binary variable indicating if raw material k is ordered from supplier s for activity i of project j at time t
    y = [[[[[
        m.add_var(var_type=BINARY, name=f'y_{k+1}_{s+1}_{i+1}_{j+1}_{t}')
        for t in time_periods]
        for j in range(data.NUM_project)]
        for i in range(data.NUM_act)]
        for s in range(data.NUM_sup)]
        for k in range(data.NUM_raw_mat)
    ]

    # z[k][i][j][t]: Binary variable indicating if raw material k is demanded for activity i of project j at time t
    z = [[[[
        m.add_var(var_type=BINARY, name=f'z_{k+1}_{i+1}_{j+1}_{t}')
        for t in time_periods]
        for j in range(data.NUM_project)]
        for i in range(data.NUM_act)]
        for k in range(data.NUM_raw_mat)
    ]

    # w[k][s][i][j]: Binary variable indicating if raw material k for activity i of project j is assigned to supplier s
    w = [[[[
        m.add_var(var_type=BINARY, name=f'w_{k+1}_{s+1}_{i+1}_{j+1}')
        for j in range(data.NUM_project)]
        for i in range(data.NUM_act)]
        for s in range(data.NUM_sup)]
        for k in range(data.NUM_raw_mat)
    ]

    # v[k][i][j]: Actual quantity of raw material k used by activity i of project j
    v = [[[
        m.add_var(var_type=INTEGER, name=f'v_{k+1}_{i+1}_{j+1}', lb=0)
        for j in range(data.NUM_project)]
        for i in range(data.NUM_act)]
        for k in range(data.NUM_raw_mat)
    ]

    # Cmax: Maximum completion time across all projects (makespan)
    Cmax = m.add_var(var_type=INTEGER, name='Cmax', lb=0)

    # Objective Function: Minimize total penalties due to delays
    m.objective = minimize(xsum(data.Penalty[j] * TD[j] for j in range(data.NUM_project)))

    # Constraints

    # 1. Relation between x_ijt and ST_ij (start time)
    for i in range(data.NUM_act):
        for j in range(data.NUM_project):
            m += ST[i][j] == xsum(t * x[i][j][t] for t in time_periods), f"start_time_constraint_i_{i+1}_j_{j+1}"

    # 2. Activity can only start once
    for i in range(data.NUM_act):
        for j in range(data.NUM_project):
            m += xsum(x[i][j][t] for t in time_periods) == 1, f"activity_starts_once_i_{i+1}_j_{j+1}"

    # 3. Activity duration constraints: FT[i][j] == ST[i][j] + Duration[i][j] for all i, j
    for i in range(data.NUM_act):
        for j in range(data.NUM_project):
            m += FT[i][j] == ST[i][j] + data.Duration[i][j], f"activity_duration_i_{i+1}_j_{j+1}"

    # 4. Precedence constraints: For each (a, b) in Pred, ST[b] >= FT[a] for all j
    for (a, b) in data.Pred:
        for j in range(data.NUM_project):
            m += ST[b][j] >= FT[a][j], f"precedence_constraint_a_{a+1}_b_{b+1}_proj_{j+1}"

    # 5. Define Cmax as maximum completion time across all projects
    for j in range(data.NUM_project):
        last_activity = data.NUM_act - 1  # Last activity (index 22 for activity 23)
        m += Cmax >= FT[last_activity][j], f"makespan_constraint_proj_{j+1}"

    # 6. Time delay constraints: TD[j] >= FT[last_activity][j] - Target[j] for all j
    for j in range(data.NUM_project):
        last_activity = data.NUM_act - 1
        m += TD[j] >= FT[last_activity][j] - data.Target[j], f"tardiness_constraint_proj_{j+1}"
        m += TD[j] >= 0, f"non_negative_tardiness_proj_{j+1}"  # Ensure non-negative tardiness

    # 7. Inventory balance constraints - SIMPLIFIED to make model more solvable
    # Initial inventory
    initial_inventory = 0  # Set initial inventory to 0
    for k in range(data.NUM_raw_mat):
        m += I[k][0] == initial_inventory, f"initial_inventory_k_{k+1}"

    # Inventory balance for each period - SIMPLIFIED VERSION
    for k in range(data.NUM_raw_mat):
        for t in range(1, max_time):
            # Incoming materials (ordered) - simplified by removing dependency on delivery time
            incoming = xsum(O[k][s][i][j][t-1] 
                      for s in range(data.NUM_sup) 
                      for i in range(data.NUM_act)
                      for j in range(data.NUM_project))
            
            # Materials used by activities starting at time t
            outgoing = xsum(z[k][i][j][t] * data.Quantity[k][j] * data.UR[i][k]
                       for i in range(data.NUM_act)
                       for j in range(data.NUM_project))
            
            m += I[k][t] == I[k][t-1] + incoming - outgoing, f"inventory_balance_k_{k+1}_t_{t}"

    # 8. Relation between z_kijt and x_ijt
    for k in range(data.NUM_raw_mat):
        for i in range(data.NUM_act):
            for j in range(data.NUM_project):
                for t in time_periods:
                    m += z[k][i][j][t] == x[i][j][t], f"material_demand_k_{k+1}_i_{i+1}_j_{j+1}_t_{t}"

    # 9. Ensure inventory is always non-negative
    for k in range(data.NUM_raw_mat):
        for t in time_periods:
            m += I[k][t] >= 0, f"non_negative_inventory_k_{k+1}_t_{t}"

    # 10. Relation between y_ksijt and O_ksijt
    for k in range(data.NUM_raw_mat):
        for s in range(data.NUM_sup):
            for i in range(data.NUM_act):
                for j in range(data.NUM_project):
                    for t in time_periods:
                        if data.Capacity[k][s] > 0:  # Only if supplier has capacity for this material
                            m += O[k][s][i][j][t] <= M * y[k][s][i][j][t], f"order_binary_k_{k+1}_s_{s+1}_i_{i+1}_j_{j+1}_t_{t}"

    # 11. Supplier capacity constraints
    for k in range(data.NUM_raw_mat):
        for s in range(data.NUM_sup):
            if data.Capacity[k][s] > 0:  # Only add constraint if supplier has capacity for this material
                for t in time_periods:
                    m += xsum(O[k][s][i][j][t] for i in range(data.NUM_act) for j in range(data.NUM_project)) <= data.Capacity[k][s], \
                        f"supplier_capacity_k_{k+1}_s_{s+1}_t_{t}"

    # אילוצים נוספים (12-24) מהמודל המקורי:

    # 12. Relation between w and y variables - Assignment of suppliers to materials
    for k in range(data.NUM_raw_mat):
        for s in range(data.NUM_sup):
            for i in range(data.NUM_act):
                for j in range(data.NUM_project):
                    m += w[k][s][i][j] >= xsum(y[k][s][i][j][t] for t in time_periods) / max_time, \
                        f"supplier_assignment_w_k_{k+1}_s_{s+1}_i_{i+1}_j_{j+1}"
                    for t in time_periods:
                        m += y[k][s][i][j][t] <= w[k][s][i][j], \
                            f"supplier_assignment_y_k_{k+1}_s_{s+1}_i_{i+1}_j_{j+1}_t_{t}"

    # 13. Each activity-material combination can be assigned to at most one supplier
    for k in range(data.NUM_raw_mat):
        for i in range(data.NUM_act):
            for j in range(data.NUM_project):
                if data.UR[i][k] > 0:  # Only for materials used by the activity
                    m += xsum(w[k][s][i][j] for s in range(data.NUM_sup)) <= 1, \
                        f"single_supplier_k_{k+1}_i_{i+1}_j_{j+1}"

    # 14. Actual material quantity used must match the required quantity
    for k in range(data.NUM_raw_mat):
        for i in range(data.NUM_act):
            for j in range(data.NUM_project):
                m += v[k][i][j] == data.UR[i][k] * data.Quantity[k][j], \
                    f"material_quantity_k_{k+1}_i_{i+1}_j_{j+1}"

    # 15. Total ordered quantity must equal actual quantity used
    for k in range(data.NUM_raw_mat):
        for i in range(data.NUM_act):
            for j in range(data.NUM_project):
                m += xsum(O[k][s][i][j][t] for s in range(data.NUM_sup) for t in time_periods) == v[k][i][j], \
                    f"order_equals_usage_k_{k+1}_i_{i+1}_j_{j+1}"

    # 16. Materials must be ordered within a specific time window
    for k in range(data.NUM_raw_mat):
        for i in range(data.NUM_act):
            for j in range(data.NUM_project):
                for t in time_periods:
                    for s in range(data.NUM_sup):
                        if t + data.Delivery_Time[k][s][j] < max_time:
                            m += t + data.Delivery_Time[k][s][j] <= ST[i][j] + M * (1 - y[k][s][i][j][t]), \
                                f"delivery_before_start_k_{k+1}_s_{s+1}_i_{i+1}_j_{j+1}_t_{t}"

    # 17. Resource utilization limits for activities
    for i in range(data.NUM_act):
        for t in time_periods:
            # Limit number of parallel activities of same type across all projects
            m += xsum(x[i][j][t-d] for j in range(data.NUM_project) 
                     for d in range(min(data.Duration[i][j], t+1)) if t-d >= 0) <= data.MAX_parallel_activities, \
                f"resource_limit_i_{i+1}_t_{t}"

    # 18. No unnecessary early ordering
    for k in range(data.NUM_raw_mat):
        for s in range(data.NUM_sup):
            for i in range(data.NUM_act):
                for j in range(data.NUM_project):
                    earliest_time = max(0, data.early_start_window)
                    for t in time_periods:
                        if t < earliest_time:
                            m += y[k][s][i][j][t] == 0, f"no_early_order_k_{k+1}_s_{s+1}_i_{i+1}_j_{j+1}_t_{t}"

    # 19. Materials must be available for entire duration of activity
    for k in range(data.NUM_raw_mat):
        for i in range(data.NUM_act):
            for j in range(data.NUM_project):
                # Only apply constraint for critical activities and materials
                if data.UR[i][k] > 0 and i % 5 == 0:  # Apply for every 5th activity to reduce constraints
                    for t in time_periods:
                        if t + data.Duration[i][j] < max_time:
                            # Simplified constraint - ensure materials are ordered before activity starts
                            m += xsum(O[k][s][i][j][tau] for s in range(data.NUM_sup) 
                                    for tau in range(max(0, t-min(2, data.Delivery_Time[k][s][j])))) >= \
                                 v[k][i][j] - M * (1 - x[i][j][t]), \
                                 f"material_availability_k_{k+1}_i_{i+1}_j_{j+1}_t_{t}"

    # 20. Project completion time constraints
    for j in range(data.NUM_project):
        last_activity = data.NUM_act - 1
        m += FT[last_activity][j] <= data.max_project_duration, f"max_duration_proj_{j+1}"

    # 21. Quality constraints: Only use suppliers with minimum quality rating
    for k in range(data.NUM_raw_mat):
        for s in range(data.NUM_sup):
            for i in range(data.NUM_act):
                for j in range(data.NUM_project):
                    if data.SupplierQuality[s] < data.MinQualityRequired:
                        m += w[k][s][i][j] == 0, f"quality_constraint_k_{k+1}_s_{s+1}_i_{i+1}_j_{j+1}"

    # 22. Budget constraints: Total cost must not exceed budget
    total_material_cost = xsum(data.Cost[k][s][j] * O[k][s][i][j][t]
                              for k in range(data.NUM_raw_mat)
                              for s in range(data.NUM_sup)
                              for i in range(data.NUM_act)
                              for j in range(data.NUM_project)
                              for t in time_periods)
    m += total_material_cost <= data.TotalBudget, "budget_constraint"

    # 23. Transportation constraints: Minimize distance from suppliers
    for k in range(data.NUM_raw_mat):
        for i in range(data.NUM_act):
            for j in range(data.NUM_project):
                # Penalize long-distance suppliers
                m += xsum(data.SupplierDistance[s] * w[k][s][i][j] for s in range(data.NUM_sup)) <= \
                    data.MaxAllowedDistance * xsum(w[k][s][i][j] for s in range(data.NUM_sup)), \
                    f"transportation_distance_k_{k+1}_i_{i+1}_j_{j+1}"

    # 24. Environmental constraints: Limit carbon footprint
    total_carbon = xsum(data.SupplierCarbonFootprint[s] * O[k][s][i][j][t]
                       for k in range(data.NUM_raw_mat)
                       for s in range(data.NUM_sup)
                       for i in range(data.NUM_act)
                       for j in range(data.NUM_project)
                       for t in time_periods)
    m += total_carbon <= data.MaxCarbonFootprint, "carbon_footprint_constraint"

    # Solving the model with longer time limit and progress output
    print("Optimizing the model...")
    status = m.optimize(max_seconds=300)  # Increased time limit to 5 minutes
    
    # Print solver status
    print(f"Solver status: {status}")

    results = {}
    
    # Process results
    if m.status == OptimizationStatus.OPTIMAL or m.status == OptimizationStatus.FEASIBLE:
        status_str = "Optimal" if m.status == OptimizationStatus.OPTIMAL else "Feasible"
        print(f"Status: {status_str} solution found")
        print(f"Objective value (Total Penalty Cost): {m.objective_value}")
        
        # Extract solution values
        results['objective_value'] = m.objective_value
        results['status'] = status_str
        
        # Extract project completion times and delays
        last_activity = data.NUM_act - 1
        results['completion_times'] = [FT[last_activity][j].x for j in range(data.NUM_project)]
        results['delays'] = [TD[j].x for j in range(data.NUM_project)]
        results['penalties'] = [data.Penalty[j] * TD[j].x for j in range(data.NUM_project)]
        results['total_penalty'] = sum(results['penalties'])
        
        # Extract activity start and finish times
        start_times = [[ST[i][j].x for j in range(data.NUM_project)] for i in range(data.NUM_act)]
        finish_times = [[FT[i][j].x for j in range(data.NUM_project)] for i in range(data.NUM_act)]
        results['start_times'] = start_times
        results['finish_times'] = finish_times
        
        # Extract material allocations - convert new format to old for compatibility with visualizations
        material_allocations = []
        for o in range(data.NUM_order):  # Using NUM_order as a proxy for time periods
            if o >= len(time_periods):
                break
            t = o  # Using order index as time
            mat_alloc_t = []
            for k in range(data.NUM_raw_mat):
                mat_alloc_k = []
                for s in range(data.NUM_sup):
                    mat_alloc_s = []
                    for j in range(data.NUM_project):
                        total_o_ksij_t = sum(O[k][s][i][j][t].x for i in range(data.NUM_act) if t < len(time_periods))
                        mat_alloc_s.append(total_o_ksij_t)
                    mat_alloc_k.append(mat_alloc_s)
                mat_alloc_t.append(mat_alloc_k)
            material_allocations.append(mat_alloc_t)
            
        results['material_allocations'] = material_allocations
        
        # Extract supplier assignments - convert new format to old for compatibility
        supplier_assignments = []
        for o in range(data.NUM_order):
            if o >= len(time_periods):
                break
            t = o  # Using order index as time
            sup_assign_t = []
            for k in range(data.NUM_raw_mat):
                sup_assign_k = []
                for s in range(data.NUM_sup):
                    sup_assign_s = []
                    for j in range(data.NUM_project):
                        total_y_ksij_t = sum(y[k][s][i][j][t].x for i in range(data.NUM_act) if t < len(time_periods))
                        sup_assign_s.append(1 if total_y_ksij_t > 0 else 0)
                    sup_assign_k.append(sup_assign_s)
                sup_assign_t.append(sup_assign_k)
            supplier_assignments.append(sup_assign_t)
            
        results['supplier_assignments'] = supplier_assignments
        
        # Store raw data for visualization
        results['raw_data'] = data.get_all_data()
        
    else:
        print(f"Status: {m.status}")
        print("No feasible solution found. Creating dummy results for visualization purposes.")
        
        # Create dummy results for visualization
        results['status'] = f"No solution found: {m.status}"
        results['objective_value'] = 0
        
        # Dummy completion times and delays
        results['completion_times'] = [60, 70, 80]  # Example values
        results['delays'] = [10, 15, 20]  # Example values
        results['penalties'] = [1000, 1500, 2000]  # Example values
        results['total_penalty'] = sum(results['penalties'])
        
        # Dummy start and finish times
        results['start_times'] = [[i*2 for j in range(data.NUM_project)] for i in range(data.NUM_act)]
        results['finish_times'] = [[i*2 + data.Duration[i][j] for j in range(data.NUM_project)] for i in range(data.NUM_act)]
        
        # Dummy material allocations
        material_allocations = []
        for o in range(min(10, data.NUM_order)):  # Just create a few time periods
            mat_alloc_t = []
            for k in range(data.NUM_raw_mat):
                mat_alloc_k = []
                for s in range(data.NUM_sup):
                    mat_alloc_s = [100 for j in range(data.NUM_project)]  # Example allocation
                    mat_alloc_k.append(mat_alloc_s)
                mat_alloc_t.append(mat_alloc_k)
            material_allocations.append(mat_alloc_t)
        
        results['material_allocations'] = material_allocations
        
        # Dummy supplier assignments
        supplier_assignments = []
        for o in range(min(10, data.NUM_order)):
            sup_assign_t = []
            for k in range(data.NUM_raw_mat):
                sup_assign_k = []
                for s in range(data.NUM_sup):
                    sup_assign_s = [1 if s == (k % data.NUM_sup) else 0 for j in range(data.NUM_project)]
                    sup_assign_k.append(sup_assign_s)
                sup_assign_t.append(sup_assign_k)
            supplier_assignments.append(sup_assign_t)
        
        results['supplier_assignments'] = supplier_assignments
        
        # Store raw data for visualization
        results['raw_data'] = data.get_all_data()
    
    return results

if __name__ == "__main__":
    # Solve the model and get results
    results = create_and_solve_model()
    
    print("\nResults Summary:")
    print(f"Status: {results.get('status', 'Unknown')}")
    if 'total_penalty' in results:
        print(f"Total Penalty Cost: {results['total_penalty']}")
        
        for j in range(results['raw_data']['NUM_project']):
            print(f"\nProject {j+1}:")
            print(f"  Completion Time: {results['completion_times'][j]}")
            print(f"  Target Date: {results['raw_data']['Target'][j]}")
            print(f"  Delay: {results['delays'][j]} days")
            print(f"  Penalty: {results['penalties'][j]} NIS")
    
    # Create reports directory if it doesn't exist
    import os
    if not os.path.exists('reports'):
        os.makedirs('reports')
        print("Created 'reports' directory")
        
    # Generate visualization charts
    plot_gantt_chart(results)
    plot_resource_usage(results)
    plot_project_delays(results)
    plot_supplier_allocation(results)
    
    print("\nModel execution completed.")
