"""
Data structures for the infrastructure project management model.
This module contains classes to organize and access the model parameters.
"""

class ModelData:
    """
    Class to store all data parameters for the optimization model
    """
    def __init__(self):
        # Number of projects
        self.NUM_project = 3
        
        # Number of activities per project
        self.NUM_act = 23
        
        # Number of raw material types
        self.NUM_raw_mat = 4
        
        # Number of suppliers
        self.NUM_sup = 4
        
        # Number of orders
        self.NUM_order = 20
        
        # Precedence relationships between activities
        # Each tuple (a,b) means that activity a must be completed before activity b can start
        # Activities are indexed from 0 to NUM_act-1 for easier Python indexing
        self.Pred = [
            (0, 1), (0, 2), (0, 3), (1, 4), (2, 5), (3, 6), (3, 7),
            (4, 8), (5, 8), (6, 9), (7, 9), (8, 10), (9, 10),
            (10, 11), (10, 12), (10, 13), (11, 14), (12, 15), (13, 16),
            (14, 17), (15, 17), (16, 18), (17, 19), (18, 19),
            (19, 20), (19, 21), (20, 22), (21, 22)
        ]
        
        # Duration of each activity for each project
        self.Duration = [
            [5, 7, 6],     # Activity 1 (index 0)
            [8, 10, 9],    # Activity 2 (index 1)
            [10, 12, 11],  # Activity 3 (index 2)
            [7, 9, 8],     # Activity 4 (index 3)
            [12, 15, 13],  # Activity 5 (index 4)
            [9, 11, 10],   # Activity 6 (index 5)
            [8, 10, 9],    # Activity 7 (index 6)
            [11, 14, 12],  # Activity 8 (index 7)
            [15, 18, 16],  # Activity 9 (index 8)
            [10, 12, 11],  # Activity 10 (index 9)
            [14, 17, 15],  # Activity 11 (index 10)
            [6, 8, 7],     # Activity 12 (index 11)
            [9, 11, 10],   # Activity 13 (index 12)
            [12, 15, 13],  # Activity 14 (index 13)
            [8, 10, 9],    # Activity 15 (index 14)
            [7, 9, 8],     # Activity 16 (index 15)
            [10, 12, 11],  # Activity 17 (index 16)
            [13, 16, 14],  # Activity 18 (index 17)
            [9, 11, 10],   # Activity 19 (index 18)
            [15, 18, 16],  # Activity 20 (index 19)
            [6, 8, 7],     # Activity 21 (index 20)
            [10, 12, 11],  # Activity 22 (index 21)
            [8, 10, 9]     # Activity 23 (index 22)
        ]
        
        # Target completion dates for each project
        self.Target = [418, 370, 423]
        
        # Penalty cost per day of delay for each project
        self.Penalty = [100, 120, 90]
        
        # Usage rate of raw materials for each activity
        # Each row represents an activity, each column a raw material type
        # Value indicates how much of the material is needed for the activity
        self.UR = [
            [0.5, 0.3, 0.2, 0.0],   # Activity 1
            [0.4, 0.2, 0.3, 0.1],   # Activity 2
            [0.3, 0.3, 0.3, 0.1],   # Activity 3
            [0.0, 0.4, 0.3, 0.3],   # Activity 4
            [0.2, 0.2, 0.3, 0.3],   # Activity 5
            [0.3, 0.3, 0.2, 0.2],   # Activity 6
            [0.4, 0.2, 0.1, 0.3],   # Activity 7
            [0.2, 0.3, 0.2, 0.3],   # Activity 8
            [0.1, 0.3, 0.3, 0.3],   # Activity 9
            [0.3, 0.3, 0.3, 0.1],   # Activity 10
            [0.2, 0.2, 0.3, 0.3],   # Activity 11
            [0.4, 0.2, 0.2, 0.2],   # Activity 12
            [0.3, 0.3, 0.3, 0.1],   # Activity 13
            [0.2, 0.2, 0.3, 0.3],   # Activity 14
            [0.1, 0.2, 0.4, 0.3],   # Activity 15
            [0.3, 0.2, 0.3, 0.2],   # Activity 16
            [0.3, 0.3, 0.2, 0.2],   # Activity 17
            [0.2, 0.3, 0.3, 0.2],   # Activity 18
            [0.3, 0.2, 0.3, 0.2],   # Activity 19
            [0.2, 0.3, 0.3, 0.2],   # Activity 20
            [0.3, 0.3, 0.2, 0.2],   # Activity 21
            [0.3, 0.2, 0.3, 0.2],   # Activity 22
            [0.2, 0.3, 0.2, 0.3]    # Activity 23
        ]
        
        # Quantity of raw materials needed for each project
        # Each row represents a raw material, each column a project
        self.Quantity = [
            [1000, 1200, 1100],   # Raw Material 1
            [800, 1000, 900],     # Raw Material 2
            [1200, 1400, 1300],   # Raw Material 3
            [900, 1100, 1000]     # Raw Material 4
        ]
        
        # Capacity of each supplier for each raw material
        # Each row represents a raw material, each column a supplier
        self.Capacity = [
            [500, 600, 550, 450],   # Raw Material 1
            [400, 500, 450, 350],   # Raw Material 2
            [600, 700, 650, 550],   # Raw Material 3
            [450, 550, 500, 400]    # Raw Material 4
        ]
        
        # Cost of raw materials from each supplier for each project
        # First index: raw material, Second index: supplier, Third index: project
        self.Cost = [
            [[50, 55, 52], [48, 53, 50], [52, 57, 54], [46, 51, 48]],   # Raw Material 1
            [[40, 44, 42], [38, 42, 40], [42, 46, 44], [36, 40, 38]],   # Raw Material 2
            [[60, 66, 63], [58, 64, 61], [62, 68, 65], [56, 62, 59]],   # Raw Material 3
            [[45, 50, 47], [43, 48, 45], [47, 52, 49], [41, 46, 43]]    # Raw Material 4
        ]
        
        # Delivery time of raw materials from each supplier for each project
        # First index: raw material, Second index: supplier, Third index: project
        self.Delivery_Time = [
            [[5, 7, 6], [4, 6, 5], [6, 8, 7], [7, 9, 8]],   # Raw Material 1
            [[4, 6, 5], [3, 5, 4], [5, 7, 6], [6, 8, 7]],   # Raw Material 2
            [[6, 8, 7], [5, 7, 6], [7, 9, 8], [8, 10, 9]],  # Raw Material 3
            [[5, 7, 6], [4, 6, 5], [6, 8, 7], [7, 9, 8]]    # Raw Material 4
        ]
        
        # פרמטרים חדשים עבור האילוצים החדשים (12-24)
        
        # מספר מקסימלי של פעילויות מקבילות מאותו סוג
        self.MAX_parallel_activities = 2
        
        # חלון זמן מוקדם ביותר להזמנת חומרים (ביחס להתחלה)
        self.early_start_window = 15
        
        # משך מקסימלי מותר לפרויקט
        self.max_project_duration = 450
        
        # איכות הספקים (ציון מ-0 עד 100)
        self.SupplierQuality = [85, 90, 80, 75]
        
        # ציון איכות מינימלי נדרש
        self.MinQualityRequired = 70
        
        # תקציב כולל מקסימלי
        self.TotalBudget = 500000
        
        # מרחק מהספקים (בק"מ)
        self.SupplierDistance = [120, 80, 150, 200]
        
        # מרחק מקסימלי מותר
        self.MaxAllowedDistance = 180
        
        # טביעת רגל פחמנית לכל ספק (יחידת CO2 לטון חומר)
        self.SupplierCarbonFootprint = [2.5, 2.0, 3.0, 3.5]
        
        # סף מקסימלי מותר של טביעת רגל פחמנית
        self.MaxCarbonFootprint = 400000
        
    def get_all_data(self):
        """
        Return all data as a dictionary for easier access
        """
        return {
            'NUM_project': self.NUM_project,
            'NUM_act': self.NUM_act,
            'NUM_raw_mat': self.NUM_raw_mat,
            'NUM_sup': self.NUM_sup,
            'NUM_order': self.NUM_order,
            'Pred': self.Pred,
            'Duration': self.Duration,
            'Target': self.Target,
            'Penalty': self.Penalty,
            'UR': self.UR,
            'Quantity': self.Quantity,
            'Capacity': self.Capacity,
            'Cost': self.Cost,
            'Delivery_Time': self.Delivery_Time,
            'MAX_parallel_activities': self.MAX_parallel_activities,
            'early_start_window': self.early_start_window,
            'max_project_duration': self.max_project_duration,
            'SupplierQuality': self.SupplierQuality,
            'MinQualityRequired': self.MinQualityRequired,
            'TotalBudget': self.TotalBudget,
            'SupplierDistance': self.SupplierDistance,
            'MaxAllowedDistance': self.MaxAllowedDistance,
            'SupplierCarbonFootprint': self.SupplierCarbonFootprint,
            'MaxCarbonFootprint': self.MaxCarbonFootprint
        }

    def get_project_names(self):
        """Return a list of project names"""
        return [f'Project {j+1}' for j in range(self.NUM_project)]
    
    def get_raw_material_names(self):
        """Return a list of raw material names"""
        return [f'Material {k+1}' for k in range(self.NUM_raw_mat)]
    
    def get_supplier_names(self):
        """Return a list of supplier names"""
        return [f'Supplier {s+1}' for s in range(self.NUM_sup)]
    
    def get_activity_names(self):
        """Return a list of activity names"""
        return [f'Activity {i+1}' for i in range(self.NUM_act)]
