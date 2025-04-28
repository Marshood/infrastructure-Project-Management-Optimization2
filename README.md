# Infrastructure Project Management Optimization

This repository contains a Python implementation of an optimization model for managing parallel infrastructure projects with supplier selection. The model is designed to minimize the total penalty costs due to delays in project completion.

## Overview

The model deals with the following scenario:
- Multiple infrastructure projects need to be managed simultaneously
- Each project consists of multiple activities with precedence relationships
- Raw materials need to be ordered from various suppliers with different capacities
- Projects have target completion dates with penalties for delays
- Suppliers have limited capacities and different delivery times

## Features

- Optimization model based on Mixed Integer Programming
- Visualization of project schedules (Gantt charts)
- Resource usage analysis
- Project delays and penalties visualization
- Supplier allocation visualization
- Critical path analysis

## Requirements

```
pip install -r requirements.txt
```

## Usage

Run the main script to solve the model and generate visualizations:

```bash
python main.py
```

## Model Description

The mathematical model minimizes the total penalty costs due to delays in project completion. The model considers:

- Multiple projects with activities and precedence relationships
- Raw material requirements for each project
- Supplier selection with capacity constraints
- Order scheduling
- Delivery times

### Decision Variables

- Start and finish times for activities
- Raw material quantities ordered from each supplier
- Assignment of suppliers to projects
- Order times for raw materials

### Constraints

- Precedence relationships between activities
- Supplier capacity constraints
- Material demand constraints
- Material arrival before activity start
- Project completion time calculations
- Delay calculations

### Objective Function

Minimize the sum of penalty costs for all projects:

```
min ∑(Penalty[j] * Delay[j]) for all projects j
```

## Visualization

The code generates the following visualizations:

1. **Gantt Chart**: Shows the schedule of activities for each project
2. **Resource Usage**: Shows the allocation of raw materials to each project
3. **Project Delays**: Compares target dates with actual completion times and associated penalties
4. **Supplier Allocation**: Shows how raw materials are allocated from suppliers to projects
5. **Critical Path**: Highlights the critical activities in each project

## Model Parameters

The model uses the following parameters:
- Duration of activities for each project
- Raw material requirements for each project
- Supplier capacities for each material
- Target completion dates for each project
- Penalties for delays for each project
- Delivery times for materials from suppliers
- Precedence relationships between activities
