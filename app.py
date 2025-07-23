from flask import Flask, request, jsonify
from flask_restx import Api, Resource, fields
import pulp
import numpy as np
from typing import Dict, List, Optional
from dataclasses import dataclass
import json
import os
import jwt

app = Flask(__name__)

@app.before_request
def before_request_func():
    secret_key = os.environ.get('JWT_TOKEN_SECRET')
    if secret_key:
        auth_header = request.headers.get('Authorization')
        if not auth_header:
            return jsonify({"message": "Missing authorization header"}), 401
        
        try:
            token_type, token = auth_header.split(' ')
            if token_type.lower() != 'bearer':
                return jsonify({"message": "Invalid token type"}), 401
            
            jwt.decode(token, secret_key, algorithms=["HS256"])
        except jwt.ExpiredSignatureError:
            return jsonify({"message": "Token has expired"}), 401
        except jwt.InvalidTokenError:
            return jsonify({"message": "Invalid token"}), 401
        except Exception as e:
            return jsonify({"message": str(e)}), 401

api = Api(app, version='1.0', title='EV Charging Optimization API', 
          description='Mixed Integer Linear Programming model for EV charging optimization')

# Namespace for the API
ns = api.namespace('optimize', description='EV Charging Optimization Operations')

# Input models for API documentation
battery_config_model = api.model('BatteryConfig', {
    's_min': fields.Float(required=True, description='Minimum state of charge (Wh)'),
    's_max': fields.Float(required=True, description='Maximum state of charge (Wh)'),
    's_initial': fields.Float(required=True, description='Initial state of charge (Wh)'),
    'c_max': fields.Float(required=True, description='Maximum charge power (W)'),
    'd_max': fields.Float(required=True, description='Maximum discharge power (W)'),
    'p_a': fields.Float(required=True, description='Value per Wh at end of horizon')
})

time_series_model = api.model('TimeSeries', {
    'gt': fields.List(fields.Float, required=True, description='Required total energy at each time step (Wh)'),
    'ft': fields.List(fields.Float, required=True, description='Forecasted energy production at each time step (Wh)'),
    'p_N': fields.List(fields.Float, required=True, description='Price per Wh taken from grid at each time step'),
    'p_E': fields.List(fields.Float, required=True, description='Price per Wh fed into grid at each time step'),
    'b_goal': fields.List(fields.List(fields.Float), required=False, description='Goal state of charge for each battery at each time step (Wh) - nested list [battery][time_step]'),
})

optimization_input_model = api.model('OptimizationInput', {
    'batteries': fields.List(fields.Nested(battery_config_model), required=True, description='Battery configurations'),
    'time_series': fields.Nested(time_series_model, required=True, description='Time series data'),
    'eta_c': fields.Float(required=False, default=0.95, description='Charging efficiency'),
    'eta_d': fields.Float(required=False, default=0.95, description='Discharging efficiency'),
})

# Output models
battery_result_model = api.model('BatteryResult', {
    'charging_power': fields.List(fields.Float, description='Charging power at each time step (W)'),
    'discharging_power': fields.List(fields.Float, description='Discharging power at each time step (W)'),
    'state_of_charge': fields.List(fields.Float, description='State of charge at each time step (Wh)')
})

optimization_result_model = api.model('OptimizationResult', {
    'status': fields.String(description='Optimization status'),
    'objective_value': fields.Float(description='Optimal objective function value'),
    'batteries': fields.List(fields.Nested(battery_result_model), description='Battery optimization results'),
    'grid_import': fields.List(fields.Float, description='Energy imported from grid at each time step (Wh)'),
    'grid_export': fields.List(fields.Float, description='Energy exported to grid at each time step (Wh)'),
    'flow_direction': fields.List(fields.Integer, description='Binary flow direction (1=export, 0=import)')
})

@dataclass
class BatteryConfig:
    s_min: float
    s_max: float
    s_initial: float
    c_min: float
    c_max: float
    d_max: float
    p_a: float

@dataclass
class TimeSeriesData:
    gt: List[float]  # Required total energy
    ft: List[float]  # Forecasted production
    p_N: List[float]  # Import prices
    p_E: List[float]  # Export prices
    b_goal: Optional[List[List[float]]] = None  # Goal state of charge for each battery (Wh)

class EVChargingOptimizer:
    def __init__(self, batteries: List[BatteryConfig], time_series: TimeSeriesData, 
                 eta_c: float = 0.95, eta_d: float = 0.95, M: float = 1e6):
        self.batteries = batteries
        self.time_series = time_series
        self.eta_c = eta_c
        self.eta_d = eta_d
        self.M = M
        self.T = len(time_series.gt)
        self.problem = None
        self.variables = {}
        
    def create_model(self):
        """Create the MILP model"""
        # Create problem
        self.problem = pulp.LpProblem("EV_Charging_Optimization", pulp.LpMaximize)
        
        # Time steps
        time_steps = range(self.T)
        
        # Decision variables
        # Charging power variables
        self.variables['c'] = {}
        for i, bat in enumerate(self.batteries):
            self.variables['c'][i] = [
                pulp.LpVariable(f"c_{i}_{t}", lowBound=bat.c_min, upBound=bat.c_max)
                for t in time_steps
            ]
        
        # Discharging power variables
        self.variables['d'] = {}
        for i, bat in enumerate(self.batteries):
            self.variables['d'][i] = [
                pulp.LpVariable(f"d_{i}_{t}", lowBound=0, upBound=bat.d_max)
                for t in time_steps
            ]
        
        # State of charge variables
        self.variables['s'] = {}
        for i, bat in enumerate(self.batteries):
            self.variables['s'][i] = [
                pulp.LpVariable(f"s_{i}_{t}", lowBound=bat.s_min, upBound=bat.s_max)
                for t in time_steps
            ]
        
        # Grid import/export variables
        self.variables['n'] = [pulp.LpVariable(f"n_{t}", lowBound=0) for t in time_steps]
        self.variables['e'] = [pulp.LpVariable(f"e_{t}", lowBound=0) for t in time_steps]
        
        # Binary flow direction variables (only when p_N <= p_E)
        self.variables['y'] = []
        for t in time_steps:
            if self.time_series.p_N[t] <= self.time_series.p_E[t]:
                self.variables['y'].append(pulp.LpVariable(f"y_{t}", cat='Binary'))
            else:
                self.variables['y'].append(None)
        
        # Objective function (1): Maximize economic benefit
        objective = 0
        
        # Grid import cost (negative because we want to minimize cost)
        for t in time_steps:
            objective -= self.variables['n'][t] * self.time_series.p_N[t]
        
        # Grid export revenue
        for t in time_steps:
            objective += self.variables['e'][t] * self.time_series.p_E[t]
        
        # Final state of charge value
        for i, bat in enumerate(self.batteries):
            objective += self.variables['s'][i][-1] * bat.p_a
        
        self.problem += objective
        
        # Constraints
        self._add_constraints()
        
    def _add_constraints(self):
        """Add all constraints to the model"""
        time_steps = range(self.T)
        
        # Constraint (2): Power balance
        for t in time_steps:
            battery_net_power = 0
            for i, bat in enumerate(self.batteries):
                battery_net_power += (-self.variables['c'][i][t] + 
                                    self.variables['d'][i][t])
            
            self.problem += (battery_net_power + self.time_series.ft[t] + 
                           self.variables['n'][t] == 
                           self.variables['e'][t] + self.time_series.gt[t])
        
        # Constraint (3): Battery dynamics
        for i, bat in enumerate(self.batteries):
            # Initial state of charge
            if len(time_steps) > 0:
                self.problem += (self.variables['s'][i][0] == 
                               bat.s_initial + self.eta_c * self.variables['c'][i][0] -
                               (1/self.eta_d) * self.variables['d'][i][0])
            
            # State of charge evolution
            for t in range(1, self.T):
                self.problem += (self.variables['s'][i][t] == 
                               self.variables['s'][i][t-1] + 
                               self.eta_c * self.variables['c'][i][t] -
                               (1/self.eta_d) * self.variables['d'][i][t])
        
        # Constraints (4)-(5): Grid flow direction (only when p_N <= p_E)
        for t in time_steps:
            if self.variables['y'][t] is not None:  # Only when p_N <= p_E
                # Export constraint
                self.problem += self.variables['e'][t] <= self.M * self.variables['y'][t]
                # Import constraint
                self.problem += self.variables['n'][t] <= self.M * (1 - self.variables['y'][t])
        
        # Constraint (6): Battery SOC goal constraints (for t > 0)
        if self.time_series.b_goal is not None:
            for i, b_goal in enumerate(self.time_series.b_goal):
                for t in range(1, self.T):
                    if b_goal[t] > 0:
                        self.problem += (self.variables['s'][i][t] >= b_goal[t])
    
    def solve(self) -> Dict:
        """Solve the optimization problem and return results"""
        if self.problem is None:
            self.create_model()
        
        # Solve the problem
        solver = pulp.PULP_CBC_CMD(msg=0)  # Silent solver
        self.problem.solve(solver)
        
        # Extract results
        status = pulp.LpStatus[self.problem.status]
        
        if status == 'Optimal':
            result = {
                'status': status,
                'objective_value': pulp.value(self.problem.objective),
                'batteries': [],
                'grid_import': [pulp.value(var) for var in self.variables['n']],
                'grid_export': [pulp.value(var) for var in self.variables['e']],
                'flow_direction': []
            }
            
            # Extract battery results
            for i, bat in enumerate(self.batteries):
                battery_result = {
                    'charging_power': [pulp.value(var) for var in self.variables['c'][i]],
                    'discharging_power': [pulp.value(var) for var in self.variables['d'][i]],
                    'state_of_charge': [pulp.value(var) for var in self.variables['s'][i]]
                }
                result['batteries'].append(battery_result)
            
            # Extract flow direction
            for y_var in self.variables['y']:
                if y_var is not None:
                    result['flow_direction'].append(int(pulp.value(y_var)))
                else:
                    result['flow_direction'].append(0)  # Default to import when constraint not active
            
            return result
        else:
            return {
                'status': status,
                'objective_value': None,
                'batteries': [],
                'grid_import': [],
                'grid_export': [],
                'flow_direction': []
            }

@ns.route('/charge-schedule')
class OptimizeCharging(Resource):
    @api.expect(optimization_input_model)
    @api.marshal_with(optimization_result_model)
    def post(self):
        """
        Optimize EV charging schedule using MILP
        
        This endpoint solves a Mixed Integer Linear Programming problem to optimize
        EV charging schedules considering battery constraints, grid prices, and energy demands.
        """
        try:
            data = request.get_json()
            
            # Validate input data
            if not data:
                api.abort(400, "No input data provided")
            
            # Parse battery configurations
            batteries = []
            for bat_data in data['batteries']:
                batteries.append(BatteryConfig(
                    s_min=bat_data['s_min'],
                    s_max=bat_data['s_max'],
                    s_initial=bat_data['s_initial'],
                    c_min=bat_data['c_min'],
                    c_max=bat_data['c_max'],
                    d_max=bat_data['d_max'],
                    p_a=bat_data['p_a']
                ))
            
            # Parse time series data
            time_series = TimeSeriesData(
                gt=data['time_series']['gt'],
                ft=data['time_series']['ft'],
                p_N=data['time_series']['p_N'],
                p_E=data['time_series']['p_E'],
                b_goal=data['time_series'].get('b_goal')
            )

            # Validate time series lengths
            lengths = [len(time_series.gt), len(time_series.ft), 
                      len(time_series.p_N), len(time_series.p_E)]

            # Validate b_goal if provided
            if time_series.b_goal is not None:
                if len(time_series.b_goal) > len(batteries):
                    api.abort(400, f"Battery goals must have same or lower length than batteries ({len(batteries)}), got {len(time_series.b_goal)}")
                for goal in time_series.b_goal:
                    lengths.append(len(goal))

            if len(set(lengths)) > 1:
                api.abort(400, "All time series must have the same length")
            
        except KeyError as e:
            api.abort(400, f"Missing required field: {str(e)}")
        except (TypeError, ValueError) as e:
            api.abort(400, f"Invalid data format: {str(e)}")
        
        try:
            # Create and solve optimizer
            optimizer = EVChargingOptimizer(
                batteries=batteries,
                time_series=time_series,
                eta_c=data.get('eta_c', 0.95),
                eta_d=data.get('eta_d', 0.95),
                M=1e6
            )
            
            result = optimizer.solve()
            return result
            
        except Exception as e:
            api.abort(500, f"Optimization failed: {str(e)}")

@ns.route('/health')
class Health(Resource):
    def get(self):
        """Health check endpoint"""
        return {'status': 'healthy', 'message': 'EV Charging MILP API is running'}

# Example data endpoint for testing
@ns.route('/example')
class ExampleData(Resource):
    def get(self):
        """Get example input data for testing the optimization"""
        example_data = {
            "batteries": [
                {
                    "s_min": 5000,
                    "s_max": 50000,
                    "s_initial": 15000,
                    "c_min": 4200,
                    "c_max": 11000,
                    "d_max": 0,
                    "p_a": 0.25
                },
                {
                    "s_min": 1000,
                    "s_max": 8000,
                    "s_initial": 5000,
                    "c_min": 0,
                    "c_max": 5000,
                    "d_max": 5000,
                    "p_a": 0.20
                }
            ],
            "time_series": {
                "gt": [3000, 4000, 5000, 4500, 3500, 3000],  # Required energy
                "ft": [2000, 6000, 8000, 7000, 4000, 1000],  # PV production
                "p_N": [0.30, 0.25, 0.20, 0.22, 0.28, 0.32],  # Import prices
                "p_E": [0.15, 0.12, 0.10, 0.11, 0.14, 0.16],  # Export prices
                "b_goal": [
                    [0, 0, 40000, 0, 0, 0]  # Battery 1 goals
                ]
            },
            "eta_c": 0.95,
            "eta_d": 0.95
        }
        return example_data

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=7050)