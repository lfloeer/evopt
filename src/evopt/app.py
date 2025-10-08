import os

import jwt
from flask import Flask, jsonify, request
from flask_restx import Api, Resource, fields

from .optimizer import BatteryConfig, OptimizationStrategy, Optimizer, TimeSeriesData

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

            payload = jwt.decode(token, secret_key, algorithms=["HS256"])
            print("subject:", payload.get('sub'))
        except jwt.ExpiredSignatureError:
            return jsonify({"message": "Token has expired"}), 401
        except jwt.InvalidTokenError:
            return jsonify({"message": "Invalid token"}), 401
        except Exception as e:
            return jsonify({"message": str(e)}), 401


api = Api(app, version='1.0', title='EV Charging Optimization API',
          description='Mixed Integer Linear Programming model for EV charging optimization',
          validate=True)

# Namespace for the API
ns = api.namespace('optimize', description='EV Charging Optimization Operations')

# Input models for API documentation
strategy_model = api.model('OptimizationStrategy', {
    'charging_strategy': fields.String(required=False, description='Sets a strategy for charging in situations where choices are cost neutral.'),
    'discharging_strategy': fields.String(required=False, description='Sets a strategy for discharging in situations where choices are cost neutral.')
})

battery_config_model = api.model('BatteryConfig', {
    'charge_from_grid': fields.Boolean(required=False, description='Controls whether the battery can be charged from the grid.'),
    'discharge_to_grid': fields.Boolean(required=False, description='Controls whether the battery can discharge to grid.'),
    's_min': fields.Float(required=True, description='Minimum state of charge (Wh)'),
    's_max': fields.Float(required=True, description='Maximum state of charge (Wh)'),
    's_initial': fields.Float(required=True, description='Initial state of charge (Wh)'),
    'p_demand': fields.List(fields.Float, required=False, description='Minimum charge demand per time step (Wh)'),
    's_goal': fields.List(fields.Float, required=False, description='Goal state of charge at each time step (Wh)'),
    'c_min': fields.Float(required=True, description='Minimum charge power (W)'),
    'c_max': fields.Float(required=True, description='Maximum charge power (W)'),
    'd_max': fields.Float(required=True, description='Maximum discharge power (W)'),
    'p_a': fields.Float(required=True, description='Monetary value per Wh at end of the optimization horizon')
})

time_series_model = api.model('TimeSeries', {
    'dt': fields.List(fields.Float, required=True, description='duration in seconds for each time step (s)'),
    'gt': fields.List(fields.Float, required=True, description='Required energy for home consumption at each time step (Wh)'),
    'ft': fields.List(fields.Float, required=True, description='Forecasted solar generation at each time step (Wh)'),
    'p_N': fields.List(fields.Float, required=True, description='Price per Wh taken from grid at each time step'),
    'p_E': fields.List(fields.Float, required=True, description='Remuneration per Wh fed into grid at each time step'),
})

optimization_input_model = api.model('OptimizationInput', {
    'strategy': fields.Nested(strategy_model, required=False, description='Optimization strategy'),
    'batteries': fields.List(fields.Nested(battery_config_model), required=True, description='Battery configurations'),
    'time_series': fields.Nested(time_series_model, required=True, description='Time series data'),
    'eta_c': fields.Float(required=False, default=0.95, description='Charging efficiency'),
    'eta_d': fields.Float(required=False, default=0.95, description='Discharging efficiency'),
})

# Output models
battery_result_model = api.model('BatteryResult', {
    'charging_power': fields.List(fields.Float, description='Optimal charging energy at each time step (Wh)'),
    'discharging_power': fields.List(fields.Float, description='Optimal discharging energy at each time step (Wh)'),
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


@ns.route('/charge-schedule')
class OptimizeCharging(Resource):
    @api.expect(optimization_input_model, validate=True)
    @api.marshal_with(optimization_result_model)
    def post(self):
        """
        Optimize EV charging schedule using MILP

        This endpoint solves a Mixed Integer Linear Programming problem to optimize
        EV charging schedules considering battery constraints, grid prices, and energy demands.
        """
        try:
            data = api.payload

            # Parse strategy items with default values
            strat_data = data.get('strategy', {})
            charging_strat = strat_data.get('charging_strategy', 'none')
            discharging_strat = strat_data.get('discharging_strategy', 'none')

            strategy = OptimizationStrategy(
                charging_strategy=charging_strat,
                discharging_strategy=discharging_strat
            )

            # Parse battery configurations
            batteries = []
            for bat_data in data['batteries']:
                batteries.append(BatteryConfig(
                    charge_from_grid=bat_data.get('charge_from_grid', False),
                    discharge_to_grid=bat_data.get('discharge_to_grid', False),
                    s_min=bat_data['s_min'],
                    s_max=bat_data['s_max'],
                    s_initial=bat_data['s_initial'],
                    p_demand=bat_data.get('p_demand'),
                    s_goal=bat_data.get('s_goal'),
                    c_min=bat_data['c_min'],
                    c_max=bat_data['c_max'],
                    d_max=bat_data['d_max'],
                    p_a=bat_data['p_a'],
                ))

            # Parse time series data
            time_series = TimeSeriesData(
                dt=data['time_series']['dt'],
                gt=data['time_series']['gt'],
                ft=data['time_series']['ft'],
                p_N=data['time_series']['p_N'],
                p_E=data['time_series']['p_E'],
            )

            # Validate time series lengths
            lengths = [len(time_series.gt), len(time_series.ft),
                       len(time_series.p_N), len(time_series.p_E)]

            # Validate p_demand if provided
            for bat in batteries:
                if bat.p_demand is not None:
                    lengths.append(len(bat.p_demand))

            # Validate s_goal if provided
            for bat in batteries:
                if bat.s_goal is not None:
                    lengths.append(len(bat.s_goal))

            if len(set(lengths)) > 1:
                api.abort(400, "All time series must have the same length")

        except Exception as e:
            api.abort(400, f"Invalid data format: {str(e)}")

        try:
            # Create and solve optimizer
            optimizer = Optimizer(
                strategy=strategy,
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


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=7050)
