from analytics import analytics_app
from flask import request,jsonify
import yfinance as yf
import numpy as np
import random
from datetime import datetime,timedelta
from time import time
import matplotlib.pyplot as plt



analytics_data = {
    "retrieve_sig_vars9599":{},
    "retrieve_avg_vars9599":{},
    "sig_profit_loss":{},
    "tot_profit_loss":{},
    "time":0,
    "cost":0

}


@analytics_app.route('/initialize_warmup', methods=['POST'])
def initialize_warmup():
    return jsonify({'initialize_warmup': 'ok'})

@analytics_app.route('/prepare_scaling', methods=['GET'])
def prepare_scaling():
    return jsonify({'response': 'ok'})

@analytics_app.route('/retrieve_warmup_cost', methods=['GET'])
def retrieve_warmup_cost():
    warmup_cost = random.uniform(10, 20)
    warmup_billable_time = random.uniform(200, 300)
    return jsonify({'warmup_billable_time': warmup_billable_time, 'warmup_cost': warmup_cost})

@analytics_app.route('/list_available_endpoints', methods=['GET'])
def list_available_endpoints():
    endpoints = [
        'http://example.com/endpoint1',
        'http://example.com/endpoint2',
        'http://example.com/endpoint3'
    ]
    return jsonify({'endpoints': endpoints})

@analytics_app.route('/perform_stock_analysis', methods=['POST'])
def perform_stock_analysis():

    start_time = time()

    h = request.json.get('h')
    d = request.json.get('d')
    t = request.json.get('t')
    p = request.json.get('p')

    end_date = datetime.now().date()
    start_date = end_date - timedelta(days=p+1)

    price_history = yf.download(["MSFT"],str(start_date),str(end_date))
    price_history = price_history['Close'].to_list()

    if p > len(price_history):
        return jsonify(error="p is larger than the price history"), 400

    # Calculate mean and standard deviation
    mu = np.mean(price_history)
    sigma = np.std(price_history)

    # Initialize lists to store the risk values
    var95_risks = []
    var99_risks = []
    profits = []

    # Generate d data points
    data_points = np.random.normal(mu, sigma, d)

    # Calculate simulated returns
    returns = (data_points[1:] - data_points[:-1]) / data_points[:-1]

    # Calculate risk values for each signal
    for i in range(0, d, h):
        # Calculate risk values
        if t == "sell":
            price_difference = price_history[-p] - price_history[-1]
        elif t == "buy":
            price_difference = price_history[-1] - price_history[-p]
        else:
            return jsonify(error="Invalid value for t. Use 'buy' or 'sell'."), 400

        var95 = np.percentile(returns[i:i+h], [5, 95])
        var99 = np.percentile(returns[i:i+h], [0.5, 99.5])


        # Append the risk values to the lists
        var95_risks.append(var95.tolist())
        var99_risks.append(var99.tolist())

        # Calculate profit/loss
        if t == "sell":
            profit = price_difference * data_points[i]
        elif t == "buy":
            profit = -price_difference * data_points[i]
        else:
            return jsonify(error="Invalid value for t. Use 'buy' or 'sell'."), 400

        profits.append(profit)


    # Calculate the average risk values
    avg_var95 = np.mean(var95_risks, axis=0)
    avg_var99 = np.mean(var99_risks, axis=0)

    global analytics_data

    sig_var_response = {"var95":var95_risks,"var99": var99_risks}
    analytics_data["retrieve_sig_vars9599"] = sig_var_response

    avg_vars9599_response = {"avg_var95":avg_var95.tolist(),"avg_var99": avg_var99.tolist()}
    analytics_data["retrieve_avg_vars9599"] = avg_vars9599_response
    
    sig_profit_loss_response = {"profit_loss:":profits}
    analytics_data["sig_profit_loss"] = sig_profit_loss_response

    tot_profit_loss_response = {"profit_loss:":sum(profits)}
    analytics_data["tot_profit_loss"] = tot_profit_loss_response

    end_time = time()
    analytics_data['time'] = end_time - start_time
    analytics_data['cost'] = analytics_data['time'] * 0.72

    response = jsonify({"response": "Success"})
    return response

@analytics_app.route('/retrieve_significant_variables_9599', methods=['GET'])
def retrieve_significant_variables_9599():
    return jsonify(analytics_data["retrieve_sig_vars9599"])

@analytics_app.route('/retrieve_average_variables_9599', methods=['GET'])
def retrieve_average_variables_9599():
    return jsonify(analytics_data["retrieve_avg_vars9599"])

@analytics_app.route('/retrieve_significant_profit_loss', methods=['GET'])
def retrieve_significant_profit_loss():
    return jsonify(analytics_data["sig_profit_loss"])

@analytics_app.route('/retrieve_total_profit_loss', methods=['GET'])
def retrieve_total_profit_loss():
    return jsonify(analytics_data["tot_profit_loss"])

@analytics_app.route('/retrieve_chart_link', methods=['GET'])
def retrieve_chart_link():

    data = analytics_data["retrieve_sig_vars9599"]
    file_path = 'analytics/static/chart.png'

    if "var95" in data and "var99" in data:
        plt.plot(data["var95"], label="var95")
        plt.plot(data["var99"], label="var99")

        plt.xlabel("X-axis")
        plt.ylabel("Y-axis")
        plt.title("Chart of var95 and var99")
        plt.legend()

        plt.savefig(file_path)
    else:
        file_path = ""

    url = file_path
    return jsonify({'chart_link': url})

@analytics_app.route('/retrieve_time_cost', methods=['GET'])
def retrieve_time_cost():
    return jsonify({'retrieve_time': analytics_data['time'], 'retrieve_cost': analytics_data['cost']})

@analytics_app.route('/retrieve_audit_data', methods=['GET'])
def retrieve_audit_data():
    return jsonify({'audit_data': 'ok'})

@analytics_app.route('/restart_instance', methods=['GET'])
def restart_instance():
    global analytics_data
    analytics_data = {
            "retrieve_sig_vars9599":{},
            "retrieve_avg_vars9599":{},
            "sig_profit_loss":{},
            "tot_profit_loss":{},
            "time":0,
            "cost":0
            }
    return jsonify({'restart_instance': True})

@analytics_app.route('/shutdown_instance', methods=['GET'])
def shutdown_instance():
    global analytics_data
    analytics_data = {
            "retrieve_sig_vars9599":{},
            "retrieve_avg_vars9599":{},
            "sig_profit_loss":{},
            "tot_profit_loss":{},
            "time":0,
            "cost":0
        }
    return jsonify({'shutdown_instance': True})

@analytics_app.route('/shutdown_scaled_cluster', methods=['GET'])
def shutdown_scaled_cluster():
    return jsonify({'shutdown_scaled_cluster': True})






