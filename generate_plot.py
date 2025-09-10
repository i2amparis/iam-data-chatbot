#!/usr/bin/env python3
"""
Generate plot for timeseries emissions per model
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from simple_plotter import simple_plot_query

def generate_emissions_plot():
    """Generate plot for emissions per model"""

    # Sample data (simulating real IAM data)
    model_data = [
        {"modelName": "MESSAGEix-GLOBIOM", "description": "Integrated assessment model"},
        {"modelName": "REMIND-MAgPIE", "description": "Energy-economy model"},
        {"modelName": "AIM/CGE", "description": "Computable general equilibrium model"},
        {"modelName": "GCAM", "description": "Global change assessment model"},
        {"modelName": "IMAGE", "description": "Integrated model to assess the global environment"}
    ]

    ts_data = [
        {
            "variable": "CO2 emissions",
            "modelName": "MESSAGEix-GLOBIOM",
            "scenario": "SSP2-Base",
            "unit": "Mt CO2/yr",
            "2020": 35000,
            "2030": 32000,
            "2040": 28000,
            "2050": 22000
        },
        {
            "variable": "CO2 emissions",
            "modelName": "REMIND-MAgPIE",
            "scenario": "SSP2-Base",
            "unit": "Mt CO2/yr",
            "2020": 36000,
            "2030": 33000,
            "2040": 29000,
            "2050": 23000
        },
        {
            "variable": "CO2 emissions",
            "modelName": "AIM/CGE",
            "scenario": "SSP2-Base",
            "unit": "Mt CO2/yr",
            "2020": 34000,
            "2030": 31000,
            "2040": 27000,
            "2050": 21000
        },
        {
            "variable": "CO2 emissions",
            "modelName": "GCAM",
            "scenario": "SSP2-Base",
            "unit": "Mt CO2/yr",
            "2020": 37000,
            "2030": 34000,
            "2040": 30000,
            "2050": 24000
        },
        {
            "variable": "CO2 emissions",
            "modelName": "IMAGE",
            "scenario": "SSP2-Base",
            "unit": "Mt CO2/yr",
            "2020": 33000,
            "2030": 30000,
            "2040": 26000,
            "2050": 20000
        }
    ]

    query = "Can you plot the timeseries for these emissions per model?"
    result = simple_plot_query(query, model_data, ts_data)
    print(result)

if __name__ == "__main__":
    generate_emissions_plot()
