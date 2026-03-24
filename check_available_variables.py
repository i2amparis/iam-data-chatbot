import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from dotenv import load_dotenv
from main import IAMParisBot

def check_available_variables():
    load_dotenv(override=True)
    
    bot = IAMParisBot(streaming=False)
    
    # Fetch test data from API
    print("Fetching test data from API...")
    test_payload = {"workspace_code": ["study-1"], "limit": 50}
    try:
        data = bot.fetch_json(bot.env["REST_API_FULL"], payload=test_payload, cache=False)
        print(f"Retrieved {len(data)} records")
        
        if data:
            # Extract unique variables
            variables = set()
            scenarios = set()
            regions = set()
            models = set()
            
            for record in data:
                variables.add(record.get('variable', 'N/A'))
                scenarios.add(record.get('scenario', 'N/A'))
                regions.add(record.get('region', 'N/A'))
                models.add(record.get('modelName', 'N/A'))
            
            print(f"\nUnique variables ({len(variables)}):")
            for var in sorted(variables):
                print(f"  - {var}")
            
            print(f"\nUnique scenarios ({len(scenarios)}):")
            for scenario in sorted(scenarios):
                print(f"  - {scenario}")
            
            print(f"\nUnique regions ({len(regions)}):")
            for region in sorted(regions):
                print(f"  - {region}")
            
            print(f"\nUnique models ({len(models)}):")
            for model in sorted(models):
                print(f"  - {model}")
            
            return variables, scenarios, regions, models
        else:
            print("No data retrieved")
            return set(), set(), set(), set()
            
    except Exception as e:
        print(f"Error: {e}")
        return set(), set(), set(), set()

if __name__ == "__main__":
    check_available_variables()
