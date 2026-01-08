# 提出结果 单个json
# import json
# import os
# from pathlib import Path

# # Define paths
# workspace_storage_path = Path("/home/wentian/HWtest/AutoOpt/workspace/storage")
# result_path = Path("/home/wentian/HWtest/AutoOpt/Result")

# # Ensure Result directory exists
# result_path.mkdir(exist_ok=True)

# # Process each problem folder
# for i in range(1, 13):  # problem_1 to problem_12
#     problem_folder = workspace_storage_path / f"problem_{i}"
#     team_json_path = problem_folder / "team.json"
    
#     if not team_json_path.exists():
#         print(f"Warning: {team_json_path} does not exist, skipping...")
#         continue
    
#     print(f"Processing {team_json_path}...")
    
#     try:
#         # Read the team.json file
#         with open(team_json_path, 'r', encoding='utf-8') as f:
#             team_data = json.load(f)
        
#         # Navigate to the storage field
#         storage_list = team_data['env']['roles']['model_llm']['rc']['memory']['storage']
        
#         # Extract the first three items' content
#         if len(storage_list) < 3:
#             print(f"Warning: {team_json_path} has less than 3 items in storage, skipping...")
#             continue
        
#         # Create new JSON structure with renamed fields
#         extracted_data = {
#             "problem_id": f"problem_{i}",
#             "UserRequirement": storage_list[0]['content'],
#             "Thinking": storage_list[1]['content'],
#             "Formulation": storage_list[2]['content']
#         }
        
#         # Save to Result folder
#         output_file = result_path / f"problem_{i}_extracted.json"
#         with open(output_file, 'w', encoding='utf-8') as f:
#             json.dump(extracted_data, f, indent=4, ensure_ascii=False)
        
#         print(f"Successfully created {output_file}")
        
#     except Exception as e:
#         print(f"Error processing {team_json_path}: {str(e)}")
#         continue

# print("\nExtraction complete!")


#  提取出结果是合并后的json
import json
import os
from pathlib import Path

# Define paths
workspace_storage_path = Path("/home/wentian/HWtest/AutoOpt/workspace/storage")
result_path = Path("/home/wentian/HWtest/AutoOpt/Result")

# Ensure Result directory exists
result_path.mkdir(exist_ok=True)

# List to store all extracted data
all_problems_data = []

# Process each problem folder
for i in range(1, 31):  # problem_1 to problem_12
    problem_folder = workspace_storage_path / f"problem_{i}"
    team_json_path = problem_folder / "team.json"
    
    if not team_json_path.exists():
        print(f"Warning: {team_json_path} does not exist, skipping...")
        continue
    
    print(f"Processing {team_json_path}...")
    
    try:
        # Read the team.json file
        with open(team_json_path, 'r', encoding='utf-8') as f:
            team_data = json.load(f)
        
        # Navigate to the storage field
        storage_list = team_data['env']['roles']['model_llm']['rc']['memory']['storage']
        
        # Extract the first three items' content
        if len(storage_list) < 3:
            print(f"Warning: {team_json_path} has less than 3 items in storage, skipping...")
            continue
        
        # Create new JSON structure with renamed fields
        extracted_data = {
            "problem_id": f"problem_{i}",
            "UserRequirement": storage_list[0]['content'],
            "Thinking": storage_list[1]['content'],
            "Formulation": storage_list[2]['content']
        }
        
        # Add to the list
        all_problems_data.append(extracted_data)
        print(f"Successfully extracted data for problem_{i}")
        
    except Exception as e:
        print(f"Error processing {team_json_path}: {str(e)}")
        continue

# Save all data to a single JSON file
output_file = result_path / "test_new_01.json"
with open(output_file, 'w', encoding='utf-8') as f:
    json.dump(all_problems_data, f, indent=4, ensure_ascii=False)

print(f"\nSuccessfully created {output_file}")
print(f"Total problems extracted: {len(all_problems_data)}")
