import json

path = '../dataserver/axcell/medium/paperDict_label_ano_edit-20231229.json'
with open(path, 'r') as f:
    data = json.load(f)

query_count = 0
pair_count = 0

for title in data:
    if "similar_label" in data[title]:
        query_flag = False
        for labels in data[title]["similar_label"]:
            if len(labels) > 0:
                pair_count += 1
                query_flag = True
        
        query_count += 1

print(f"query_count: {query_count}")
print(f"pair_coumt: {pair_count}")
