
import json
import requests
headers = {
    "user-agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/95.0.4638.54 Safari/537.36 Edg/95.0.1020.40",
}

LABEL_MAP = requests.get(
    "https://raw.githubusercontent.com/anishathalye/imagenet-simple-labels/master/imagenet-simple-labels.json"
,headers=headers).json()


filename = './Label_map.json'
with open(filename,'w') as file_obj:
    json.dump(LABEL_MAP,file_obj)

with open('./Label_map.json') as file_obj:
    test = json.load(file_obj)
print(LABEL_MAP==test)
