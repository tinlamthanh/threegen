# import requests
# import base64

# # Define the URL of the endpoint
# url = "http://localhost:8094/validate/"

# # Open the .obj file in binary mode
# with open("test.obj", "rb") as f:
#     # Read the file data
#     file_data = f.read()

# # Encode the binary data as a base64 string
# file_data_base64 = base64.b64encode(file_data).decode('utf-8')

# # Define the data to send to the endpoint
# data = {
#     "prompt": "pink bicycle",
#     "data": file_data_base64,
# }

# # Send a POST request to the endpoint
# response = requests.post(url, json=data, timeout=300)

# # Print the response
# print(response.json())

import argparse
import requests
from time import time

# Create the parser
parser = argparse.ArgumentParser(description="Send a prompt to an endpoint.")

# Add the arguments
parser.add_argument("prompt", metavar="mode", type=str)

# Parse the arguments
args = parser.parse_args()


# Send a POST request to the endpoint
start_time = time()
gen_response = requests.post("http://localhost:8093/generate/", data={
    "prompt": args.prompt,
}, timeout=600)
print(f"[INFO] It took: {(time() - start_time) / 60.0} min to create model")


# val_response = requests.post("http://localhost:8094/validate/", json={
#     "prompt": "pink bicycle",
#     "data": gen_response.content
# }, timeout=600)

# print(val_response.json())

# Print the response