# https://roboworld.pl
# Marek Augustyn
# Date: 2025-03-01
# Base on the code from the Google AI Platform
# This script is used for testing the fine-tuned model.
# It loads the model from the account that will be used for prediction.
# It tests the model and prints the results.
# It could be used for model generate-num-313 or generate-num-8379.
#First model is trained to give you next element in the sequence.
#Second model is trained to give you previous element in the sequence.
import google.generativeai as genai
import os


api_key = os.getenv("GOOGLE_API_KEY")
# print(api_key)# get the API key from the environment variable

if not api_key:
    raise ValueError("API key not found. Please set the GOOGLE_API_KEY environment variable.")

genai.configure(api_key=os.environ['GOOGLE_API_KEY']) # Get the key from the GOOGLE_API_KEY env variable

"""You can check you existing tuned models with the `genai.list_tuned_model` method."""
for i, m in zip(range(15), genai.list_tuned_models()):
  print(m.name)


# load model from the account that will be used for prediction
# model = genai.GenerativeModel(model_name=f'tunedModels/generate-num-313')# result for prediction next bigger number
model = genai.GenerativeModel(model_name=f'tunedModels/generate-num-8379') # result for prediction next smaller number
print(f"Using model: {model}")

#test the model and print the results

result = model.generate_content('61')
print(result.text)

result = model.generate_content('1000')
print(result.text)

result = model.generate_content('12347876')
print(result.text)

result = model.generate_content('nine')
print(result.text)

result = model.generate_content('pięć')   # Polish 5
print(result.text)                        # Polish 6 is "sześć" , Polish 4 is "cztery"

result = model.generate_content('dos')    # Spanish numeral 2
print(result.text)                        # Spanish numeral 1 is "uno", Spanish numeral 3 is "tres"

result = model.generate_content('due')    # Italy numeral 2
print(result.text)                        # Italy numeral 1 is "uno", Italy numeral 3 is "tre"

result = model.generate_content('tre')    # Italy numeral 3
print(result.text)                        # Italy numeral 2 is "due", Italy numeral 4 is "quattro"

result = model.generate_content('七')      # Japanese 7
print(result.text)                         # Japanese 6 is "六"

result = model.generate_content('セブン')  # Japanese seven
print(result.text)                         # Japanese 6 is "六", Japanese 8 is "八"

