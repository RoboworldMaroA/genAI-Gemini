"""## Delete the model

You can clean up your tuned model list by deleting models you no longer need. Use the `genai.delete_tuned_model` method to delete a model. If you canceled any tuning jobs, you may want to delete those as their performance may be unpredictable.
"""
import google.generativeai as genai
import os

name = "generate-num-2402"


api_key = os.getenv("GOOGLE_API_KEY")
# print(api_key)# get the API key from the environment variable

if not api_key:
    raise ValueError("API key not found. Please set the GOOGLE_API_KEY environment variable.")

genai.configure(api_key=os.environ['GOOGLE_API_KEY']) # Get the key from the GOOGLE_API_KEY env variable

"""You can check you existing tuned models with the `genai.list_tuned_model` method."""
for i, m in zip(range(15), genai.list_tuned_models()):
  print(m.name)


#
genai.delete_tuned_model(f'tunedModels/{name}')

"""The model no longer exists:"""

try:
  m = genai.get_tuned_model(f'tunedModels/{name}')
  print(m)
except Exception as e:
  print(e)
  pass


