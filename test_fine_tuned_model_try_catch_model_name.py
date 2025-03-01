"""### Import libraries"""
# This version display loss function curve and save it to a file.
# 
#  
import google.generativeai as genai
import os
import logging
# from google.colab import userdata
# genai.configure(api_key=userdata.get('GOOGLE_API_KEY'))

api_key = os.getenv("GOOGLE_API_KEY")
print(api_key)

if not api_key:
    raise ValueError("API key not found. Please set the GOOGLE_API_KEY environment variable.")

genai.configure(api_key=os.environ['GOOGLE_API_KEY']) # Get the key from the GOOGLE_API_KEY env variable

"""You can check you existing tuned models with the `genai.list_tuned_model` method."""

for i, m in zip(range(5), genai.list_tuned_models()):
  print(m.name)

"""## Create tuned model

To create a tuned model, you need to pass your dataset to the model in the `genai.create_tuned_model` method. You can do this be directly defining the input and output values in the call or importing from a file into a dataframe to pass to the method.

"""

# model = genai.GenerativeModel(model_name=f'tunedModels/generate-num-313')#good result
model = genai.GenerativeModel(model_name=f'tunedModels/generate-num-313')
print(f"Using model: {model}")
logging.info(f"Using model: {model}")

description_tuned_model = genai.get_tuned_model(f'tunedModels/generate-num-313')
print("Description of the tuned model:")
print(description_tuned_model.description)


try:
    model = genai.GenerativeModel(model_name=f'tunedModels/generate-num-313')
    logging.info(f"Model {model} loaded successfully.")
except Exception as e:
    logging.error(f"Error loading model {model}: {e}")
    raise

# Generate content using the model
try:
    result = model.generate_content('61')
    print(result.text)

    result = model.generate_content('1234551')
    print(result.text)

    result = model.generate_content('nine')
    print(result.text)

    result = model.generate_content('piec') # Polish 5
    print(result.text)                        # Polish 6 is "sześć"

    result = model.generate_content('IV')    # Roman numeral 4
    print(result.text)                        # Roman numeral 5 is V

    result = model.generate_content('七')  # Japanese 7
    print(result.text)                     # Japanese 8 is 八!
except Exception as e:
    logging.error(f"Error generating content: {e}")
    raise



"""It really seems to have picked up the task despite the limited examples, but "next" is a simple concept,
 see the [tuning guide](https://ai.google.dev/gemini-api/docs/model-tuning) for more guidance on improving performance.

## Update the description

You can update the description of your tuned model any time using the `genai.update_tuned_model` method.
"""

# genai.update_tuned_model(f'tunedModels/{name}', {"description":"This is my model."});

# model = genai.get_tuned_model(f'tunedModels/{name}')

# print(model.description)

"""## Delete the model

You can clean up your tuned model list by deleting models you no longer need. Use the `genai.delete_tuned_model` method to delete a model. If you canceled any tuning jobs, you may want to delete those as their performance may be unpredictable.
"""

# genai.delete_tuned_model(f'tunedModels/{name}')

"""The model no longer exists:"""

# try:
#   m = genai.get_tuned_model(f'tunedModels/{name}')
#   print(m)
# except Exception as e:
#   print(f"{type(e)}: {e}")