"""### Import libraries"""
# This version display loss function curve and save it to a file.
# It also shows how to update the description of the model and how to delete the model.
# It also shows how to cancel the tuning job.
# It also shows how to generate content with the tuned model.
# It also shows how to list the tuned models.
# It also shows how to create a tuned model.
# It also shows how to check the tuning progress.
# It also shows how to evaluate the model.
# It also shows how to delete the model.
# It also shows how to update the description of the model.\
# 
#  It predict next number when user give a number as input.
#  Number can be written or in digit.
#  It can predict next number in sequence.
# Input could be in different languages like Japan, Polish French etc.
import google.generativeai as genai
import os
# from google.colab import userdata
# genai.configure(api_key=userdata.get('GOOGLE_API_KEY'))

# add first API key to the system using command:
# export GOOGLE_API_KEY="your_api_key_here"

#Than get this value in the program using os.getenv("GOOGLE_API_KEY")
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

For this example, you will tune a model to generate the previous number in the sequence. 
For example, if the input is `1`, the model should output `0`. 
If the input is `one hundred`, the output should be `ninety nine`.
"""

base_model = [
    m for m in genai.list_models()
    if "createTunedModel" in m.supported_generation_methods and
    "flash" in m.name][0]
base_model

import random
name = f'generate-num-{random.randint(0,10000)}'
operation = genai.create_tuned_model(
    # You can use a tuned model here too. Set `source_model="tunedModels/..."`
    source_model=base_model.name,
    training_data=[
        {
             'text_input': '1',
             'output': '0',
        },{
             'text_input': '3',
             'output': '2',
        },{
             'text_input': '-3',
             'output': '-4',
        },{
             'text_input': 'twenty two',
             'output': 'twenty one',
        },
        {
             'text_input': '251',
             'output': '250',
        },{
             'text_input': '2345',
             'output': '2544',
        },{
             'text_input': 'twenty nine',
             'output': 'twenty eight',
        },
        {
             'text_input': '324567',
             'output': '324566',
        },
        {
             'text_input': 'two hundred',
             'output': 'one hundred ninety nine',
        },{
             'text_input': 'ninety nine',
             'output': 'ninety eight',
        },{
             'text_input': '8',
             'output': '7',
        },{
             'text_input': '-98',
             'output': '-99',
        },{
             'text_input': '1,000',
             'output': '999',
        },{
             'text_input': '10,100,001',
             'output': '10,100,000',
        },{
             'text_input': 'thirteen',
             'output': 'twelve',
        },{
             'text_input': 'eighty',
             'output': 'seventy nine',
        },{
             'text_input': 'one',
             'output': 'zero',
        },{
             'text_input': 'three',
             'output': 'two',
        },{
             'text_input': 'seven',
             'output': 'six',
        }
    ],
    id = name,
    # epoch_count = 100,
    epoch_count = 100,
    batch_size=4,
    learning_rate=0.001,
)

"""Your tuned model is immediately added to the list of tuned models, but its status is set to "creating" while the model is tuned."""

model = genai.get_tuned_model(f'tunedModels/{name}')
print(model)

model.state

"""### Check tuning progress

Use `metadata` to check the state:
"""

print(operation.metadata)

"""Wait for the training to finish using `operation.result()`, or `operation.wait_bar()`"""

import time

for status in operation.wait_bar():
  time.sleep(30)

"""You can cancel your tuning job any time using the `cancel()` method. Uncomment the line below and run the code cell to cancel your job before it finishes."""

# operation.cancel()

"""Once the tuning is complete, you can view the loss curve from the tuning results. The [loss curve](https://ai.google.dev/gemini-api/docs/model-tuning#recommended_configurations) shows how much the model's predictions deviate from the ideal outputs."""

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

model = operation.result()

snapshots = pd.DataFrame(model.tuning_task.snapshots)

sns.lineplot(data=snapshots, x = 'epoch', y='mean_loss')



# Save the plot to a file
plt.savefig('loss_curve.png')

# Display the plot
plt.show()

"""## Evaluate your model

You can use the `genai.generate_content` method and specify the name of your model to test your model performance.
"""

model = genai.GenerativeModel(model_name=f'tunedModels/{name}')

result = model.generate_content('55')
print(result.text)

result = model.generate_content('123455')
print(result.text)

result = model.generate_content('four')
print(result.text)

result = model.generate_content('cinq') # French 5
print(result.text)                        # French 4 is "quatre"

result = model.generate_content('IV')    # Roman numeral 3
print(result.text)                        # Roman numeral 4 is IV

result = model.generate_content('八')  # Japanese 8
print(result.text)                     # Japanese 7 is '七'

"""It really seems to have picked up the task despite the limited examples,
 but "next" is a simple concept,
 see the [tuning guide](https://ai.google.dev/gemini-api/docs/model-tuning) 
 for more guidance on improving performance.

## Update the description

You can update the description of your tuned model any time using the `genai.update_tuned_model` method.
"""

genai.update_tuned_model(f'tunedModels/{name}', {"description":"Fine tune - decrease one."});

model = genai.get_tuned_model(f'tunedModels/{name}')
print("Model:")
print(model)
print("Description of the model:")
print(model.description)

"""## Delete the model

You can clean up your tuned model list by deleting models you no longer need. 
Use the `genai.delete_tuned_model` method to delete a model.
If you canceled any tuning jobs, 
you may want to delete those as their performance may be unpredictable.
"""

# genai.delete_tuned_model(f'tunedModels/{name}')

"""The model no longer exists:"""

# try:
#   m = genai.get_tuned_model(f'tunedModels/{name}')
#   print(m)
# except Exception as e:
#   print(f"{type(e)}: {e}")