# iam-data-chatbot
ChatGPT bot to provide access and analysis to IAM result data

~~~~~INFO~~~~
create conda enviroment 
conda create -n botenv python=3.10

conda activation
conda activate botenv

install required packages
pip install openai pandas openpyxl faiss-cpu numpy

save them
pip freeze > requirements.txt


run the bot to get answers
python final.py