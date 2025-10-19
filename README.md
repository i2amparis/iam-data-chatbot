# iam-data-chatbot
ChatGPT bot to provide access and analysis to IAM result data

~~~~~INFO~~~~
# create conda enviroment
conda create -n botenv python=3.10

# conda activation
conda activate botenv

# install required packages
pip install openai pandas openpyxl faiss-cpu numpy requests matplotlib

#save them
pip freeze > requirements.txt

#install packages
pip install -r requirements.txt

#run the bot to get answers from data
python mainv2.py

#run the bot to get answers from api
python mainv2.py --query "your question here"

#run the bot to get answers with graph
python mainv2.py --debug
