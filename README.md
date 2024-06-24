This is a small project for rag and openAI demostration using AWS Opensearch as vector database.

### Config
* Export below environment variables: 
* AWS_PROFILE: your aws credential 
* OPENAI_API_KEY: your openAI key 
* USERNAME: UI username 
* PASSWORD: UI password 

### Run your app
* Activate venv: `source venv/bin/activate` 
* Install required packages: `python3 -m pip install .` 
* Deploy UI: `python3 src/ui/app.py` 
* You app will be deploy on local host at port 8080
