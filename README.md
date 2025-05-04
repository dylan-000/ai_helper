# Terraria AI Helper
### Description:
Llama3.2 AI helper companion for Terraria that operates on a RAG system with planning and other experimental tool-calling features.

### Requirements:
- Ollama installed with `Llama3.2:latest` and `nomic-embed-text` pulled
- Ollama package should also be installed to a new Conda env
- ChromaDB, Langchain, Flask, and other packages within the project should also be installed
### Usage:
- All of the necessary data to run the RAG system is included in this repo. Run `create_vectory_store.py` to create the Chromadb collection that the model will run on.
- Navigate to the Tmodloader directory on your computer and clone this repository to the 'ModSources' subdirectory.
- Open TmodLoader.
- Click the 'Workshop' button.
- Click 'Develop Mods'.
- You should see the repository you cloned listed as a mod source here. Click the 'Build + Reload' option. This will incorporate the Mod into your game.
- Navigate to vscode or an ide of your choice and run `api_server.py`. This will start the local server that the mod interfaces with.
- Go back to TmodLoader and load into a game.
- Use `aihelp <prompt>` to ask the model questions. Answers to your prompts will appear in the chat in-game.