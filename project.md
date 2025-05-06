# Project Report

## 1. Base System Functionality

The AI Terraria Guide system has been developed to handle a diverse range of user scenarios effectively. The model (Llama3.2) functions on a foundation of an extensive RAG library built from the Terraria Wiki, which we were able to web-scrape. The primary functionalities include:

* **Item crafting guidance** – The system provides step-by-step recipes and crafting station requirements.
* **Boss strategy support** – The model is able to recall information on the bosses in the game and offer advice on how the player should go about defeating the boss.
* **Game progression advice** – Players receive recommendations on what to pursue next.
* **Item and Entity Lookup** – Detailed descriptions and contextual background can be retrieved on demand for most items and entities in the game. In Terraria, there are over 5400 unique items, 28 different NPCs, and over 540 entities in the game.

Here are some examples of the various scenarios that our model can handle:
<Br>**Q**: How do I make a spelunker potion? 
<Br>**A**: 
```
To make a Spelunker Potion, you'll need to gather the following ingredients:

1. Blinkroot
2. Moonglow Seeds
3. Bottled Water
4. Gold Ore (or Platinum Ore)

You can find Blinkroot by exploring underground and keeping an eye out for it. Moonglow Seeds are easy to obtain in large quantities by planting them in Clay Pots, which only require nighttime to bloom.

The other ingredients, Bottled Water and either Gold Ore or Platinum Ore, are relatively easy to acquire through passive means while mining.
```

<Br>**Q**: How do I get the Rod of Discord? What does it do?
<Br>**A**: 
```
The Rod of Discord! To get your hands on this handy tool, you'll need to focus on defeating Chaos Elementals in Hardmode. These mobs have a 0.2% chance (or 0.25% in some cases) of dropping the Rod of Discord.

Since Chaos Elementals are relatively rare and can be found in caves, you might want to explore these areas with caution and a trusty pickaxe. Be prepared for a fight!

Once you've managed to obtain the Rod of Discord, you'll notice that it has a unique ability: instantly teleporting you to your mouse cursor using Shimmer Transmutation. However, keep an eye out - each use will inflict the Chaos State debuff on you, causing damage over time.

If you're wondering how to get rid of this debuff or what to do next with the Rod of Discord, I recommend checking out a more advanced Terraria guide or online resource for further information.

Do you have any other questions about the Rod of Discord or would you like some tips on how to farm Chaos Elementals?
```

<Br>**Q**: How do I enter into Hardmode?
<Br>**A**: 
```
To enter Hardmode in Terraria, you need to defeat the Wall of Flesh in the Underworld.

Here's a step-by-step guide:

1. Head to the Underworld: Make sure you have a way to access the Underworld, such as using a portal or finding a hidden entrance.
2. Locate and fight the Wall of Flesh: The Wall of Flesh is a formidable boss that can be found in the Underworld. Be prepared for a challenging fight!
3. Defeat the Wall of Flesh: After defeating the Wall of Flesh, you'll receive new weapons and accessories to boost your early Hardmode survivability.
4. Wait for the message: Once you've defeated the Wall of Flesh, a message will appear confirming that your world has officially entered Hardmode.

That's it! You should now be in Hardmode, ready to face the challenges that come with it.
```


## 2. Prompt Engineering and Model Parameter Choice

For the model's integration into our system, we used the following parameters for response generations:
* **Temperature**: 0.7 -> This allowed for a good balance between creativity and reliable behavior from the model given various prompts
* **Max Tokens**: 1024 -> This allowed the model to have more detailed responses that could incorporate more information from our RAG queries.
* **Number of Results**: 3 -> Although not a model-specific parameter, this parameter was defined in our query processor class (`processor.py`) and regulates the number of contexts pulled from the RAG system. We found that pulling the top-3 was optimal because we noticed that after 3 contexts, the information decreased significantly in relevancy and was confusing the model.

Our AI system is more of an assistant-style implementation designed to help the player, so the prompts were designed for that purpose rather than a more narrative-based approach that would coincide with the DnD style project. 

In `processor.py`, our system uses a prompt to guide a planning process for the query processing pipeline. In the `plan_query()` method, we use a tailored prompt to get the model to define a set of steps to guide the retrieval of useful information or optionally call tools. 

Another method (`format_rag_prompt()` located in `llm_utils.py`) was designed with a specific prompt to get the model to interpret the contexts retrieved from the RAG database.

Also in `llm_utils.py` is the `format_tool_prompt()`, which features a prompt specifically designed to allow our model to interface with tools.

Please navigate to the mentioned files to check the prompts that were implemented into our system.

## 3. Tools Usage

We attempted to define a set of tools that our model could interact with that would enhance the generation of the responses. We had originally thought that we could implement more dynamic and beneficial tools, such as the mod featuring an endpoint that the model would be able to send an async request to gather game-save specific data; however, some technical errors were faced, so we had to focus on more fundamental mechanics to our system.

Regardless, we still tried to implement a set of tools to experiment with tool-calling as a feature, so hopefully in the future, they could lay the groundwork for more helpful ones. The following tools were defined:

* `item_lookup`: Fetches information about specific items.
* `crafting_recipe`: Retrieves the recipe and components needed for item creation.
* `boss_info`: Provides background and strategies for individual bosses.
* `progression_check`: Offers tailored suggestions based on current game progress.

Each tool was defined using a JSON schema and integrated into the reasoning pipeline. Here’s an example tool call:

```json
{
  "tool": "crafting_recipe",
  "parameters": {"item_name": "Night's Edge"}
}
```

More on the tools can be found in the `tools.py`.

## 4. Planning & Reasoning

The system employed a multi-step reasoning approach, organized as follows:

1. **Query planning**: The LLM analyzed user questions to generate search terms and assess tool use.
2. **Information retrieval**: ChromaDB was queried to return top-ranked context chunks.
3. **Tool invocation**: If needed, tools were automatically called based on LLM response.
4. **Final synthesis**: Retrieved context and tool outputs were assembled into a natural language response.

This chain-of-thought pipeline helped the AI adapt its behavior based on the complexity of the query. For example, asking “How do I beat the Wall of Flesh as a ranged character?” triggers a reasoning chain involving class-specific advice and biome preparation.

## 5. RAG Implementation

The Ollama 3.2 model that we were using was never trained on Terraria-specific data. Because of this, it was critical that we have an extensive RAG dataset that our system could use. 

There is already a huge repository of information on the Terraria Wikipedia; however, harnessing this data and storing it for our model to use was trickier than we expected. Initially, we planned to create a web-scraper to pull data from every page of the wikipedia but many of the site's images and subject-based formatting made that difficult--combine that with the fact that there are 55,000 total pages over the entire wiki . . . and we knew we had to adjust our approach. 

In the end, we decided to narrow our focus to two main categories: items and entities. The page format stays relatively consistent across entries which allowed us to create a unique webscraper specific to each category. The scrapers were designed to automatically format each page's data to fit directly into our RAG dataset, which avoided the need to manually adjust data before being used by our model. In total we scraped and formatted nearly 6000 pages for our RAG: 5400 for items and 540 for entities.

With this data, our RAG was strong enough for our model to answer many of the most commonly asked questions about things like weapon options, crafting recipes, enemy stats, and drop chances.  

## 6. Additional Tools / Innovation

The creative aspect of our project is that we figured out a way to incorporate a helpful AI assistant into Terraria. 

Terraria is a game that has been around for a while and is close to the hearts of our team members -- each of us having countless hours into the game. The aspect of Terraria that many of us agree is tedious is that it has so many items and entities, so it is seemingly required to have the Wiki page opened to be able to do anything that you want to do. 

As a way of streamlining that aspect of the game, we thought that a RAG based AI assistant built on web-scraped wiki data would be able to provide a better sense of immersion in the game so the player doesn't have to constantly have the wiki open in another browser. Additionally, they would not have to search for items or guides online and could instead ask a simple question to the assistant, which would be able to recall the information from the ChromaDB collection.

The tricky part of this process was figuring out how to integrate the locally running Ollama model with the mod and formatting the model in a way that was asynchronous, so the game would not stop running while a user asked a question.

We landed on using a Flask server with a singular `/ask` endpoint for asking the model questions from the game. From the mod side, or client in this case, an in-game command was created that would send a simple async HTTP POST request to the Flask server, which would then respond back with the relevant information on what the user asked and include that in the game chat.

We would have liked to continue implementing more endpoints for additional tool-calling features, but as mentioned before, we wanted to make sure that the model's core functionality was the best we could have gotten it.

## 7. Code Quality & Modular Design

In order to maintain modularity, we created sub classes to encapsulate code and keep our main api_server script clean and easy to understand. The first of which is our QueryProcessor class which has all model interaction code. This class handles the steps of each query: planning, RAG, tool use, and generating the final response. This QueryProcessor class is further segmented into more methods found in our llm_utils script that handle formatting prompts and responses to match our desired structures. 

We also have a subclass that handles accessing the vector store for our RAG dataset. These vectors are updated once and stored within the mod files so that the server does not need to recreate them each time it is started. Also, the system is created so that we can add new collections of data if we wanted our system to have knowledge of other popular games, like Minecraft, Stardew Valley, Dungeons and Dragons, or Kingdom Come Deliverance.