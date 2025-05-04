using System;
using System.Collections.Generic;
using System.Linq;
using System.Net.Http;
using System.Text;
using System.Text.Json;
using System.Threading.Tasks;
using Terraria.Chat;
using Terraria.ID;
using Terraria.ModLoader;

namespace ai_helper.common
{
	// Please read https://github.com/tModLoader/tModLoader/wiki/Basic-tModLoader-Modding-Guide#mod-skeleton-contents for more information about the various files in a mod.
	public class ChatCommand : ModCommand
	{
		public override string Command => "aihelp";
		public override string Description => "Interface with locally running Ollama Model";

		public override string Usage => "/aihelp <question>";
		public override CommandType Type => CommandType.Chat;
		
		private record ResponseDto(string response);
		// Defining the action of our command
		public override async void Action(CommandCaller caller, string input, string[] args) {
			try 
			{
				// Define our URL to the server and set up our HTTP Client
				// Here we will later put in our own url to the locally running endpoint that will return the model response
				var url = "http://127.0.0.1:5000/ask";
				HttpClient client = new HttpClient();
				
				// Create the request body
				using StringContent content = new(
					JsonSerializer.Serialize( new
					{
						question = input
					}),
					Encoding.UTF8,
					"application/json");

				// Send a get request to the endpoint and await the response
				using HttpResponseMessage response = await client.PostAsync(url, content);
				response.EnsureSuccessStatusCode();

				// Decode the JSON response
    			var body = await response.Content.ReadAsStringAsync();
				var result = JsonSerializer.Deserialize<ResponseDto>(body);
				
				// Reply to the caller with the response, i.e. print out the response in the chat
				caller.Reply(result?.response ?? "Empty reply from server");

			} catch (Exception ex) 
			{
				caller.Reply($"AIHELP ERROR: {ex.Message}");
			}
		}
	}
}
