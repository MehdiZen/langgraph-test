import "dotenv/config";
import { TavilySearchResults } from "@langchain/community/tools/tavily_search";
import { ChatOpenAI } from "@langchain/openai";
import {
  HumanMessage,
  AIMessage,
  MessageContent,
} from "@langchain/core/messages";
import { ToolNode } from "@langchain/langgraph/prebuilt";
import { StateGraph, MessagesAnnotation } from "@langchain/langgraph";
import RapportJSON from "./assets/rapport.json";
import GoodRapportJSON from "./assets/goodRapport.json";
import TresholdJSON from "./assets/treshold.json";
import readline from "node:readline";

const rapport = JSON.stringify(RapportJSON);
const goodRapport = JSON.stringify(GoodRapportJSON);
const treshold = JSON.stringify(TresholdJSON);

const tools = [new TavilySearchResults({ maxResults: 3 })];
const toolNode = new ToolNode(tools);

const model = new ChatOpenAI({
  model: "gpt-4o-mini",
  temperature: 0,
});
// .bindTools(tools);

function shouldContinue({ messages }: typeof MessagesAnnotation.State) {
  const lastMessage = messages[messages.length - 1] as AIMessage;

  // if (lastMessage.tool_calls?.length) {
  //   return "tools";
  // }
  return "__end__";
}

async function callModel(state: typeof MessagesAnnotation.State) {
  const response = await model.invoke(state.messages);
  console.clear();
  console.log("Requesting an agent")
  return { messages: [response] };
}

async function analyzeGivenData(state: typeof MessagesAnnotation.State) {
  console.clear();
  console.log("Retrieving data...");
  const response = await model.invoke([
    ...state.messages,
    new HumanMessage(
      "Analyze the provided values."
    ),
  ]);
  return { messages: [response] };
}

async function compareTresholds(state: typeof MessagesAnnotation.State) {
  console.clear();
  console.log("Analyzing data...");
  const response = await model.invoke([
    ...state.messages,
    new HumanMessage(
      "Now you have to use the provided values and compare them against the thresholds object. Identify only those values that exceed the defined thresholds, are offline or are considered dangerous. If no values exceed a critical threshold or are considered dangerous, respond only with 'R.A.S.'. Here's the treshold : " +
        treshold
    ),
  ]);
  return { messages: [response] };
}

async function recommendations(state: typeof MessagesAnnotation.State) {
  console.clear();
  console.log("Critical value(s) found.\nGenerating recommendations...");
  const response = await model.invoke([
    ...state.messages,
    new HumanMessage(
      "Tell me how many objects were analyzed, how many values per categories were considered dangerous or exceeded the tresholds and give me recommendations on how to correct those values. I want it in this format : The category (and each service status if needed) | number of problems | 2 to 3 exemples of the highest values exceeding(unless it's offline). Then under this line write a recommendation. Here's an exemple :  'Disk Usage | Total Exceedances: 6 instances exceeded 90%. | Highest Values: 90%, 94%, 96%.' \n - Recommendation: Regularly monitor disk usage and implement data archiving strategies to free up space. Consider upgrading to larger disks or implementing a more efficient storage solution to handle high disk usage.' And don't write anything else"
    ),
  ]);
  return { messages: [response] };
}

async function asktheman(): Promise<string> {
  return new Promise((resolve) => {
    console.clear();
    const rl = readline.createInterface({
      input: process.stdin,
      output: process.stdout,
    });
    rl.question(
      "No issues found. Everything is normal.\nWould you still like to have recommendations? Y/N: ",
      (answer) => {
        answer = answer.toLowerCase();
        if (answer == "yes" || answer == "y") {
          resolve("recommendations");
        } else if (answer == "no" || answer == "n") {
          resolve("__end__");
        } else {
          console.log("Invalid input. Aborting.");
          resolve("__end__");
        }
        rl.close();
      }
    );
    rl.close();
  });
}

async function shouldRecommend({ messages }: typeof MessagesAnnotation.State) {
  if (messages[messages.length - 1].content.includes("R.A.S." as any)) {
    const result = (await asktheman()) as string;
    return result;
  }

  return "recommendations";
}

const workflow = new StateGraph(MessagesAnnotation)
  .addNode("agent", callModel)
  .addNode("analyze", analyzeGivenData)
  .addNode("compareTresholds", compareTresholds)
  .addNode("recommendations", recommendations)
  .addEdge("__start__", "agent")
  .addEdge("agent", "analyze")
  .addEdge("analyze", "compareTresholds")
  .addConditionalEdges("analyze", shouldRecommend)
  .addNode("tools", toolNode)
  .addEdge("tools", "agent")
  .addConditionalEdges("agent", shouldContinue)

const app = workflow.compile();

const finalState = await app.invoke({
  messages: [new HumanMessage(rapport)],
});

console.clear();
console.log(finalState.messages[finalState.messages.length - 1].content);

// TODO: Possibilitée de réponse de l'utilisateur à la fin du process, pour demander peut être + d'indications sur comment réaliser une des recommendation