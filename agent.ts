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

const tools = [new TavilySearchResults({ maxResults: 3 })];
const toolNode = new ToolNode(tools);

const model = new ChatOpenAI({
  model: "gpt-4o-mini",
  temperature: 0,
}).bindTools(tools);

function shouldContinue({ messages }: typeof MessagesAnnotation.State) {
  const lastMessage = messages[messages.length - 1] as AIMessage;

  if (lastMessage.tool_calls?.length) {
    return "tools";
  }
  return "__end__";
}

async function callModel(state: typeof MessagesAnnotation.State) {
  const response = await model.invoke(state.messages);

  return { messages: [response] };
}
async function analyze(state: typeof MessagesAnnotation.State) {
  const response = await model.invoke([
    ...state.messages,
    new HumanMessage(
      "Analyze the provided values and identify only those that are abnormally high or dangerous. If no values exceed a critical threshold or are considered dangerous Do not provide any additional explanation if everything is normal. Respond only with 'R.A.S.'. ignore values that, while potentially high, do not pose a significant risk or anomaly."
    ),
  ]);
  return { messages: [response] };
}

//Todo: Mieux structurer l'output
async function recommendations(state: typeof MessagesAnnotation.State) {
  const response = await model.invoke([
    ...state.messages,
    new HumanMessage("Give me recommendations on how to correct those values"),
  ]);
  return { messages: [response] };
}

async function shouldRecommend({ messages }: typeof MessagesAnnotation.State) {
  if (messages[messages.length - 1].content.includes("R.A.S." as any)) {
    messages[messages.length - 1].content = "No issues found. Everything is normal.";
    //Todo: ajouter un input user pour savoir s'il veut quand même des recommandations ?
    return "__end__";
  }
  return "recommendations";
}

const workflow = new StateGraph(MessagesAnnotation)
  .addNode("agent", callModel)
  .addNode("analyze", analyze)
  .addNode("recommendations", recommendations)
  .addEdge("__start__", "agent")
  .addEdge("agent", "analyze")
  .addConditionalEdges("analyze", shouldRecommend)
  .addNode("tools", toolNode)
  .addEdge("tools", "agent")
  .addConditionalEdges("agent", shouldContinue);

const app = workflow.compile();
const rapport = JSON.stringify(RapportJSON);
const goodRapport = JSON.stringify(GoodRapportJSON);

const finalState = await app.invoke({
  messages: [new HumanMessage(rapport)],
});

console.log(finalState.messages[finalState.messages.length - 1].content);
