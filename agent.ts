import 'dotenv/config'
import { TavilySearchResults } from "@langchain/community/tools/tavily_search";
import { ChatOpenAI } from "@langchain/openai";
import { HumanMessage, AIMessage } from "@langchain/core/messages";
import { ToolNode } from "@langchain/langgraph/prebuilt";
import { StateGraph, MessagesAnnotation } from "@langchain/langgraph";
import RapportJSON from "./assets/rapport.json"
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

const workflow = new StateGraph(MessagesAnnotation)
  .addNode("agent", callModel)
  .addEdge("__start__", "agent") 
  .addNode("tools", toolNode)
  .addEdge("tools", "agent")
  .addConditionalEdges("agent", shouldContinue);

const app = workflow.compile();
const rapport = JSON.stringify(RapportJSON);

const finalState = await app.invoke({
  messages: [new HumanMessage("Take a look at this json :" + rapport)],
});
console.log(finalState.messages[finalState.messages.length - 1].content);

const nextState = await app.invoke({
  messages: [...finalState.messages, new HumanMessage("Analyse the values and find the ones that are not normal (maybe too high or too low)")],
});
console.log(nextState.messages[nextState.messages.length - 1].content);

const lastState = await app.invoke({
  messages: [...finalState.messages, new HumanMessage("Give me recommandation on how to correct those values")],
});
console.log(lastState.messages[lastState.messages.length - 1].content);