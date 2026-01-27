# from langchain_classic.agents import Agent
from dotenv import load_dotenv

load_dotenv()

from langchain_classic import hub
from langchain_classic.agents import AgentExecutor
from langchain_classic.agents.react.agent import create_react_agent
from langchain_openai import ChatOpenAI
from langchain_tavily import TavilySearch

# from langchain_core.output_parsers.pydantic import PydanticOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableLambda

from prompt import REACT_PROMPT_WITH_FORMAT_INSTRUCTIONS
from schemas import AgentRespone



tools = [TavilySearch()]
llm = ChatOpenAI(model="gpt-4o-mini")
# react_prompt = hub.pull("hwchase17/react")
# output_parser = PydanticOutputParser(pydantic_object = AgentRespone)

structured_llm = llm.with_structured_output(AgentRespone)

react_prompt_with_format_instructions = PromptTemplate(
    template = REACT_PROMPT_WITH_FORMAT_INSTRUCTIONS,
    input_variables = ["input","agent_scratchpad","tool_names"]
).partial(format_instructions = "")


agent = create_react_agent(
    tools=tools,
    llm=llm,
    prompt=react_prompt_with_format_instructions,
)


agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True,handle_parsing_errors=True
)
extract_output = RunnableLambda(lambda x : x["output"])
# parse_output = RunnableLambda(lambda x : output_parser.parse(x))



chain = agent_executor | extract_output | structured_llm


def main():
    print("Hello from langchain-course!")
    result = chain.invoke(
        input={
            "input": "Top Goal Scorer in Soccerr in entire history of the sport with number of goals and assists"
        }
    )
    print(result)


if __name__ == "__main__":
    main()
