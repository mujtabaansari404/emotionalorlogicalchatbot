from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage, AIMessage
from typing import Annotated, Literal
from typing_extensions import TypedDict
from pydantic import BaseModel, Field

from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages

load_dotenv()

llm = init_chat_model(
    "openai/gpt-oss-120b",
    model_provider="groq"
)


class MessageClassifier(BaseModel):
    message_type: Literal["emotional", "logical"] = Field(
        ...,
        description="emotional or logical classification"
    )


class State(TypedDict):
    messages: Annotated[list, add_messages]
    message_type: str | None


def classify_message(state: State):
    last_msg = state["messages"][-1]

    classifier = llm.with_structured_output(MessageClassifier)

    result = classifier.invoke([
        {
            "role": "system",
            "content": (
                "Classify message as:\n"
                "- emotional: feelings, stress, personal issues\n"
                "- logical: facts, reasoning, solutions"
            )
        },
        {
            "role": "user",
            "content": last_msg.content
        }
    ])

    return {"message_type": result.message_type}

def therapist_agent(state: State):
    last_msg = state["messages"][-1]

    reply = llm.invoke([
        {
            "role": "system",
            "content": (
                "You are a compassionate therapist. "
                "Be empathetic, validate emotions, and guide reflection."
            )
        },
        {
            "role": "user",
            "content": last_msg.content
        }
    ])

    return {
        "messages": [
            AIMessage(content=reply.content)
        ]
    }


def logical_agent(state: State):
    last_msg = state["messages"][-1]

    reply = llm.invoke([
        {
            "role": "system",
            "content": (
                "You are a logical assistant. "
                "Give clear, factual, structured answers."
            )
        },
        {
            "role": "user",
            "content": last_msg.content
        }
    ])

    return {
        "messages": [
            AIMessage(content=reply.content)
        ]
    }


def route(state: State):
    return state.get("message_type", "logical")

graph_builder = StateGraph(State)

graph_builder.add_node("classifier", classify_message)
graph_builder.add_node("therapist", therapist_agent)
graph_builder.add_node("logical", logical_agent)

graph_builder.add_edge(START, "classifier")

graph_builder.add_conditional_edges(
    "classifier",
    route,
    {
        "emotional": "therapist",
        "logical": "logical"
    }
)

graph_builder.add_edge("therapist", END)
graph_builder.add_edge("logical", END)

graph = graph_builder.compile()


def run_chatbot():
    state = {
        "messages": [],
        "message_type": None
    }

    while True:
        user_input = input("\nYou: ")

        if user_input.lower() in ["exit", "quit"]:
            print("Bye 👋")
            break

        # add user message
        state["messages"].append(
            HumanMessage(content=user_input)
        )

        # run graph
        state = graph.invoke(state)

        # print last AI response
        print("\nAssistant:", state["messages"][-1].content)


if __name__ == "__main__":
    run_chatbot()