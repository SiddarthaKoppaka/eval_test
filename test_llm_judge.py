import os
from dotenv import load_dotenv

from langchain_aws.chat_models.bedrock import ChatBedrock
from langchain_core.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.pydantic_v1 import BaseModel, Field, validator
from langchain_core.runnables import RunnableLambda, RunnablePassthrough

from main import generate_structured_story

load_dotenv()

# ---- Bedrock credentials ----
AWS_ACCESS_KEY_ID = os.getenv("AWS_ACCESS_KEY_ID")
AWS_SECRET_ACCESS_KEY = os.getenv("AWS_SECRET_ACCESS_KEY")
AWS_SESSION_TOKEN = os.getenv("AWS_SESSION_TOKEN")
BEDROCK_REGION = os.getenv("BEDROCK_REGION", "us-east-1")
MODEL_ID = os.getenv("BEDROCK_MODEL_ID", "us.anthropic.claude-3-5-sonnet-20240620-v1:0")

# ---- Bedrock LLM (Judge) ----
llm_judge = ChatBedrock(
    model_id=MODEL_ID,
    region_name=BEDROCK_REGION,
    aws_access_key_id=AWS_ACCESS_KEY_ID,
    aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
    aws_session_token=AWS_SESSION_TOKEN,
)

# ---- Output schema ----
class EvalScores(BaseModel):
    Hallucination: int = Field(..., ge=1, le=100)
    Completeness: int = Field(..., ge=1, le=100)
    Relevancy: int = Field(..., ge=1, le=100)

    @validator("*", pre=True)
    def to_int(cls, v):
        try:
            return int(round(float(v)))
        except Exception:
            return v

parser = JsonOutputParser(pydantic_object=EvalScores)

# ---- Prompts (no raw JSON braces in the template!) ----
EVAL_SYSTEM = (
    "You are an intelligent LLM Response Evaluator. Evaluate the story model's response "
    "against the original user query. Check for Hallucinations, Relevancy, and Completeness.\n\n"
    "{format_instructions}"
)

EVAL_HUMAN = (
    "User Query:\n{user_query}\n\n"
    "Story Model Response:\n{story_response}\n"
)

eval_prompt = ChatPromptTemplate.from_messages(
    [
        SystemMessagePromptTemplate.from_template(EVAL_SYSTEM),
        HumanMessagePromptTemplate.from_template(EVAL_HUMAN),
    ]
).partial(format_instructions=parser.get_format_instructions())

# ---- Chain pieces ----
def _gen_story(inputs: dict) -> str:
    return generate_structured_story(inputs["user_query"])

# Full chain: user_query -> story -> judge -> parsed JSON
judge_chain = (
    RunnablePassthrough.assign(story_response=RunnableLambda(_gen_story))
    | eval_prompt
    | llm_judge
    | parser
)

def evaluate_answer(user_query: str) -> dict:
    result: EvalScores = judge_chain.invoke({"user_query": user_query})
    return result

# Optional: evaluate an already-generated response
direct_eval_chain = eval_prompt | llm_judge | parser

def evaluate_provided_answer(user_query: str, story_response: str) -> dict:
    result: EvalScores = direct_eval_chain.invoke(
        {"user_query": user_query, "story_response": story_response}
    )
    return result


q = " a time-traveling botanist discovering an extinct flower."
print(evaluate_answer(q))
