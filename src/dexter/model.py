import os
import time
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import ChatPromptTemplate
from pydantic import BaseModel
from typing import Type, List, Optional
from langchain_core.tools import BaseTool
from langchain_core.messages import AIMessage
from google.api_core.exceptions import GoogleAPIError

from dexter.prompts import DEFAULT_SYSTEM_PROMPT

# Load environment variables
load_dotenv()

# Initialize the Gemini client
# Make sure your GOOGLE_API_KEY is set in your .env
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash", 
    temperature=0, 
    google_api_key=os.getenv("GOOGLE_API_KEY"),
    convert_system_message_to_human=True  # Important for Gemini compatibility
)

def call_llm(
    prompt: str,
    system_prompt: Optional[str] = None,
    output_schema: Optional[Type[BaseModel]] = None,
    tools: Optional[List[BaseTool]] = None,
) -> AIMessage:
  final_system_prompt = system_prompt if system_prompt else DEFAULT_SYSTEM_PROMPT
  
  prompt_template = ChatPromptTemplate.from_messages([
      ("system", final_system_prompt),
      ("user", "{prompt}")
  ])

  runnable = llm
  if output_schema:
      # Gemini 2.5 requires explicit json_mode or json_schema method
      try:
          runnable = llm.with_structured_output(output_schema, method="json_schema")
      except:
          # Fallback to default method if json_schema is not supported
          runnable = llm.with_structured_output(output_schema)
  elif tools:
      # Bind tools for Gemini function calling
      runnable = llm.bind_tools(tools)
  
  chain = prompt_template | runnable
  
  # Retry logic for transient connection errors
  for attempt in range(3):
      try:
          return chain.invoke({"prompt": prompt})
      except GoogleAPIError as e:
          if attempt == 2:  # Last attempt
              raise
          time.sleep(0.5 * (2 ** attempt))  # 0.5s, 1s backoff
