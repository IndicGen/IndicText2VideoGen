from typing import Type
from crewai.tools import BaseTool
from pydantic import BaseModel, Field


class AudioToolInput(BaseModel):
    """Input schema for MyCustomTool."""
    temple: str = Field(..., description="Name of the temple")

class AudioTool(BaseTool):
    name: str = "Vector store extractor"
    description: str = "This tool is used for extracting the content from the vector store given the name of the temple"
    args_schema: Type[BaseModel] = AudioToolInput
    def _run(self, temple: str) -> str:
        
        return {"Message":"Audio is generated for the specific temple"}