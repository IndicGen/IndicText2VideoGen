from typing import Type
from crewai.tools import BaseTool
from pydantic import BaseModel, Field
from utils.vectorstore import VectorStoreHandler
from typing import List,ClassVar


class ExtractorToolInput(BaseModel):
    """Input schema for MyCustomTool."""
    temple: str = Field(..., description="Name of the temple")

class ExtractorTool(BaseTool):
    name: str = "Vector store extractor"
    description: str = "This tool is used for extracting the content from the vector store given the name of the temple"
    args_schema: Type[BaseModel] = ExtractorToolInput
    vector_store: ClassVar[VectorStoreHandler] = VectorStoreHandler()

    def _run(self, temple: str) -> List[str]:
        return {"temple_info": self.vector_store.get_documents(case_id = temple)}