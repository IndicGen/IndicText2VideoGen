import logging
from typing import Type, List, ClassVar
from crewai.tools import BaseTool
from pydantic import BaseModel, Field
from utils.vectorstore import VectorStoreHandler
from utils.logger_config import logger

class ExtractorToolInput(BaseModel):
    """Input schema for ExtractorTool."""
    temple: str = Field(..., description="Name of the temple")

class ExtractorTool(BaseTool):
    name: str = "Vector Store Extractor"
    description: str = "Extracts content from the vector store given the name of the temple."
    args_schema: Type[BaseModel] = ExtractorToolInput
    vector_store: ClassVar[VectorStoreHandler] = VectorStoreHandler()

    def _run(self, temple: str) -> List[str]:
        logger.info(f"Extracting content for temple: {temple}")
        try:
            temple_info = self.vector_store.get_documents(case_id=temple)
            logger.info(f"Successfully retrieved {len(temple_info)} documents for temple: {temple}")
            return {"temple_info": temple_info}
        except Exception as e:
            logger.error(f"Error retrieving documents for temple {temple}: {str(e)}")
            return {"error": str(e)}