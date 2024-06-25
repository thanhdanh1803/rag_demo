from typing import Any, Optional, List
from langchain_core.vectorstores import VectorStoreRetriever
from langchain_core.documents import Document
from langchain_core.runnables.config import RunnableConfig
from langchain_core.pydantic_v1 import PrivateAttr


class FakeStoreRetriever(VectorStoreRetriever):
    _documents: List[Document] = PrivateAttr()

    def __init__(self, documents: List[Document], /, **data: Any):
        try:
            super().__init__(**data)
        except Exception as _:
            pass
        self._documents = documents

    def invoke(
        self, input: str, config: Optional[RunnableConfig] = None, **kwargs: Any
    ) -> List[Document]:
        return self._documents

    async def ainvoke(
        self,
        input: str,
        config: Optional[RunnableConfig] = None,
        **kwargs: Any,
    ) -> List[Document]:
        return self._documents
