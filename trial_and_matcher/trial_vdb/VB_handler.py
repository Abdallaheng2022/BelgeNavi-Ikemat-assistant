from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Union
from dataclasses import dataclass
from enum import Enum
import numpy as np
from qdrant_client import QdrantClient
from dotenv import load_dotenv
import os

env_path = os.path.join('.env')
load_dotenv(env_path)


class VectorDBProvider(Enum):
    PINECONE = "pinecone"
    AZURE_AI_SEARCH = "azure_ai_search"
    WEAVIATE = "weaviate"
    QDRANT= "qdrant"
    CHROMA = "chroma"
    FAISS = "faiss"
    


@dataclass
class VectorDocument:
    """Standardized document representation"""
    id: str
    vector: List[float]
    metadata: Optional[Dict[str, Any]] = None
    content: Optional[str] = None

@dataclass
class SearchResult:
    """Standardized search result"""
    id: str
    score: float
    metadata: Optional[Dict[str, Any]] = None
    content: Optional[str] = None

@dataclass
class SearchQuery:
    """Standardized search query"""
    vector: List[float]
    top_k: int = 10
    filter_conditions: Optional[Dict[str, Any]] = None
    include_metadata: bool = True
    include_content: bool = True

class VectorDBInterface(ABC):
    """Abstract base class for all vector database providers"""
    
    @abstractmethod
    def connect(self, **kwargs) -> bool:
        """Establish connection to the vector database"""
        pass
    
    @abstractmethod
    def create_index(self, index_name: str, dimension: int, **kwargs) -> bool:
        """Create a new index/collection"""
        pass
    
    @abstractmethod
    def delete_index(self, index_name: str) -> bool:
        """Delete an index/collection"""
        pass
    
    @abstractmethod
    def insert(self, index_name: str, documents: List[VectorDocument]) -> bool:
        """Insert documents into the index"""
        pass
    
    @abstractmethod
    def search(self, index_name: str, query: SearchQuery) -> List[SearchResult]:
        """Search for similar vectors"""
        pass
    
    @abstractmethod
    def delete(self, index_name: str, document_ids: List[str]) -> bool:
        """Delete documents by IDs"""
        pass
    
    @abstractmethod
    def update(self, index_name: str, documents: List[VectorDocument]) -> bool:
        """Update existing documents"""
        pass
    
    @abstractmethod
    def get_stats(self, index_name: str) -> Dict[str, Any]:
        """Get index statistics"""
        pass

class VectorDBConfig:
    """Configuration class for different providers"""
    
    def __init__(self, provider: VectorDBProvider, **kwargs):
        self.provider = provider
        self.config = kwargs
    
    @classmethod
    def pinecone(cls, api_key: str, environment: str, **kwargs):
        return cls(VectorDBProvider.PINECONE, 
                  api_key=api_key, 
                  environment=environment, 
                  **kwargs)
    
    @classmethod
    def azure_ai_search(cls, endpoint: str, api_key: str, **kwargs):
        return cls(VectorDBProvider.AZURE_AI_SEARCH, 
                  endpoint=endpoint, 
                  api_key=api_key, 
                  **kwargs)
    @classmethod
    def qdrant(cls, url: str = "http://localhost:6333", api_key: Optional[str] = None, **kwargs):
        return cls(VectorDBProvider.QDRANT,
                  url=url,
                  api_key=api_key,
                  **kwargs)

    @classmethod
    def chroma(cls, host: str = "localhost", port: int = 8000, **kwargs):
        return cls(VectorDBProvider.CHROMA, 
                  host=host, 
                  port=port, 
                  **kwargs)
    
    @classmethod
    def faiss(cls, index_path: str = "./faiss_index", **kwargs):
        return cls(VectorDBProvider.FAISS, 
                  index_path=index_path, 
                  **kwargs)

class VectorDBWrapper:
    """Main wrapper class that provides unified interface"""
    
    def __init__(self, config: VectorDBConfig):
        self.config = config
        self.provider = self._create_provider()
    
    def _create_provider(self) -> VectorDBInterface:
        """Factory method to create appropriate provider instance"""
        if self.config.provider == VectorDBProvider.PINECONE:
            return PineconeProvider(**self.config.config)
        elif self.config.provider == VectorDBProvider.AZURE_AI_SEARCH:
            return AzureAISearchProvider(**self.config.config)
        elif self.config.provider == VectorDBProvider.CHROMA:
            return ChromaProvider(**self.config.config)
        elif self.config.provider == VectorDBProvider.FAISS:
            return FaissProvider(**self.config.config)
        elif self.config.provider == VectorDBProvider.QDRANT:
            return QdrantProvider(**self.config.config)
        else:
            raise ValueError(f"Unsupported provider: {self.config.provider}")
    
    def connect(self) -> bool:
        """Connect to the vector database"""
        return self.provider.connect()
    
    def create_index(self, index_name: str, dimension: int, **kwargs) -> bool:
        """Create a new index"""
        return self.provider.create_index(index_name, dimension, **kwargs)
    
    def insert(self, index_name: str, documents: List[VectorDocument]) -> bool:
        """Insert documents"""
        return self.provider.insert(index_name, documents)
    
    def search(self, index_name: str, query: SearchQuery) -> List[SearchResult]:
        """Search for similar vectors"""
        return self.provider.search(index_name, query)
    
    def delete(self, index_name: str, document_ids: List[str]) -> bool:
        """Delete documents"""
        return self.provider.delete(index_name, document_ids)
    
    def update(self, index_name: str, documents: List[VectorDocument]) -> bool:
        """Update documents"""
        return self.provider.update(index_name, documents)
    
    def get_stats(self, index_name: str) -> Dict[str, Any]:
        """Get index statistics"""
        return self.provider.get_stats(index_name)

# Provider implementations (stubs for now)
class PineconeProvider(VectorDBInterface):
    def __init__(self, api_key: str, environment: str, **kwargs):
        self.api_key = api_key
        self.environment = environment
        self.client = None
    
    def connect(self) -> bool:
        # Implementation would use pinecone-client
        print(f"Connecting to Pinecone with environment: {self.environment}")
        return True
    
    def create_index(self, index_name: str, dimension: int, **kwargs) -> bool:
        print(f"Creating Pinecone index: {index_name} with dimension: {dimension}")
        return True
    
    def delete_index(self, index_name: str) -> bool:
        print(f"Deleting Pinecone index: {index_name}")
        return True
    
    def insert(self, index_name: str, documents: List[VectorDocument]) -> bool:
        print(f"Inserting {len(documents)} documents to Pinecone index: {index_name}")
        return True
    
    def search(self, index_name: str, query: SearchQuery) -> List[SearchResult]:
        print(f"Searching Pinecone index: {index_name} with top_k: {query.top_k}")
        # Mock results
        return [SearchResult(id="doc1", score=0.95), SearchResult(id="doc2", score=0.87)]
    
    def delete(self, index_name: str, document_ids: List[str]) -> bool:
        print(f"Deleting {len(document_ids)} documents from Pinecone index: {index_name}")
        return True
    
    def update(self, index_name: str, documents: List[VectorDocument]) -> bool:
        print(f"Updating {len(documents)} documents in Pinecone index: {index_name}")
        return True
    
    def get_stats(self, index_name: str) -> Dict[str, Any]:
        return {"provider": "pinecone", "index": index_name, "document_count": 1000}

class AzureAISearchProvider(VectorDBInterface):
    def __init__(self, endpoint: str, api_key: str, **kwargs):
        self.endpoint = endpoint
        self.api_key = api_key
        self.client = None
    
    def connect(self) -> bool:
        print(f"Connecting to Azure AI Search at: {self.endpoint}")
        return True
    
    def create_index(self, index_name: str, dimension: int, **kwargs) -> bool:
        print(f"Creating Azure AI Search index: {index_name} with dimension: {dimension}")
        return True
    
    def delete_index(self, index_name: str) -> bool:
        print(f"Deleting Azure AI Search index: {index_name}")
        return True
    
    def insert(self, index_name: str, documents: List[VectorDocument]) -> bool:
        print(f"Inserting {len(documents)} documents to Azure AI Search index: {index_name}")
        return True
    
    def search(self, index_name: str, query: SearchQuery) -> List[SearchResult]:
        print(f"Searching Azure AI Search index: {index_name} with top_k: {query.top_k}")
        return [SearchResult(id="doc1", score=0.92), SearchResult(id="doc2", score=0.84)]
    
    def delete(self, index_name: str, document_ids: List[str]) -> bool:
        print(f"Deleting {len(document_ids)} documents from Azure AI Search index: {index_name}")
        return True
    
    def update(self, index_name: str, documents: List[VectorDocument]) -> bool:
        print(f"Updating {len(documents)} documents in Azure AI Search index: {index_name}")
        return True
    
    def get_stats(self, index_name: str) -> Dict[str, Any]:
        return {"provider": "azure_ai_search", "index": index_name, "document_count": 800}

class QdrantProvider(VectorDBInterface):
    def __init__(self, url: str, api_key: Optional[str] = None, **kwargs):
        self.url = url
        self.api_key = api_key
        self.client = None
    
    def connect(self) -> Optional[QdrantClient]:
        try:         
                qdrant_client = QdrantClient(
                    url=self.url,
                    api_key=self.api_key
                )
                if qdrant_client:
                   
                    return True
                else:
                    return False
        except:
               return False
          
 
    
    def create_index(self, index_name: str, dimension: int, **kwargs) -> bool:
        print(f"Creating Qdrant collection: {index_name} with dimension: {dimension}")
        return True
    
    def delete_index(self, index_name: str) -> bool:
        print(f"Deleting Qdrant collection: {index_name}")
        return True
    
    def insert(self, index_name: str, documents: List[VectorDocument]) -> bool:
        print(f"Inserting {len(documents)} documents to Qdrant collection: {index_name}")
        return True
    
    def search(self, index_name: str, query: SearchQuery) -> List[SearchResult]:
        print(f"Searching Qdrant collection: {index_name} with top_k: {query.top_k}")
        return [SearchResult(id="doc1", score=0.96), SearchResult(id="doc2", score=0.91)]
    
    def delete(self, index_name: str, document_ids: List[str]) -> bool:
        print(f"Deleting {len(document_ids)} documents from Qdrant collection: {index_name}")
        return True
    
    def update(self, index_name: str, documents: List[VectorDocument]) -> bool:
        print(f"Updating {len(documents)} documents in Qdrant collection: {index_name}")
        return True
    
    def get_stats(self, index_name: str) -> Dict[str, Any]:
        return {"provider": "qdrant", "collection": index_name, "document_count": 1500}

# Example usage

class ChromaProvider(VectorDBInterface):
    def __init__(self, host: str, port: int, **kwargs):
        self.host = host
        self.port = port
        self.client = None
    
    def connect(self) -> bool:
        print(f"Connecting to Chroma at: {self.host}:{self.port}")
        return True
    
    def create_index(self, index_name: str, dimension: int, **kwargs) -> bool:
        print(f"Creating Chroma collection: {index_name} with dimension: {dimension}")
        return True
    
    def delete_index(self, index_name: str) -> bool:
        print(f"Deleting Chroma collection: {index_name}")
        return True
    
    def insert(self, index_name: str, documents: List[VectorDocument]) -> bool:
        print(f"Inserting {len(documents)} documents to Chroma collection: {index_name}")
        return True
    
    def search(self, index_name: str, query: SearchQuery) -> List[SearchResult]:
        print(f"Searching Chroma collection: {index_name} with top_k: {query.top_k}")
        return [SearchResult(id="doc1", score=0.89), SearchResult(id="doc2", score=0.81)]
    
    def delete(self, index_name: str, document_ids: List[str]) -> bool:
        print(f"Deleting {len(document_ids)} documents from Chroma collection: {index_name}")
        return True
    
    def update(self, index_name: str, documents: List[VectorDocument]) -> bool:
        print(f"Updating {len(documents)} documents in Chroma collection: {index_name}")
        return True
    
    def get_stats(self, index_name: str) -> Dict[str, Any]:
        return {"provider": "chroma", "collection": index_name, "document_count": 500}

class FaissProvider(VectorDBInterface):
    def __init__(self, index_path: str, **kwargs):
        self.index_path = index_path
        self.index = None
    
    def connect(self) -> bool:
        print(f"Connecting to FAISS at path: {self.index_path}")
        return True
    
    def create_index(self, index_name: str, dimension: int, **kwargs) -> bool:
        print(f"Creating FAISS index: {index_name} with dimension: {dimension}")
        return True
    
    def delete_index(self, index_name: str) -> bool:
        print(f"Deleting FAISS index: {index_name}")
        return True
    
    def insert(self, index_name: str, documents: List[VectorDocument]) -> bool:
        print(f"Inserting {len(documents)} documents to FAISS index: {index_name}")
        return True
    
    def search(self, index_name: str, query: SearchQuery) -> List[SearchResult]:
        print(f"Searching FAISS index: {index_name} with top_k: {query.top_k}")
        return [SearchResult(id="doc1", score=0.94), SearchResult(id="doc2", score=0.88)]
    
    def delete(self, index_name: str, document_ids: List[str]) -> bool:
        print(f"Deleting {len(document_ids)} documents from FAISS index: {index_name}")
        return True
    
    def update(self, index_name: str, documents: List[VectorDocument]) -> bool:
        print(f"Updating {len(documents)} documents in FAISS index: {index_name}")
        return True
    
    def get_stats(self, index_name: str) -> Dict[str, Any]:
        return {"provider": "faiss", "index": index_name, "document_count": 2000}

# Example usage
if __name__ == "__main__":
    qdrant_config = VectorDBConfig.qdrant(
       url=os.getenv("QD_END_POINT"),
       api_key=os.getenv("QD_API_KEY")
      )  
     
    db=VectorDBWrapper(qdrant_config)
    if db.connect():
        print("Qdrant connected sucess")
   
    """
    # Example with Pinecone
    pinecone_config = VectorDBConfig.pinecone(
        api_key="your-api-key",
        environment="us-west1-gcp"
    )
    
    # Example with Pinecone
    azure_config = VectorDBConfig.azure_ai_search(
        api_key="your-api-key",
        environment="us-west1-gcp"
    )
    
    db = VectorDBWrapper(pinecone_config)
    db.connect()
    
     



    db = VectorDBWrapper(pinecone_config)
    db.connect()
    
    # Create sample documents
    documents = [
        VectorDocument(
            id="doc1",
            vector=[0.1, 0.2, 0.3],
            metadata={"category": "tech", "author": "John"},
            content="This is a tech document"
        ),
        VectorDocument(
            id="doc2", 
            vector=[0.4, 0.5, 0.6],
            metadata={"category": "science", "author": "Jane"},
            content="This is a science document"
        )
    ]
    
    # Basic operations
    db.create_index("test-index", dimension=3)
    db.insert("test-index", documents)
    
    # Search
    query = SearchQuery(
        vector=[0.1, 0.2, 0.3],
        top_k=5,
        filter_conditions={"category": "tech"}
    )
    results = db.search("test-index", query)
    
    print("Search results:")
    for result in results:
        print(f"ID: {result.id}, Score: {result.score}")
    
    # Get stats
    stats = db.get_stats("test-index")
    print(f"Index stats: {stats}")
    """