from pprint import pprint
from typing import List

from langchain.retrievers import EnsembleRetriever
from langchain_chroma import Chroma

# from langchain_community.chat_models import ChatOllama
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.llms.ollama import Ollama
from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document
import pandas as pd
from tqdm import tqdm

template = """Utilice las siguientes piezas de contexto para responder la pregunta al final.
Si no sabe la respuesta, simplemente diga que no la sabe, no intente inventar una respuesta.
Utilice tres oraciones como máximo y mantenga la respuesta lo más concisa posible.
Di siempre "¡gracias por preguntar!" al final de la respuesta.

{context}

Pregunta: {question}

Respuesta:"""


class TabiPipeline:
    vector_store_persistent_path = "assets/vector_store"

    def __init__(
        self,
        book_path: str = "assets/TABI.pdf",
        ollama_embeddings: str = "nomic-embed-text",
        llm_model: str = "llama3.1:8b",
    ):
        self.embeddings = OllamaEmbeddings(model=ollama_embeddings)
        self.book_path = book_path
        self.vector_store: Chroma = Chroma(
            collection_name="tabi",
            embedding_function=self.embeddings,
            persist_directory=self.vector_store_persistent_path,
        )
        self.bm25_retriever: BM25Retriever = None
        self.llm = Ollama(
            model=llm_model,
            temperature=0.1,
            # repeat_last_n=-1,
            top_k=20,
            top_p=0.5,
        )

        # inicializar los índices
        self.create_book_bm25()
        self.create_book_vector_store()

    @property
    def loader(self):
        return PyPDFLoader(self.book_path)

    @property
    def ensemble_retriever(self):
        return EnsembleRetriever(
            retrievers=[self.bm25_retriever, self.chroma_retriever],
            weights=[0.5, 0.5],
        )

    @property
    def chroma_retriever(self):
        return self.vector_store.as_retriever()

    def create_book_vector_store(
        self,
    ):
        if self.vector_store._client.get_collection("tabi").count() > 0:
            return

        pages = self.loader.load_and_split()
        self.vector_store.add_documents(pages)

    def create_book_bm25(self):
        pages = self.loader.load_and_split()
        self.bm25_retriever = BM25Retriever.from_documents(pages)

    def get_template_parsed(self, docs: List[Document]) -> str:
        template = """CONTEXTO {index}\n\n {content}"""
        return 10 * "_" + "\n\n".join(
            [
                template.format(content=doc.page_content, index=index)
                for index, doc in enumerate(docs)
            ]
        )

    def rag_with_chroma(self, question: str) -> dict:
        output = self.vector_store.similarity_search(question, k=3)

        return dict(
            question=question,
            response=self.llm.invoke(
                template.format(
                    context=self.get_template_parsed(output),
                    question=question,
                )
            ),
        )

    def rag_with_bm25(self, question: str) -> dict:
        output = self.bm25_retriever.invoke(
            question,
            config={"configurable": {"search_kwargs_bm25": {"k": 3}}},
        )

        return dict(
            question=question,
            response=self.llm.invoke(
                template.format(
                    context=self.get_template_parsed(output),
                    question=question,
                )
            ),
        )

    def rag_with_ensemble(self, question: str) -> dict:
        output = self.ensemble_retriever.invoke(question)

        return dict(
            question=question,
            response=self.llm.invoke(
                template.format(
                    context=self.get_template_parsed(output),
                    question=question,
                )
            ),
        )


if __name__ == "__main__":
    tp = TabiPipeline()

    questions = [
        "¿Qué es machine learning?",
        "¿Qué es el Business Intelligence?",
        "¿Qué es una dimensión?",
        "¿Qué es un cubo?",
        "¿Quién es Ralph Kimball?",
    ]
    answers = []
    for question in tqdm(questions):
        answers.append(
            dict(
                question=question,
                bm25_response=tp.rag_with_bm25(question)["response"],
                chroma_response=tp.rag_with_chroma(question)["response"],
            )
        )

    pprint(answers)
    pd.DataFrame(answers).to_parquet("assets/preguntas_resueltas.parquet")
