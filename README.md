# NLP-Based Research Analytics Dashboard & RAG Chatbot  
Discover India Program | FLAME University

This project builds an intelligent research analytics platform for FLAME University’s Discover India Program (DIP), designed to improve accessibility and exploration of student research papers. The system integrates an interactive analytics dashboard with a Retrieval-Augmented Generation (RAG)–based conversational chatbot.

## Problem Statement
Searching across large collections of academic research is time-consuming and inefficient, particularly when insights are spread across multiple documents. Traditional keyword-based search fails to support thematic, geographic, and cross-document analysis.

## System Overview
The platform consists of two core components:
1. **Research Analytics Dashboard** for exploratory analysis
2. **RAG-Based Question Answering Chatbot** for conversational access to research

## Research Analytics Dashboard
- Applied **Natural Language Processing (NLP)** techniques to extract and visualize key themes, geographic references, and culturally relevant keywords from research papers.
- Built interactive visualizations to support rapid comparison across documents, regions, and research themes.
- Enabled students, faculty, and administrators to explore the research corpus without manual document review.

## RAG-Based Conversational Chatbot
- Implemented a **Retrieval-Augmented Generation (RAG)** pipeline using **Mistral embeddings** and **Pinecone vector database**.
- Indexed document chunks into a semantic vector space and used **cosine similarity** for relevant context retrieval.
- Generated **accurate, citation-linked responses**, ensuring answers remain grounded in the original research documents and minimizing hallucinations.
- Supported cross-document querying, allowing users to synthesize insights from multiple papers in a single interaction.

## Tools & Technologies
- Python
- Natural Language Processing (NLP)
- Mistral Embeddings
- Pinecone Vector Database
- Retrieval-Augmented Generation (RAG)
- Cosine Similarity
- Data Visualization Libraries

## Impact
- Significantly improved research discoverability across the DIP corpus.
- Reduced time required to extract insights from multiple research papers.
- Demonstrated a scalable framework for institutional knowledge management.

## Future Scope
- Expand support for additional document types and metadata.
- Improve ranking using feedback-based relevance scoring.
- Deploy as a university-wide academic research assistant.
