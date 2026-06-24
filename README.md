# NoteForge

- after running some experiments in my private RAG repos I am making this one from scratch in public, showcasing all the steps I am taking
- I will also work on optimisng the outputs and reducing the hallucinations

- this file will serve as my personal documentation

### Todos:

#### Todos for v1:

- [done] Initialise a project

- [done] Project Structure

- [done] Document pre-processing

- [done] Vectorstore

- [done] State

- [done] Nodes and Graph structure
  - designing the whole workflow

- [done] deploy the basic version

#### Todos for v2:

- [done] Tavily
  - replaced Wikipedia

- [done] Evaluation
  - [done] Using Notebooklm to test the outputs for now
  - [done] Tool selection issue found and fixed (agent bypassed retriever for Tavily)
  - [done] Created Evaluations/ with bug → plan → fix workflow

- [to-do] Corrective strategies
  - [done] Implement MMR
    - in ./src/vector_store/store.py : `fetch_k = 30` for fetching 30 candidates and `lambda_mult = 0.7` to balance relevancy v/s diversity (0 -> pure diversity, 1 -> pure similarity)
  - [next] Better chunking
