# VectorDB — Build a Vector Database from Scratch in Python

A fully working **Vector Database** built from scratch in Python with a beautiful web UI.  
Implements **HNSW**, **KD-Tree**, and **Brute Force** search algorithms side-by-side, plus a **RAG pipeline** powered by a local LLM via Ollama.

> Built as an educational project to show how production vector databases like Pinecone, Weaviate, and Chroma actually work under the hood.

---

## What This Project Does

| Feature                     | Description                                                                     |
| --------------------------- | ------------------------------------------------------------------------------- |
| **3 Search Algorithms**     | HNSW (production-grade), KD-Tree, Brute Force — run all three and compare speed |
| **3 Distance Metrics**      | Cosine similarity, Euclidean distance, Manhattan distance                       |
| **16D Demo Vectors**        | 20 pre-loaded semantic vectors across 4 categories (CS, Math, Food, Sports)     |
| **2D PCA Scatter Plot**     | Live visualization of semantic space — watch clusters form                      |
| **Real Document Embedding** | Paste any text → Ollama embeds it with `nomic-embed-text` (768D)                |
| **RAG Pipeline**            | Ask questions about your documents → HNSW retrieves context → local LLM answers |
| **Full REST API**           | CRUD endpoints: insert, delete, search, benchmark, hnsw-info                    |

---

## How It Works

```
Your Text
    │
    ▼
Ollama (nomic-embed-text)          ← converts text to a 768-dimensional vector
    │
    ▼
HNSW Index (Python)                ← indexes the vector in a multilayer graph
    │
    ▼
Semantic Search                    ← finds nearest neighbors in vector space
    │
    ▼
Ollama (llama3.2)                  ← reads retrieved chunks, generates an answer
    │
    ▼
Answer
```

**HNSW (Hierarchical Navigable Small World)** is the same algorithm used by Pinecone, Weaviate, Chroma, and Milvus. It builds a multilayer graph where each layer is progressively sparser — searches start at the top layer and zoom in, achieving O(log N) complexity instead of O(N) for brute force.

---

## Prerequisites

You need **3 things** installed on your Windows laptop:

1. **Python 3.11+** (for running the Flask backend)
2. **Git** (to clone the repository)
3. **Ollama** (runs the local AI models for embeddings & generation)

---

## Step-by-Step Setup (Windows)

### Step 1 — Install Python

1. Go to **https://www.python.org** and download **Python 3.11** or higher
2. Run the installer
3. **IMPORTANT:** Check the box **"Add Python to PATH"** before clicking Install
4. Open **PowerShell** and verify:

```powershell
python --version
```

You should see `Python 3.11.x` or higher.

---

### Step 2 — Install Git

1. Go to **https://git-scm.com/download/win** and download Git for Windows
2. Run the installer with default settings
3. Verify in PowerShell:

```powershell
git --version
```

---

### Step 3 — Install Ollama (Local AI Models)

1. Go to **https://ollama.com** and click **Download for Windows**
2. Run the installer
3. Ollama starts automatically in the system tray
4. Open **PowerShell** and pull the two required models:

```powershell
C:\Users\YourUsername\AppData\Local\Programs\Ollama\ollama.exe pull nomic-embed-text
```

_(~274 MB — embedding model)_

```powershell
C:\Users\YourUsername\AppData\Local\Programs\Ollama\ollama.exe pull llama3.2
```

_(~2 GB — language model for generation)_

5. Verify Ollama is running:

```powershell
C:\Users\YourUsername\AppData\Local\Programs\Ollama\ollama.exe list
```

You should see both models listed.

> **Minimum specs:** 8GB RAM recommended. The models use ~3GB total.

---

### Step 4 — Clone the Repository

Open **PowerShell** and run:

```powershell
git clone https://github.com/YOUR_USERNAME/VectorDB.git
cd VectorDB
```

---

### Step 5 — Create a Python Virtual Environment

Inside the `VectorDB` folder:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

If you get an execution policy error, run:

```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

Then activate the venv again.

---

### Step 6 — Install Dependencies

```powershell
pip install -r requirements.txt
```

This installs:

- **Flask** — web server
- **requests** — HTTP client for Ollama API

---

### Step 7 — Run the Server

```powershell
python app.py
```

You should see:

```
=== VectorDB Engine ===
http://localhost:8080
20 demo vectors | 16 dims | HNSW+KD-Tree+BruteForce
Ollama: ONLINE
 * Running on http://127.0.0.1:8080
```

**Open your browser** and go to:

```
http://localhost:8080
```

---

## Using the Application

### Tab 1: Search (Demo Vectors)

- Type any concept in the search box: `binary tree`, `sushi`, `basketball`, `calculus`
- Choose your algorithm: **HNSW**, **KD-Tree**, or **Brute Force**
- Choose distance metric: **Cosine**, **Euclidean**, or **Manhattan**
- Click **⚡ SEARCH** — results appear with distances, the matching point glows on the scatter plot
- Click **▶ COMPARE ALL ALGOS** to run all 3 algorithms and compare their speed

**The scatter plot** shows all 20 vectors projected to 2D using PCA. Notice how the 4 semantic categories (CS, Math, Food, Sports) form distinct clusters — this is what "semantic similarity" looks like visually.

### Tab 2: Documents (Real Embeddings)

This uses Ollama to generate **real 768-dimensional embeddings** from any text.

1. Type a title (e.g., `Operating Systems Notes`)
2. Paste any text — lecture notes, textbook paragraphs, Wikipedia articles
3. Click **⚡ EMBED & INSERT**
4. Long documents are automatically split into overlapping 250-word chunks
5. Each chunk gets its own embedding and is stored in a separate HNSW index

### Tab 3: Ask AI (RAG Pipeline)

1. Make sure you have inserted some documents in Tab 2 first
2. Type a question about your documents
3. Click **🤖 ASK AI**

What happens behind the scenes:

```
1. Your question → embedded with nomic-embed-text (768D vector)
2. HNSW search → finds 3 most semantically similar chunks
3. Retrieved chunks → sent as context to llama3.2
4. llama3.2 → generates an answer based only on your documents
```

The answer streams in with a typewriter effect. Click the **context chips** to see exactly which chunks the AI used.

---

## REST API Reference

The server exposes a full REST API at `http://localhost:8080`.

### Demo Vector Endpoints

| Method   | Endpoint                                          | Description                          |
| -------- | ------------------------------------------------- | ------------------------------------ |
| `GET`    | `/search?v=f1,f2,...&k=5&metric=cosine&algo=hnsw` | K-NN search                          |
| `POST`   | `/insert`                                         | Insert a demo vector                 |
| `DELETE` | `/delete/:id`                                     | Delete by ID                         |
| `GET`    | `/items`                                          | List all demo vectors                |
| `GET`    | `/benchmark?v=...&k=5&metric=cosine`              | Compare all 3 algorithms             |
| `GET`    | `/hnsw-info`                                      | HNSW graph structure and layer stats |
| `GET`    | `/stats`                                          | Database statistics                  |

### Document & RAG Endpoints

| Method   | Endpoint          | Body                           | Description                  |
| -------- | ----------------- | ------------------------------ | ---------------------------- |
| `POST`   | `/doc/insert`     | `{"title":"...","text":"..."}` | Embed and store document     |
| `GET`    | `/doc/list`       | —                              | List all stored documents    |
| `DELETE` | `/doc/delete/:id` | —                              | Delete document chunk        |
| `POST`   | `/doc/ask`        | `{"question":"...","k":3}`     | RAG: retrieve + generate     |
| `GET`    | `/status`         | —                              | Ollama status and model info |

### Example: Search via curl

```powershell
curl "http://localhost:8080/search?v=0.9,0.8,0.7,0.6,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1&k=3&metric=cosine&algo=hnsw"
```

### Example: Ask a question via curl

```powershell
curl -X POST http://localhost:8080/doc/ask `
  -H "Content-Type: application/json" `
  -d '{"question":"What is dynamic programming?","k":3}'
```

---

## Project Structure

```
VectorDB/
├── app.py          ← Flask backend (HNSW, KD-Tree, BruteForce, REST API, RAG)
├── index.html      ← Frontend (PCA scatter plot, chat UI, document manager, benchmark)
├── requirements.txt ← Python dependencies (Flask, requests)
└── README.md       ← This file
```

### Architecture (app.py)

```
BruteForce          O(N·d)      Exact, baseline search
KDTree              O(log N)    Exact, axis-aligned partitioning
HNSW                O(log N)    Approximate, multilayer small-world graph

VectorDB            Unified interface over all 3 (16D demo vectors)
DocumentDB          HNSW-only index for real Ollama embeddings (768D)
OllamaClient        HTTP client to Ollama's /api/embeddings + /api/generate
```

### Key Classes & Methods

**VectorDB** — Manages 16-dimensional demo vectors

- `insert(metadata, category, embedding)` — Add vector to all 3 indices
- `search(query_vec, k, metric, algo)` — Search by algorithm
- `benchmark(query_vec, k, metric)` — Time all 3 algorithms
- `remove(id)` — Delete vector from all indices

**DocumentDB** — Manages high-dimensional document embeddings

- `insert(title, text, embedding)` — Store document chunks with embeddings
- `search(query_embedding, k)` — Find top-k similar documents
- `remove(id)` — Delete document

**OllamaClient** — Wraps Ollama HTTP API

- `embed(text)` → returns 768D vector from nomic-embed-text
- `generate(prompt)` → returns LLM response from llama3.2
- `is_available()` → checks if Ollama is running

---

## Algorithm Deep Dive

### HNSW (Hierarchical Navigable Small World)

Nodes are inserted into a multilayer graph. Each node randomly gets assigned a maximum layer. Layer 0 has all nodes with many connections; higher layers have fewer nodes (exponentially fewer) with longer-range connections.

**Insert:** Start at the top layer, greedily find the nearest node, drop a layer, repeat. At each layer from your assigned max down to 0, run a beam search (ef_construction=200) and connect to the M nearest neighbors bidirectionally.

**Search:** Same greedy descent from top layer. At layer 0, expand to ef nearest candidates using a priority queue.

**Why it's fast:** The upper layers act like a highway — you quickly get to the right neighborhood, then zoom in at layer 0.

### KD-Tree (K-Dimensional Tree)

Binary space partitioning. Each node splits space along one dimension (cycling through all dimensions). Search prunes entire subtrees when the closest possible point in that subtree can't beat the current best — the "ball within hyperslab" check.

**Weakness:** Degrades with high dimensions (curse of dimensionality). Works well for ≤20D, becomes close to brute force at 768D.

### Why HNSW Wins at High Dimensions

KD-Tree pruning relies on axis-aligned distance bounds. In high dimensions, almost all the space is near the boundary of the hypersphere — no subtrees get pruned. HNSW's graph-based approach doesn't have this problem.

---

## Common Issues

| Problem                      | Fix                                                                                                          |
| ---------------------------- | ------------------------------------------------------------------------------------------------------------ |
| `Ollama: OFFLINE` in header  | Ollama not running. Run `C:\Users\YourUsername\AppData\Local\Programs\Ollama\ollama.exe serve` in a terminal |
| Embedding takes forever      | Ollama is downloading the model on first use, wait a couple minutes                                          |
| `ModuleNotFoundError: Flask` | Virtual environment not activated. Run `.venv\Scripts\Activate.ps1`                                          |
| Port 8080 already in use     | Kill the process: `netstat -ano \| findstr :8080` then `taskkill /PID <pid> /F`                              |
| LLM answer is slow           | Normal — llama3.2 takes 10–30s on a laptop CPU. Use llama3.2:1b for faster answers                           |

### Use a Smaller/Faster LLM

If llama3.2 is too slow on your laptop, switch to the 1B model:

```powershell
C:\Users\YourUsername\AppData\Local\Programs\Ollama\ollama.exe pull llama3.2:1b
```

Then edit `app.py` and change line ~310:

```python
self.gen_model = "llama3.2:1b"   # change from "llama3.2"
```

Restart the server.

---

## Performance Tips

1. **Batch embeddings** — The Ollama API is slow for single texts. For bulk document insertion, consider batching 10+ documents per request.

2. **Use cosine similarity** — Fastest for high-dimensional embeddings (768D). Euclidean and Manhattan are slower.

3. **Tune HNSW parameters** — In `app.py`, the HNSW is initialized with `M=16, ef_build=200`. Reduce M for faster insert, increase for better search quality.

4. **KD-Tree weakness** — At 768D, KD-Tree becomes brute force. Only use it for debugging or for ≤20D data.

5. **Document chunking** — The default is 250-word chunks with 30-word overlap. Adjust in [app.py](app.py) function `chunk_text()` for your domain.

---

## Development Notes

### Adding a New Search Algorithm

1. Create a class with `insert(item)`, `knn(query, k, dist_fn)`, and `remove(id)` methods.
2. Add to `VectorDB.__init__()` and call from `insert()`, `remove()`, `search()`.
3. Update the `/benchmark` endpoint to time it.

### Running Tests

Check the `/status` endpoint:

```powershell
curl http://localhost:8080/status
```

Returns:

```json
{
  "ollamaAvailable": true,
  "embedModel": "nomic-embed-text",
  "genModel": "llama3.2",
  "docCount": 5,
  "docDims": 768,
  "demoCount": 20,
  "demoDims": 16
}
```

### Debugging

Enable Flask debug mode by editing `app.py`:

```python
app.run(host="0.0.0.0", port=8080, debug=True)
```

### Threading

The `VectorDB` and `DocumentDB` use Python `threading.Lock()` to ensure thread-safe access. Safe for concurrent requests.

---

## License

MIT — use this however you want.
