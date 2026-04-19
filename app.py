from flask import Flask, request, jsonify, send_from_directory, make_response
import heapq
import math
import random
import threading
import time
import os
import requests

DIMS = 16

class VectorItem:
    def __init__(self, id, metadata, category, emb):
        self.id = id
        self.metadata = metadata
        self.category = category
        self.emb = emb

class BruteForce:
    def __init__(self):
        self.items = []

    def insert(self, v):
        self.items.append(v)

    def knn(self, q, k, dist):
        results = [(dist(q, v.emb), v.id) for v in self.items]
        results.sort(key=lambda x: x[0])
        return results[:k]

    def remove(self, id):
        self.items = [v for v in self.items if v.id != id]

class KDNode:
    def __init__(self, item):
        self.item = item
        self.left = None
        self.right = None

class KDTree:
    def __init__(self, dims):
        self.root = None
        self.dims = dims

    def destroy(self, node):
        if node is None:
            return
        self.destroy(node.left)
        self.destroy(node.right)

    def ins(self, node, v, d):
        if node is None:
            return KDNode(v)
        ax = d % self.dims
        if v.emb[ax] < node.item.emb[ax]:
            node.left = self.ins(node.left, v, d + 1)
        else:
            node.right = self.ins(node.right, v, d + 1)
        return node

    def insert(self, v):
        self.root = self.ins(self.root, v, 0)

    def knn(self, q, k, dist):
        heap = []
        self._knn(self.root, q, k, 0, dist, heap)
        results = [(-d, id) for d, id in heap]
        results.sort(key=lambda x: x[0])
        return results

    def _knn(self, node, q, k, d, dist, heap):
        if node is None:
            return
        dn = dist(q, node.item.emb)
        if len(heap) < k or dn < -heap[0][0]:
            heapq.heappush(heap, (-dn, node.item.id))
            if len(heap) > k:
                heapq.heappop(heap)
        ax = d % self.dims
        diff = q[ax] - node.item.emb[ax]
        closer = node.left if diff < 0 else node.right
        farther = node.right if diff < 0 else node.left
        self._knn(closer, q, k, d + 1, dist, heap)
        if len(heap) < k or abs(diff) < -heap[0][0]:
            self._knn(farther, q, k, d + 1, dist, heap)

    def rebuild(self, items):
        self.destroy(self.root)
        self.root = None
        for v in items:
            self.insert(v)

class HNSW:
    class Node:
        def __init__(self, item, max_layer):
            self.item = item
            self.max_layer = max_layer
            self.nbrs = [[] for _ in range(max_layer + 1)]

    def __init__(self, m=16, ef_build=200):
        self.G = {}
        self.M = m
        self.M0 = 2 * m
        self.ef_build = ef_build
        self.mL = 1.0 / math.log(float(m))
        self.top_layer = -1
        self.entry_pt = -1
        self.rng = random.Random(42)

    def rand_level(self):
        return int(math.floor(-math.log(self.rng.random()) * self.mL))

    def search_layer(self, q, ep, ef, lyr, dist):
        vis = set()
        d0 = dist(q, self.G[ep].item.emb)
        vis.add(ep)
        cands = [(d0, ep)]
        found = [(-d0, ep)]

        while cands:
            cd, cid = heapq.heappop(cands)
            if len(found) >= ef and cd > -found[0][0]:
                break
            if lyr >= len(self.G[cid].nbrs):
                continue
            for nid in self.G[cid].nbrs[lyr]:
                if nid in vis or nid not in self.G:
                    continue
                vis.add(nid)
                nd = dist(q, self.G[nid].item.emb)
                if len(found) < ef or nd < -found[0][0]:
                    heapq.heappush(cands, (nd, nid))
                    heapq.heappush(found, (-nd, nid))
                    if len(found) > ef:
                        heapq.heappop(found)

        result = [(-d, id) for d, id in found]
        result.sort(key=lambda x: x[0])
        return result

    def select_nbrs(self, cands, max_m):
        return [id for _, id in cands[:max_m]]

    def insert(self, item, dist):
        node_id = item.id
        lvl = self.rand_level()
        self.G[node_id] = HNSW.Node(item, lvl)

        if self.entry_pt == -1:
            self.entry_pt = node_id
            self.top_layer = lvl
            return

        ep = self.entry_pt
        for lc in range(self.top_layer, lvl, -1):
            if lc < len(self.G[ep].nbrs):
                W = self.search_layer(item.emb, ep, 1, lc, dist)
                if W:
                    ep = W[0][1]
        for lc in range(min(self.top_layer, lvl), -1, -1):
            W = self.search_layer(item.emb, ep, self.ef_build, lc, dist)
            max_m = self.M0 if lc == 0 else self.M
            sel = self.select_nbrs(W, max_m)
            self.G[node_id].nbrs[lc] = sel

            for nid in sel:
                if nid not in self.G:
                    continue
                if lc >= len(self.G[nid].nbrs):
                    self.G[nid].nbrs.extend([[] for _ in range(lc + 1 - len(self.G[nid].nbrs))])
                conn = self.G[nid].nbrs[lc]
                conn.append(node_id)
                if len(conn) > max_m:
                    ds = [(dist(self.G[nid].item.emb, self.G[c].item.emb), c)
                          for c in conn if c in self.G]
                    ds.sort(key=lambda x: x[0])
                    self.G[nid].nbrs[lc] = [c for _, c in ds[:max_m]]
            if W:
                ep = W[0][1]
        if lvl > self.top_layer:
            self.top_layer = lvl
            self.entry_pt = node_id

    def knn(self, q, k, ef, dist):
        if self.entry_pt == -1:
            return []
        ep = self.entry_pt
        for lc in range(self.top_layer, 0, -1):
            if lc < len(self.G[ep].nbrs):
                W = self.search_layer(q, ep, 1, lc, dist)
                if W:
                    ep = W[0][1]
        W = self.search_layer(q, ep, max(ef, k), 0, dist)
        return W[:k]

    def remove(self, id):
        if id not in self.G:
            return
        for nid, nd in self.G.items():
            for layer in nd.nbrs:
                while id in layer:
                    layer.remove(id)
        del self.G[id]
        if self.entry_pt == id:
            self.entry_pt = next(iter(self.G), -1)
            self.top_layer = max((n.max_layer for n in self.G.values()), default=-1)

    def get_info(self):
        nodes_per_layer = []
        edges_per_layer = []
        nodes = []
        edges = []
        max_layer = max((nd.max_layer for nd in self.G.values()), default=-1)
        total_layers = max_layer + 1
        nodes_per_layer = [0] * total_layers
        edges_per_layer = [0] * total_layers

        for node_id, nd in self.G.items():
            nodes.append({
                "id": node_id,
                "metadata": nd.item.metadata,
                "category": nd.item.category,
                "maxLayer": nd.max_layer,
            })
            for lc in range(0, nd.max_layer + 1):
                nodes_per_layer[lc] += 1
                if lc < len(nd.nbrs):
                    for nid in nd.nbrs[lc]:
                        if node_id < nid:
                            edges_per_layer[lc] += 1
                            edges.append({"src": node_id, "dst": nid, "layer": lc})

        return {
            "topLayer": self.top_layer,
            "nodeCount": len(self.G),
            "nodesPerLayer": nodes_per_layer,
            "edgesPerLayer": edges_per_layer,
            "nodes": nodes,
            "edges": edges,
        }

    def size(self):
        return len(self.G)


def euclidean(a, b):
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))


def cosine(a, b):
    dot = sum(x * y for x, y in zip(a, b))
    na = sum(x * x for x in a)
    nb = sum(y * y for y in b)
    if na < 1e-9 or nb < 1e-9:
        return 1.0
    return 1.0 - dot / (math.sqrt(na) * math.sqrt(nb))


def manhattan(a, b):
    return sum(abs(x - y) for x, y in zip(a, b))


def get_dist_fn(metric):
    if metric == "cosine":
        return cosine
    if metric == "manhattan":
        return manhattan
    return euclidean


def parse_vec(s):
    return [float(x) for x in s.split(",") if x.strip()]


class VectorDB:
    def __init__(self, dims):
        self.store = {}
        self.bf = BruteForce()
        self.kdt = KDTree(dims)
        self.hnsw = HNSW(16, 200)
        self.lock = threading.Lock()
        self.next_id = 1
        self.dims = dims

    def insert(self, meta, cat, emb, dist):
        with self.lock:
            vid = self.next_id
            self.next_id += 1
            v = VectorItem(vid, meta, cat, emb)
            self.store[vid] = v
            self.bf.insert(v)
            self.kdt.insert(v)
            self.hnsw.insert(v, dist)
            return vid

    def remove(self, id):
        with self.lock:
            if id not in self.store:
                return False
            del self.store[id]
            self.bf.remove(id)
            self.hnsw.remove(id)
            self.kdt.rebuild(list(self.store.values()))
            return True

    def search(self, q, k, metric, algo):
        with self.lock:
            dfn = get_dist_fn(metric)
            start = time.perf_counter_ns()
            if algo == "bruteforce":
                raw = self.bf.knn(q, k, dfn)
            elif algo == "kdtree":
                raw = self.kdt.knn(q, k, dfn)
            else:
                raw = self.hnsw.knn(q, k, 50, dfn)
            elapsed = (time.perf_counter_ns() - start) // 1000
            hits = []
            for dist_val, vid in raw:
                if vid in self.store:
                    item = self.store[vid]
                    hits.append({
                        "id": vid,
                        "metadata": item.metadata,
                        "category": item.category,
                        "distance": dist_val,
                        "embedding": item.emb,
                    })
            return {"results": hits, "latencyUs": elapsed, "algo": algo, "metric": metric}

    def benchmark(self, q, k, metric):
        with self.lock:
            dfn = get_dist_fn(metric)
            def time_fn(fn):
                start = time.perf_counter_ns()
                fn()
                return (time.perf_counter_ns() - start) // 1000
            return {
                "bruteforceUs": time_fn(lambda: self.bf.knn(q, k, dfn)),
                "kdtreeUs": time_fn(lambda: self.kdt.knn(q, k, dfn)),
                "hnswUs": time_fn(lambda: self.hnsw.knn(q, k, 50, dfn)),
                "itemCount": len(self.store),
            }

    def all(self):
        with self.lock:
            return [v for v in self.store.values()]

    def hnsw_info(self):
        with self.lock:
            return self.hnsw.get_info()

    def size(self):
        with self.lock:
            return len(self.store)


class DocItem:
    def __init__(self, id, title, text, emb):
        self.id = id
        self.title = title
        self.text = text
        self.emb = emb


class DocumentDB:
    def __init__(self):
        self.store = {}
        self.hnsw = HNSW(16, 200)
        self.bf = BruteForce()
        self.lock = threading.Lock()
        self.next_id = 1
        self.dims = 0

    def insert(self, title, text, emb):
        with self.lock:
            if self.dims == 0:
                self.dims = len(emb)
            doc_id = self.next_id
            self.next_id += 1
            item = DocItem(doc_id, title, text, emb)
            self.store[doc_id] = item
            vi = VectorItem(doc_id, title, "doc", emb)
            self.hnsw.insert(vi, cosine)
            self.bf.insert(vi)
            return doc_id

    def search(self, q, k, max_dist=0.7):
        with self.lock:
            if not self.store:
                return []
            raw = self.bf.knn(q, k, cosine) if len(self.store) < 10 else self.hnsw.knn(q, k, 50, cosine)
            return [(dist_val, self.store[id]) for dist_val, id in raw if id in self.store and dist_val <= max_dist]

    def remove(self, id):
        with self.lock:
            if id not in self.store:
                return False
            del self.store[id]
            self.hnsw.remove(id)
            self.bf.remove(id)
            return True

    def all(self):
        with self.lock:
            return [v for v in self.store.values()]

    def size(self):
        with self.lock:
            return len(self.store)

    def get_dims(self):
        with self.lock:
            return self.dims


class OllamaClient:
    def __init__(self, host="127.0.0.1", port=11434):
        self.host = host
        self.port = port
        self.embed_model = "nomic-embed-text"
        self.gen_model = "llama3.2"

    def url(self, path):
        return f"http://{self.host}:{self.port}{path}"

    def is_available(self):
        try:
            resp = requests.get(self.url("/api/tags"), timeout=2)
            return resp.status_code == 200
        except requests.RequestException:
            return False

    def embed(self, text):
        body = {"model": self.embed_model, "prompt": text}
        try:
            resp = requests.post(self.url("/api/embeddings"), json=body, timeout=(3, 30))
            if resp.status_code != 200:
                return []
            data = resp.json()
            return data.get("embedding", [])
        except requests.RequestException:
            return []

    def generate(self, prompt):
        body = {"model": self.gen_model, "prompt": prompt, "stream": False}
        try:
            resp = requests.post(self.url("/api/generate"), json=body, timeout=(3, 180))
            if resp.status_code != 200:
                return "ERROR: Ollama unavailable. Run: ollama serve"
            data = resp.json()
            return data.get("response", "")
        except requests.RequestException:
            return "ERROR: Ollama unavailable. Run: ollama serve"


def chunk_text(text, chunk_words=250, overlap_words=30):
    words = text.split()
    if not words:
        return []
    if len(words) <= chunk_words:
        return [text]
    chunks = []
    step = chunk_words - overlap_words
    for i in range(0, len(words), step):
        chunk = " ".join(words[i:i + chunk_words])
        chunks.append(chunk)
        if i + chunk_words >= len(words):
            break
    return chunks


def load_demo(db):
    dist = get_dist_fn("cosine")
    demo_items = [
        ("Linked List: nodes connected by pointers", "cs", [0.90,0.85,0.72,0.68,0.12,0.08,0.15,0.10,0.05,0.08,0.06,0.09,0.07,0.11,0.08,0.06]),
        ("Binary Search Tree: O(log n) search and insert", "cs", [0.88,0.82,0.78,0.74,0.15,0.10,0.08,0.12,0.06,0.07,0.08,0.05,0.09,0.06,0.07,0.10]),
        ("Dynamic Programming: memoization overlapping subproblems", "cs", [0.82,0.76,0.88,0.80,0.20,0.18,0.12,0.09,0.07,0.06,0.08,0.07,0.08,0.09,0.06,0.07]),
        ("Graph BFS and DFS: breadth and depth first traversal", "cs", [0.85,0.80,0.75,0.82,0.18,0.14,0.10,0.08,0.06,0.09,0.07,0.06,0.10,0.08,0.09,0.07]),
        ("Hash Table: O(1) lookup with collision chaining", "cs", [0.87,0.78,0.70,0.76,0.13,0.11,0.09,0.14,0.08,0.07,0.06,0.08,0.07,0.10,0.08,0.09]),
        ("Calculus: derivatives integrals and limits", "math", [0.12,0.15,0.18,0.10,0.91,0.86,0.78,0.72,0.08,0.06,0.07,0.09,0.07,0.08,0.06,0.10]),
        ("Linear Algebra: matrices eigenvalues eigenvectors", "math", [0.20,0.18,0.15,0.12,0.88,0.90,0.82,0.76,0.09,0.07,0.08,0.06,0.10,0.07,0.08,0.09]),
        ("Probability: distributions random variables Bayes theorem", "math", [0.15,0.12,0.20,0.18,0.84,0.80,0.88,0.82,0.07,0.08,0.06,0.10,0.09,0.06,0.09,0.08]),
        ("Number Theory: primes modular arithmetic RSA cryptography", "math", [0.22,0.16,0.14,0.20,0.80,0.85,0.76,0.90,0.08,0.09,0.07,0.06,0.08,0.10,0.07,0.06]),
        ("Combinatorics: permutations combinations generating functions", "math", [0.18,0.20,0.16,0.14,0.86,0.78,0.84,0.80,0.06,0.07,0.09,0.08,0.06,0.09,0.10,0.07]),
        ("Neapolitan Pizza: wood-fired dough San Marzano tomatoes", "food", [0.08,0.06,0.09,0.07,0.07,0.08,0.06,0.09,0.90,0.86,0.78,0.72,0.08,0.06,0.09,0.07]),
        ("Sushi: vinegared rice raw fish and nori rolls", "food", [0.06,0.08,0.07,0.09,0.09,0.06,0.08,0.07,0.86,0.90,0.82,0.76,0.07,0.09,0.06,0.08]),
        ("Ramen: noodle soup with chashu pork and soft-boiled eggs", "food", [0.09,0.07,0.06,0.08,0.08,0.09,0.07,0.06,0.82,0.78,0.90,0.84,0.09,0.07,0.08,0.06]),
        ("Tacos: corn tortillas with carnitas salsa and cilantro", "food", [0.07,0.09,0.08,0.06,0.06,0.07,0.09,0.08,0.78,0.82,0.86,0.90,0.06,0.08,0.07,0.09]),
        ("Croissant: laminated pastry with buttery flaky layers", "food", [0.06,0.07,0.10,0.09,0.10,0.06,0.07,0.10,0.85,0.80,0.76,0.82,0.09,0.07,0.10,0.06]),
        ("Basketball: fast-paced shooting dribbling slam dunks", "sports", [0.09,0.07,0.08,0.10,0.08,0.09,0.07,0.06,0.08,0.07,0.09,0.06,0.91,0.85,0.78,0.72]),
        ("Football: tackles touchdowns field goals and strategy", "sports", [0.07,0.09,0.06,0.08,0.09,0.07,0.10,0.08,0.07,0.09,0.08,0.07,0.87,0.89,0.82,0.76]),
        ("Tennis: racket volleys groundstrokes and Wimbledon serves", "sports", [0.08,0.06,0.09,0.07,0.07,0.08,0.06,0.09,0.09,0.06,0.07,0.08,0.83,0.80,0.88,0.82]),
        ("Chess: openings endgames tactics strategic board game", "sports", [0.25,0.20,0.22,0.18,0.22,0.18,0.20,0.15,0.06,0.08,0.07,0.09,0.80,0.84,0.78,0.90]),
        ("Swimming: butterfly freestyle backstroke Olympic competition", "sports", [0.06,0.08,0.07,0.09,0.08,0.06,0.09,0.07,0.10,0.08,0.06,0.07,0.85,0.82,0.86,0.80]),
    ]
    for meta, cat, emb in demo_items:
        db.insert(meta, cat, emb, dist)


def make_cors_response(response):
    response.headers["Access-Control-Allow-Origin"] = "*"
    response.headers["Access-Control-Allow-Methods"] = "GET, POST, DELETE, OPTIONS"
    response.headers["Access-Control-Allow-Headers"] = "Content-Type"
    return response

app = Flask(__name__, static_folder=".")

@app.after_request
def after_request(response):
    return make_cors_response(response)

@app.route("/", methods=["GET"])
def index():
    return send_from_directory(os.getcwd(), "index.html")

@app.route("/search", methods=["GET"])
def search():
    v = request.args.get("v", "")
    q = parse_vec(v)
    if len(q) != DIMS:
        return jsonify({"error": f"need {DIMS}D vector"}), 400
    try:
        k = int(request.args.get("k", 5))
    except ValueError:
        k = 5
    metric = request.args.get("metric", "cosine")
    algo = request.args.get("algo", "hnsw")
    return jsonify(db.search(q, k, metric, algo))

@app.route("/insert", methods=["POST"])
def insert():
    body = request.get_json(silent=True) or {}
    meta = body.get("metadata", "")
    cat = body.get("category", "")
    emb = body.get("embedding", [])
    if not meta or not isinstance(emb, list) or len(emb) != DIMS:
        return jsonify({"error": "invalid body"}), 400
    vid = db.insert(meta, cat, emb, get_dist_fn("cosine"))
    return jsonify({"id": vid})

@app.route("/delete/<int:item_id>", methods=["DELETE"])
def delete(item_id):
    ok = db.remove(item_id)
    return jsonify({"ok": ok})

@app.route("/items", methods=["GET"])
def items():
    result = []
    for item in db.all():
        result.append({
            "id": item.id,
            "metadata": item.metadata,
            "category": item.category,
            "embedding": item.emb,
        })
    return jsonify(result)

@app.route("/benchmark", methods=["GET"])
def benchmark():
    v = request.args.get("v", "")
    q = parse_vec(v)
    if len(q) != DIMS:
        return jsonify({"error": f"need {DIMS}D vector"}), 400
    try:
        k = int(request.args.get("k", 5))
    except ValueError:
        k = 5
    metric = request.args.get("metric", "cosine")
    return jsonify(db.benchmark(q, k, metric))

@app.route("/hnsw-info", methods=["GET"])
def hnsw_info():
    return jsonify(db.hnsw_info())

@app.route("/doc/insert", methods=["POST"])
def doc_insert():
    body = request.get_json(silent=True) or {}
    title = body.get("title", "")
    text = body.get("text", "")
    if not title or not text:
        return jsonify({"error": "need title and text"}), 400
    chunks = chunk_text(text, 250, 30)
    ids = []
    for i, chunk in enumerate(chunks):
        emb = ollama.embed(chunk)
        if not emb:
            return jsonify({
                "error": "Ollama unavailable. Install from https://ollama.com then run: ollama pull nomic-embed-text && ollama pull llama3.2"
            }), 500
        chunk_title = f"{title} [{i+1}/{len(chunks)}]" if len(chunks) > 1 else title
        ids.append(doc_db.insert(chunk_title, chunk, emb))
    return jsonify({"ids": ids, "chunks": len(chunks), "dims": doc_db.get_dims()})

@app.route("/doc/delete/<int:item_id>", methods=["DELETE"])
def doc_delete(item_id):
    ok = doc_db.remove(item_id)
    return jsonify({"ok": ok})

@app.route("/doc/list", methods=["GET"])
def doc_list():
    docs = []
    for doc in doc_db.all():
        preview = doc.text[:120] + ("…" if len(doc.text) > 120 else "")
        docs.append({
            "id": doc.id,
            "title": doc.title,
            "preview": preview,
            "words": len(doc.text.split()),
        })
    return jsonify(docs)

@app.route("/doc/search", methods=["POST"])
def doc_search():
    body = request.get_json(silent=True) or {}
    question = body.get("question", "")
    k = body.get("k", 3)
    if not question:
        return jsonify({"error": "need question"}), 400
    q_emb = ollama.embed(question)
    if not q_emb:
        return jsonify({"error": "Ollama unavailable"}), 500
    hits = doc_db.search(q_emb, k)
    return jsonify({
        "contexts": [
            {"id": item.id, "title": item.title, "distance": round(dist_val, 4)}
            for dist_val, item in hits
        ]
    })

@app.route("/doc/ask", methods=["POST"])
def doc_ask():
    body = request.get_json(silent=True) or {}
    question = body.get("question", "")
    k = body.get("k", 3)
    if not question:
        return jsonify({"error": "need question"}), 400
    q_emb = ollama.embed(question)
    if not q_emb:
        return jsonify({"error": "Ollama unavailable"}), 500
    hits = doc_db.search(q_emb, k)
    ctx = "\n\n".join(f"[{i+1}] {item.title}:\n{item.text}" for i, (_, item) in enumerate(hits))
    prompt = (
        "You are a helpful assistant. Answer the user's question directly. "
        "Use the provided context if it contains relevant information. "
        "If it doesn't, just use your own general knowledge. "
        "IMPORTANT: Do NOT mention the 'context', 'provided text', or say things like 'the context doesn't mention'. "
        "Just answer the question naturally.\n\n"
        f"Context:\n{ctx}\n\nQuestion: {question}\n\nAnswer:"
    )
    answer = ollama.generate(prompt)
    return jsonify({
        "answer": answer,
        "model": ollama.gen_model,
        "contexts": [
            {"id": item.id, "title": item.title, "text": item.text, "distance": round(dist_val, 4)}
            for dist_val, item in hits
        ],
        "docCount": doc_db.size(),
    })

@app.route("/status", methods=["GET"])
def status():
    up = ollama.is_available()
    return jsonify({
        "ollamaAvailable": up,
        "embedModel": ollama.embed_model,
        "genModel": ollama.gen_model,
        "docCount": doc_db.size(),
        "docDims": doc_db.get_dims(),
        "demoDims": DIMS,
        "demoCount": db.size(),
    })

@app.route("/stats", methods=["GET"])
def stats():
    return jsonify({
        "count": db.size(),
        "dims": DIMS,
        "algorithms": ["bruteforce", "kdtree", "hnsw"],
        "metrics": ["euclidean", "cosine", "manhattan"],
    })

if __name__ == "__main__":
    db = VectorDB(DIMS)
    doc_db = DocumentDB()
    ollama = OllamaClient()
    load_demo(db)
    print("=== VectorDB Engine ===")
    print("http://localhost:8080")
    print(f"{db.size()} demo vectors | {DIMS} dims | HNSW+KD-Tree+BruteForce")
    print("Ollama: ONLINE" if ollama.is_available() else "Ollama: OFFLINE (install from ollama.com)")
    app.run(host="0.0.0.0", port=8080)
