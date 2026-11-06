"""Python conversion of the supplied C++ VectorDB/RAG demo.

Run: python vector_db.py
Optional: pip install numpy requests
The HTTP server uses only the Python standard library. Ollama calls require requests.
"""
from __future__ import annotations
import json, math, random, threading, time, urllib.request, urllib.error
from collections import defaultdict
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse, parse_qs
from typing import Any, Optional


def distance(a, b, metric="euclidean"):
    if len(a) != len(b):
        raise ValueError("vectors must have the same dimension")
    if metric == "cosine":
        dot = sum(x*y for x, y in zip(a,b)); na = math.sqrt(sum(x*x for x in a)); nb = math.sqrt(sum(y*y for y in b))
        return 1.0 - (dot/(na*nb) if na and nb else 0.0)
    if metric == "manhattan": return sum(abs(x-y) for x,y in zip(a,b))
    return math.sqrt(sum((x-y)**2 for x,y in zip(a,b)))

@dataclass
class VectorItem:
    id: str
    vector: list[float]
    metadata: dict[str, Any] = field(default_factory=dict)

class BruteForce:
    def __init__(self, metric="euclidean"): self.metric=metric; self.items=[]
    def build(self, items): self.items=list(items)
    def search(self, query, k):
        return sorted(((distance(query,x.vector,self.metric),x) for x in self.items),key=lambda z:z[0])[:max(0,k)]

@dataclass
class KDNode:
    item: VectorItem
    axis: int
    left: Optional['KDNode']=None
    right: Optional['KDNode']=None

class KDTree:
    def __init__(self, metric="euclidean"): self.metric=metric; self.root=None; self.items=[]; self.dim=0
    def build(self, items):
        self.items=list(items); self.dim=len(items[0].vector) if items else 0
        def rec(arr, depth):
            if not arr:return None
            axis=depth%self.dim; arr.sort(key=lambda x:x.vector[axis]); mid=len(arr)//2
            node=KDNode(arr[mid],axis); node.left=rec(arr[:mid],depth+1); node.right=rec(arr[mid+1:],depth+1); return node
        self.root=rec(self.items[:],0)
    def search(self, query, k):
        # Exact search; pruning uses the split-plane lower bound for L2.
        best=[]
        def add(item):
            best.append((distance(query,item.vector,self.metric),item)); best.sort(key=lambda z:z[0]); del best[max(0,k):]
        def rec(node):
            if node is None:return
            add(node.item); axis=node.axis; delta=query[axis]-node.item.vector[axis]
            near,far=(node.left,node.right) if delta<0 else (node.right,node.left)
            rec(near)
            threshold=best[-1][0] if len(best)>=k and k>0 else float('inf')
            if self.metric != "euclidean" or delta*delta <= threshold: rec(far)
        if k>0:rec(self.root)
        return best

class HNSW:
    """Small educational graph index. Builds a bounded neighbor graph and searches it best-first."""
    def __init__(self, metric="euclidean", m=8, ef=64): self.metric=metric; self.m=m; self.ef=ef; self.items=[]; self.graph=[]
    def build(self, items):
        self.items=list(items); self.graph=[[] for _ in items]
        for i,item in enumerate(items):
            # Connect each new item to its M nearest earlier items, then add reciprocal edges.
            candidates=sorted((distance(item.vector,items[j].vector,self.metric),j) for j in range(i))[:self.m]
            for _,j in candidates: self.graph[i].append(j); self.graph[j].append(i)
        for i in range(len(self.graph)):
            self.graph[i]=list(dict.fromkeys(self.graph[i]))
    def search(self, query, k):
        if not self.items or k<=0:return []
        visited=set(); frontier=[(distance(query,self.items[0].vector,self.metric),0)]; found=[]
        while frontier and len(visited)<max(self.ef,k):
            d,i=frontier.pop(0)
            if i in visited:continue
            visited.add(i); found.append((d,self.items[i]))
            for j in self.graph[i]:
                if j not in visited: frontier.append((distance(query,self.items[j].vector,self.metric),j))
            frontier.sort(key=lambda x:x[0])
        return sorted(found,key=lambda x:x[0])[:k]
    def info(self): return {"items":len(self.items),"edges":sum(map(len,self.graph))//2,"m":self.m,"ef":self.ef}

class VectorDB:
    def __init__(self, metric="euclidean", index_type="bruteforce"):
        self.metric=metric; self.index_type=index_type; self.items={}; self.lock=threading.RLock(); self.index=None
        self._rebuild()
    def _make_index(self):
        if self.index_type.lower() in ("kdtree","kd-tree","kd_tree"): return KDTree(self.metric)
        if self.index_type.lower()=="hnsw": return HNSW(self.metric)
        return BruteForce(self.metric)
    def _rebuild(self):
        self.index=self._make_index(); self.index.build(list(self.items.values()))
    def insert(self, item_id, vector, metadata=None):
        with self.lock:
            self.items[str(item_id)]=VectorItem(str(item_id),list(map(float,vector)),dict(metadata or {})); self._rebuild()
    def delete(self,item_id):
        with self.lock:
            existed=self.items.pop(str(item_id),None) is not None; self._rebuild(); return existed
    def search(self,query,k=5):
        with self.lock: results=self.index.search(list(map(float,query)),int(k))
        return [{"id":item.id,"distance":dist,"vector":item.vector,"metadata":item.metadata} for dist,item in results]
    def list_items(self):
        with self.lock:return [{"id":x.id,"vector":x.vector,"embedding":x.vector,"metadata":x.metadata,
                 "name":x.metadata.get("name",x.metadata.get("text",x.id)),
                 "category":x.metadata.get("category","default")} for x in self.items.values()]
    def benchmark(self, queries=None, k=5):
        items=list(self.items.values()); queries=queries or [x.vector for x in items[:min(10,len(items))]]
        start=time.perf_counter()
        for q in queries:self.search(q,k)
        elapsed=(time.perf_counter()-start)*1000
        return {"index":self.index_type,"items":len(items),"queries":len(queries),"total_ms":elapsed,"avg_ms":elapsed/len(queries) if queries else 0}
    def stats(self):return {"count":len(self.items),"dimension":len(next(iter(self.items.values())).vector) if self.items else 0,"metric":self.metric,"index_type":self.index_type}


def chunk_text(text, chunk_size=500, overlap=50):
    text=str(text); chunk_size=max(1,int(chunk_size)); overlap=max(0,min(int(overlap),chunk_size-1)); chunks=[]; start=0
    while start<len(text):
        end=min(len(text),start+chunk_size); chunks.append(text[start:end])
        if end==len(text):break
        start=end-overlap
    return chunks

class OllamaClient:
    def __init__(self, base_url="http://localhost:11434", embedding_model="nomic-embed-text", generation_model="llama3.2:latest"):
        self.base_url=base_url.rstrip("/"); self.embedding_model=embedding_model; self.generation_model=generation_model
    def _post(self,path,payload):
        data=json.dumps(payload).encode(); req=urllib.request.Request(self.base_url+path,data=data,headers={"Content-Type":"application/json"})
        with urllib.request.urlopen(req,timeout=120) as response:return json.loads(response.read().decode())
    def embed(self,text):
        try:
            result=self._post("/api/embeddings",{"model":self.embedding_model,"prompt":text}); return result.get("embedding",[])
        except Exception as e: raise RuntimeError(f"Ollama embedding request failed: {e}") from e
    def generate(self,prompt):
        try:
            result=self._post("/api/generate",{"model":self.generation_model,"prompt":prompt,"stream":False}); return result.get("response","")
        except Exception as e: raise RuntimeError(f"Ollama generation request failed: {e}")

@dataclass
class DocItem:
    id: str
    text: str
    metadata: dict[str,Any]=field(default_factory=dict)
    vector: list[float]=field(default_factory=list)

class DocumentDB:
    def __init__(self,client=None,metric="cosine",index_type="bruteforce",chunk_size=500,overlap=50):
        self.client=client or OllamaClient(); self.vdb=VectorDB(metric,index_type); self.docs={}; self.chunk_size=chunk_size; self.overlap=overlap; self.lock=threading.RLock()
    def insert(self,doc_id,text,metadata=None):
        chunks=chunk_text(text,self.chunk_size,self.overlap); ids=[]
        with self.lock:
            for i,chunk in enumerate(chunks):
                cid=f"{doc_id}_{i}"; vector=self.client.embed(chunk); meta=dict(metadata or {}); meta.update({"document_id":str(doc_id),"chunk_index":i,"text":chunk})
                self.docs[cid]=DocItem(cid,chunk,meta,vector); self.vdb.insert(cid,vector,meta); ids.append(cid)
        return {"id":str(doc_id),"chunks":len(ids),"chunk_ids":ids}
    def delete(self,doc_id):
        with self.lock:
            keys=[k for k,d in self.docs.items() if d.metadata.get("document_id")==str(doc_id)]
            for k in keys:self.docs.pop(k,None); self.vdb.delete(k)
            return len(keys)
    def list(self):
        grouped=defaultdict(list)
        for d in self.docs.values():grouped[d.metadata.get("document_id")].append(d)
        return [{"id":doc_id,"chunks":len(chunks),"metadata":chunks[0].metadata if chunks else {},
                 "title":(chunks[0].metadata.get("title") if chunks else None) or str(doc_id),
                 "preview":(chunks[0].text[:180] if chunks else ""),
                 "words":sum(len(d.text.split()) for d in chunks)} for doc_id,chunks in grouped.items()]
    def search(self,query,k=5):
        vector=self.client.embed(query); return self.vdb.search(vector,k)
    def ask(self,question,k=5):
        results=self.search(question,k); context="\n\n".join(r.get("metadata",{}).get("text","") for r in results)
        prompt=f"Answer using the context below. If the answer is not in the context, say you do not know.\n\nContext:\n{context}\n\nQuestion: {question}\nAnswer:"
        contexts=[{"title":r.get("metadata",{}).get("title",r.get("metadata",{}).get("document_id","Document")),
                   "text":r.get("metadata",{}).get("text",""),"distance":float(r.get("distance",0)),
                   "id":r.get("id")} for r in results]
        return {"answer":self.client.generate(prompt),"model":self.client.generation_model,
                "contexts":contexts,"sources":results}

# Demo data corresponding to the original demo's sample-vector concept.
def load_demo(db):
    rng=random.Random(42)
    samples=[("apple",[1.0,0.2,0.1]),("banana",[0.9,0.1,0.2]),("orange",[0.8,0.3,0.1]),("car",[0.1,0.9,0.2]),("truck",[0.2,0.8,0.3]),("bus",[0.1,0.7,0.4])]
    for name,vec in samples:
        db.insert(name,list(vec)+[0.0]*13,{"name":name,"category":"demo"})

class APIHandler(BaseHTTPRequestHandler):
    db=None; docs=None
    def log_message(self,*args):pass
    def _send(self,status,payload):
        raw=json.dumps(payload,ensure_ascii=False).encode(); self.send_response(status); self.send_header("Content-Type","application/json; charset=utf-8"); self.send_header("Access-Control-Allow-Origin","*"); self.send_header("Access-Control-Allow-Headers","Content-Type"); self.send_header("Access-Control-Allow-Methods","GET,POST,DELETE,OPTIONS"); self.send_header("Content-Length",str(len(raw))); self.end_headers(); self.wfile.write(raw)
    def _body(self):
        try:return json.loads(self.rfile.read(int(self.headers.get("Content-Length",0))) or b"{}")
        except Exception:return {}
    def do_OPTIONS(self):self.send_response(204); self.send_header("Access-Control-Allow-Origin","*"); self.send_header("Access-Control-Allow-Headers","Content-Type"); self.send_header("Access-Control-Allow-Methods","GET,POST,DELETE,OPTIONS"); self.end_headers()
    def do_GET(self):
        path=urlparse(self.path).path; qs=parse_qs(urlparse(self.path).query)
        try:
            if path=="/":return self._send(200,{"message":"VectorDB API","routes":["/search","/insert","/items","/benchmark","/hnsw-info","/doc/insert","/doc/list","/doc/search","/doc/ask","/status","/stats"]})
            if path=="/items":return self._send(200,self.db.list_items())
            if path=="/benchmark":
                outputs={}
                for name in ("bruteforce","kdtree","hnsw"):
                    idx=VectorDB(self.db.metric,name); idx.items=dict(self.db.items); idx._rebuild()
                    start=time.perf_counter()
                    for item in list(idx.items.values())[:10]: idx.search(item.vector,5)
                    outputs[name+"Us"]=(time.perf_counter()-start)*1_000_000
                return self._send(200,outputs)
            if path=="/hnsw-info":
                info=self.db.index.info() if isinstance(self.db.index,HNSW) else {"index":self.db.index_type,"supported":False,"items":len(self.db.items),"edges":0}
                info.setdefault("nodesPerLayer",[info.get("items",0)]); info.setdefault("edgesPerLayer",[info.get("edges",0)])
                return self._send(200,info)
            if path=="/status":
                available=False
                try: available=urllib.request.urlopen(self.docs.client.base_url+"/api/tags",timeout=2).status==200
                except Exception: pass
                return self._send(200,{"status":"ok","stats":self.db.stats(),"ollamaAvailable":available,
                    "embedModel":self.docs.client.embedding_model,"genModel":self.docs.client.generation_model,
                    "docCount":len(self.docs.list()),"docDims":self.docs.vdb.stats().get("dimension",0)})
            if path=="/stats":return self._send(200,self.db.stats())
            if path.startswith("/delete/"):return self._send(200,{"deleted":self.db.delete(path[len('/delete/'):])})
            if path=="/doc/list":return self._send(200,self.docs.list())
            if path.startswith("/doc/delete/"):return self._send(200,{"deleted_chunks":self.docs.delete(path[len('/doc/delete/'):])})
            if path=="/doc/search":
                q=qs.get("q",[""])[0]; k=int(qs.get("k",[5])[0]); return self._send(200,self.docs.search(q,k))
            if path=="/doc/ask":
                q=qs.get("q",[""])[0]; k=int(qs.get("k",[5])[0]); return self._send(200,self.docs.ask(q,k))
            if path=="/search":
                q=qs.get("q",[None])[0]
                if q is None:
                    body=self._body(); q=body.get("vector",body.get("query",[])); k=body.get("k",5)
                else:k=int(qs.get("k",[5])[0]); q=json.loads(q) if q.startswith("[") else [float(x) for x in q.split(",")]
                return self._send(200,self.db.search(q,k))
            return self._send(404,{"error":"Not found"})
        except Exception as e:return self._send(400,{"error":str(e)})
    def do_DELETE(self):
        path=urlparse(self.path).path
        try:
            if path.startswith("/delete/"):
                return self._send(200,{"deleted":self.db.delete(path[len("/delete/"):] )})
            if path.startswith("/doc/delete/"):
                return self._send(200,{"deleted_chunks":self.docs.delete(path[len("/doc/delete/"):] )})
            return self._send(404,{"error":"Not found"})
        except Exception as e:
            return self._send(400,{"error":str(e)})

    def do_POST(self):
        path=urlparse(self.path).path; body=self._body()
        try:
            if path=="/insert":
                item_id=body.get("id",str(time.time_ns())); vector=body.get("vector",body.get("embedding",[])); meta=body.get("metadata",{})
                if isinstance(meta,str): meta={"name":meta}
                if body.get("category"): meta["category"]=body.get("category")
                self.db.insert(item_id,vector,meta); return self._send(200,{"inserted":True,"id":item_id})
            if path=="/doc/insert":
                meta=body.get("metadata",{})
                if not isinstance(meta,dict): meta={"title":body.get("title",str(meta))}
                if body.get("title"): meta.setdefault("title",body.get("title"))
                result=self.docs.insert(body.get("id",body.get("title",str(time.time_ns()))),body.get("text",body.get("content","")),meta)
                result["dims"]=self.docs.vdb.stats().get("dimension",0)
                return self._send(200,result)
            return self._send(404,{"error":"Not found"})
        except Exception as e:return self._send(400,{"error":str(e)})


def main():
    import argparse
    parser=argparse.ArgumentParser(description="VectorDB / RAG demo server")
    parser.add_argument("--host",default="0.0.0.0"); parser.add_argument("--port",type=int,default=8080)
    parser.add_argument("--metric",choices=["euclidean","cosine","manhattan"],default="euclidean")
    parser.add_argument("--index",choices=["bruteforce","kdtree","hnsw"],default="bruteforce")
    parser.add_argument("--ollama",default="http://localhost:11434"); parser.add_argument("--embedding-model",default="nomic-embed-text"); parser.add_argument("--generation-model",default="llama3.2:latest")
    parser.add_argument("--no-demo",action="store_true"); args=parser.parse_args()
    db=VectorDB(args.metric,args.index)
    if not args.no_demo:load_demo(db)
    client=OllamaClient(args.ollama,args.embedding_model,args.generation_model)
    docs=DocumentDB(client,args.metric,args.index)
    APIHandler.db=db; APIHandler.docs=docs
    server=ThreadingHTTPServer((args.host,args.port),APIHandler)
    print(f"VectorDB server listening at http://{args.host}:{args.port}")
    try:server.serve_forever()
    except KeyboardInterrupt:pass
    finally:server.server_close()

if __name__=="__main__":main()
