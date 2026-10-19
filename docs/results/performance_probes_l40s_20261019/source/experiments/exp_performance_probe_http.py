"""Loopback transport control with a separate CPU-only client process.

This deliberately simple fixed-batch HF service is not vLLM or a production API.
The client observes only returned token IDs, event sizes and its own clock.
The server trace is used only for subsequent measurement diagnosis.
"""
from __future__ import annotations
import argparse
import dataclasses
import http.client
import json
import random
import socket
import subprocess
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

from experiments.exp_performance_probe_gpu import Case, Engine, DEFAULT_MODEL_PATH, manifest, file_hash


def client(port, destination, session):
    cases = []
    for rep in range(2):
        for length in (512, 8192):
            for batch in (1, 4):
                cases.append((Case("http",session,length,batch,outputs=32,mode="greedy",execution="eager",repetition=rep),1))
        for arm,chunk in (("truncate",1),("full",8),("truncate",8)):
            cases.append((Case("http",session,8192,1,condition=arm,outputs=32,
                               mode="greedy",execution="eager",repetition=rep),chunk))
    random.Random(9519+session).shuffle(cases)
    with Path(destination).open("x") as output:
        for order,(case,chunk) in enumerate(cases):
            request_id = case.key + f"_chunk{chunk}"
            conn = http.client.HTTPConnection("127.0.0.1",port,timeout=300)
            conn.connect()
            conn.sock.setsockopt(socket.IPPROTO_TCP,socket.TCP_NODELAY,1)
            payload = json.dumps({"request_id":request_id,"case":dataclasses.asdict(case),"chunk":chunk})
            conn.request("POST","/probe",body=payload,headers={"Content-Type":"application/json"})
            start = time.perf_counter_ns()
            response = conn.getresponse()
            if response.status != 200:
                raise RuntimeError(response.status)
            events = []
            for line in response:
                received = time.perf_counter_ns()
                event = json.loads(line)
                if "error" in event:
                    raise RuntimeError(event["error"])
                events.append({"received_ns_since_send":received-start,**event})
            conn.close()
            assert events and sum(len(x["steps"]) for x in events)==case.outputs
            first=events[0]["received_ns_since_send"]/1e6
            last=events[-1]["received_ns_since_send"]/1e6
            row={"request_id":request_id,**dataclasses.asdict(case),"chunk":chunk,
                 "session":session,"order":order,"client_concurrency":1,"actual_batch":case.batch,
                 "status":"complete","ttft_ms":first,"last_event_ms":last,
                 "events":events,"transport":"loopback NDJSON token-ID frames; one request per fixed batch",
                 "timer_origin":"after request send completes"}
            output.write(json.dumps(row)+"\n"); output.flush()
            print(json.dumps({"request_id":request_id,"ttft_ms":first,"last_event_ms":last}),flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output",type=Path)
    parser.add_argument("--session",type=int,default=0)
    parser.add_argument("--model-path",default=str(DEFAULT_MODEL_PATH))
    parser.add_argument("--client-port",type=int)
    parser.add_argument("--client-output")
    args=parser.parse_args()
    if args.client_port:
        client(args.client_port,args.client_output,args.session)
        return
    if args.output is None:
        parser.error("--output is required")
    args.output.mkdir(parents=True,exist_ok=False)
    engine=Engine(args.model_path)
    engine.warmup()
    info=manifest(engine,args)
    info["args"]["output"]=str(args.output)
    info["http_scope"]="loopback only; CPU client subprocess; C=1; batch sent as explicit test setting"
    (args.output/"manifest.json").write_text(json.dumps(info,indent=2))
    server_log=(args.output/"server.jsonl").open("x")

    class Handler(BaseHTTPRequestHandler):
        def log_message(self,*a):
            pass
        def do_POST(self):
            self.connection.setsockopt(socket.IPPROTO_TCP,socket.TCP_NODELAY,1)
            request=json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            case=Case(**request["case"])
            chunk=request["chunk"]
            self.send_response(200)
            self.send_header("Content-Type","application/x-ndjson")
            self.send_header("Connection","close")
            self.end_headers()
            buffered=[]
            def emit(step,ids):
                buffered.append({"step":step,"token_ids":ids})
                if len(buffered)>=chunk or step==case.outputs-1:
                    frame={"steps":list(buffered)}
                    self.wfile.write((json.dumps(frame)+"\n").encode())
                    self.wfile.flush()
                    buffered.clear()
            try:
                result=engine.run(case,emit=emit)
                if result["status"]!="complete":
                    raise RuntimeError(result)
                result["request_id"]=request["request_id"]
                result["chunk"]=chunk
                server_log.write(json.dumps(result)+"\n");server_log.flush()
            except Exception as exc:
                self.wfile.write((json.dumps({"error":repr(exc)})+"\n").encode());self.wfile.flush()
                server_log.write(json.dumps({"request_id":request["request_id"],"error":repr(exc)})+"\n")
                server_log.flush()
            self.close_connection=True

    server=HTTPServer(("127.0.0.1",0),Handler)
    thread=threading.Thread(target=server.serve_forever,daemon=True)
    thread.start()
    started=time.perf_counter()
    try:
        subprocess.run([sys.executable,"-m","experiments.exp_performance_probe_http",
            "--client-port",str(server.server_port),"--client-output",str(args.output/"client.jsonl"),
            "--session",str(args.session)],check=True)
    finally:
        server.shutdown();server.server_close();thread.join();server_log.close()
    info["elapsed_seconds"]=time.perf_counter()-started
    info["client_sha256"]=file_hash(args.output/"client.jsonl")
    info["server_sha256"]=file_hash(args.output/"server.jsonl")
    (args.output/"manifest.json").write_text(json.dumps(info,indent=2))


if __name__=="__main__":
    main()
