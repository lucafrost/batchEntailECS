from OpenSearchQueries import knnClaimsByTopicExclNFS, nonEntailedClaimsQuery, randomNonEntailedClaimsQuery
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from torch.utils.data import Dataset, DataLoader
from opensearchpy import OpenSearch, helpers
# from IPython.display import clear_output
from datetime import datetime
from pprint import pprint
from wasabi import msg
from tqdm import tqdm
import numpy as np
import logging
import random
import torch
import boto3
import json
import time
import ast

# connect to OpenSearch cluster
shh = boto3.client("secretsmanager", region_name="eu-west-2")
creds = shh.get_secret_value(SecretId='dev/OpenSearch/lucaRW')['SecretString']
usr, pwd = ast.literal_eval(creds).values()
client = OpenSearch("https://opensearch.whisp.dev",
                    http_auth=(usr, pwd),
                    use_ssl=True, timeout=120)

# init model, tokenizer
tokenizer = AutoTokenizer.from_pretrained("ynie/roberta-large-snli_mnli_fever_anli_R1_R2_R3-nli", use_fast=True)
model = AutoModelForSequenceClassification.from_pretrained("ynie/roberta-large-snli_mnli_fever_anli_R1_R2_R3-nli")
device = "cuda:0" if torch.cuda.is_available() else "cpu"
model = model.to(device)

def batch_generator(premises, hypotheses, batch_size, max_length):
    for i in range(0, len(hypotheses), batch_size):
        yield tokenizer(premises[i:i+batch_size], hypotheses[i:i+batch_size], max_length=max_length, return_token_type_ids=True, truncation=True, padding="longest", return_tensors="pt").to(device)
    
def batch_inference(model, encoded_batches):
    for batch in encoded_batches:
        with torch.no_grad():
            outputs = model(**batch)
            yield torch.softmax(outputs[0], dim=1).tolist()

# MAIN FUNCTION
def bulkEntail(num_docs, sample_size):
    max_length = 256
    batch_size = 32
    
    timing = {
        "makePayload": 0,
        "doInf": 0,
        "formatResult": 0,
        "updateDocs": 0,
        "total": 0
    }
    
    start = time.time()
    # get premise claims using randomNonEntailedClaimsQuery
    premiseClaims = client.search(index="wpc2", body=randomNonEntailedClaimsQuery(), size=num_docs)["hits"]["hits"]
    
    premises = []
    hypotheses = []
    docIds = []
    
    queries = []
    for p in tqdm(premiseClaims): # for each premise claim
        # get topic of the claim's parent article
        topic = client.get(index="wpc2", id=p["_routing"])["_source"]["topic"]
        vector = p["_source"]["vector"]
        # add search query to the list of queries
        queries.append({"index": "wpc2"})
        queries.append(knnClaimsByTopicExclNFS(topic, vector, size=250))

    # use the msearch API to send all queries in a single request
    response = client.msearch(body='\n'.join([json.dumps(q) for q in queries]), request_timeout=120)

    # process the response
    for idx, hit in enumerate(response["responses"]):
        hypClaims = hit["hits"]["hits"]
        hypClaims = [doc for doc in hypClaims if doc["_score"] <= 1.95]
        if len(hypClaims) > sample_size:
            scores = [d["_score"] for d in hypClaims]
            score_counts, score_bins = np.histogram(scores, bins=20)
            # weights are inverse to the distribution of scores
            score_weights = 1 / score_counts[score_bins.searchsorted(scores, side='left') - 1]
            score_weights /= np.sum(score_weights)
            # use weighted scores to get a sample of 50 claims
            sampledDocs = random.choices(hypClaims, weights=score_weights, k=50)
            for d in sampledDocs: # append all to list
                premises.append(p["_source"]["cleanTriple"])
                hypotheses.append(d["_source"]["cleanTriple"])
                docIds.append([p["_id"], d["_id"]])
        else: # if returned kNN hypotheses length < 50
            for hc in hypClaims:
                premises.append(p["_source"]["cleanTriple"])
                hypotheses.append(hc["_source"]["cleanTriple"])
                docIds.append([p["_id"], hc["_id"]])
    end = time.time()
    print(f"[1.] Queries + make payload took {end-start} secs")
    timing["makePayload"] = end-start
    msg.info(f"Created inference payload...")
    
    # do inference
    start = time.time()
    encoded_batches = batch_generator(premises,
                                      hypotheses,
                                      batch_size, 
                                      max_length)
    results = batch_inference(model, encoded_batches)
    flatResults = [item for sublist in results for item in sublist] # flatten results from batches
    msg.good(f"inference done!")
    end = time.time()
    print(f"[2.] Inference took {end-start} secs")
    timing["doInf"] = end-start
    
    # process entailment field
    # doc_results dict where KEY is the ID of the premise claim
    # & value is a list of dicts where each dict is the entailment
    # data between premise-hypothesis claims
    # 
    # doc_results[claimId] = [
    #     {
    #         "claimId": "...", # ID of hypothesis
    #         "verdict": "entail" | "neutral" | "contradict"
    #         "classConfidence": {
    #             "entail": 0.000,
    #             "neutral": 0.000,
    #             "contradict": 0.000
    #         }
    #     },
    #     {...}
    # ]
    
    start = time.time()
    doc_results = {}
    for tup in docIds:
        doc1, doc2 = tup
        if doc1 not in doc_results:
            doc_results[doc1] = []
    # Assign results
    for idx, result in enumerate(flatResults):
        doc_id, hyp_id = docIds[idx][0], docIds[idx][1]
        hypText = hypotheses[idx]
        classConf = {
            "entail": result[0],
            "neutral": result[1],
            "contradict": result[2]
        }
        verdict = max(classConf, key=classConf.get)
        if verdict != "neutral":
            doc_results[doc_id].append({
                "claimId": hyp_id,
                "verdict": verdict,
                "claimText": hypText,
                "classConfidence": classConf
            })
    end = time.time()
    print(f"[3.] Output-doc association took {end-start} secs")
    timing["formatResult"] = end-start
    
    # UPDATE DOCS IN OPENSEARCH
    start = time.time()
    bulk_updates = []
    for k, v in doc_results.items():
        query = {
          "query": {
            "ids": {
              "values": [k]
            }
          }
        }
        routing = client.search(index="wpc2", body=query, _source=False)["hits"]["hits"][0]["_routing"]
        agg = []
        for calc in v:
            if calc["verdict"] == "entail":
                agg.append(calc["classConfidence"]["entail"])
            elif calc["verdict"] == "contradict":
                agg.append(-calc["classConfidence"]["contradict"])
        bulk_updates.append({ # add the update data to `bulk_updates`
            "_op_type": "update",
            "_index": "wpc2",
            "_id": k,
            "_routing": routing,
            "doc": {
                "entailStatus": "aggV1-jupyterTest",
                "consensus": {
                    "aggregate": sum(agg)/len(agg),
                    "allCalcs": v
                }
            }
        })
    # send bulk request
    response = helpers.bulk(client, bulk_updates)
    end = time.time()
    timing["updateDocs"] = end-start
    
    timing["total"] = sum(list(timing.values()))
    timing["tpc"] = timing["total"]/num_docs
                          
    # delete fields we ain't using anymore for CUDA memory
    del results, flatResults, premises, hypotheses, premiseClaims, hypClaims, doc_results, response, queries
    return timing

if __name__ == "__main__":
    # disable INFO logging
    logger = logging.getLogger()
    logger.setLevel(logging.CRITICAL)
    
    docsAvl = 1
    SIZE = 150
    n_proc = 0
    while docsAvl > 0:
        docsAvl = client.count(index="wpc2", body=nonEntailedClaimsQuery())["count"]
        # call the bulkEntail func from above
        timing = bulkEntail(SIZE, 50)
        print("##### ITERATION COMPLETE #####")
        pprint(timing)