def nonEntailedClaimsQuery():
    """
    Query to return all claims (child docs)
    without the `entailStatus` field.
    """
    return {
      "query": {
        "bool": {
          "must_not": [
            {
              "match": {
                "entailStatus": "V2-BugSquash"
              }
            },
              {
                  "term": {
                    "claimworthy.label.keyword": "Non-Factual Statement (NFS)"
                  }
                }
          ],
          "must": [
            {
              "match": {
                "docRelations": "claim"
              }
            }
          ]
        }
      }
    }

def randomNonEntailedClaimsQuery():
    """
    Query to return all claims (child docs)
    without the `entailStatus` field.
    
    Random Scoring
    """
    return {
      "query": {
        "function_score": {
          "query": {
            "bool": {
              "must": [
                {
                  "match": {
                    "docRelations": "claim"
                  }
                }
              ],
              "must_not": [
                {
                  "match": {
                    "entailStatus": "V2-BugSquash"
                  }
                },
                {
                  "term": {
                    "claimworthy.label.keyword": {
                      "value": "Non-Factual Statement (NFS)"
                    }
                  }
                }
              ]
            }
          },
          "functions": [
            {
              "random_score": {}
            }
          ]
        }
      }
    }


def knnClaimsByTopic(topic, vector, size=10):
    """
    Query to perform exact k-NN matching on
    claims in OpenSearch based on a specified
    topic.
    
    Args:
        topic (str)
        vector (list)
    """
    return {
      "size": size,
      "query": {
        "script_score": {
          "query": {
            "has_parent": {
              "parent_type": "article",
              "query": {
                "match": {
                  "topic": topic
                }
              }
            }
          },
          "script": {
            "source": "knn_score",
            "lang": "knn",
            "params": {
              "field": "vector",
              "query_value": vector,
              "space_type": "cosinesimil"
            }
          }
        }
      }
    }
    
def knnClaimsByTopicExclNFS(topic, vector, size=10):
    """
    Query to perform exact k-NN matching on
    claims in OpenSearch based on a specified
    topic. This query EXCLUDES claims marked
    as Non-Factual by ClaimBuster.
    
    Args:
        topic (str)
        vector (list)
    """
    return {
      "size": size,
      "min_score": 1.5,
      "query": {
        "script_score": {
          "query": {
            "bool": {
              "must": [
                {
                  "has_parent": {
                    "parent_type": "article",
                    "query": {
                      "match": {
                        "topic": topic
                      }
                    }
                  }
                }
              ],
              "must_not": [
                {
                  "term": {
                    "claimworthy.label.keyword": "Non-Factual Statement (NFS)"
                  }
                }
              ]
            }
          },
          "script": {
            "source": "knn_score",
            "lang": "knn",
            "params": {
              "field": "vector",
              "query_value": vector,
              "space_type": "cosinesimil"
            }
          }
        }
      }
    }