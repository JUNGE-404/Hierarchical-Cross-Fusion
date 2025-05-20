import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import argparse
from typing import Any
import mteb
import json
import torch
from datasets import DatasetDict

import numpy as np

from model.model_main import MyEmbeddingModel

class ModelWrapper:
    def __init__(self, model, task_to_instructions):

        self.task_to_instructions = task_to_instructions
        self.model = model

    def encode(
        self,
        sentences: list[str],
        *,
        prompt_name: str = None,
        **kwargs: Any,  # noqa
    ) -> np.ndarray:
        if prompt_name is not None:
            instruction = (
                self.task_to_instructions[prompt_name]
                if self.task_to_instructions
                and prompt_name in self.task_to_instructions
                else None
            )
        else:
            instruction = ""

        sentences = [[instruction, sentence] for sentence in sentences]
        return self.model.encode(sentences)

    def encode_corpus(
        self,
        corpus: list[dict[str, str]] | dict[str, list[str]] | list[str],
        prompt_name: str = None,
        **kwargs: Any,
    ) -> np.ndarray:
        sentences = [["", sentence] for sentence in sentences]
        if "request_qid" in kwargs:
            kwargs.pop("request_qid")
        return self.model.encode(sentences)

    def encode_queries(self, queries: list[str], **kwargs: Any) -> np.ndarray:
        return self.encode(queries)

def subsample_retrieval_task(task, max_samples):
    split = "test"
    query_doc_pairs = []
    for query_id in task.queries.get(split, {}):
        relevant_docs = task.relevant_docs.get(split, {}).get(query_id, {})
        for doc_id, score in relevant_docs.items():
            if score >= 1:
                query_doc_pairs.append((query_id, doc_id))
                if len(query_doc_pairs) >= max_samples:
                    break
        if len(query_doc_pairs) >= max_samples:
            break

    sampled_pairs = query_doc_pairs[:max_samples]
    sampled_corpus = {split: {}}
    sampled_queries = {split: {}}
    sampled_relevant_docs = {split: {}}
    
    selected_docs = set()
    selected_queries = set()
    for q_id, d_id in sampled_pairs:
        selected_queries.add(q_id)
        selected_docs.add(d_id)
    
    for doc_id in selected_docs:
        if doc_id in task.corpus.get(split, {}):
            sampled_corpus[split][doc_id] = task.corpus[split][doc_id]
    
    for q_id in selected_queries:
        if q_id in task.queries.get(split, {}):
            sampled_queries[split][q_id] = task.queries[split][q_id]
    
    for q_id, d_id in sampled_pairs:
        if q_id not in sampled_relevant_docs[split]:
            sampled_relevant_docs[split][q_id] = {}
        sampled_relevant_docs[split][q_id][d_id] = task.relevant_docs[split][q_id][d_id]
    
    task.corpus = sampled_corpus
    task.queries = sampled_queries
    task.relevant_docs = sampled_relevant_docs

    return task

def universal_subsample(task, max_samples):
    if hasattr(task, 'queries') and hasattr(task, 'corpus'):
        return subsample_retrieval_task(task, max_samples)
    elif isinstance(task.dataset, (DatasetDict, dict)):
        samples = min(max_samples, len(task.dataset["test"]))
        task.dataset["test"] = task.dataset["test"].select(range(samples))
        return task
    elif hasattr(task.dataset, "take"):
        samples = min(max_samples, len(task.dataset["test"]))
        task.dataset["test"] = task.dataset["test"].take(samples)
        return task
    else:
        raise ValueError(f"Unsupported task type: {type(task)}")

if __name__ == "__main__":
    model_name = "Llama-3.2-1B"
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        default="/data/hf/meta-llama/Llama-3.2-1B",
    )
    parser.add_argument(
        "--checkpoint_name_or_path",
        type=str,
        default=f"output/{model_name}/hcf/checkpoint-2000",
    )
    parser.add_argument(
        "--enable_lora",
        type=bool,
        default=False,
    )
    parser.add_argument(
        "--enable_bidirectional",
        type=bool,
        default=True,
    )
    parser.add_argument(
        "--task_name", 
        type=str,
        nargs="+",
        default=["ArguAna", "FEVER", "DBPedia", "HotpotQA", "NFCorpus", "NQ", "SciFact", "Touche2020"],
    )
    parser.add_argument(
        "--task_to_instructions_fp",
        type=str,
        default="test_configs/mteb/task_to_instructions.json",
    )
    parser.add_argument(
        "--output_dir", 
        type=str,
        default=f"results/{model_name}/hcf/checkpoint-2000",
    )

    args = parser.parse_args()

    task_to_instructions = None
    if args.task_to_instructions_fp is not None:
        with open(args.task_to_instructions_fp, "r") as f:
            task_to_instructions = json.load(f)

    embeding_model = MyEmbeddingModel.from_pretrained(
        model_name_or_path=args.model_name_or_path,
        checkpoint_name_or_path=args.checkpoint_name_or_path,
        enable_bidirectional=args.enable_bidirectional,
        enable_lora=args.enable_lora,
        device_map="cuda" if torch.cuda.is_available() else "cpu",
        torch_dtype=torch.bfloat16,
    )

    model = ModelWrapper(model=embeding_model, task_to_instructions=task_to_instructions)
    tasks = mteb.get_tasks(tasks=args.task_name)
    evaluation = mteb.MTEB(tasks=tasks)

    # sample data
    max_samples = 20000
    for i in range(len(evaluation.tasks)):
        evaluation.tasks[i].load_data()
        evaluation.tasks[i] = universal_subsample(evaluation.tasks[i], max_samples)

    results = evaluation.run(model, output_folder=args.output_dir, eval_splits=["test"])
