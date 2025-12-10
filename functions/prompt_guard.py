from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import multiprocessing as mp

def classify_text_safety(text: str,model_id="meta-llama/Llama-Guard-3-1B") -> tuple[str, str]:
    """
    classify safety of a text string (not using parallel processing)

    returns ('safe',x) | ('unsafe',x)
    where x is the LLM output before parsing to extract safety classification  
    """
    
    # load model and tokenizer
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    tokenizer = AutoTokenizer.from_pretrained(model_id)

    # create conversation from prompt
    conversation = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": text}
            ]
        }
    ]

    # create input
    input_ids = tokenizer.apply_chat_template(
        conversation, return_tensors="pt"
    ).to(model.device)

    # get output
    prompt_len = input_ids.shape[1]
    output = model.generate(
        input_ids,
        max_new_tokens=50,
        pad_token_id=0,
    )

    # extract new tokens from guard
    generated_tokens = output[:, prompt_len:]
    out = tokenizer.decode(generated_tokens[0]).lstrip()

    return "safe",out if out == "safe<|eot_id|>" else "unsafe", out

# MULTI-PROCESSING FUNCTIONS

def safety_worker(worker_id, task_queue, result_queue, model_id, num_gpus):
    """Worker process that runs prompts on assigned GPU"""
    device_id = worker_id % num_gpus
    device = f"cuda:{device_id}"
    
    # Load model on this worker's GPU
    model = AutoModelForCausalLM.from_pretrained(
        model_id, 
        torch_dtype=torch.float16
    ).to(device)
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    
    print(f"Worker {worker_id} loaded on {device}")
    
    while True:
        task = task_queue.get()
        if task is None:  # Stop signal
            break
        
        idx, user_prompt = task
        
        try:
            # create conversation input
            conversation = [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": user_prompt}
                    ]
                }
            ]
            # create input
            input_ids = tokenizer.apply_chat_template(
                conversation, return_tensors="pt"
            ).to(model.device)

            # get output
            prompt_len = input_ids.shape[1]
            output = model.generate(
                input_ids,
                max_new_tokens=50,
                pad_token_id=0,
            )

            # extract new tokens from guard
            generated_tokens = output[:, prompt_len:]
            out = tokenizer.decode(generated_tokens[0]).lstrip()
            safety_class = "safe" if out == "safe<|eot_id|>" else "unsafe"
            result_queue.put((idx, user_prompt, safety_class, None))
        except Exception as e:
            print(f"Worker {worker_id} error: {e}")
            result_queue.put((idx, user_prompt, None, str(e)))
    
    # Cleanup
    del model
    torch.cuda.empty_cache()
    print(f"Worker {worker_id} finished")

def run_parallel_safety(prompts: list[str], model_id="meta-llama/Llama-Guard-3-1B", num_workers=None) -> list[tuple[str,str,str]]:
    """Run prompts in parallel using multiprocessing"""
    if num_workers is None:
        num_workers = min(torch.cuda.device_count(), len(prompts))
    
    num_gpus = torch.cuda.device_count()
    print(f"Using {num_workers} workers across {num_gpus} GPUs")
    print(f"Processing {len(prompts)} prompts")
    
    # Use 'spawn' context for better compatibility with Jupyter and CUDA
    ctx = mp.get_context('spawn')
    
    # Create queues
    task_queue = ctx.Queue()
    result_queue = ctx.Queue()
    
    # Start worker processes
    processes = []
    for i in range(num_workers):
        p = ctx.Process(
            target=safety_worker,
            args=(i, task_queue, result_queue, model_id, num_gpus)
        )
        p.start()
        processes.append(p)
    
    # Add tasks to queue
    for idx, prompt in enumerate(prompts):
        task_queue.put((idx, prompt))
    
    # Add stop signals
    for _ in range(num_workers):
        task_queue.put(None)
    
    # Collect results
    results = []
    for i in range(len(prompts)):
        result = result_queue.get()
        results.append(result)
        print(f"Collected result {i+1}/{len(prompts)}")
    
    # Wait for all processes to finish
    for p in processes:
        p.join()
    
    # Sort results by original index
    results.sort(key=lambda x: x[0])
    
    return [(prompt, response, error) for _, prompt, response, error in results]