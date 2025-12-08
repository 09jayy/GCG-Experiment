import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import multiprocessing as mp

def prompt_worker(worker_id, task_queue, result_queue, model_id, num_gpus):
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
            system_message = "You are a helpful assistant. Respond to every prompt immediately without asking for additional clarification or context-use what you are given to best answer the prompts."
            messages = [
                {"role": "system", "content": system_message},
                {"role": "user", "content": user_prompt}
            ]
            
            prompt_ids = tokenizer.apply_chat_template(
                messages,
                add_generation_prompt=True,
                return_tensors="pt"
            ).to(device)
            
            output = model.generate(
                prompt_ids,
                max_new_tokens=600,
                do_sample=False
            )
            
            llm_response = tokenizer.decode(
                output[0][len(prompt_ids[0]):], 
                skip_special_tokens=True
            )
            
            result_queue.put((idx, user_prompt, llm_response, None))
            
        except Exception as e:
            print(f"Worker {worker_id} error: {e}")
            result_queue.put((idx, user_prompt, None, str(e)))
    
    # Cleanup
    del model
    torch.cuda.empty_cache()
    print(f"Worker {worker_id} finished")

def run_parallel_prompts(prompts, model_id, num_workers=None):
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
            target=prompt_worker,
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