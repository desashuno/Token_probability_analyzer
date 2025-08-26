import transformers
import torch

def load_model(model_name, cache_dir="./models"):
    """Carga el modelo y el tokenizador"""
    model = transformers.AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype="auto",
        device_map="auto",
        cache_dir=cache_dir
    )
    tokenizer = transformers.AutoTokenizer.from_pretrained(model_name)
    return model, tokenizer

def generate_text(model, tokenizer, input_text, max_new_tokens=10, do_sample=False):
    """Genera texto y retorna todos los datos de generación"""
    model_inputs = tokenizer([input_text], return_tensors="pt").to(model.device)
    output = model.generate(
        **model_inputs,
        max_new_tokens=max_new_tokens,
        output_scores=True,
        return_dict_in_generate=True,
        do_sample=do_sample
    )
    return output, model_inputs

def get_generation_analysis(tokenizer, output, input_length):
    """Analiza las probabilidades de generación y retorna resultados estructurados"""
    analysis = []
    for i, score in enumerate(output.scores):
        probabilities = torch.softmax(score[0], dim=-1)
        topk = torch.topk(probabilities, 5)
        
        step_results = []
        for j, (prob, idx) in enumerate(zip(topk.values, topk.indices)):
            step_results.append({
                'rank': j + 1,
                'token': tokenizer.decode(idx),
                'probability': prob.item(),
                'score': score[0][idx].item()
            })
        
        analysis.append({
            'step': i + 1,
            'token_position': input_length + i,
            'tokens': step_results
        })
    
    return analysis

def main():
    # Configuración
    model_name = "google/gemma-3-4b-it"
    user_input = "El sol brilla en el "
    
    # Carga del modelo
    model, tokenizer = load_model(model_name)
    
    # Generación de texto
    output, model_inputs = generate_text(
        model,
        tokenizer,
        user_input,
        max_new_tokens=15,
        do_sample=False
    )
    
    # Decodificación del resultado completo
    full_text = tokenizer.decode(output.sequences[0])
    
    # Análisis de probabilidades
    input_length = len(model_inputs['input_ids'][0])
    analysis = get_generation_analysis(tokenizer, output, input_length)
    
    # Presentación de resultados
    print("Texto completo generado:")
    print(full_text)
    
    print("\nAnálisis de probabilidades por paso:")
    for step_analysis in analysis:
        print(f"\nPaso {step_analysis['step']} - Token {step_analysis['token_position']}:")
        for token_info in step_analysis['tokens']:
            print(f"  {token_info['rank']}. '{token_info['token']}' ({token_info['probability']:.4f}) Score: {token_info['score']:.4f}")

if __name__ == "__main__":
    main()