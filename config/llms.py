
# Google Gemini Models
def get_google_models(modality: str) -> dict:

    google_text_models = {
        "Gemini 1.5 Flash" : "gemini-1.5-flash",
        "Gemini 1.5 Pro" : "gemini-1.5-pro",
    }
    
    google_imaging_models = {
        "Gemini 2.0 Flash Preview Image Generation" : "gemini-2.0-flash-preview-image-generation",
    }

    if modality == "text":
        return google_text_models
    elif modality == "image":
        return google_imaging_models
    else:
        raise ValueError(f"Unknown modality: {modality}")


# Ollama Models
def get_ollama_models() -> dict:
    
    ollama_models: dict[str] = {}

    def available_models() -> list:
        import ollama
        ollama_client = ollama.Client()
        llm = ollama_client.list()
        model_list = [i['model'] for i in llm['models']]
        return model_list

    try:
        installed_ollama_model: list[str] = available_models()
    except Exception as e:
        raise RuntimeError(e)
    
    for model in installed_ollama_model:
        model_name: str = model.split(":")[0]
        version: str = model.split(":")[1]
        model_view_name: str = model_name.title() + f" ({version})" if version == "latest" else model_name.title() + f" ({version} Parameters)"
        ollama_models[model_view_name] = model
    
    return ollama_models